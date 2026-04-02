"""
Survey Builder Agent — Streamlit Web App
Converts study reference docs into fully deployed Typeform surveys.
"""

import json
import re
import textwrap

import anthropic
import requests
import streamlit as st

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Survey Builder", page_icon="📋", layout="wide")

# ─── Load secrets ─────────────────────────────────────────────────────────────
if "ANTHROPIC_API_KEY" not in st.secrets or "TYPEFORM_API_TOKEN" not in st.secrets:
    st.error("Missing API keys. Go to your app settings on Streamlit Cloud:")
    st.markdown("""
1. Click **Manage app** (bottom-right corner)
2. Click **Settings** → **Secrets**
3. Paste the following (with your real keys):

```toml
ANTHROPIC_API_KEY = "sk-ant-your-key-here"
TYPEFORM_API_TOKEN = "tfp_your-typeform-token-here"
```

4. Click **Save** — the app will reboot automatically.
""")
    st.stop()

ANTHROPIC_API_KEY = st.secrets["ANTHROPIC_API_KEY"]
TYPEFORM_API_TOKEN = st.secrets["TYPEFORM_API_TOKEN"]

TYPEFORM_BASE = "https://api.typeform.com"
TYPEFORM_HEADERS = {
    "Authorization": f"Bearer {TYPEFORM_API_TOKEN}",
    "Content-Type": "application/json",
}
WORKSPACE_URL_PREFIX = "https://617tetg12vx.typeform.com/to/"

# ─── System prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""\
You are a Typeform survey builder agent. You receive a study reference document
and you output a COMPLETE, VALID JSON object that can be POSTed directly to the
Typeform Create Form API (POST https://api.typeform.com/forms).

Follow EVERY rule below with zero exceptions.

===============================================================================
RULE 1: ENDINGS ARE ALWAYS url_redirect
===============================================================================

Never use thankyou_screen type. Spectrum requires auto-redirects.

The title field on each ending must match its purpose:
- ending_a_quality  -> "title": "Quality"
- ending_b_terminate -> "title": "Terminate"
- ending_c_complete -> "title": "Complete"

CORRECT format:
{
    "ref": "ending_b_terminate",
    "title": "Terminate",
    "type": "url_redirect",
    "properties": {
        "redirect_url": "https://spectrumsurveys.com/surveydone?st=18&transaction_id={{hidden:transaction_id}}"
    }
}

NEVER use "Redirecting..." or "Thank you!" as the title.
NEVER use button_mode, button_text, show_button, or share_icons in endings.

===============================================================================
RULE 2: EVERY SURVEY GETS EXACTLY THREE ENDINGS
===============================================================================

| Ending       | Ref                  | Status Code | When to Use                           |
|-------------|----------------------|-------------|---------------------------------------|
| A - Quality  | ending_a_quality     | st=20       | Trap failures, speeders, nonsense     |
| B - Terminate| ending_b_terminate   | st=18       | Screened out (didn't qualify)         |
| C - Complete | ending_c_complete    | st=21       | Successfully finished the survey      |

Redirect URLs:
  A: https://spectrumsurveys.com/surveydone?st=20&transaction_id={{hidden:transaction_id}}
  B: https://spectrumsurveys.com/surveydone?st=18&transaction_id={{hidden:transaction_id}}
  C: https://spectrumsurveys.com/surveydone?st=21&transaction_id={{hidden:transaction_id}}&ps_hash=XXXX

===============================================================================
RULE 3: ALWAYS INCLUDE THE HIDDEN FIELD
===============================================================================

"hidden": ["transaction_id"]

Every redirect URL must include transaction_id={{hidden:transaction_id}}.

===============================================================================
RULE 4: ENDING C ALWAYS NEEDS ps_hash=XXXX
===============================================================================

Use XXXX as a placeholder. The user replaces it with the real hash from Spectrum.

===============================================================================
RULE 5: THE LAST SURVEY QUESTION ALWAYS JUMPS TO ending_c_complete
===============================================================================

The very last question in the main survey flow must have an "always" logic rule
pointing to ending_c_complete.

===============================================================================
RULE 6: "always" MUST BE THE LAST ACTION
===============================================================================

Logic actions are evaluated top-to-bottom. First match wins.
Put specific conditions BEFORE "always" fallbacks.

===============================================================================
RULE 7: CONDITIONAL QUESTIONS NEED BOTH A SKIP AND A REJOIN
===============================================================================

If a question only appears under certain conditions:
1. On the trigger question: matching answer -> follow-up, other -> skip past it
2. On the follow-up question: always -> rejoin the main flow

===============================================================================
RULE 8: CROSS-QUESTION ROUTING REFERENCES THE EARLIER FIELD
===============================================================================

When routing at Question X based on answer to earlier Question Y,
the vars must reference Question Y's field and choices - not Question X.

===============================================================================
RULE 9: SCREENER TERMINATE PATTERN
===============================================================================

For screener questions where certain answers disqualify, use the allow-list:
- List every passing choice explicitly -> next question
- Final "always" action -> ending_b_terminate

===============================================================================
RULE 10: SECTIONS ARE LABELS, NOT SCREENS
===============================================================================

CRITICAL: When a study reference has headings like "Section 1: Eligibility",
those are INTERNAL LABELS ONLY. NEVER create statement fields for section headings.
The respondent flows directly from one question to the next.

Only use statement fields when the study explicitly calls for an intro screen,
a definition block, or a transition message that the respondent needs to read.

===============================================================================
FIELD TYPES
===============================================================================

| Type             | Use                    | Key Properties                                    |
|-----------------|------------------------|---------------------------------------------------|
| multiple_choice  | Single or multi select | allow_multiple_selection, allow_other_choice, randomize |
| long_text        | Open-ended text box    | -                                                 |
| short_text       | One-line text input    | -                                                 |
| statement        | Info/transition screen | description, button_text, hide_marks: true        |
| yes_no           | DO NOT USE - use multiple_choice with Yes/No choices instead | -                |
| opinion_scale    | Number scale (1-5,1-10)| start_at_one, steps, labels: {left, right}        |
| rating           | Star rating            | steps                                             |
| email            | Email with validation  | -                                                 |
| date             | Date picker            | -                                                 |
| dropdown         | Dropdown select        | same as multiple_choice                           |

===============================================================================
REF NAMING CONVENTION
===============================================================================

- Questions: sectionprefix_descriptor -> s1_age, a2_why_chose
- Choices: questionprefix_shortlabel -> s1_under18, s4_lilly, b6_yes
- Endings: always ending_a_quality, ending_b_terminate, ending_c_complete
- Statements (rare): stmt_intro, stmt_definition - only when the respondent
  needs to read something, NEVER for section headings

===============================================================================
FORM PAYLOAD SKELETON
===============================================================================

{
    "title": "SURVEY_TITLE",
    "settings": {
        "language": "en",
        "progress_bar": "percentage",
        "show_progress_bar": true,
        "is_public": true,
        "meta": {"title": "SURVEY_TITLE", "description": "..."}
    },
    "hidden": ["transaction_id"],
    "welcome_screens": [{
        "ref": "welcome",
        "title": "...",
        "properties": {"description": "...", "show_button": true, "button_text": "Start"}
    }],
    "thankyou_screens": [
        {"ref": "ending_a_quality",   "title": "Quality",   "type": "url_redirect", "properties": {"redirect_url": "https://spectrumsurveys.com/surveydone?st=20&transaction_id={{hidden:transaction_id}}"}},
        {"ref": "ending_b_terminate", "title": "Terminate", "type": "url_redirect", "properties": {"redirect_url": "https://spectrumsurveys.com/surveydone?st=18&transaction_id={{hidden:transaction_id}}"}},
        {"ref": "ending_c_complete",  "title": "Complete",  "type": "url_redirect", "properties": {"redirect_url": "https://spectrumsurveys.com/surveydone?st=21&transaction_id={{hidden:transaction_id}}&ps_hash=XXXX"}}
    ],
    "fields": [],
    "logic": []
}

===============================================================================
STUDY REFERENCE MAPPING
===============================================================================

When you receive a study reference, map these sections:
1. Terminate Summary / [Terminate] markers -> screener logic -> ending_b_terminate
2. Conditional Logic / branching -> skip/show rules with rejoin jumps
3. Question list with types -> fields array
4. Last question in the main survey -> always jump to ending_c_complete
5. [Retro only] markers -> these are NOT terminates, they pass through normally

===============================================================================
OUTPUT FORMAT
===============================================================================

You MUST output ONLY a single valid JSON object - the complete Typeform form
payload. No explanation, no markdown, no code fences, no commentary.
Just the raw JSON object starting with { and ending with }.

All fields must have "validations": {"required": true}.

Double-check:
- All logic jump targets reference valid field refs or ending refs
- "always" is always the LAST action in any logic rule
- Cross-question routing vars reference the EARLIER field
- No statement fields for section headings
- Ending titles are "Quality", "Terminate", "Complete" (not "Redirecting...")
- Every redirect URL includes transaction_id={{hidden:transaction_id}}
- NEVER use yes_no field type - always use multiple_choice with Yes/No choices
- Logic conditions always use "op": "is" with vars referencing the field and choice
""")


# ─── Helper functions ─────────────────────────────────────────────────────────

def fix_conditions(payload):
    """Fix condition format to match Typeform's expected structure."""
    for rule in payload.get("logic", []):
        for action in rule.get("actions", []):
            cond = action.get("condition", {})
            if cond.get("op") == "always" and "vars" not in cond:
                cond["vars"] = []
            if cond.get("op") == "is" and "value" in cond:
                choice = cond.pop("value")
                if "vars" not in cond:
                    cond["vars"] = []
                cond["vars"].append(choice)
            if cond.get("op") == "is" and "values" in cond:
                values = cond.pop("values")
                if "vars" not in cond:
                    cond["vars"] = []
                if isinstance(values, list):
                    cond["vars"].extend(values)
                else:
                    cond["vars"].append(values)
    return payload


def extract_json(text):
    """Extract JSON from Claude's response."""
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in response")
    return json.loads(text[start : end + 1])


def validate_payload(payload):
    """Validate the payload against all rules. Returns list of errors."""
    errors = []

    if "hidden" not in payload or "transaction_id" not in payload.get("hidden", []):
        errors.append("Missing hidden field: transaction_id")

    endings = {ts.get("ref"): ts for ts in payload.get("thankyou_screens", [])}
    for ref, expected_title in [
        ("ending_a_quality", "Quality"),
        ("ending_b_terminate", "Terminate"),
        ("ending_c_complete", "Complete"),
    ]:
        if ref not in endings:
            errors.append(f"Missing ending: {ref}")
        else:
            ts = endings[ref]
            if ts.get("type") != "url_redirect":
                errors.append(f"{ref} type is '{ts.get('type')}', should be 'url_redirect'")
            if ts.get("title") != expected_title:
                errors.append(f"{ref} title is '{ts.get('title')}', should be '{expected_title}'")
            url = ts.get("properties", {}).get("redirect_url", "")
            if "transaction_id={{hidden:transaction_id}}" not in url:
                errors.append(f"{ref} redirect_url missing transaction_id placeholder")

    for field in payload.get("fields", []):
        if field.get("type") == "statement":
            title = field.get("title", "").lower()
            if any(title.startswith(f"section {i}") for i in range(1, 10)):
                errors.append(f"Rule 10 violation: statement '{field.get('ref')}' is a section heading")
        if field.get("type") == "yes_no":
            errors.append(f"yes_no field type used on '{field.get('ref')}' — should be multiple_choice")

    return errors


def generate_payload(study_doc):
    """Send study doc to Claude and get back a Typeform payload."""
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Build the Typeform payload for this study reference:\n\n{study_doc}",
            }
        ],
    )
    raw = response.content[0].text
    payload = extract_json(raw)
    payload = fix_conditions(payload)
    return payload


def deploy_to_typeform(payload):
    """POST the payload to Typeform."""
    r = requests.post(f"{TYPEFORM_BASE}/forms", headers=TYPEFORM_HEADERS, json=payload)
    if r.status_code not in (200, 201):
        return None, r.status_code, r.text
    result = r.json()
    form_id = result.get("id", "unknown")
    form_url = result.get("_links", {}).get("display", f"{WORKSPACE_URL_PREFIX}{form_id}")
    return (form_id, form_url), r.status_code, None


def verify_form(form_id):
    """GET the form and verify endings."""
    r = requests.get(f"{TYPEFORM_BASE}/forms/{form_id}", headers=TYPEFORM_HEADERS)
    if r.status_code != 200:
        return None, []
    form = r.json()
    results = []
    for ts in form.get("thankyou_screens", []):
        ref = ts.get("ref", "?")
        ttype = ts.get("type", "?")
        url = ts.get("properties", {}).get("redirect_url", "NONE")
        status = "OK" if ttype == "url_redirect" else "FAIL"
        if ref == "default_tys":
            status = "SKIP"
        results.append({"ref": ref, "type": ttype, "url": url, "status": status})

    for field in form.get("fields", []):
        if field.get("type") == "statement":
            title = field.get("title", "").lower()
            if any(title.startswith(f"section {i}") for i in range(1, 10)):
                results.append({"ref": field.get("ref"), "type": "statement", "url": "N/A", "status": "FAIL — section heading"})

    return form, results


# ─── UI ───────────────────────────────────────────────────────────────────────

st.title("Survey Builder Agent")
st.caption("Paste a study reference doc and deploy it to Typeform in one click.")

# Initialize session state
if "payload" not in st.session_state:
    st.session_state.payload = None
if "deployed" not in st.session_state:
    st.session_state.deployed = None

# Step 1: Input
st.header("1. Paste Study Reference")
study_doc = st.text_area(
    "Study reference document",
    height=400,
    placeholder="Paste your full study reference here...\n\nSection 1: Eligibility\n\nAre you a U.S. resident?\nYes\nNo [Terminate]\n\n..."
)

# Step 2: Generate
if st.button("Generate Survey Payload", type="primary", disabled=not study_doc.strip()):
    with st.spinner("Sending to Claude... (30-60 seconds for complex surveys)"):
        try:
            payload = generate_payload(study_doc)
            st.session_state.payload = payload
            st.session_state.deployed = None
        except Exception as e:
            st.error(f"Failed to generate payload: {e}")
            st.session_state.payload = None

# Step 3: Review
if st.session_state.payload:
    payload = st.session_state.payload
    st.header("2. Review")

    # Validation
    errors = validate_payload(payload)
    if errors:
        st.warning("Validation warnings:")
        for e in errors:
            st.write(f"- {e}")
    else:
        st.success("All validation checks passed.")

    # Summary
    col1, col2, col3 = st.columns(3)
    col1.metric("Questions", len(payload.get("fields", [])))
    col2.metric("Logic Rules", len(payload.get("logic", [])))
    col3.metric("Endings", len(payload.get("thankyou_screens", [])))

    # Question list
    with st.expander("Questions", expanded=True):
        for i, f in enumerate(payload.get("fields", []), 1):
            ftype = f.get("type", "?")
            title = f.get("title", "?")
            st.write(f"**{i}.** `{ftype}` — {title}")

    # Logic rules
    with st.expander("Logic Rules"):
        for rule in payload.get("logic", []):
            ref = rule.get("ref", "?")
            targets = []
            for a in rule.get("actions", []):
                to = a.get("details", {}).get("to", {})
                op = a.get("condition", {}).get("op", "?")
                targets.append(f"{op} -> {to.get('type','?')}:{to.get('value','?')}")
            st.write(f"**{ref}:** {' | '.join(targets)}")

    # Endings
    with st.expander("Endings"):
        for ts in payload.get("thankyou_screens", []):
            ref = ts.get("ref", "?")
            title = ts.get("title", "?")
            ttype = ts.get("type", "?")
            url = ts.get("properties", {}).get("redirect_url", "NONE")
            st.write(f"**{ref}** — title: `{title}` | type: `{ttype}` | url: `{url[:80]}...`")

    # Raw JSON
    with st.expander("Raw JSON Payload"):
        st.json(payload)

    # Step 4: Deploy
    st.header("3. Deploy")
    if st.button("Deploy to Typeform", type="primary"):
        with st.spinner("Deploying..."):
            result, status_code, error = deploy_to_typeform(payload)
            if result:
                form_id, form_url = result
                st.session_state.deployed = {"form_id": form_id, "form_url": form_url}
            else:
                st.error(f"Deployment failed (HTTP {status_code}): {error[:500]}")

# Step 5: Results
if st.session_state.deployed:
    info = st.session_state.deployed
    form_id = info["form_id"]
    form_url = info["form_url"]

    st.header("4. Deployed")
    st.success("Survey is live!")

    st.code(f"""FORM ID:    {form_id}
LIVE URL:   {form_url}
EDIT URL:   https://admin.typeform.com/form/{form_id}/create

SPECTRUM SURVEY LINK (paste into panel):
  {form_url}#transaction_id={{transaction_id}}

REMINDER: Replace XXXX in Ending C with your ps_hash from Spectrum!""")

    # Verify
    with st.spinner("Verifying endings..."):
        form, results = verify_form(form_id)
        if results:
            st.subheader("Ending Verification")
            for r in results:
                icon = "✅" if r["status"] == "OK" else ("⏭️" if r["status"] == "SKIP" else "❌")
                st.write(f"{icon} **{r['ref']}** — type: `{r['type']}` | {r['status']}")
            if all(r["status"] in ("OK", "SKIP") for r in results):
                st.success("All checks passed!")
            else:
                st.warning("Some checks failed — review above.")
