import time
import json
import requests
import streamlit as st

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
DEPLOYMENT_URL = "http://localhost:2024"
ASSISTANT_ID = "agent"


# -------------------------------------------------------------------
# REST helpers
# -------------------------------------------------------------------
def create_thread() -> str:
    resp = requests.post(
        f"{DEPLOYMENT_URL}/threads",
        json={},  # required, otherwise 422
        headers={"Content-Type": "application/json"},
    )
    resp.raise_for_status()
    data = resp.json()
    return data["thread_id"]


def submit_run(thread_id: str, input_data: dict) -> str:
    payload = {
        "assistant_id": ASSISTANT_ID,
        "input": input_data,
        "stream_mode": "updates",
    }
    resp = requests.post(
        f"{DEPLOYMENT_URL}/threads/{thread_id}/runs",
        json=payload,
        headers={"Content-Type": "application/json"},
    )
    resp.raise_for_status()
    data = resp.json()
    return data["run_id"]


def get_run_state(thread_id: str, run_id: str) -> dict:
    resp = requests.get(
        f"{DEPLOYMENT_URL}/threads/{thread_id}/runs/{run_id}",
        headers={"Content-Type": "application/json"},
    )
    resp.raise_for_status()
    return resp.json()


def get_thread_state(thread_id: str) -> dict:
    """Fetch thread; this is where state/final_report lives in your setup."""
    resp = requests.get(
        f"{DEPLOYMENT_URL}/threads/{thread_id}",
        headers={"Content-Type": "application/json"},
    )
    resp.raise_for_status()
    return resp.json()


def _find_key_recursive(obj, key: str):
    """DFS search for key in nested dict/list structure."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                return v
            found = _find_key_recursive(v, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _find_key_recursive(item, key)
            if found is not None:
                return found
    return None


def extract_final_report_from_thread(thread_state: dict) -> str:
    """
    Try to extract `final_report` from the thread object.
    Your State has final_report: CombinedPlan, but it’s persisted by the app,
    so we search the whole thread JSON.
    """
    final = _find_key_recursive(thread_state, "final_report")
    if final is None:
        # fallback: show entire thread state
        final = thread_state

    try:
        return "```json\n" + json.dumps(final, indent=2, default=str) + "\n```"
    except TypeError:
        return str(final)


# -------------------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------------------
st.set_page_config(page_title="FlowCheck Agent", layout="wide")
st.title("FlowCheck Orchestrator–Worker (REST)")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "last_output" not in st.session_state:
    st.session_state.last_output = ""
if "last_issue" not in st.session_state:
    st.session_state.last_issue = ""

issue_text = st.text_area(
    "Issue / context (large input allowed)",
    height=200,
    placeholder="Paste logs, events, user reports, etc...",
)

run_button = st.button("Run")

if run_button and issue_text.strip():
    if st.session_state.thread_id is None:
        try:
            st.session_state.thread_id = create_thread()
        except Exception as e:
            st.error(f"Error creating thread: {e}")
            st.stop()

    thread_id = st.session_state.thread_id

    input_payload = {
        "issue": issue_text,
        "messages": [
            {
                "role": "user",
                "content": issue_text,
            }
        ],
    }

    try:
        run_id = submit_run(thread_id, input_payload)
    except Exception as e:
        st.error(f"Error submitting run: {e}")
        st.stop()

    st.session_state.last_issue = issue_text

    status_placeholder = st.empty()
    step = 0
    run_state = None

    with st.spinner("Running graph..."):
        while True:
            try:
                run_state = get_run_state(thread_id, run_id)
            except Exception as e:
                st.error(f"Error polling run: {e}")
                break

            status = run_state.get("status", "unknown")
            step += 1
            status_placeholder.write(f"Status: **{status}** · Poll #{step}")

            if status in ("success", "failed", "error", "cancelled"):
                break

            time.sleep(1)

    status_placeholder.empty()

    if run_state is not None:
        status = run_state.get("status", "unknown")
        if status == "success":
            st.success("Run completed successfully.")

            try:
                thread_state = get_thread_state(thread_id)
                final_report_md = extract_final_report_from_thread(thread_state)
            except Exception as e:
                st.error(f"Error fetching thread state: {e}")
                final_report_md = "No final_report found."

            st.session_state.last_output = final_report_md
        else:
            st.error(f"Run finished with status: {status}")
            st.code(json.dumps(run_state, indent=2))


if st.session_state.last_output:
    st.markdown("### Final report")
    st.markdown(st.session_state.last_output)

