import streamlit as st
from dashboard_rendering import (
    render_dashboard_summary,
    render_feedback_summary,
    render_recent_runs,
    render_run_trends,
    render_stored_run,
    select_run_to_inspect,
)

from lib.monitoring_store import MonitoringStore


@st.cache_resource
def get_monitoring_store() -> MonitoringStore:
    store = MonitoringStore.from_dotenv()
    store.initialize_schema()
    return store


st.set_page_config(page_title="RAG Monitoring", layout="wide")
st.title("Course Assistant Monitoring")
st.caption("Usage, cost, latency, feedback, and the history of each agent run.")

try:
    store = get_monitoring_store()
    summary = store.get_monitoring_summary()
    feedback_summary = store.get_feedback_summary()
    recent_runs = store.list_recent_agent_runs(limit=100)
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not load monitoring data from Postgres: {exc}")
    st.stop()

render_dashboard_summary(summary)
render_feedback_summary(feedback_summary)

if not recent_runs:
    st.info("No agent runs have been stored yet. Ask a question in the chat first.")
    st.stop()

render_run_trends(recent_runs)
render_recent_runs(recent_runs)

selected_run_id = select_run_to_inspect(recent_runs)
selected_run = store.get_agent_run(selected_run_id)
if selected_run is None:
    st.warning("The selected run no longer exists.")
else:
    render_stored_run(selected_run)
