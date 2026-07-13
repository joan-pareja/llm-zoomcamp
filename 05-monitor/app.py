import streamlit as st
from assistant import create_assistant
from run_rendering import render_run_details, render_run_summary

from lib.agentic_rag import AgenticRAG
from lib.monitoring_store import MonitoringStore
from lib.types import FAQDocument


@st.cache_resource
def get_assistant() -> AgenticRAG[FAQDocument]:
    return create_assistant()


@st.cache_resource
def get_monitoring_store() -> MonitoringStore:
    store = MonitoringStore.from_dotenv()
    store.initialize_schema()
    return store


assistant = get_assistant()

st.title("Course Assistant")

user_input = st.text_input("Enter your question:")

if st.button("Ask"):
    question = user_input.strip()

    if not question:
        st.warning("Enter a question first.")
    else:
        try:
            with st.spinner("Processing..."):
                result = assistant.find_and_reply(question)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not generate an answer: {exc}")
        else:
            st.success("Answer generated.")
            st.write(result.answer)
            render_run_summary(result.metrics)
            render_run_details(result)

            try:
                run_id = get_monitoring_store().save_agent_run(question, result)
            except Exception as exc:  # noqa: BLE001
                st.warning(
                    f"Answer generated, but the run could not be saved to Postgres: {exc}"
                )
            else:
                st.caption(f"Stored run {run_id}")
