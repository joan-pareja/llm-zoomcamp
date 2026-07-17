from uuid import UUID

import streamlit as st
from assistant import create_assistant
from judge import RelevanceVerdict, evaluate_relevance
from run_rendering import render_run_details, render_run_summary

from lib.agentic_rag import AgenticRAG, AgentRun
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


def _generate_answer(
    assistant: AgenticRAG[FAQDocument],
    question: str,
) -> AgentRun | None:
    try:
        with st.spinner("Processing..."):
            return assistant.find_and_reply(question)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not generate an answer: {exc}")
        return None


def _save_run(question: str, result: AgentRun) -> UUID | None:
    try:
        return get_monitoring_store().save_agent_run(question, result)
    except Exception as exc:  # noqa: BLE001
        st.warning(
            f"Answer generated, but the run could not be saved to Postgres: {exc}"
        )
        return None


def _evaluate_answer(
    assistant: AgenticRAG[FAQDocument],
    question: str,
    result: AgentRun,
) -> RelevanceVerdict | None:
    try:
        return evaluate_relevance(
            client=assistant.llm_client,
            question=question,
            answer=result.answer,
        ).result
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Run stored, but the judge could not evaluate it: {exc}")
        return None


def _save_judge_feedback(run_id: UUID, verdict: RelevanceVerdict) -> None:
    try:
        get_monitoring_store().save_feedback(
            run_id,
            source="judge",
            relevance=verdict.relevance,
            explanation=verdict.explanation,
        )
    except Exception as exc:  # noqa: BLE001
        st.warning(
            "Judge evaluation completed, but it could not be saved "
            f"to Postgres: {exc}"
        )


def _handle_question(
    assistant: AgenticRAG[FAQDocument],
    user_input: str,
) -> None:
    if "feedback_run_id" in st.session_state:
        del st.session_state["feedback_run_id"]

    question = user_input.strip()
    if not question:
        st.warning("Enter a question first.")
        return

    result = _generate_answer(assistant, question)
    if result is None:
        return

    st.success("Answer generated.")
    st.write(result.answer)
    render_run_summary(result.metrics)
    render_run_details(result)

    run_id = _save_run(question, result)
    if run_id is None:
        return

    st.session_state["feedback_run_id"] = run_id
    st.caption(f"Stored run {run_id}")

    verdict = _evaluate_answer(assistant, question, result)
    if verdict is None:
        return

    st.write(f"Relevance: {verdict.relevance}")
    st.write(f"Judge explanation: {verdict.explanation}")
    _save_judge_feedback(run_id, verdict)


def _save_user_feedback(run_id: UUID, score: int) -> None:
    try:
        get_monitoring_store().save_feedback(
            run_id,
            source="user",
            score=score,
        )
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Could not save feedback to Postgres: {exc}")
    else:
        st.success("Thanks for the feedback!")


def _render_feedback_controls() -> None:
    run_id = st.session_state.get("feedback_run_id")
    if not isinstance(run_id, UUID):
        return

    st.write("Was this answer helpful?")
    positive, negative = st.columns(2)

    with positive:
        positive_feedback = st.button("+1", key=f"positive-{run_id}")

    with negative:
        negative_feedback = st.button("-1", key=f"negative-{run_id}")

    score = 1 if positive_feedback else -1 if negative_feedback else None
    if score is None:
        return

    _save_user_feedback(run_id, score)


assistant = get_assistant()

st.title("Course Assistant")
user_input = st.text_input("Enter your question:")

if st.button("Ask"):
    _handle_question(assistant, user_input)

_render_feedback_controls()
