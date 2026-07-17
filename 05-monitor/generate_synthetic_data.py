"""Continuously generate synthetic monitoring runs and feedback."""

import random
import time
from datetime import UTC, datetime

from judge import Relevance

from lib.agentic_rag import AgentRun
from lib.llm import DEFAULT_MODEL, ModelInput
from lib.metrics import AgentRunMetrics, ModelCallMetrics, UsagePrice
from lib.monitoring_store import MonitoringStore

SyntheticExample = tuple[str, str, Relevance]

SYNTHETIC_EXAMPLES: tuple[SyntheticExample, ...] = (
    (
        "How do I install Docker?",
        "Download Docker Desktop for your operating system and follow its setup guide.",
        "RELEVANT",
    ),
    (
        "Can I join late, and can I still earn a certificate?",
        "You can join after the course has started.",
        "PARTLY_RELEVANT",
    ),
    (
        "What are the course prerequisites?",
        "Basic Python and command-line familiarity are enough to get started.",
        "RELEVANT",
    ),
    (
        "Where do I submit homework?",
        "Office hours are held weekly; check the course calendar.",
        "NON_RELEVANT",
    ),
    (
        "When are office hours?",
        "Check the course calendar for the current weekly office-hours schedule.",
        "RELEVANT",
    ),
)


def _build_synthetic_run(question: str, answer: str) -> AgentRun:
    input_tokens = random.randint(50, 200)
    output_tokens = random.randint(50, 300)
    duration_seconds = random.uniform(0.5, 5.0)
    message_history: ModelInput = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    model_call = ModelCallMetrics(
        model=DEFAULT_MODEL,
        input_tokens=input_tokens,
        cached_input_tokens=0,
        output_tokens=output_tokens,
        reasoning_output_tokens=0,
        duration_seconds=duration_seconds,
        price=UsagePrice(
            input_cost_usd=input_tokens * 0.75 / 1_000_000,
            output_cost_usd=output_tokens * 4.50 / 1_000_000,
        ),
        completed_at=datetime.now(UTC),
    )
    return AgentRun(
        answer=answer,
        metrics=AgentRunMetrics(
            model_call_metrics=(model_call,),
            tool_calls_count=1,
            duration_seconds=duration_seconds,
        ),
        message_history=message_history,
    )


def generate_one(store: MonitoringStore) -> None:
    question, answer, relevance = random.choice(SYNTHETIC_EXAMPLES)
    run_id = store.save_agent_run(question, _build_synthetic_run(question, answer))

    if random.random() < 0.7:
        store.save_feedback(
            run_id,
            source="judge",
            relevance=relevance,
            explanation=f"Synthetic verdict: {relevance.lower()}.",
        )

    if random.random() < 0.5:
        store.save_feedback(
            run_id,
            source="user",
            score=random.choice((1, 1, 1, 1, -1)),
        )

    print(f"Stored synthetic run {run_id}", flush=True)


def generate_live(store: MonitoringStore, interval_seconds: float = 1.0) -> None:
    print("Generating synthetic data (Ctrl+C to stop)...", flush=True)
    while True:
        generate_one(store)
        time.sleep(interval_seconds)


def main() -> None:
    store = MonitoringStore.from_dotenv()
    try:
        store.initialize_schema()
        generate_live(store)
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        store.close()


if __name__ == "__main__":
    main()
