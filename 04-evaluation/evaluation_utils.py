from collections.abc import Callable, Sequence
from concurrent.futures import Executor, Future
from dataclasses import dataclass
from time import sleep
from typing import TypeVar

from tqdm.auto import tqdm

from lib.agentic_rag import AgentRunStats, AgenticRAG
from lib.types import Document


InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")

__all__ = [
    "AgenticRAGAnswer",
    "answer_with_usage",
    "map_progress",
]


@dataclass
class AgenticRAGAnswer:
    answer: str
    stats: AgentRunStats


def answer_with_usage[TDocument: Document](
    rag: AgenticRAG[TDocument],
    question: str,
) -> AgenticRAGAnswer:
    answer = rag.find_and_reply(question)

    return AgenticRAGAnswer(
        answer=answer,
        stats=rag.last_run_stats,
    )


def map_progress(
    pool: Executor,
    seq: Sequence[InputT],
    f: Callable[[InputT], OutputT],
    submit_delay_seconds: float = 0.0,
) -> list[OutputT]:
    results: list[OutputT] = []

    with tqdm(total=len(seq)) as progress:
        futures: list[Future[OutputT]] = []

        for el in seq:
            future = pool.submit(f, el)
            future.add_done_callback(lambda _future: progress.update())
            futures.append(future)

            if submit_delay_seconds > 0:
                sleep(submit_delay_seconds)

        for future in futures:
            result = future.result()
            results.append(result)

    return results
