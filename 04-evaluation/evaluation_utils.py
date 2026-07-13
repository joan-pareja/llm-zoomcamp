from collections.abc import Callable, Sequence
from concurrent.futures import Executor, Future
from time import sleep
from typing import TypeVar

from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)
from tqdm.auto import tqdm

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")

__all__ = [
    "call_with_retry",
    "map_progress",
]


def call_with_retry(
    call: Callable[[], OutputT],
    max_attempts: int = 3,
) -> OutputT:
    for attempt in range(max_attempts):
        try:
            return call()
        except (
            APIConnectionError,
            APITimeoutError,
            InternalServerError,
            RateLimitError,
        ):
            if attempt == max_attempts - 1:
                raise
            sleep(2**attempt)

    raise RuntimeError("Retry loop finished without returning or raising.")


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
