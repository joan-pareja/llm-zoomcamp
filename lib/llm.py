"""Reusable LLM call and usage-cost helpers."""

import os
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TypeVar

from openai import OpenAI
from openai.types.responses import Response, ResponseInputParam, ResponseUsage, ToolParam


DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL_NAME", "gpt-5.4-mini")

StructuredOutputT = TypeVar("StructuredOutputT")


@dataclass
class UsagePrice:
    input_cost: float
    output_cost: float
    total_cost: float


def calc_price(usage: ResponseUsage) -> UsagePrice:
    input_price_per_million = 0.75
    output_price_per_million = 4.50

    input_cost = (usage.input_tokens / 1_000_000) * input_price_per_million
    output_cost = (usage.output_tokens / 1_000_000) * output_price_per_million
    total_cost = input_cost + output_cost

    return UsagePrice(
        input_cost=input_cost,
        output_cost=output_cost,
        total_cost=total_cost,
    )


def calc_total_price(usages: Iterable[ResponseUsage | None]) -> float:
    total_cost = 0.0

    for usage in usages:
        if usage is None:
            continue

        cost = calc_price(usage)
        total_cost = total_cost + cost.total_cost

    return total_cost


def call_llm(
    client: OpenAI,
    messages: ResponseInputParam,
    model: str = DEFAULT_OPENAI_MODEL,
    tools: Iterable[ToolParam] | None = None,
) -> Response:
    if tools is None:
        return client.responses.create(
            model=model,
            input=messages,
        )

    else:
        return client.responses.create(
            model=model,
            input=messages,
            tools=tools,
        )


def call_structured_llm(
    client: OpenAI,
    instructions: str,
    user_prompt: str,
    output_type: type[StructuredOutputT],
    model: str = DEFAULT_OPENAI_MODEL,
) -> tuple[StructuredOutputT, ResponseUsage]:
    messages: ResponseInputParam = [
        {"role": "developer", "content": instructions},
        {"role": "user", "content": user_prompt},
    ]

    response = client.responses.parse(
        model=model,
        input=messages,
        text_format=output_type,
    )

    if response.output_parsed is None:
        raise RuntimeError("The LLM response did not include parsed output.")

    if response.usage is None:
        raise RuntimeError("The LLM response did not include usage metadata.")

    return response.output_parsed, response.usage


def call_structured_llm_with_retry(
    client: OpenAI,
    instructions: str,
    user_prompt: str,
    output_type: type[StructuredOutputT],
    model: str = DEFAULT_OPENAI_MODEL,
    max_retries: int = 3,
) -> tuple[StructuredOutputT, ResponseUsage]:
    for attempt in range(max_retries):
        try:
            return call_structured_llm(
                client,
                instructions,
                user_prompt,
                output_type,
                model=model,
            )
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)

    raise RuntimeError("Structured LLM call failed without raising an exception.")
