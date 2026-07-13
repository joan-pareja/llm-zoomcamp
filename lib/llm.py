"""Reusable, instrumented LLM calls."""

import os
from collections.abc import Iterable
from dataclasses import dataclass
from time import perf_counter
from typing import TypeAlias, cast

from openai import OpenAI
from openai.types.responses import (
    Response,
    ResponseInputItemParam,
    ResponseInputParam,
    ResponseOutputItem,
    ToolParam,
)

from .metrics import (
    ModelCallMetrics,
    calculate_price,
    validate_model_pricing,
)

DEFAULT_MODEL = os.getenv("OPENAI_MODEL_NAME", "gpt-5.4-mini")
ModelInput: TypeAlias = list[ResponseInputItemParam | ResponseOutputItem]


@dataclass(frozen=True)
class ModelCall[ResultT]:
    result: ResultT
    metrics: ModelCallMetrics


class StructuredOutputError(RuntimeError):
    """A billed structured response that could not be parsed."""

    def __init__(self, message: str, metrics: ModelCallMetrics) -> None:
        super().__init__(message)
        self.metrics = metrics


def _as_response_input(messages: ModelInput) -> ResponseInputParam:
    """Present a supported mixed conversation as the SDK's narrower input type.

    The Responses API accepts previous ``response.output`` items in subsequent
    input. ``ModelInput`` models that runtime contract, while the generated SDK
    annotation only describes parameter-shaped input items. Keep the necessary
    cast at this boundary so callers retain the honest, stricter shared type.
    This function changes only the static type; it does not transform the list.
    """
    return cast(ResponseInputParam, messages)


def _build_call_metrics(
    response: Response,
    duration_seconds: float,
    pricing_model: str,
) -> ModelCallMetrics:
    usage = response.usage
    if usage is None:
        raise RuntimeError("The LLM response did not include usage metadata.")

    return ModelCallMetrics(
        model=response.model,
        input_tokens=usage.input_tokens,
        cached_input_tokens=usage.input_tokens_details.cached_tokens,
        output_tokens=usage.output_tokens,
        reasoning_output_tokens=usage.output_tokens_details.reasoning_tokens,
        duration_seconds=duration_seconds,
        price=calculate_price(pricing_model, usage),
    )


def call_llm(
    client: OpenAI,
    messages: ModelInput,
    model: str = DEFAULT_MODEL,
    tools: Iterable[ToolParam] | None = None,
) -> ModelCall[Response]:
    validate_model_pricing(model)
    started_at = perf_counter()

    if tools is None:
        response = client.responses.create(
            model=model,
            input=_as_response_input(messages),
        )
    else:
        response = client.responses.create(
            model=model,
            input=_as_response_input(messages),
            tools=tools,
        )

    duration_seconds = perf_counter() - started_at
    return ModelCall(
        result=response,
        metrics=_build_call_metrics(response, duration_seconds, model),
    )


def call_structured_llm[StructuredOutputT](
    client: OpenAI,
    instructions: str,
    user_prompt: str,
    output_type: type[StructuredOutputT],
    model: str = DEFAULT_MODEL,
) -> ModelCall[StructuredOutputT]:
    validate_model_pricing(model)
    messages: ModelInput = [
        {"role": "developer", "content": instructions},
        {"role": "user", "content": user_prompt},
    ]

    started_at = perf_counter()
    response = client.responses.parse(
        model=model,
        input=_as_response_input(messages),
        text_format=output_type,
    )
    duration_seconds = perf_counter() - started_at
    metrics = _build_call_metrics(cast(Response, response), duration_seconds, model)

    if response.output_parsed is None:
        raise StructuredOutputError(
            "The LLM response did not include parsed output.",
            metrics,
        )

    return ModelCall(
        result=response.output_parsed,
        metrics=metrics,
    )
