"""Usage and cost metrics shared by LLM-backed workflows."""

from dataclasses import dataclass, field
from datetime import UTC, datetime

from openai.types.responses import ResponseUsage

_MODEL_RATES_USD_PER_MILLION = {
    "gpt-5.4-mini": (0.75, 0.075, 4.50),
    "gpt-5.4-mini-2026-03-17": (0.75, 0.075, 4.50),
}


@dataclass(frozen=True)
class UsagePrice:
    input_cost_usd: float
    output_cost_usd: float

    @property
    def total_cost_usd(self) -> float:
        return self.input_cost_usd + self.output_cost_usd


@dataclass(frozen=True)
class ModelCallMetrics:
    model: str
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    reasoning_output_tokens: int
    duration_seconds: float
    price: UsagePrice
    completed_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class AgentRunMetrics:
    model_call_metrics: tuple[ModelCallMetrics, ...]
    tool_calls_count: int
    duration_seconds: float

    @property
    def input_tokens(self) -> int:
        return sum(call.input_tokens for call in self.model_call_metrics)

    @property
    def cached_input_tokens(self) -> int:
        return sum(call.cached_input_tokens for call in self.model_call_metrics)

    @property
    def output_tokens(self) -> int:
        return sum(call.output_tokens for call in self.model_call_metrics)

    @property
    def reasoning_output_tokens(self) -> int:
        return sum(call.reasoning_output_tokens for call in self.model_call_metrics)

    @property
    def total_tokens(self) -> int:
        return sum(call.total_tokens for call in self.model_call_metrics)

    @property
    def total_cost_usd(self) -> float:
        return sum(call.price.total_cost_usd for call in self.model_call_metrics)


def calculate_price(model: str, usage: ResponseUsage) -> UsagePrice:
    """Calculate the standard API price for a supported model response."""
    rates = _MODEL_RATES_USD_PER_MILLION.get(model)
    if rates is None:
        raise ValueError(f"Pricing is not configured for model: {model}")

    input_rate, cached_input_rate, output_rate = rates
    cached_input_tokens = usage.input_tokens_details.cached_tokens
    uncached_input_tokens = usage.input_tokens - cached_input_tokens

    input_cost = (
        uncached_input_tokens * input_rate + cached_input_tokens * cached_input_rate
    ) / 1_000_000
    output_cost = usage.output_tokens * output_rate / 1_000_000

    return UsagePrice(
        input_cost_usd=input_cost,
        output_cost_usd=output_cost,
    )


def validate_model_pricing(model: str) -> None:
    """Fail before an API call when its model has no configured USD price."""
    if model not in _MODEL_RATES_USD_PER_MILLION:
        raise ValueError(f"Pricing is not configured for model: {model}")
