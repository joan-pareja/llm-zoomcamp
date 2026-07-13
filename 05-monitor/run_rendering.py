from typing import Protocol

import streamlit as st

from lib.json_utils import convert_to_jsonable
from lib.metrics import AgentRunMetrics, ModelCallMetrics


class RunDetails(Protocol):
    @property
    def metrics(self) -> AgentRunMetrics: ...

    @property
    def message_history(self) -> object: ...


def generate_model_call_row(
    index: int,
    metrics: ModelCallMetrics,
) -> dict[str, object]:
    return {
        "call": index,
        "model": metrics.model,
        "completed_at": metrics.completed_at.isoformat(),
        "duration_s": round(metrics.duration_seconds, 3),
        "input_tokens": metrics.input_tokens,
        "cached_input_tokens": metrics.cached_input_tokens,
        "output_tokens": metrics.output_tokens,
        "reasoning_output_tokens": metrics.reasoning_output_tokens,
        "total_tokens": metrics.total_tokens,
        "input_cost_usd": round(metrics.price.input_cost_usd, 8),
        "output_cost_usd": round(metrics.price.output_cost_usd, 8),
        "total_cost_usd": round(metrics.price.total_cost_usd, 8),
    }


def generate_run_metrics_dict(metrics: AgentRunMetrics) -> dict[str, object]:
    return {
        "duration_seconds": round(metrics.duration_seconds, 3),
        "model_call_count": len(metrics.model_call_metrics),
        "tool_calls_count": metrics.tool_calls_count,
        "input_tokens": metrics.input_tokens,
        "cached_input_tokens": metrics.cached_input_tokens,
        "output_tokens": metrics.output_tokens,
        "reasoning_output_tokens": metrics.reasoning_output_tokens,
        "total_tokens": metrics.total_tokens,
        "total_cost_usd": round(metrics.total_cost_usd, 8),
    }


def render_run_summary(metrics: AgentRunMetrics) -> None:
    duration, cost, model_calls, tool_calls = st.columns(4)
    duration.metric("Response time", f"{metrics.duration_seconds:.2f}s")
    cost.metric("Cost", f"${metrics.total_cost_usd:.6f}")
    model_calls.metric("Model calls", str(len(metrics.model_call_metrics)))
    tool_calls.metric("Tool calls", str(metrics.tool_calls_count))

    input_tokens, output_tokens, total_tokens = st.columns(3)
    input_tokens.metric("Input tokens", str(metrics.input_tokens))
    output_tokens.metric("Output tokens", str(metrics.output_tokens))
    total_tokens.metric("Total tokens", str(metrics.total_tokens))


def render_run_details(run: RunDetails) -> None:
    with st.expander("Run metrics"):
        st.json(generate_run_metrics_dict(run.metrics))

    with st.expander("Model call metrics"):
        rows = [
            generate_model_call_row(index, metrics)
            for index, metrics in enumerate(run.metrics.model_call_metrics, start=1)
        ]
        st.dataframe(  # pyright: ignore[reportUnknownMemberType]
            rows,
            hide_index=True,
            width="stretch",
        )

    with st.expander("Message history"):
        st.json(convert_to_jsonable(run.message_history))
