from collections.abc import Sequence
from datetime import datetime
from uuid import UUID

import pandas as pd
import streamlit as st
from run_rendering import render_run_details, render_run_summary

from lib.monitoring_store import (
    MonitoringSummary,
    StoredAgentRun,
    StoredAgentRunSummary,
)


def generate_run_trend_data(
    runs: Sequence[StoredAgentRunSummary],
) -> pd.DataFrame:
    rows = [
        {
            "created_at": run.created_at,
            "cost_usd": run.total_cost_usd,
            "duration_seconds": run.duration_seconds,
            "input_tokens": run.input_tokens,
            "cached_input_tokens": run.cached_input_tokens,
            "output_tokens": run.output_tokens,
        }
        for run in reversed(runs)
    ]
    return pd.DataFrame(rows)


def generate_recent_run_rows(
    runs: Sequence[StoredAgentRunSummary],
) -> list[dict[str, object]]:
    return [
        {
            "time": run.created_at,
            "question": run.question,
            "duration_s": round(run.duration_seconds, 3),
            "cost_usd": round(run.total_cost_usd, 8),
            "total_tokens": run.total_tokens,
            "model_calls": run.model_call_count,
            "tool_calls": run.tool_calls_count,
        }
        for run in runs
    ]


def render_dashboard_summary(summary: MonitoringSummary) -> None:
    average_cost_usd = (
        summary.total_cost_usd / summary.agent_run_count
        if summary.agent_run_count
        else 0.0
    )

    run_count, duration, total_cost, average_cost = st.columns(4)
    run_count.metric("Agent runs", f"{summary.agent_run_count:,}")
    duration.metric("Average response time", f"{summary.average_duration_seconds:.2f}s")
    total_cost.metric("Total cost", f"${summary.total_cost_usd:.6f}")
    average_cost.metric("Average cost per run", f"${average_cost_usd:.6f}")

    st.caption(f"Average token usage: {summary.average_total_tokens:,.0f} per run")


def render_run_trends(runs: Sequence[StoredAgentRunSummary]) -> None:
    st.subheader("Run trends")
    st.caption(
        "Each point is one completed agent run, ordered from oldest to newest."
    )
    trend_data = generate_run_trend_data(runs)

    cost, duration = st.columns(2)
    with cost:
        st.markdown("**Cost per run**")
        st.line_chart(  # pyright: ignore[reportUnknownMemberType]
            trend_data,
            x="created_at",
            y="cost_usd",
            x_label="Completed at",
            y_label="USD",
        )
    with duration:
        st.markdown("**Response time per run**")
        st.line_chart(  # pyright: ignore[reportUnknownMemberType]
            trend_data,
            x="created_at",
            y="duration_seconds",
            x_label="Completed at",
            y_label="Seconds",
        )

    st.markdown("**Token usage per run**")
    st.line_chart(  # pyright: ignore[reportUnknownMemberType]
        trend_data,
        x="created_at",
        y=["input_tokens", "cached_input_tokens", "output_tokens"],
        x_label="Completed at",
        y_label="Tokens",
    )


def render_recent_runs(runs: Sequence[StoredAgentRunSummary]) -> None:
    st.subheader("Recent runs")
    st.dataframe(  # pyright: ignore[reportUnknownMemberType]
        generate_recent_run_rows(runs),
        hide_index=True,
        width="stretch",
        column_config={
            "time": st.column_config.DatetimeColumn("Completed at"),
            "question": st.column_config.TextColumn("Question", width="large"),
            "duration_s": st.column_config.NumberColumn(
                "Response time", format="%.3f s"
            ),
            "cost_usd": st.column_config.NumberColumn("Cost", format="$%.8f"),
            "total_tokens": st.column_config.NumberColumn("Tokens", format="%d"),
            "model_calls": st.column_config.NumberColumn("Model calls", format="%d"),
            "tool_calls": st.column_config.NumberColumn("Tool calls", format="%d"),
        },
    )


def select_run_to_inspect(runs: Sequence[StoredAgentRunSummary]) -> UUID:
    run_by_id = {run.run_id: run for run in runs}
    selected_run_id = st.selectbox(
        "Inspect a run",
        options=list(run_by_id),
        format_func=lambda run_id: _format_run_option(run_by_id[run_id]),
    )
    return selected_run_id


def render_stored_run(run: StoredAgentRun) -> None:
    st.subheader("Run details")
    st.markdown("**Question**")
    st.write(run.question)
    st.markdown("**Answer**")
    st.write(run.answer)
    render_run_summary(run.metrics)
    render_run_details(run)


def _format_run_option(run: StoredAgentRunSummary) -> str:
    timestamp = _format_timestamp(run.created_at)
    question = run.question if len(run.question) <= 80 else f"{run.question[:77]}..."
    return f"{timestamp} - {question}"


def _format_timestamp(value: datetime) -> str:
    return value.astimezone().strftime("%Y-%m-%d %H:%M:%S")
