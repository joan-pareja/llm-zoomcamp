"""Persistence helpers for agent run monitoring."""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast
from uuid import UUID, uuid4

from dotenv import dotenv_values
from psycopg import Connection, connect
from psycopg.rows import class_row
from psycopg.types.json import Jsonb

from .agentic_rag import AgentRun
from .json_utils import JSONValue, convert_to_jsonable
from .metrics import AgentRunMetrics, ModelCallMetrics, UsagePrice

_MAX_RECENT_AGENT_RUNS = 200

_CREATE_AGENT_RUNS_SQL = """
CREATE TABLE IF NOT EXISTS agent_runs (
    run_id uuid PRIMARY KEY,
    question text NOT NULL,
    answer text NOT NULL,
    duration_seconds double precision NOT NULL,
    model_call_count integer NOT NULL,
    tool_calls_count integer NOT NULL,
    input_tokens integer NOT NULL,
    cached_input_tokens integer NOT NULL,
    output_tokens integer NOT NULL,
    reasoning_output_tokens integer NOT NULL,
    total_tokens integer NOT NULL,
    total_cost_usd double precision NOT NULL,
    message_history jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
"""

_CREATE_MODEL_CALL_METRICS_SQL = """
CREATE TABLE IF NOT EXISTS model_call_metrics (
    run_id uuid NOT NULL REFERENCES agent_runs (run_id) ON DELETE CASCADE,
    call_index integer NOT NULL,
    model text NOT NULL,
    input_tokens integer NOT NULL,
    cached_input_tokens integer NOT NULL,
    output_tokens integer NOT NULL,
    reasoning_output_tokens integer NOT NULL,
    duration_seconds double precision NOT NULL,
    input_cost_usd double precision NOT NULL,
    output_cost_usd double precision NOT NULL,
    total_cost_usd double precision NOT NULL,
    completed_at timestamptz NOT NULL,
    PRIMARY KEY (run_id, call_index)
);
"""

_CREATE_FEEDBACK_SQL = """
CREATE TABLE IF NOT EXISTS feedback (
    feedback_id uuid PRIMARY KEY,
    run_id uuid NOT NULL REFERENCES agent_runs (run_id) ON DELETE CASCADE,
    source text NOT NULL,
    relevance text,
    explanation text,
    score integer,
    created_at timestamptz NOT NULL DEFAULT now()
);
"""

_CREATE_AGENT_RUNS_CREATED_AT_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS agent_runs_created_at_idx
ON agent_runs (created_at DESC);
"""

@dataclass(frozen=True)
class StoredAgentRunSummary:
    run_id: UUID
    question: str
    answer: str
    duration_seconds: float
    model_call_count: int
    tool_calls_count: int
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    reasoning_output_tokens: int
    total_tokens: int
    total_cost_usd: float
    created_at: datetime


@dataclass(frozen=True)
class MonitoringSummary:
    agent_run_count: int
    average_duration_seconds: float
    total_cost_usd: float
    average_total_tokens: float


@dataclass(frozen=True)
class FeedbackSummary:
    relevant_count: int
    partly_relevant_count: int
    non_relevant_count: int
    thumbs_up_count: int
    thumbs_down_count: int


@dataclass(frozen=True)
class StoredAgentRun:
    run_id: UUID
    question: str
    answer: str
    metrics: AgentRunMetrics
    message_history: JSONValue
    created_at: datetime


@dataclass(frozen=True)
class _StoredAgentRunRecord:
    run_id: UUID
    question: str
    answer: str
    duration_seconds: float
    tool_calls_count: int
    message_history: object
    created_at: datetime


@dataclass(frozen=True)
class _ModelCallMetricsRecord:
    model: str
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    reasoning_output_tokens: int
    duration_seconds: float
    input_cost_usd: float
    output_cost_usd: float
    completed_at: datetime


@dataclass(frozen=True)
class MonitoringDatabaseSettings:
    host: str
    port: int
    db_name: str
    user: str
    password: str
    sslmode: str = "disable"

    @classmethod
    def from_environment(cls) -> "MonitoringDatabaseSettings":
        """Read settings from the current process environment."""
        return cls(
            host=os.getenv("MONITORING_POSTGRES_HOST", "localhost"),
            port=int(os.getenv("MONITORING_POSTGRES_PORT", "5433")),
            db_name=os.getenv("MONITORING_POSTGRES_DB", "rag_monitoring"),
            user=os.getenv("MONITORING_POSTGRES_USER", "rag_monitoring"),
            password=os.getenv("MONITORING_POSTGRES_PASSWORD", ""),
            sslmode=os.getenv("MONITORING_POSTGRES_SSLMODE", "disable"),
        )

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, str | None],
    ) -> "MonitoringDatabaseSettings":
        """Build settings from an explicit mapping such as dotenv_values()."""
        return cls(
            host=values.get("MONITORING_POSTGRES_HOST") or "localhost",
            port=int(values.get("MONITORING_POSTGRES_PORT") or "5433"),
            db_name=values.get("MONITORING_POSTGRES_DB") or "rag_monitoring",
            user=values.get("MONITORING_POSTGRES_USER") or "rag_monitoring",
            password=values.get("MONITORING_POSTGRES_PASSWORD") or "",
            sslmode=values.get("MONITORING_POSTGRES_SSLMODE") or "disable",
        )

    @classmethod
    def from_dotenv(
        cls,
        dotenv_path: str | Path = ".env",
    ) -> "MonitoringDatabaseSettings":
        """Read settings directly from a dotenv file without mutating os.environ."""
        return cls.from_mapping(dotenv_values(dotenv_path))

    def connect(self) -> Connection[Any]:
        return connect(
            host=self.host,
            port=self.port,
            dbname=self.db_name,
            user=self.user,
            password=self.password,
            sslmode=self.sslmode,
        )


@dataclass
class MonitoringStore:
    connection: Connection[Any]

    @classmethod
    def from_environment(cls) -> "MonitoringStore":
        return cls(connection=MonitoringDatabaseSettings.from_environment().connect())

    @classmethod
    def from_dotenv(cls, dotenv_path: str | Path = ".env") -> "MonitoringStore":
        return cls(connection=MonitoringDatabaseSettings.from_dotenv(dotenv_path).connect())

    def close(self) -> None:
        self.connection.close()

    def initialize_schema(self) -> None:
        with self.connection.transaction():
            self.connection.execute(_CREATE_AGENT_RUNS_SQL)
            self.connection.execute(_CREATE_MODEL_CALL_METRICS_SQL)
            self.connection.execute(_CREATE_FEEDBACK_SQL)
            self.connection.execute(_CREATE_AGENT_RUNS_CREATED_AT_INDEX_SQL)

    def list_recent_agent_runs(
        self,
        limit: int = 20,
    ) -> list[StoredAgentRunSummary]:
        if limit <= 0:
            raise ValueError("limit must be greater than zero")

        bounded_limit = min(limit, _MAX_RECENT_AGENT_RUNS)
        with self.connection.transaction():
            with self.connection.cursor(
                row_factory=class_row(StoredAgentRunSummary)
            ) as cursor:
                return cursor.execute(
                    """
                    SELECT
                        run_id,
                        question,
                        answer,
                        duration_seconds,
                        model_call_count,
                        tool_calls_count,
                        input_tokens,
                        cached_input_tokens,
                        output_tokens,
                        reasoning_output_tokens,
                        total_tokens,
                        total_cost_usd,
                        created_at
                    FROM agent_runs
                    ORDER BY created_at DESC
                    LIMIT %(limit)s
                    """,
                    {"limit": bounded_limit},
                ).fetchall()

    def get_monitoring_summary(self) -> MonitoringSummary:
        with self.connection.transaction():
            with self.connection.cursor(
                row_factory=class_row(MonitoringSummary)
            ) as cursor:
                row = cursor.execute(
                    """
                    SELECT
                        COUNT(*) AS agent_run_count,
                        COALESCE(
                            AVG(duration_seconds), 0.0
                        )::double precision AS average_duration_seconds,
                        COALESCE(
                            SUM(total_cost_usd), 0.0
                        )::double precision AS total_cost_usd,
                        COALESCE(
                            AVG(total_tokens), 0.0
                        )::double precision AS average_total_tokens
                    FROM agent_runs
                    """
                ).fetchone()
        if row is None:
            raise RuntimeError("Monitoring summary query returned no row")

        return row

    def get_feedback_summary(self) -> FeedbackSummary:
        with self.connection.transaction():
            with self.connection.cursor(
                row_factory=class_row(FeedbackSummary)
            ) as cursor:
                row = cursor.execute(
                    """
                    SELECT
                        COUNT(*) FILTER (
                            WHERE source = 'judge' AND relevance = 'RELEVANT'
                        ) AS relevant_count,
                        COUNT(*) FILTER (
                            WHERE source = 'judge'
                                AND relevance = 'PARTLY_RELEVANT'
                        ) AS partly_relevant_count,
                        COUNT(*) FILTER (
                            WHERE source = 'judge' AND relevance = 'NON_RELEVANT'
                        ) AS non_relevant_count,
                        COUNT(*) FILTER (
                            WHERE source = 'user' AND score > 0
                        ) AS thumbs_up_count,
                        COUNT(*) FILTER (
                            WHERE source = 'user' AND score < 0
                        ) AS thumbs_down_count
                    FROM feedback
                    """
                ).fetchone()
        if row is None:
            raise RuntimeError("Feedback summary query returned no row")

        return row

    def get_agent_run(self, run_id: UUID) -> StoredAgentRun | None:
        with self.connection.transaction():
            with self.connection.cursor(
                row_factory=class_row(_StoredAgentRunRecord)
            ) as cursor:
                row = cursor.execute(
                    """
                    SELECT
                        run_id,
                        question,
                        answer,
                        duration_seconds,
                        tool_calls_count,
                        message_history,
                        created_at
                    FROM agent_runs
                    WHERE run_id = %(run_id)s
                    """,
                    {"run_id": run_id},
                ).fetchone()
            if row is None:
                return None

            metrics = AgentRunMetrics(
                model_call_metrics=self._list_model_call_metrics(run_id),
                tool_calls_count=row.tool_calls_count,
                duration_seconds=row.duration_seconds,
            )

        return StoredAgentRun(
            run_id=row.run_id,
            question=row.question,
            answer=row.answer,
            metrics=metrics,
            message_history=cast(JSONValue, row.message_history),
            created_at=row.created_at,
        )

    def _list_model_call_metrics(
        self,
        run_id: UUID,
    ) -> tuple[ModelCallMetrics, ...]:
        with self.connection.cursor(
            row_factory=class_row(_ModelCallMetricsRecord)
        ) as cursor:
            rows = cursor.execute(
                """
                SELECT
                    model,
                    input_tokens,
                    cached_input_tokens,
                    output_tokens,
                    reasoning_output_tokens,
                    duration_seconds,
                    input_cost_usd,
                    output_cost_usd,
                    completed_at
                FROM model_call_metrics
                WHERE run_id = %(run_id)s
                ORDER BY call_index
                """,
                {"run_id": run_id},
            ).fetchall()

        model_call_metrics: list[ModelCallMetrics] = []
        for row in rows:
            model_call_metrics.append(
                ModelCallMetrics(
                    model=row.model,
                    input_tokens=row.input_tokens,
                    cached_input_tokens=row.cached_input_tokens,
                    output_tokens=row.output_tokens,
                    reasoning_output_tokens=row.reasoning_output_tokens,
                    duration_seconds=row.duration_seconds,
                    price=UsagePrice(
                        input_cost_usd=row.input_cost_usd,
                        output_cost_usd=row.output_cost_usd,
                    ),
                    completed_at=row.completed_at,
                )
            )

        return tuple(model_call_metrics)

    def save_agent_run(self, question: str, run: AgentRun) -> UUID:
        run_id = uuid4()
        metrics = run.metrics

        with self.connection.transaction():
            self.connection.execute(
                """
                INSERT INTO agent_runs (
                    run_id,
                    question,
                    answer,
                    duration_seconds,
                    model_call_count,
                    tool_calls_count,
                    input_tokens,
                    cached_input_tokens,
                    output_tokens,
                    reasoning_output_tokens,
                    total_tokens,
                    total_cost_usd,
                    message_history
                )
                VALUES (
                    %(run_id)s,
                    %(question)s,
                    %(answer)s,
                    %(duration_seconds)s,
                    %(model_call_count)s,
                    %(tool_calls_count)s,
                    %(input_tokens)s,
                    %(cached_input_tokens)s,
                    %(output_tokens)s,
                    %(reasoning_output_tokens)s,
                    %(total_tokens)s,
                    %(total_cost_usd)s,
                    %(message_history)s
                )
                """,
                {
                    "run_id": run_id,
                    "question": question,
                    "answer": run.answer,
                    "duration_seconds": metrics.duration_seconds,
                    "model_call_count": len(metrics.model_call_metrics),
                    "tool_calls_count": metrics.tool_calls_count,
                    "input_tokens": metrics.input_tokens,
                    "cached_input_tokens": metrics.cached_input_tokens,
                    "output_tokens": metrics.output_tokens,
                    "reasoning_output_tokens": metrics.reasoning_output_tokens,
                    "total_tokens": metrics.total_tokens,
                    "total_cost_usd": metrics.total_cost_usd,
                    "message_history": Jsonb(convert_to_jsonable(run.message_history)),
                },
            )

            for index, model_call in enumerate(metrics.model_call_metrics, start=1):
                self.connection.execute(
                    """
                    INSERT INTO model_call_metrics (
                        run_id,
                        call_index,
                        model,
                        input_tokens,
                        cached_input_tokens,
                        output_tokens,
                        reasoning_output_tokens,
                        duration_seconds,
                        input_cost_usd,
                        output_cost_usd,
                        total_cost_usd,
                        completed_at
                    )
                    VALUES (
                        %(run_id)s,
                        %(call_index)s,
                        %(model)s,
                        %(input_tokens)s,
                        %(cached_input_tokens)s,
                        %(output_tokens)s,
                        %(reasoning_output_tokens)s,
                        %(duration_seconds)s,
                        %(input_cost_usd)s,
                        %(output_cost_usd)s,
                        %(total_cost_usd)s,
                        %(completed_at)s
                    )
                    """,
                    {
                        "run_id": run_id,
                        "call_index": index,
                        "model": model_call.model,
                        "input_tokens": model_call.input_tokens,
                        "cached_input_tokens": model_call.cached_input_tokens,
                        "output_tokens": model_call.output_tokens,
                        "reasoning_output_tokens": model_call.reasoning_output_tokens,
                        "duration_seconds": model_call.duration_seconds,
                        "input_cost_usd": model_call.price.input_cost_usd,
                        "output_cost_usd": model_call.price.output_cost_usd,
                        "total_cost_usd": model_call.price.total_cost_usd,
                        "completed_at": model_call.completed_at,
                    },
                )

        return run_id

    def save_feedback(
        self,
        run_id: UUID,
        source: str,
        *,
        relevance: str | None = None,
        explanation: str | None = None,
        score: int | None = None,
    ) -> UUID:
        feedback_id = uuid4()

        with self.connection.transaction():
            self.connection.execute(
                """
                INSERT INTO feedback (
                    feedback_id,
                    run_id,
                    source,
                    relevance,
                    explanation,
                    score
                )
                VALUES (
                    %(feedback_id)s,
                    %(run_id)s,
                    %(source)s,
                    %(relevance)s,
                    %(explanation)s,
                    %(score)s
                )
                """,
                {
                    "feedback_id": feedback_id,
                    "run_id": run_id,
                    "source": source,
                    "relevance": relevance,
                    "explanation": explanation,
                    "score": score,
                },
            )

        return feedback_id
