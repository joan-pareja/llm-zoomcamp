"""dlt REST API pipeline: load agent trace logs from the test-agent-traces API into DuckDB."""

from collections.abc import Iterator
from typing import Any, cast

import dlt
import requests
from dlt.hub import run
from dlt.sources.rest_api import rest_api_resources
from dlt.sources.rest_api.typing import RESTAPIConfig


@dlt.source(name="agent_traces_source")
def agent_traces_source(
    base_url: str = dlt.config.value,
    max_rows: int = 20000,
) -> Any:
    """Load agent trace logs from the test-agent-traces API.

    Args:
        base_url: API base URL. Auto-loaded from .dlt/config.toml.
        max_rows: Maximum number of log rows to load (offset-paginator cap).
    """
    config: RESTAPIConfig = {
        "client": {
            "base_url": base_url,
        },
        "resources": [
            {
                "name": "logs",
                "endpoint": {
                    "path": "logs",
                    "params": {
                        "limit": 1000,
                    },
                    "paginator": {
                        "type": "offset",
                        "limit": 1000,
                        "total_path": "total",
                        "maximum_offset": max_rows,
                    },
                    "data_selector": "logs",
                },
            },
        ],
    }
    yield from rest_api_resources(config)


@run.pipeline("agent_traces_pipeline")
def load_logs() -> None:
    """Load agent trace logs into the dltHub Platform playground destination.

    base_url is read from .dlt/config.toml under [sources.agent_traces_source].
    Uses the "playground" destination (a managed, persistent store) instead of
    "duckdb" — DuckDB's local file is wiped between runtime job runs.
    """
    pipeline = dlt.pipeline(
        pipeline_name="agent_traces_pipeline",
        destination="playground",
        dataset_name="agent_traces_dataset",
    )

    load_info = pipeline.run(agent_traces_source(), write_disposition="replace")
    print(load_info)
    print(pipeline.last_trace.last_normalize_info)


LOGFIRE_RECORDS_SQL = "SELECT * FROM records ORDER BY start_timestamp DESC"


@dlt.source(name="logfire_source")
def logfire_source(
    base_url: str = dlt.config.value,
    read_token: str = dlt.secrets.value,
    min_timestamp: str = "2025-01-01T00:00:00Z",
    sql: str = LOGFIRE_RECORDS_SQL,
    row_limit: int = 10000,
) -> Any:
    """Load spans from the Logfire Query API into the `records` resource.

    The Query API returns columnar JSON ({"columns": [{"name", "values"}]}),
    so the resource transposes it into row dicts for dlt.
    """

    @dlt.resource(name="records", write_disposition="replace", max_table_nesting=0)
    def records() -> Iterator[dict[str, Any]]:
        response = requests.get(
            base_url + "query",
            headers={
                "Authorization": f"Bearer {read_token}",
                "Accept": "application/json",
            },
            params={"sql": sql, "min_timestamp": min_timestamp, "limit": row_limit},
            timeout=60,
        )
        response.raise_for_status()
        columns = cast("list[dict[str, Any]]", response.json()["columns"])
        names = [str(column["name"]) for column in columns]
        value_lists = [column["values"] for column in columns]
        for values in zip(*value_lists):
            yield dict(zip(names, values))

    yield records()


def load_logfire_traces() -> dlt.Pipeline:
    """Load Logfire spans into a local DuckDB dataset named `agent_traces`."""
    pipeline = dlt.pipeline(
        pipeline_name="logfire_pipeline",
        destination="duckdb",
        dataset_name="agent_traces",
    )
    load_info = pipeline.run(logfire_source(), write_disposition="replace")
    print(load_info)
    print(pipeline.last_trace.last_normalize_info)
    return pipeline


if __name__ == "__main__":
    load_logs()
