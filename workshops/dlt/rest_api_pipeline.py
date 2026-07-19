"""dlt REST API pipeline: load agent trace logs from the test-agent-traces API into DuckDB."""

from typing import Any

import dlt
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


if __name__ == "__main__":
    load_logs()
