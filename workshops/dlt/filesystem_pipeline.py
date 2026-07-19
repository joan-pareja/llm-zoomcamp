"""dlt filesystem pipeline: load Claude Code session logs (JSONL) into DuckDB."""

import dlt
from dlt.sources.filesystem import filesystem, read_jsonl


def load_logs() -> None:
    """Load Claude Code session log files into DuckDB.

    bucket_url is read from .dlt/config.toml under [sources.filesystem].
    file_glob is set inline so it lives next to the code that depends on it.
    """
    pipeline = dlt.pipeline(
        pipeline_name="claude_logs_pipeline",
        destination="duckdb",
        dataset_name="claude_logs",
        dev_mode=True,
    )

    reader = (filesystem(file_glob="**/*.jsonl") | read_jsonl()).with_name("logs")
    # tool_use_result and attachment are per-tool polymorphic blobs (Bash output,
    # file diffs, glob matches, notebook cells, ...) — flattening them explodes the
    # root table's width and spins off a child table per nested list. Keep them as
    # JSON columns, queryable via DuckDB's JSON functions, instead of unnesting.
    reader.apply_hints(
        columns={
            "tool_use_result": {"data_type": "json"},
            "attachment": {"data_type": "json"},
        }
    )

    load_info = pipeline.run(reader, write_disposition="replace")
    print(load_info)
    print(pipeline.last_trace.last_normalize_info)


if __name__ == "__main__":
    load_logs()
