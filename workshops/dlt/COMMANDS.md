# dlt workshop — command reference

Run from `workshops/dlt/`. This subproject has its own `pyproject.toml`/`.venv`
(scoped deps: `dlt`, `dlthub`, `marimo`, `altair`, `ibis-framework`, `pyarrow`,
`pandas`, `fastmcp`), separate from the repo root's — a bare `uv run` here uses it.

## Setup (one-time)

```bash
uvx dlthub-init@latest   # scaffold a new dltHub workspace (already done here)
uv sync                  # install dependencies
uv run dlthub ai status  # confirm the workspace is healthy
```

## Local development

```bash
uv run python filesystem_pipeline.py                 # load local Claude Code logs -> DuckDB
uv run python rest_api_pipeline.py                    # load agent-traces API -> DuckDB
uv run dlthub local show                              # browse local DuckDB data in a dashboard
uv run marimo edit dashboards/<file>.py --no-token    # open a dashboard locally
```

`Pyright`/`Ruff` are repo-wide dev tools, not part of this subproject's own deps —
run them from the repo root, or with `--project "<repo root>"` from in here.

## Deploy to dltHub Platform

```bash
uv run dlthub login                     # one-time device-code login
uv run dlthub workspace connect         # link this project to a workspace
uv run dlthub show                      # open the workspace in the platform UI
uv run dlthub deploy                    # ship the current code as a new deployment
uv run dlthub run <file_or_job>         # run a pipeline job on the cloud
```

## Share a dashboard

```bash
uv run dlthub job publish <dashboard_job_name>   # make it publicly accessible via URL
uv run dlthub job list                           # confirm jobs (and schedules) are registered
```
