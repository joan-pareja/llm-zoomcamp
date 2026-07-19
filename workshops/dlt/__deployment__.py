"""Agent traces workspace — ingest agent trace logs from the test-agent-traces API into DuckDB."""

import dashboards.agent_traces_pipeline_dashboard as agent_traces_dashboard
from rest_api_pipeline import load_logs

__all__ = ["load_logs", "agent_traces_dashboard"]
