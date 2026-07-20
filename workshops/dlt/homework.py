import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # dlt workshop — homework

    A FAQ agent built with **Pydantic AI** over our own `lib` search tools
    (sqlitesearch), instrumented with **Logfire**, whose traces we then load
    into DuckDB with **dlt**.

    - **Q1** — run the agent on *"How do I run Ollama locally?"* and count the spans.
    - **Q2** — load the Logfire traces with dlt; how many tables appear?
    - **Q3** — total input tokens for the Q1 run.
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    notebook_dir = str(Path().resolve())
    repo_root = str(Path().resolve().parents[1])
    for path in (repo_root, notebook_dir):
        if path not in sys.path:
            sys.path.insert(0, path)

    import dlt
    import logfire
    import marimo as mo
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )
    from pydantic_ai import Agent
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    from lib import load_faq_documents
    from lib.search import SQLiteLexicalSearchTool

    return (
        Agent,
        InMemorySpanExporter,
        OpenAIChatModel,
        OpenAIProvider,
        SQLiteLexicalSearchTool,
        SimpleSpanProcessor,
        dlt,
        load_faq_documents,
        logfire,
        mo,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## Setup — search tool over the course FAQ

    We reuse our own `SQLiteLexicalSearchTool` from `lib.search`, backed by a
    persisted sqlitesearch index of the DataTalks.Club FAQ.
    """)
    return


@app.cell
def _(SQLiteLexicalSearchTool, load_faq_documents):
    faq_documents = load_faq_documents()
    search_tool = SQLiteLexicalSearchTool.from_documents(
        documents=faq_documents,
        text_fields=["question", "answer", "section"],
        keyword_fields=["course"],
        db_path="faq_lexical_index.db",
    )
    return (search_tool,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Setup — the Pydantic AI agent

    The agent owns the tool-calling loop; our search tool is registered as a
    plain tool. The OpenAI key and model are read explicitly from dlt secrets
    and config, then passed into the provider.
    """)
    return


@app.cell
def _(Agent, OpenAIChatModel, OpenAIProvider, dlt, search_tool):
    from lib.types import FAQDocument

    provider = OpenAIProvider(api_key=dlt.secrets["openai.api_key"])
    model = OpenAIChatModel(dlt.config["openai.model"], provider=provider)

    def search(query: str) -> list[FAQDocument]:
        return search_tool.search(query)

    faq_agent = Agent(
        model,
        instructions=(
            "You answer questions about DataTalks.Club courses. Always call the "
            "search tool to ground your answer in the FAQ, then reply concisely."
        ),
        tools=[search],
    )
    return (faq_agent,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Logfire instrumentation

    `instrument_pydantic_ai()` auto-traces the whole agent run. We also attach
    a local in-memory span exporter via `additional_span_processors`, so the
    same spans that go to Logfire cloud are counted here for Q1.
    """)
    return


@app.cell
def _(InMemorySpanExporter, SimpleSpanProcessor, dlt, logfire):
    span_collector = InMemorySpanExporter()

    logfire.configure(
        token=dlt.secrets["sources.logfire_source.write_token"],
        service_name="faq-agent",
        environment="homework",
        additional_span_processors=[SimpleSpanProcessor(span_collector)],
    )
    logfire.instrument_pydantic_ai()
    return (span_collector,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Q1 — First trace

    Run the agent on the homework query and count the spans it produces.
    """)
    return


@app.cell
async def _(faq_agent, span_collector):
    span_collector.clear()
    q1_result = await faq_agent.run("How do I run Ollama locally?")
    q1_spans = span_collector.get_finished_spans()
    q1_span_count = len(q1_spans)
    q1_span_names = [span.name for span in q1_spans]
    return q1_result, q1_span_count, q1_span_names


@app.cell
def _(mo, q1_result, q1_span_count, q1_span_names):
    mo.md(f"""
    **Answer:** {q1_result.output}

    **Spans captured:** {q1_span_count}

    **Span names:** {", ".join(q1_span_names)}

    **Usage:** {q1_result.usage.input_tokens} input · {q1_result.usage.output_tokens} output · {q1_result.usage.requests} request(s)
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Q2 — Load the Logfire traces into DuckDB

    Reuse `load_logfire_traces` (our dlt REST source over Logfire's Query API)
    to load into a local DuckDB dataset named `agent_traces`, then count the
    tables dlt created. Give Logfire a few seconds to ingest the Q1 run first;
    re-run this cell if the load comes back empty.
    """)
    return


@app.cell
def _():
    from rest_api_pipeline import load_logfire_traces

    logfire_pipeline = load_logfire_traces()
    return (logfire_pipeline,)


@app.cell
def _(logfire_pipeline):
    with logfire_pipeline.sql_client() as _client:
        _rows = _client.execute_sql(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'agent_traces' ORDER BY table_name"
        )
    q2_table_names = [row[0] for row in _rows]
    q2_table_count = len(q2_table_names)
    return q2_table_count, q2_table_names


@app.cell
def _(mo, q2_table_count, q2_table_names):
    mo.md(f"""
    **Tables created:** {q2_table_count}

    {", ".join(q2_table_names)}
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Q3 — Total input tokens for the Q1 run

    Token usage lives in the JSON `attributes` column (kept as JSON to avoid the
    nested-table explosion). Read the latest agent run's aggregated input tokens
    and cross-check against Pydantic AI's own usage total.
    """)
    return


@app.cell
def _(logfire_pipeline):
    with logfire_pipeline.sql_client() as _client:
        _rows = _client.execute_sql(
            "SELECT json_extract(attributes, "
            "'$.\"gen_ai.aggregated_usage.input_tokens\"') "
            "FROM agent_traces.records WHERE span_name LIKE 'invoke_agent%' "
            "ORDER BY start_timestamp DESC LIMIT 1"
        )
    q3_input_tokens = int(_rows[0][0]) if _rows and _rows[0][0] is not None else None
    return (q3_input_tokens,)


@app.cell
def _(mo, q1_result, q3_input_tokens):
    mo.md(f"""
    **Total input tokens (from Logfire via dlt):** {q3_input_tokens}

    **Cross-check (Pydantic AI usage):** {q1_result.usage.input_tokens}
    """)
    return


if __name__ == "__main__":
    app.run()
