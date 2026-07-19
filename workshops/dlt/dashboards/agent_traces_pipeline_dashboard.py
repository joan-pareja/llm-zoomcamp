import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import altair as alt
    import dlt
    import marimo as mo

    return alt, dlt, mo


@app.cell
def _(dlt):
    # destination/dataset_name must be explicit here — deployed notebooks
    # can't rely on dlt.attach() alone to resolve the pipeline's saved state.
    pipeline = dlt.attach(
        "agent_traces_pipeline",
        destination="playground",
        dataset_name="agent_traces_dataset",
    )
    dataset = pipeline.dataset()
    return (dataset,)


@app.cell
def _(mo):
    mo.md("""
    # Agent Traces API usage report
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Activity over time
    """)
    return


@app.cell
def _(dataset):
    df_chart1 = dataset("""
        SELECT
            DATE_TRUNC('hour', timestamp) AS hour,
            COUNT(*) AS log_count
        FROM logs
        GROUP BY 1
        ORDER BY 1
    """).df()
    return (df_chart1,)


@app.cell
def _(alt, df_chart1):
    _chart = (
        alt.Chart(df_chart1)
        .mark_line(point=True)
        .encode(
            x=alt.X("hour:T", title="Hour"),
            y=alt.Y("log_count:Q", title="Log entries"),
            tooltip=["hour:T", "log_count:Q"],
        )
        .properties(title="Activity Over Time")
    )
    _chart
    return


@app.cell
def _(mo):
    mo.md("""
    ## Log entries by type
    """)
    return


@app.cell
def _(dataset):
    df_chart2 = dataset("""
        SELECT
            type,
            COUNT(*) AS entry_count
        FROM logs
        GROUP BY 1
        ORDER BY 2 DESC
    """).df()
    return (df_chart2,)


@app.cell
def _(alt, df_chart2):
    _chart = (
        alt.Chart(df_chart2)
        .mark_bar()
        .encode(
            x=alt.X("entry_count:Q", title="Entries"),
            y=alt.Y("type:N", sort="-x", title="Type"),
            tooltip=["type:N", "entry_count:Q"],
        )
        .properties(title="Log Entries by Type")
    )
    _chart
    return


@app.cell
def _(mo):
    mo.md("""
    ## Messages by model
    """)
    return


@app.cell
def _(dataset):
    df_chart3 = dataset("""
        SELECT
            message__model AS model,
            COUNT(*) AS message_count
        FROM logs
        WHERE message__model IS NOT NULL
        GROUP BY 1
        ORDER BY 2 DESC
    """).df()
    return (df_chart3,)


@app.cell
def _(alt, df_chart3):
    _chart = (
        alt.Chart(df_chart3)
        .mark_bar()
        .encode(
            x=alt.X("message_count:Q", title="Messages"),
            y=alt.Y("model:N", sort="-x", title="Model"),
            tooltip=["model:N", "message_count:Q"],
        )
        .properties(title="Messages by Model")
    )
    _chart
    return


@app.cell
def _(mo):
    mo.md("""
    ## Token usage over time
    """)
    return


@app.cell
def _(dataset):
    df_chart4 = dataset("""
        SELECT DATE_TRUNC('hour', timestamp) AS hour, 'input' AS token_type, SUM(usage__input_tokens) AS tokens
        FROM logs
        GROUP BY 1
        UNION ALL
        SELECT DATE_TRUNC('hour', timestamp) AS hour, 'output' AS token_type, SUM(usage__output_tokens) AS tokens
        FROM logs
        GROUP BY 1
        ORDER BY 1
    """).df()
    return (df_chart4,)


@app.cell
def _(alt, df_chart4):
    _chart = (
        alt.Chart(df_chart4)
        .mark_line(point=True)
        .encode(
            x=alt.X("hour:T", title="Hour"),
            y=alt.Y("tokens:Q", title="Tokens"),
            color=alt.Color("token_type:N", title="Direction"),
            tooltip=["hour:T", "token_type:N", "tokens:Q"],
        )
        .properties(title="Token Usage Over Time")
    )
    _chart
    return


@app.cell
def _(mo):
    mo.md("""
    ## Top projects
    """)
    return


@app.cell
def _(dataset):
    df_chart5 = dataset("""
        SELECT
            regexp_extract(cwd, '([^/]+)$') AS project,
            COUNT(*) AS entry_count
        FROM logs
        WHERE cwd IS NOT NULL
        GROUP BY 1
        ORDER BY 2 DESC
    """).df()
    return (df_chart5,)


@app.cell
def _(alt, df_chart5):
    _chart = (
        alt.Chart(df_chart5)
        .mark_bar()
        .encode(
            x=alt.X("entry_count:Q", title="Entries"),
            y=alt.Y("project:N", sort="-x", title="Project"),
            tooltip=["project:N", "entry_count:Q"],
        )
        .properties(title="Activity by Project")
    )
    _chart
    return


@app.cell
def _(mo):
    mo.md("""
    ## Estimated cost over time
    """)
    return


@app.cell
def _(dataset):
    df_chart6 = dataset("""
        WITH priced AS (
            SELECT
                DATE_TRUNC('hour', timestamp) AS hour,
                message__model AS model,
                CASE message__model
                    WHEN 'claude-haiku-4-5-20251001' THEN 1.00
                    WHEN 'claude-sonnet-5' THEN 3.00
                    WHEN 'claude-opus-4-8' THEN 5.00
                    WHEN 'claude-fable-5' THEN 10.00
                END AS input_price_per_mtok,
                CASE message__model
                    WHEN 'claude-haiku-4-5-20251001' THEN 5.00
                    WHEN 'claude-sonnet-5' THEN 15.00
                    WHEN 'claude-opus-4-8' THEN 25.00
                    WHEN 'claude-fable-5' THEN 50.00
                END AS output_price_per_mtok,
                usage__input_tokens AS input_tokens,
                usage__output_tokens AS output_tokens
            FROM logs
            WHERE message__model IS NOT NULL
        )
        SELECT
            hour,
            model,
            SUM(
                COALESCE(input_tokens, 0) * input_price_per_mtok
                + COALESCE(output_tokens, 0) * output_price_per_mtok
            ) / 1000000.0 AS estimated_cost_usd
        FROM priced
        GROUP BY 1, 2
        ORDER BY 1, 2
    """).df()
    return (df_chart6,)


@app.cell
def _(alt, df_chart6):
    _chart = (
        alt.Chart(df_chart6)
        .mark_line(point=True)
        .encode(
            x=alt.X("hour:T", title="Hour"),
            y=alt.Y("estimated_cost_usd:Q", title="Estimated cost (USD)"),
            color=alt.Color("model:N", title="Model"),
            tooltip=["hour:T", "model:N", "estimated_cost_usd:Q"],
        )
        .properties(title="Estimated Cost Over Time")
    )
    _chart
    return


if __name__ == "__main__":
    app.run()
