# Analysis Plan: agent_traces_pipeline

## Connection
pipeline: agent_traces_pipeline
dataset: agent_traces_dataset
destination: duckdb

## Profile Summary
| table | rows | key columns | notes |
|-------|------|-------------|-------|
| logs | 20000 | type, timestamp, session_id, message__role, message__model, usage__input_tokens, usage__output_tokens, cwd, git_branch | no PII (synthetic cwd paths, no real usernames); no cache-token fields (this API doesn't report cache usage) |
| logs__message__content | 19668 | type, text, name (tool), id | one row per message content block, for rows where content is a list rather than a plain string |

Anomalies: `message__role`/`message__model`/`message__stop_reason` are null on ~35% of rows — expected, only `assistant`-type rows populate them (35% of rows are `type='user'`). Data spans only ~38 hours (2026-01-01 00:00 to 2026-01-02 14:53), so hourly grain is used instead of daily. `cwd` has 5 distinct clean values already (no path normalization needed, unlike the local-logs pipeline).

## Questions
1. [x] How has activity changed over time? → Chart 1
2. [x] What's the breakdown of log entries by type? → Chart 2
3. [x] Which models have been used, and how often? → Chart 3
4. [x] How has token usage trended over time? → Chart 4
5. [x] Which projects get the most activity? → Chart 5
6. [x] What's the estimated dollar cost, by model, over time? → Chart 6

## Data Gaps
(none)

## Chart 1: Activity Over Time
question: How has activity changed over time?
type: line
x: timestamp (hourly)
y: count(*)
source: logs

```sql
SELECT
    DATE_TRUNC('hour', timestamp) AS hour,
    COUNT(*) AS log_count
FROM logs
GROUP BY 1
ORDER BY 1
```

```altair
alt.Chart(df).mark_line(point=True).encode(
    x=alt.X("hour:T", title="Hour"),
    y=alt.Y("log_count:Q", title="Log entries"),
    tooltip=["hour:T", "log_count:Q"],
).properties(title="Activity Over Time")
```

## Chart 2: Log Entries by Type
question: What's the breakdown of log entries by type?
type: bar
x: count(*)
y: type
source: logs

```sql
SELECT
    type,
    COUNT(*) AS entry_count
FROM logs
GROUP BY 1
ORDER BY 2 DESC
```

```altair
alt.Chart(df).mark_bar().encode(
    x=alt.X("entry_count:Q", title="Entries"),
    y=alt.Y("type:N", sort="-x", title="Type"),
    tooltip=["type:N", "entry_count:Q"],
).properties(title="Log Entries by Type")
```

## Chart 3: Messages by Model
question: Which models have been used, and how often?
type: bar
x: count(*)
y: message__model
source: logs

```sql
SELECT
    message__model AS model,
    COUNT(*) AS message_count
FROM logs
WHERE message__model IS NOT NULL
GROUP BY 1
ORDER BY 2 DESC
```

```altair
alt.Chart(df).mark_bar().encode(
    x=alt.X("message_count:Q", title="Messages"),
    y=alt.Y("model:N", sort="-x", title="Model"),
    tooltip=["model:N", "message_count:Q"],
).properties(title="Messages by Model")
```

## Chart 4: Token Usage Over Time
question: How has token usage trended over time?
type: line
x: timestamp (hourly)
y: sum(tokens)
source: logs

```sql
SELECT DATE_TRUNC('hour', timestamp) AS hour, 'input' AS token_type, SUM(usage__input_tokens) AS tokens
FROM logs
GROUP BY 1
UNION ALL
SELECT DATE_TRUNC('hour', timestamp) AS hour, 'output' AS token_type, SUM(usage__output_tokens) AS tokens
FROM logs
GROUP BY 1
ORDER BY 1
```

```altair
alt.Chart(df).mark_line(point=True).encode(
    x=alt.X("hour:T", title="Hour"),
    y=alt.Y("tokens:Q", title="Tokens"),
    color=alt.Color("token_type:N", title="Direction"),
    tooltip=["hour:T", "token_type:N", "tokens:Q"],
).properties(title="Token Usage Over Time")
```

## Chart 5: Top Projects
question: Which projects get the most activity?
type: bar
x: count(*)
y: project (extracted from cwd)
source: logs

```sql
SELECT
    regexp_extract(cwd, '([^/]+)$') AS project,
    COUNT(*) AS entry_count
FROM logs
WHERE cwd IS NOT NULL
GROUP BY 1
ORDER BY 2 DESC
```

```altair
alt.Chart(df).mark_bar().encode(
    x=alt.X("entry_count:Q", title="Entries"),
    y=alt.Y("project:N", sort="-x", title="Project"),
    tooltip=["project:N", "entry_count:Q"],
).properties(title="Activity by Project")
```

## Chart 6: Estimated Cost Over Time
question: What's the estimated dollar cost, by model, over time?
type: line
x: timestamp (hourly)
y: sum(estimated_cost_usd)
source: logs

Pricing is hardcoded per model (standard API pricing, per 1M tokens — source: Anthropic Claude API pricing reference, same as the local-logs report). This API doesn't report cache-token usage, so no cache multipliers are needed here.

| Model | Input $/1M | Output $/1M |
|-------|-----------|-------------|
| claude-haiku-4-5-20251001 | $1.00 | $5.00 |
| claude-sonnet-5 | $3.00 | $15.00 |
| claude-opus-4-8 | $5.00 | $25.00 |
| claude-fable-5 | $10.00 | $50.00 |

```sql
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
```

```altair
alt.Chart(df).mark_line(point=True).encode(
    x=alt.X("hour:T", title="Hour"),
    y=alt.Y("estimated_cost_usd:Q", title="Estimated cost (USD)"),
    color=alt.Color("model:N", title="Model"),
    tooltip=["hour:T", "model:N", "estimated_cost_usd:Q"],
).properties(title="Estimated Cost Over Time")
```
