# Analysis Plan: claude_logs_pipeline

## Connection
pipeline: claude_logs_pipeline
dataset: claude_logs
destination: duckdb

## Profile Summary
| table | rows | key columns | notes |
|-------|------|-------------|-------|
| logs | 1527 | type, timestamp, session_id, message__role, message__model, message__usage__* , cwd | PII: cwd embeds local Windows username; tool_use_result/attachment are JSON-typed blobs (not unnested) |
| logs__message__content | 1060 | type, text, name (tool), tool_use_id | one row per message content block |
| logs__message__usage__iterations | 654 | input_tokens, output_tokens, type | per-iteration token usage |

Anomalies: `message__role` and `message__model` are null on ~31%/56% of rows — expected, since only `assistant`/`user` message rows populate them (other rows are `queue-operation`, `attachment`, `ai-title`, etc.). `cwd` has 8 raw distinct values including case-variant duplicates and `.venv` library-code paths that belong to the same project.

## Questions
1. [x] How has Claude Code activity changed over time? → Chart 1
2. [x] What's the breakdown of log entries by type? → Chart 2
3. [x] Which models have been used, and how often? → Chart 3
4. [x] How has token usage trended over time? → Chart 4
5. [x] Which projects get the most activity? → Chart 5
6. [x] What's the estimated dollar cost, by model, over time? → Chart 6

## Data Gaps
(none)

## Chart 1: Activity Over Time
question: How has Claude Code activity changed over time?
type: line
x: timestamp (daily)
y: count(*)
source: logs

```sql
SELECT
    DATE_TRUNC('day', timestamp) AS day,
    COUNT(*) AS log_count
FROM logs
GROUP BY 1
ORDER BY 1
```

```altair
alt.Chart(df).mark_line(point=True).encode(
    x=alt.X("day:T", title="Day"),
    y=alt.Y("log_count:Q", title="Log entries"),
    tooltip=["day:T", "log_count:Q"],
).properties(title="Claude Code Activity Over Time")
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
x: timestamp (daily)
y: sum(tokens)
source: logs

```sql
SELECT DATE_TRUNC('day', timestamp) AS day, 'input' AS token_type, SUM(message__usage__input_tokens) AS tokens
FROM logs
GROUP BY 1
UNION ALL
SELECT DATE_TRUNC('day', timestamp) AS day, 'output' AS token_type, SUM(message__usage__output_tokens) AS tokens
FROM logs
GROUP BY 1
ORDER BY 1
```

```altair
alt.Chart(df).mark_line(point=True).encode(
    x=alt.X("day:T", title="Day"),
    y=alt.Y("tokens:Q", title="Tokens"),
    color=alt.Color("token_type:N", title="Direction"),
    tooltip=["day:T", "token_type:N", "tokens:Q"],
).properties(title="Token Usage Over Time")
```

## Chart 5: Top Projects
question: Which projects get the most activity?
type: bar
x: count(*)
y: project (normalized from cwd)
source: logs

```sql
SELECT
    lower(cwd) AS cwd,
    COUNT(*) AS entry_count
FROM logs
WHERE cwd IS NOT NULL
GROUP BY 1
ORDER BY 2 DESC
```

Post-aggregation normalization (raw `cwd` values are collapsed to a project folder name — dedupes case-variant paths and folds `.venv` library-code paths back into the project they belong to):

```python
import re

def project_name(cwd: str) -> str:
    match = re.search(r"[\\/](?:git|github)[\\/]([^\\/]+)", cwd)
    if match:
        return match.group(1)
    match = re.search(r"onedrive[^\\/]*[\\/]([^\\/]+)", cwd)
    if match:
        return match.group(1)
    return cwd

df["project"] = df["cwd"].apply(project_name)
df = df.groupby("project", as_index=False)["entry_count"].sum().sort_values("entry_count", ascending=False)
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
x: timestamp (daily)
y: sum(estimated_cost_usd)
source: logs

Pricing is hardcoded per model (standard API pricing, per 1M tokens, as of 2026-06-24 — source: Anthropic Claude API pricing reference). Cache write/read multipliers follow Anthropic's standard formula: 5-minute cache writes cost 1.25x the input price, 1-hour cache writes cost 2x, and cache reads cost 0.1x.

| Model | Input $/1M | Output $/1M |
|-------|-----------|-------------|
| claude-sonnet-5 | $3.00 | $15.00 |
| claude-sonnet-4-6 | $3.00 | $15.00 |
| claude-haiku-4-5-20251001 | $1.00 | $5.00 |

Note: Sonnet 5 has an introductory discount ($2.00/$10.00) active through 2026-08-31 — standard pricing is used here since it's the durable reference; actual near-term cost is somewhat lower.

```sql
WITH priced AS (
    SELECT
        DATE_TRUNC('day', timestamp) AS day,
        message__model AS model,
        CASE message__model
            WHEN 'claude-sonnet-5' THEN 3.00
            WHEN 'claude-sonnet-4-6' THEN 3.00
            WHEN 'claude-haiku-4-5-20251001' THEN 1.00
        END AS input_price_per_mtok,
        CASE message__model
            WHEN 'claude-sonnet-5' THEN 15.00
            WHEN 'claude-sonnet-4-6' THEN 15.00
            WHEN 'claude-haiku-4-5-20251001' THEN 5.00
        END AS output_price_per_mtok,
        message__usage__input_tokens AS input_tokens,
        message__usage__output_tokens AS output_tokens,
        message__usage__cache_creation__ephemeral_5m_input_tokens AS cache_write_5m_tokens,
        message__usage__cache_creation__ephemeral_1h_input_tokens AS cache_write_1h_tokens,
        message__usage__cache_read_input_tokens AS cache_read_tokens
    FROM logs
    WHERE message__model IN ('claude-sonnet-5', 'claude-sonnet-4-6', 'claude-haiku-4-5-20251001')
)
SELECT
    day,
    model,
    SUM(
        COALESCE(input_tokens, 0) * input_price_per_mtok
        + COALESCE(output_tokens, 0) * output_price_per_mtok
        + COALESCE(cache_write_5m_tokens, 0) * input_price_per_mtok * 1.25
        + COALESCE(cache_write_1h_tokens, 0) * input_price_per_mtok * 2.0
        + COALESCE(cache_read_tokens, 0) * input_price_per_mtok * 0.1
    ) / 1000000.0 AS estimated_cost_usd
FROM priced
GROUP BY 1, 2
ORDER BY 1, 2
```

```altair
alt.Chart(df).mark_line(point=True).encode(
    x=alt.X("day:T", title="Day"),
    y=alt.Y("estimated_cost_usd:Q", title="Estimated cost (USD)"),
    color=alt.Color("model:N", title="Model"),
    tooltip=["day:T", "model:N", "estimated_cost_usd:Q"],
).properties(title="Estimated Cost Over Time")
```
