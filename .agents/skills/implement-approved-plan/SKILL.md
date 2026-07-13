---
name: implement-approved-plan
description: Delegate the latest approved, bounded implementation plan to a low-cost executor. Use only when the approach is settled and the user explicitly invokes this skill; never use it for exploration or redesign.
---

Build a brief from the latest approved decisions:

- Task: requested outcome.
- Boundaries: constraints and excluded changes.
- Done: acceptance criteria and relevant checks.
- References: existing plans, specs, and useful paths.

Reference artifacts instead of copying them. Omit rationale, rejected options, logs, and facts the executor can inspect.

Spawn one `implementation_executor` with the brief and wait. If spawning fails only because its model is unavailable or unsupported, spawn `implementation_executor_mini` with the same brief and wait. Never run both concurrently or fall back for implementation failures.

On `DONE`, inspect the diff and validation before reporting completion.

On `BLOCKED`, surface the executor's evidence and recommended next step. Do not guess, redesign, or retry.
