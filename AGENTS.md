Use `uv` as the package and dependency manager for this repository.

Add every new Python file or notebook to `[tool.pyright].include` in `pyproject.toml`.

Do not run commands that can print or expand environment variable values into output. For example, avoid `docker compose config` when it would resolve secrets from `.env`.

## dltHub subprojects (e.g. `workshops/dlt`)

dltHub-scaffolded subprojects share this repo's root `pyproject.toml`/`uv` environment rather than their own. After running `dlthub pipeline init` (or similar scaffolding commands) inside one, merge any generated `requirements.txt` into the root `pyproject.toml` via `uv add` and delete `requirements.txt`, and delete any per-subproject `.gitignore` it (re)creates — it's redundant with the root one.

## Coding standards

### Naming

- Booleans read as predicates: prefix with `is_`, `has_`, `should_`, or `can_`, phrased positively (avoid `is_not_x`).

### Explicitness

- Load each credential/config value by name and pass it explicitly into the library that uses it. Avoid implicit magic — blanket `load_dotenv()` or letting a library auto-discover a token from the environment — which hides what's wired to what.

### Comments

- Don't explain things to the user in code comments — explanations belong in chat. Add a comment only when the code itself genuinely needs one (a non-obvious workaround or constraint), never to narrate intent or point at where something else lives.
