Use `uv` as the package and dependency manager for this repository.

Add every new Python file or notebook to `[tool.pyright].include` in `pyproject.toml`.

Do not run commands that can print or expand environment variable values into output. For example, avoid `docker compose config` when it would resolve secrets from `.env`.
