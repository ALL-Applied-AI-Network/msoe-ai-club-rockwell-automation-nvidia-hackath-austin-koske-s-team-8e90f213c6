# API

FastAPI starter. Python3.11+; no database or authentication setup required.

```text
uv sync --frozen
uv run uvicorn main:app --reload --host 127.0.0.1 --port 8443
uv run pytest
```

`/health` checks the process. `/docs` describes the widgets CRUD example.
Routers handle HTTP, services handle behavior and the data adapter owns disposable
in-memory objects. Replace the widgets example as the team's domain takes shape.
Restarting the process clears example data; it is not an operator audit store.
