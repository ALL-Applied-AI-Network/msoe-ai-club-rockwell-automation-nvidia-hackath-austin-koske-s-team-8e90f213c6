# Linewatch operator integration

This is the active full-stack integration built from Austin's FastAPI/React
template and Hayden's operator-dashboard prototype.

- [`../client/`](../client/README.md): live camera, image and recording input;
  recipe timelines; persisted artifacts; operator decisions and run history.
- `api/`: FastAPI with router → service → pipeline → databridge boundaries.
- `pipelines/mock.py`: an explicit non-production adapter that keeps the entire
  integration runnable while Team 1/2/3 model adapters are connected.

There is no PLC/actuator control, authentication, or production model decision
in this slice. The operator remains in the loop. SQLite metadata and filesystem
artifacts persist across restarts. Use one API process; the in-process model queue
serializes GPU work by default. The dashboard labels all mock decisions so they
cannot be mistaken for model output.

## Run locally

Use Python3.11+ and Node22.12+ (or a supported newer LTS). Open two terminals.

API, from the repository root:

```text
cd api
uv sync --frozen
uv run uvicorn main:app --reload --host 127.0.0.1 --port 8443
```

Frontend:

```text
cd client
npm ci
npm run dev
```

Open `http://localhost:3000`. The development server proxies `/api`, WebSockets,
and `/health` to the API on 8443. API documentation:
`http://127.0.0.1:8443/docs`; health: `/health`.
No page is opened automatically. Stop both processes with Ctrl+C.

For a different local API port, set `API_PROXY_TARGET` before starting Vite.
For an independently hosted API, set `VITE_API_BASE_URL` explicitly at build time.

## Integration boundaries

`POST /api/v1/sessions` creates a durable inspection run and its initial recipe
segment. `POST /api/v1/sessions/{id}/recipe-segments` adds a transition and the
`/import` endpoint accepts timestamped CSV recipe timelines.
`POST /api/v1/sessions/{id}/frames` accepts a raw JPEG/PNG/WebP body and returns
the persisted source-frame URL, optional post-pipeline rendered frame/mask URLs,
resolved recipe and operator decision. `POST /recordings` streams the original
recording into the artifact store; the browser decoder samples it with original
video timestamps and queues those frames. `GET /sessions` supplies run history,
and artifact URLs support frame review. `WS /api/v1/sessions/{id}/events` streams
events to subscribed dashboards. Failure decisions can be acknowledged. The mock
adapter intentionally leaves post-pipeline visuals empty rather than fabricating
a mask.

Implement production model adapters behind `pipelines/base.py`. Load each model
once at API startup; do not spawn a Python CLI per frame. The intended order is
Team 1 vial detector → Team 2 liquid detector → Team 3 SAM mask refinement →
recipe-aware decision policy. `api/data/linewatch.sqlite3` contains metadata and
`api/data/artifacts/{run-id}/` contains source, recording, rendered and mask files.
Both paths are runtime data and are gitignored.

## Checks

```text
cd api
uv run pytest
```

```text
cd client
npm run typecheck
npm run lint
npm run build
```

Adapted from Austin's local `full-stack-app` source. Authentication and deployment
remain out of scope; dependency locks are included for repeatable installation.
