# Frontend + API starter for the app team

Josiah and Hayden: this is Austin's `full-stack-app` template adapted into a
standalone local starter. Use it as a reference or copy it into your app branch;
it does not replace anyone's dashboard/backend work.

- [`../client/`](../client/README.md): React, TypeScript, Vite, React Router, TanStack Query and Axios.
- `api/`: FastAPI with router → service → in-memory data adapter boundaries.
- The disposable widgets example demonstrates a working UI/API round trip.
  Its state lives only in the API process and resets on restart. Use one worker.

There is no database, SQL driver, authentication requirement, Terraform,
deployment pipeline, model inference or industrial connection in this starter.
It needs no secrets. Keep real operator evidence/history requirements separate
from the disposable widget example when implementing your application.

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

Open `http://localhost:3000`. The development server proxies `/api` to the API
on8443. API documentation: `http://127.0.0.1:8443/docs`; health: `/health`.
No page is opened automatically. Stop both processes with Ctrl+C.

For a different local API port, set `API_PROXY_TARGET` before starting Vite.
For an independently hosted API, set `VITE_API_BASE_URL` explicitly at build time.

## Where to work

Client: `models/` → `services/` → `hooks/` → `pages/` and `components/`.
API: `models/` request/response contracts, `routers/` HTTP, `services/` behavior,
`databridge/` replaceable data adapters, and `dependencies.py` wiring.

Replace widgets with your operator-facing resources and Srijan's agreed model
output contracts. This starter intentionally makes no model, recipe or quality
decision. The dataset/training work remains elsewhere in the repository.

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

Adapted from Austin's local `full-stack-app` source; database/auth/deployment
examples were removed, and widget storage was replaced with a process-local
dictionary. Dependency locks are included for repeatable installation.
