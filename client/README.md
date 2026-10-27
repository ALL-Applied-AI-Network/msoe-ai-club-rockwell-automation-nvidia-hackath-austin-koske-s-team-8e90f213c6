# Linewatch client

React + TypeScript + Vite operator dashboard. It supports live cameras, dragged
image batches and dragged recordings. Recording replay uses the browser's media
decoder, samples at a configurable interval and sends the original video timestamp
with each queued frame. Timestamped recipe CSVs can be imported before or during
a run. Run-history and frame-review routes display persisted source, rendered and
mask artifacts together with the decision and resolved recipe context.

```text
npm ci
npm run dev
```

Open localhost:3000 while the API runs on127.0.0.1:8443. Vite proxies `/api`.
The warning/failure buttons are integration-test controls. Their results are
explicitly labeled simulated and are not model predictions. Camera access and
recording decoding stay in the browser; original recordings and sampled artifacts
are persisted by the local API. No PLC or actuator command is sent.

Keep resource types in `models/`, HTTP calls in `services/`, query/mutation hooks
in `hooks/`, route screens in `pages/` and reusable UI in `components/`.

```text
npm run typecheck
npm run lint
npm run build
```

Set `API_PROXY_TARGET` for another local API port, or `VITE_API_BASE_URL` for an
explicit external API base URL. Neither is required for the default local setup.
