# Client

React + TypeScript + Vite. React Router handles routes, TanStack Query owns
server state and one Axios instance calls the API. No authentication is required
by this local starter.

```text
npm ci
npm run dev
```

Open localhost:3000 while the API runs on127.0.0.1:8443. Vite proxies `/api`.
The widgets page demonstrates create/list/delete; the service/hooks also expose
update. Example data resets when the API restarts.

Keep resource types in `models/`, HTTP calls in `services/`, query/mutation hooks
in `hooks/`, route screens in `pages/` and reusable UI in `components/`.

```text
npm run typecheck
npm run lint
npm run build
```

Set `API_PROXY_TARGET` for another local API port, or `VITE_API_BASE_URL` for an
explicit external API base URL. Neither is required for the default local setup.
