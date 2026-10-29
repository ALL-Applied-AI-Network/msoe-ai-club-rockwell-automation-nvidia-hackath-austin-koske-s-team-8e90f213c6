# Operator dashboard

React/TypeScript/Vite monitor, run history and frame review. See [Docker demo](../docs/demo.md)
for the laptop launcher. For development: `npm ci`, then `npm run dev` with
`API_PROXY_TARGET=http://127.0.0.1:8001`. Production uses Nginx and the same-origin API.

The monitor accepts original image files, browser-decoded recordings and approved
browser cameras. Live uploads preserve orientation/dimensions and use PNG; only the
webcam preview is mirrored by CSS. Recordings keep a declared browser seek clock;
stills have no invented timestamps. A reference upload is explicitly unreviewed.

Run changes invalidate pending HTTP and WebSocket updates. Stopped, disconnected,
failed and stale live sources cannot retain a current normal status. Historical
source/rendered/mask artifacts remain reviewable. Sampling QA reasons are visible;
missing evidence is unavailable. The retained review queue and history pages save
attributed responses separately from product approval. Earlier frames are paginated.
Mock controls are disabled unless the API explicitly runs in mock mode.

`npm run typecheck`, `npm run lint` and `npm run build` validate the client.
