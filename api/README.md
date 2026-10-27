# Inspection API

FastAPI owns sessions, recipe timelines, queued model work, source/artifact storage,
WebSocket delivery and attributed operator responses. SQLite metadata and media files
persist across restarts. There is no PLC, sorter, valve or sensor connection.

The default adapter is `pipelines/temporal.py`: YOLO vial geometry → paired temporal
U-Net → the existing measurement QA. Models load once and inference runs off the
HTTP event loop. State is separate per run and resets on source/reference changes,
rewind and independent stills. The frozen measurement thresholds are unchanged.
Explicit `APP__PIPELINE_MODE=mock` enables labeled integration test controls.

Use [the Docker demo instructions](../docs/demo.md) for a laptop with no Python/Node
setup. For development, install the base API with `uv sync --frozen`, the CPU model
requirements with `uv pip install -r requirements-model.txt`, and point
`APP__MODEL_ASSETS` at the extracted release. `APP__REFERENCE_IMAGE` is an optional
preview reference; a run can instead upload a reference through `/reference`.
Run `uv run --no-sync uvicorn main:app --host 127.0.0.1 --port 8001`.

## Contracts

- `/api/v1/sessions`: create/list runs with explicit source mode and operator context.
- `/sessions/{id}/frames`: raw PNG/JPEG/WebP and frame metadata. Recording timestamps
  are browser seek positions, live timestamps are browser elapsed time, and stills
  have no inferred time. Model input must be original 640×480 RGB; never resize/mirror.
- `/sessions/{id}/reference`: retain an earlier aligned image and reset run QA state.
  An upload stays unreviewed; it cannot certify emptiness, alignment or recipe quality.
- `/sessions/{id}/recordings`: retain the original recording. Sampling is currently
  performed by the browser, not a native-PTS video worker.
- `/sessions/{id}/recipe-segments/import`: atomically replace the timeline using
  `recipe_id,start_s,end_s,vial` (or `start_ms,end_ms`). Half-open intervals; missing,
  overlapping or inverted times are rejected. Frame-index CSV needs an explicit clock
  mapping before import. Gaps stay unmatched.
- `/sessions/{id}/events`: newest first; `limit`, `before_frame`, `review_only` support
  complete evidence review and an outstanding review queue.
- `/sessions/{id}/events/{event_id}/acknowledgement`: persist reviewer, response and note.
- `/artifacts/{id}`: original source, rendered overlay, transparent proposal mask,
  reference and original recording bytes, each with a stored hash.

Events include actual measurement QA, reasons, sampling-mask RLE, model/reference
identities, source hashes and separate proposal/sample counts. A valid color sample
still does not establish recipe agreement. The demo has no independently qualified
recipe endpoint/completion/identity inputs, so it never approves product or issues
hardware commands. Progress/confidence are not fabricated for the model adapter.

Run one API process. The model adapter serializes inference, the queue is bounded,
and slow/disconnected WebSockets cannot indefinitely block a model result.
`cd api && uv run --no-sync pytest` runs the API contract/regression checks.
