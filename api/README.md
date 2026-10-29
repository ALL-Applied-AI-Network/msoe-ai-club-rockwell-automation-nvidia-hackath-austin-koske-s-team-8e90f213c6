# Linewatch inspection API

FastAPI service for multi-source frame ingestion, recipe timelines, persistent
run history, model work queues, and operator-in-the-loop results.

The service stores metadata in SQLite and source/rendered/mask/recording files
under `api/data/`. The React client in `../client/` consumes these endpoints.

With `APP__PIPELINE_MODE=model` and `APP__MODEL_ASSETS` pointing at the extracted
`temporal-cnn-assets-v1` release, the API uses `pipelines/temporal.py` instead of the
checkpoint bridge: the same YOLO → paired temporal U-Net → measurement QA, loaded from
the hash-checked release bundle. The GB10 kit, `compose.yaml` and `onboard.sh` use this
mode. `APP__REFERENCE_IMAGE` is an optional preview reference; a run can instead upload
one through `/reference`.

## Contracts

From the repository root:

```text
cd api
uv sync --frozen
uv run uvicorn main:app --reload --host 127.0.0.1 --port 8443
```

API documentation is available at `http://127.0.0.1:8443/docs` outside the
production environment. Health and model readiness are reported at `/health`.

## Pipeline modes

The default `mock` mode exercises ingestion, persistence, queueing and operator
states without claiming model inference. Its test controls are visible only in
mock mode.

The Temporal CNN bridge is enabled with `APP__PIPELINE_MODE=model`. It adapts
the reusable `integration/model_pipeline.py` core into the API's frame,
artifact and operator-review contracts. It runs:

1. YOLO vial detection.
2. Paired Temporal CNN liquid proposal using a qualified empty reference.
3. Same-frame vial association.
4. Stateful measurement QA, isolated per inspection run.
5. Rendered frame, mask and operator-review metadata persistence.

That core consolidates Srijan's former single-frame and batch integration
implementations; both CLI entrypoints and FastAPI now call the same functions.

## Recipe plans

`PUT /api/v1/sessions/{session_id}/recipe-plan` persists the operator's ordered
vial plan. Each step contains a recipe, vial position and optional descriptive
expected-color intent plus Tank 100/200/300/400 volumes in mL. `GET` on the same
path returns the durable order for run
review. Activating a step creates a normal recipe segment, so subsequent frames
resolve the intended vial by source time without treating the plan as a model
prediction or qualified color reference.

Copy `example.env` to `api/.env` and configure the two ignored checkpoints and
an aligned 640×480 empty-vial reference. Set
`APP__TEMPORAL_REFERENCE_QUALIFIED=true` only after that reference has been
independently reviewed. A temporal reference is not a recipe endpoint reference;
recipe acceptance remains an explicit future decision layer.

The pinned vision packages are listed in
`../setup/requirements/temporal-pool.in`. Install the appropriate PyTorch wheel
for the target CPU/CUDA host before enabling model mode. Keep one model worker
unless shared-model concurrency is deliberately validated.

## Checks

```text
cd api
uv run pytest
```

The bridge tests use a fake inference backend, so API contract and safety-state
tests do not require GPU packages or checkpoints. Actual model execution still
requires the configured assets and vision environment.

## Endpoints

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
