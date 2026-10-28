# Linewatch inspection API

FastAPI service for multi-source frame ingestion, recipe timelines, persistent
run history, model work queues, and operator-in-the-loop results.

The service stores metadata in SQLite and source/rendered/mask/recording files
under `api/data/`. The React client in `../client/` consumes these endpoints.

## Run locally

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
