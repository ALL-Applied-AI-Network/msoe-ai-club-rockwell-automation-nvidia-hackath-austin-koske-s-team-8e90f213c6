# Development

The operator app defaults to a deterministic mock pipeline. Model mode uses the
same FastAPI entrypoint and loads YOLO plus the Temporal CNN lazily on the first
frame. Compute setup and standalone model verification remain separate.

## Run liquid inference

Download and extract `temporal-cnn-assets-v1.zip` from the
[model release](https://github.com/ALL-Applied-AI-Network/msoe-ai-club-rockwell-automation-nvidia-hackath-austin-koske-s-team-8e90f213c6/releases/tag/temporal-cnn-v1.0.0).
Keep it outside the repository. Use an environment with PyTorch, NumPy, Pillow
and OpenCV; [tested versions](../setup/requirements/temporal-pool.in) are recorded
in the repository. Install the PyTorch wheel for the actual host. The Windows
and GB10 CUDA wheels differ.

From the extracted asset directory:

```bash
python verify_temporal_handoff.py --assets . --out ../verification-01
python temporal_pool.py --checkpoint models/liquid-best.pt --current examples/run4-fill1-f000339.png --reference examples/run4-fill1-f000264.png --out ../prediction-01 --device cpu
```

Both commands require fresh output paths. Verification checks asset hashes and
replays two frozen machine predictions. Inference writes `observation.json`,
`raw_mask.png`, `component_mask.png` and `proposal_mask.png`. It runs the liquid
model only, without vial detection, sampling QA or recipe comparison.

The same scripts live in `setup/vision/`. For a worker, load
`TemporalPool(checkpoint, device)` once and call `predict(current, reference)`
for each source pair. Inputs must be RGB `uint8` arrays at 640 × 480.
Use `cuda:0` only on the approved GPU host; CPU is the local default.

## Operator app

Requirements: Python 3.11+, `uv`, and Node 22.12+ or a supported newer LTS.
Use two terminals from the repository root.

```bash
cd api
uv sync --frozen
uv run uvicorn main:app --reload --host 127.0.0.1 --port 8443
```

```bash
cd client
npm ci
npm run dev
```

Open `http://localhost:3000`. Vite proxies `/api` to port 8443; API docs are at
`http://127.0.0.1:8443/docs`, health at `/health`. Stop both with Ctrl+C.
Inspection metadata is stored in SQLite and source/rendered/mask/recording
artifacts are stored under `api/data/`; both are local runtime state excluded
from Git. The dashboard supports live-camera, image and recording source setup,
recipe-segment import, run history and frame review. Recording upload currently
stores the original file and the browser decodes sampled frames with millisecond
media seek timestamps. A backend decoder that retains native PTS/time base is
still an integration step.

Recipe CSV import can be staged before a run starts, then attaches to the run
created for the selected source mode. Once a run starts its source mode is
locked; use **New Run** to change sources. The import assigns recipe context by
frame timestamp only. Descriptive color intent is not a qualified recipe color
reference or acceptance decision, and re-importing identical rows does not create
duplicates.
The dashboard also persists an ordered vial plan. Before starting a run, the
operator may add and reorder vial steps and provide descriptive expected-color
labels/swatches plus Tank 100/200/300/400 volumes. The displayed recipe total is
calculated from those inputs. During an operator-led run, **Activate** records the selected
vial as a new timestamped context segment. This is operator intent and remains
separate from measured RGB/Lab evidence and qualified recipe references.
Each run retains its initial recipe as a fallback context. Image batches currently
receive ordered timestamps at 0, 1, 2, … seconds; recording ingestion uses media
seek timestamps. A recording-timestamp CSV should therefore be paired with the
recording replay unless external image timestamps are added.

To enable real frame inference, install the recorded model requirements into the
API environment and copy `api/example.env` to the ignored `api/.env`. Set these
values to the extracted, hash-verified release assets:

```dotenv
APP__PIPELINE_MODE=model
APP__MODEL_WORKERS=1
APP__YOLO_CHECKPOINT=/absolute/path/temporal-cnn-assets-v1/models/vial-best.pt
APP__TEMPORAL_CHECKPOINT=/absolute/path/temporal-cnn-assets-v1/models/liquid-best.pt
APP__TEMPORAL_REFERENCE_IMAGE=/absolute/path/to/reviewed-aligned-empty.png
APP__TEMPORAL_REFERENCE_QUALIFIED=false
```

Install `setup/requirements/temporal-pool.in` with the API environment's package
manager before starting the server. Keep `TEMPORAL_REFERENCE_QUALIFIED=false`
until a human has independently reviewed the aligned empty-vial reference. The
release example reference is unreviewed and is suitable only for an explicitly
unqualified smoke test. `/health` reports `pipeline_mode` and `pipeline_status`;
`reference_review_required` means inference is available but recipe conclusions
must remain unavailable.

Client boundaries are types → services → hooks → pages/components. API
boundaries are models → routers → services → databridge, with dependency wiring
in `dependencies.py`. The model bridge lives in
`pipelines/temporal_cnn_bridge.py`; pipeline output crosses those existing
layers rather than bypassing persistence.

`integration/model_pipeline.py` is the single reusable implementation of model
loading, per-pair inference, vial association, measurement QA and visualization.
`integration/integration.py` is a thin one-pair CLI and
`integration/batch_integration.py` is a thin ordered-pairs CLI; neither contains
a separate copy of the inference algorithm. The FastAPI bridge imports the same
core directly.
`API_PROXY_TARGET` changes Vite's local proxy; `VITE_API_BASE_URL` selects a
separately hosted API at build time. Defaults need no credentials or database.

## Checks

From the repository root, in an environment with the relevant dependencies:

```bash
python -m unittest discover -s replay/tests -p test_geometry.py -v
python -m unittest discover -s annotation_review/tests -p 'test_*scoring.py' -v
python -m unittest discover -s setup/vision -p test_onboarding_readiness.py -v
```

These are the 16 focused checks used for the model publication. Frozen-example
replay is a separate check requiring the released assets.

For app changes:

```bash
cd api
uv run pytest
```

```bash
cd client
npm run typecheck
npm run lint
npm run build
```

## Repository layout

| Path | Responsibility |
| --- | --- |
| `api/` | HTTP contracts, services and data adapters |
| `client/` | Browser application |
| `setup/vision/` | Model definition, inference, training and experiment helpers |
| `replay/` | Geometry, sampling QA and reference/evidence checks |
| `annotation_review/` | Human review and independent scoring |
| `setup/` | Environment/bootstrap, remote runners and verification records |
| `docs/` | Maintained technical documentation |

See [compute setup](compute.md) for Windows data tools, GB10 rendering and Rosie
jobs. Research scripts can require the original experiment bundles and clocks;
the two-image asset example is not a replacement for those inputs.
