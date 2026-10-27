# Development

Inference, the app skeleton and compute setup are separate entrypoints. Running
the widgets app does not start a model worker.

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

For a laptop, use [the Docker demo](demo.md). It runs the built React client, Nginx,
FastAPI, the selected models and persisted inspection history. Mock controls need
explicit mock mode; the default never substitutes simulated success for inference.

For development, run `uv sync --frozen` in `api/`, then install the model dependencies
from `api/requirements-model.txt` into that environment. Set `APP__MODEL_ASSETS` and
optionally `APP__REFERENCE_IMAGE` to the extracted release paths. Start
`uv run --no-sync uvicorn main:app --host 127.0.0.1 --port 8001`.
In `client/`, use `npm ci`, then `API_PROXY_TARGET=http://127.0.0.1:8001 npm run dev`.
The API/router/service/databridge boundaries remain intact. Do not run `uv sync`
without the model dependencies just before the demo; synchronization removes
packages outside the base API lock.

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
