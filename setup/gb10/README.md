# GB10 deployment

Gets the released YOLO11n/U-Net checkpoints, unchanged measurement QA, FastAPI and the React
UI running on an NVIDIA GB10 (aarch64 Linux). The primary path is a GitHub checkout plus
`onboard.sh` (Docker, GPU). The backup is a USB kit that needs no Docker, Node, root or system
Python: it carries CPython 3.12.15 and installs exact, hash-locked packages.

## Primary: GitHub checkout + onboarding (Docker)

```bash
git clone -b austin-work git@github.com:ALL-Applied-AI-Network/msoe-ai-club-rockwell-automation-nvidia-hackath-austin-koske-s-team-8e90f213c6.git linewatch
cd linewatch
gh auth login                       # or: export GITHUB_TOKEN=...  (private model release)
bash setup/gb10/onboard.sh          # or: --assets /media/usb/temporal-cnn-assets-v1.zip
```

`onboard.sh` records machine facts, downloads `temporal-cnn-assets-v1.zip` from the
`temporal-cnn-v1.0.0` release and checks it against `SHA256SUMS.txt` and the checkpoint pins.
It then builds and starts the stack, runs the self-test and prints the URL. It tries, in order:

1. Docker with the GPU (`compose.gpu.yaml`, torch cu130, needs the NVIDIA Container Toolkit)
   and confirms CUDA inside the container;
2. Docker on CPU;
3. the native launcher below, reusing the built UI from the client image.

Everything is logged to `~/linewatch-onboarding/<timestamp>/` (`facts.txt`, build logs,
`selftest-report.json`, `summary.json`). Keep that folder; it is the device record. Options:
`--port`, `--cpu`, `--native`, `--no-selftest`. Stop with `docker compose stop`; never `down -v`.

## Backup: USB kit, no Docker (any account, any GB10)

```bash
tar -xzf linewatch-gb10.tar.gz        # or linewatch-gb10-offline.tar.gz from USB
cd linewatch-gb10
bash START.sh --check                 # 1. readiness: arch, GPU, internet/wheels, cameras, port
bash START.sh --selftest              # 2. install + prove outputs match our reference run
bash START.sh                         # 3. serve http://127.0.0.1:8788; open it in a local browser
```

- `--device auto` (default) uses CUDA when the GPU check passes and otherwise falls back to CPU
  with a message. Force with `--device cuda` or `--device cpu`.
- The offline kit installs from its `wheels/` folder automatically and never contacts the
  internet. The online kit downloads the same locked files from PyPI and the PyTorch index.
- `--port` changes the port if 8788 is busy. `--host 0.0.0.0` exposes the UI on the network;
  use it only where organizers allow. Inspection data persists in `~/linewatch-data` (`--data`).
- The self-test posts 40 reference frames to a temporary server and compares every frame's
  proposal/QA decision and Lab color with frozen outputs (`selftest-expected.json`). It writes
  `selftest-report.json` under the data folder. Passing shows this machine reproduces our
  software outputs; it is not an accuracy measurement.

Installs go into `.venv-gb10-cuda` / `.venv-gb10-cpu` inside the kit and are reused on later
starts. Delete the kit folder to remove everything; data stays in the data folder.

## Validated

On spark8 (GB10, Ubuntu 24.04, driver 580, team account without Docker access), a fresh
extract used the bundled Python and passed `--selftest`:

| Device | Frames agreeing | Proposals / QA | Max ΔLab vs. reference | Median per frame |
| --- | ---: | ---: | ---: | ---: |
| CUDA | 40/40 | 28 / 14 | 0.59 | 90 ms |
| CPU | 40/40 | 28 / 14 | < 1e-13 | 171 ms |

Times include HTTP, rendering and persistence; they are not camera-to-display latency.

## Building the kits (any Linux/WSL machine)

```bash
(cd client && npm ci && npm run build)
bash setup/gb10/build-kit.sh --assets /path/to/temporal-cnn-assets-v1 \
  --frames /path/to/run4-demo-frames --out linewatch-gb10.tar.gz            # ~70 MB, online install
bash setup/gb10/build-kit.sh ... --offline --out linewatch-gb10-offline.tar.gz # ~3.5 GB, no internet
```

The kit records `SOURCE_COMMIT` and `MANIFEST.sha256` (`sha256sum -c MANIFEST.sha256`). The
offline kit is too large for one GitHub release asset (2 GB limit), so move it by USB (exFAT,
not FAT32). The online kit fits a release asset.

Dependency locks: `lock-cuda.txt` / `lock-cpu.txt` are generated with
`uv pip compile api/requirements-api.txt api/requirements-model[-cuda].txt --python-version 3.12
--python-platform aarch64-unknown-linux-gnu --index-strategy unsafe-best-match --generate-hashes
--emit-index-url`. Regenerate them only deliberately, then rerun the self-test.

## Camera

LIVE CAMERA uses the browser's `getUserMedia`. The camera must be visible to a browser on the
GB10 (for example a UVC `/dev/video*` device), at 640×480 with the released fill-station view.
A different view or resolution needs its own camera profile and reviewed reference; the app
rejects other geometry rather than resizing. If the camera is not browser-visible, record
clips with the organizers' tools and use RECORDING. No industrial network, PLC or sorter
connection is made.
