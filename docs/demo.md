# Laptop demo

Docker Desktop (Linux containers), Git and the released model ZIP are sufficient.
There is no local Python, Node, CUDA or separate database setup for this path.
The default runs on CPU. For the GB10 deployment target, use the [native GB10 path](#gb10-deployment).

## Windows

1. Check out `austin-work` and start Docker Desktop.
2. Download `temporal-cnn-assets-v1.zip` from the
   [private model release](https://github.com/ALL-Applied-AI-Network/msoe-ai-club-rockwell-automation-nvidia-hackath-austin-koske-s-team-8e90f213c6/releases/tag/temporal-cnn-v1.0.0).
   You can keep the ZIP or extract it outside Git.
3. From the repository root in PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File setup/start-demo.ps1 -Assets "C:\Users\YOUR_NAME\Downloads\temporal-cnn-assets-v1.zip" -Build
```

Open **http://localhost:8788**. The launcher checks both checkpoint hashes, extracts
ZIP input to a sibling folder if needed, and starts the API and built React UI.
The first image build needs internet and downloads CPU model dependencies. Later
starts reuse the images. `-Port 8790` chooses another local port.

## Linux / WSL

```bash
export LINEWATCH_ASSETS_DIR=/absolute/path/to/temporal-cnn-assets-v1
docker compose up -d --build
```

The API is internal to Compose. Nginx exposes only loopback port 8788 and proxies
HTTP/WebSockets. SQLite and media persist in the `inspection-data` named volume;
the models are mounted read-only. No industrial source is configured or contacted.

## Rehearse

The monitor follows three steps:

1. **Next vial recipe.** Pick a recipe (or **Custom amounts…** for clear/orange/blue mL)
   and press **SET RECIPE**. This manual entry stands in for a future live recipe feed.
2. **Fill.** Choose RECORDING and drop one vial's recording, or start the camera and press
   **FINISH VIAL** when it is full. Each recording uses its first frame (the empty vial) as
   the temporal reference; this can be turned off under **Advanced setup**.
3. **Vial results.** Each vial gets IN SPEC, OUT OF SPEC or CAN'T VERIFY, its ΔE2000 to the
   intended recipe and to the last different recipe, and a **simulated** action: route to
   GOOD, divert to SCRAP (noting possible carryover when the colour shifted toward the
   previous recipe), or hold for a manual check. The operator confirms or overrides; the
   decision is saved against the vial's last frame. Nothing is sent to the PLC, sorter or
   Teledyne.

The vial colour is the median Lab of QA-valid samples in the last 6 s of the vial. Reference
colours come from the Run5 retrospective analysis (proxy endpoints for 7 of 10 recipes). A
recipe without a reference returns CAN'T VERIFY; **USE AS REFERENCE FOR THIS RECIPE** accepts
that vial as the operator's reference, stored in this browser only. Custom amounts that equal
a library recipe reuse its reference.

Demo clips are the `run5-vial*.mp4` assets of the release (one Run5 vial each, cut by
`replay/make_run5_vial_clips.py`). Enter recipe 5 → vial 2 (in spec), recipe 6 → vial 4
(out of spec, almost exactly recipe 5's colour), recipe 6 → vial 5 (still shifted toward 5).
`api/tests/rehearse_vial_flow.py` drives this sequence in a browser.

Green overlay is the liquid proposal; blue is the region QA sampled. Frames where QA
withholds colour still show their reasons. **Advanced setup** holds the sampling interval,
reference upload, raw QA evidence and the recipe timeline/CSV import. **RECIPE CHECK** shows
the full Run5 evidence. Run review keeps source, rendered and mask artifacts, accepts
per-frame responses and contains the Investigation panel.

A new camera requires its own camera profile; the app rejects geometry outside the released
640×480 profile instead of resizing. Recording seek time is not native camera PTS.

## GB10 deployment

Fastest: download `linewatch-gb10-v1.1.0.tar.gz` from the `linewatch-v1.1.0` GitHub release,
extract it and run `bash START.sh --selftest`, then `bash START.sh`.

From a GitHub checkout on the GB10, `bash setup/gb10/onboard.sh` downloads
and verifies the model release, starts Docker with the GPU (falling back to Docker CPU, then
native), runs a 40-frame reproducibility self-test and saves a device report under
`~/linewatch-onboarding/`. See [setup/gb10](../setup/gb10/README.md).

Backup without Docker: the USB kit bundles CPython 3.12 and hash-locked packages (online or
offline wheelhouse) and runs the same models, QA and UI natively:

```bash
tar -xzf linewatch-gb10.tar.gz && cd linewatch-gb10
bash START.sh --check      # readiness report
bash START.sh --selftest   # install and compare 40 reference frames with our frozen outputs
bash START.sh              # serve http://127.0.0.1:8788 (CUDA, else CPU)
```

On spark8 a fresh extract passed the self-test on CUDA and CPU (40/40 frames agreeing) and
the full browser rehearsal. See [evaluation](evaluation.md#gb10-native-run).

## Stop, inspect and move

```bash
docker compose ps
docker compose logs --tail=100 api client
docker compose stop
```

Run the launcher again to start. Do not use `down -v` unless you intend to delete
inspection history. The laptop needs the same Git revision and asset ZIP. It does
not need the experiment folders, reviewer databases or full training dataset.

The local rehearsal report under `setup/verification/` records actual image/HTTP/UI
checks and their limits; it does not establish segmentation accuracy, recipe
acceptance or performance on an untested laptop or CSI workstation.

## Investigation

The Investigation panel is in run review, not on the monitor. Its synthetic examples remain
labelled synthetic; real imports follow [the investigation guide](investigation.md).
