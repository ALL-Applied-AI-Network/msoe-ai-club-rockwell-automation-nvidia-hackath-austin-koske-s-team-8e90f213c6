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

Choose IMAGES and upload `examples/run4-fill1-f000339.png` from the model assets.
A real model overlay should appear. A single independent still cannot establish a
stable track, so unavailable sampling on the first still is expected.

Choose RECORDING and upload an approved 640×480 recording. Reduce FRAME INTERVAL SEC
to 0.2 for a short saved-clip rehearsal; this is sampling cadence, not an inference
FPS guarantee. Each recording after an existing run creates a fresh run. Use SOURCE
and POST-PIPELINE to inspect original pixels and model output, then open run review
for the separate source/rendered/mask artifacts and QA reasons. A local screenshot
or fallback video is not a physical line validation.

The default reference is the released earlier Run4 example. **It is unreviewed** and
only appropriate for previewing that same camera geometry. For another source,
choose an earlier aligned reference image before processing. Uploading does not
certify emptiness, timing or alignment. A new camera requires its own camera profile;
the app rejects geometry outside the released 640×480 profile instead of resizing.

Green overlay is the liquid proposal; blue overlay is the actual sampling region.
YOLO rectangles show vial geometry. The first frames and tiny/clear pools can have a
proposal while color QA remains unavailable. When QA passes, RGB/Lab and reasons
are retained, but recipe comparison remains unqualified until independent recipe
reference, completion and physical-vial evidence exist. No automatic product pass,
sorter/PLC command or claimed carryover diagnosis is produced.

Enter a reviewer and response in the retained review queue; history also supports
responses to old events and loading earlier frames. The response is evidence review,
not product release. Mock test buttons are disabled in the normal model demo.

Recipe CSV supports `recipe_id,start_s,end_s,vial` or explicit millisecond columns.
Intervals are [start, end); missing/overlapping/inverted timing is rejected. The old
frame-index CSV requires a reviewed source-clock mapping first. Stills have no
invented timestamp; recording seek time is not labeled native camera PTS.

## GB10 deployment

Fastest: download `linewatch-gb10-v1.0.0.tar.gz` from the `linewatch-v1.0.0` GitHub release,
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

## Investigation demo

Choose INVESTIGATION, then SYNTHETIC EXAMPLE. Switch between transient recovery,
persistent deviation, a delivery-record discrepancy and unavailable camera evidence.
The four-vial table and possible causes come from the frozen feature-level pilot;
the banner remains synthetic, separate from the camera frame and operator decision.
Choose THIS RUN to return to actual source-bound evidence; ordinary frame uploads
leave completed-vial count at zero. Real imports require the declarations and
source bindings in [the investigation guide](investigation.md). No cause is confirmed
and no product or equipment action is authorized by a hypothesis.
