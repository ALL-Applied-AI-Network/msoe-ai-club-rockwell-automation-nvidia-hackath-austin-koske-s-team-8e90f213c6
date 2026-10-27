# Laptop demo

Docker Desktop (Linux containers), Git and the released model ZIP are sufficient.
There is no local Python, Node, CUDA or separate database setup for this path.
The default runs on CPU; GPU performance is a separate approved-host test.

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

The API is internal to Compose. Nginx exposes only loopback port8788 and proxies
HTTP/WebSockets. SQLite and media persist in the `inspection-data` named volume;
the models are mounted read-only. No industrial source is configured or contacted.

## Rehearse

Choose IMAGES and upload `examples/run4-fill1-f000339.png` from the model assets.
A real model overlay should appear. A single independent still cannot establish a
stable track, so unavailable sampling on the first still is expected.

Choose RECORDING and upload an approved 640×480 recording. Reduce FRAME INTERVAL SEC
to0.2 for a short saved-clip rehearsal; this is sampling cadence, not an inference
FPS guarantee. Each recording after an existing run creates a fresh run. Use SOURCE
and POST-PIPELINE to inspect original pixels and model output, then open run review
for the separate source/rendered/mask artifacts and QA reasons. A local screenshot
or fallback video is not a physical line validation.

The default reference is the released earlier Run4 example. **It is unreviewed** and
only appropriate for previewing that same camera geometry. For another source,
choose an earlier aligned reference image before processing. Uploading does not
certify emptiness, timing or alignment. A new camera requires its own camera profile;
the app rejects geometry outside the released640×480 profile instead of resizing.

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
Intervals are[start,end); missing/overlapping/inverted timing is rejected. The old
frame-index CSV requires a reviewed source-clock mapping first. Stills have no
invented timestamp; recording seek time is not labeled native camera PTS.

## Stop, inspect and move

```bash
docker compose ps
docker compose logs --tail=100 api client
docker compose stop
```

Run the launcher again to start. Do not use `down -v` unless you intend to delete
inspection history. The laptop needs the same Git revision and asset ZIP. It does
not need the experiment folders, reviewer databases or full training dataset.

The local rehearsal report under`setup/verification/` records actual image/HTTP/UI
checks and their limits; it does not establish segmentation accuracy, recipe
acceptance or performance on an untested laptop or CSI workstation.
