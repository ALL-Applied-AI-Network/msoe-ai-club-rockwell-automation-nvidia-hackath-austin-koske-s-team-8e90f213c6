# Tracked-liquid replay correction

The earlier fixed color patch could sample the holder/background. Its 19/30 QA
coverage and example 9.56 ΔE comparison do **not** establish valid liquid color.
They are historical plumbing diagnostics. Do not reuse their reference bank.

## Launch

From the repository, after the ordinary data bootstrap:

```powershell
pwsh -File setup/run-replay.ps1 -Config replay/examples/bootstrap-vision.json -RecipePlan replay/examples/run5-demo-plan.json -Out "$HOME/ai-club-tools/results/tracked-liquid-demo"
```

Open http://127.0.0.1:8765/. The explicit config pins the ONNX model and hash.
Omitting it still runs fill detection but leaves color unavailable. CPU OpenCV
performs inference; no local PyTorch, API or GPU is required. First preparation
processes the recording; subsequent launches reuse the audited cache.

On Linux, from the repository:

```bash
bash setup/run-replay.sh --config replay/examples/bootstrap-vision.json \
  --recipe-plan replay/examples/run5-demo-plan.json
```

As before,
the server is loopback-only and operates on recordings. No industrial feed is
connected or approved by this work.

## What the geometry means

The blue vial and green liquid rectangles are **fresh model proposals**. They
are no longer static sampling patches. A vial must be uniquely under the nozzle
and freshly detected, match the previous box with IoU ≥ 0.65 and persist for
three samples. Motion resets the short track. Missing/ambiguous detections drop
the track and mask immediately; a later vial gets a new short-lived track ID.
These IDs are neither physical identities nor supplied evaluation vial IDs.

The liquid proposal must lie at least 90% inside the vial. A lower interior is
inset from the glass walls/base and upper label/holder. Saturated, unclipped
pixels form a connected region; opening and erosion reject isolated reflections.
At least 60 final pixels, sufficient coherent area, unchanged motion/holder QA,
mask-erosion sensitivity and stable post-fill colors are required. Clear/neutral
liquid is intentionally unavailable. This is a conservative **heuristic sampling
mask**, not a trained, reviewed liquid segmentation model. Colored reflections,
foam, glare, wrong teacher boxes and occlusions can still defeat it.

The old `color_roi` config field is retained only for older config compatibility;
it is unused. Holder/nozzle strips remain fixed **process diagnostics**, not vial
or liquid boundaries. The stream detector and completed-fill timing policy remain
separate from learned localization. Their event scores do not validate masks.

The live overlay is shown only on its exact sampled frame and cleared when video
advances. The frozen evidence panel is authoritative: select any actual window
sample and inspect its **exact decoded PNG, overlay and binary sampling mask**.
White pixels are the pixels used for that frame's median RGB, including clipping
and luminance trimming. A completed color is the median across accepted sample
colors; the displayed frame alone is not the aggregate. Rejected samples are
also visible with their reasons. Original decoded pixel hashes guard recovery.

## Reproduce the Run 5 diagnostic

```powershell
$python = "$HOME/ai-club-tools/envs/data-windows/Scripts/python.exe"
$out = "$HOME/ai-club-tools/results/tracked-liquid-demo"
& $python -m replay analyze --runs 5 --config replay/examples/bootstrap-vision.json --recipe-plan replay/examples/run5-demo-plan.json --out $out
& $python -m replay export-review --runs 5 --config replay/examples/bootstrap-vision.json --recipe-plan replay/examples/run5-demo-plan.json --out $out
& $python -m replay validate-detection --runs 5 --config replay/examples/bootstrap-vision.json --out "$HOME/ai-club-tools/results/tracked-liquid-event-evaluation"
```

Do not automatically choose a reference. `make-reference` still requires an
explicit source fill and availability frame before the evaluated fill starts.
The model/config/code measurement profile invalidates old fixed-patch references.
Future, current-fill, unknown-recipe and incompatible references remain rejected.

All current masks remain `geometry_review_status=unreviewed_model_proposals`.
An eligible reference may produce a **diagnostic** `delta_e2000` and
`experimental_status`, but the assessment stays `status=unavailable` until the
geometry has an implemented reviewed-evidence path. The app cannot silently
promote a confidence score to trusted geometry. No result authorizes release.

## QA review contract

`$out/run5/qa-review/manifest.json` uses
[review-manifest.schema.json](review-manifest.schema.json), version 2. All detected
fills have `samples`, `boundaries`, `measurement_available`, rejection reasons,
and `representative_frames`. `measurement_available` means heuristic window QA,
not approved segmentation or operational acceptance. Each representative has
source `path`, `mask_path`, `overlay_path` and individual file hashes. The
`contact-sheet.jpg` previews one measurement sample per fill, including rejected
ones. Every actual window sample and departure frame is exported losslessly.

Each sample includes its zero-based decoded `frame_index`, `mp4_s`, original
`source_rgb_sha256`, measured RGB/Lab, QA flags and `geometry`: boxes in source
pixels `[x0,y0,x1,y1]`, short track ID/age, proposal confidences, mask pixel count
and model hash. The **final mask** is row-major foreground RLE within `origin`
`[x,y]` and `size` `[width,height]`; each run is `[start_offset,length]`. The PNG
is the same mask, with white=sampled and black=excluded. No future frame is used
to construct an earlier mask. MP4 time is not verified physical time.

Aleyna/Zoe/Reagan can return corrected vial/liquid polygons or binary masks keyed
by recording SHA + frame index, retaining source proposals separately. Use the
manifest to identify moved vials, holder/background leakage, missed liquid,
glare/base inclusion and bad windows. Timing labels remain a separate artifact.
Reviewed geometry import/unlocking is not yet implemented; don't edit manifest
review status to pretend it is. Cameron/Hayden still own physical scene rendering;
Srithej/Josiah still own workflow/CSI confirmation. See
[bootstrap experiment](../setup/vision/README.md) for optional candidate labels
and augmentation handoff without replacing those assignments.

## Stop, interruption and restart

Recording and future approved frame sources share `MonitorSession.ingest`.
Interruptions stop a session and abandon any incomplete fill. A new session has
no track, old mask or intended recipe unless explicitly supplied again. Replay
backward seeking reads time-gated cached state and never makes a later reference
eligible earlier. Ctrl+C stops the local app; no machine control is performed.
