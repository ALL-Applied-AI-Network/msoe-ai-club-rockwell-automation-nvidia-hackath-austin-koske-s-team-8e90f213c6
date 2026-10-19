# First Vial recording replay

One local browser app and one batch pipeline, using the existing data environment.
The recordings and supplied labels stay read-only. No GPU, API key, model server,
Node runtime or frontend build is required to run it.

For the current **tracked-vial/liquid-mask correction**, start with
[VISION.md](VISION.md). The model is trained on provisional boxes, masks need
review, and old fixed-patch color comparisons must not be trusted.

For the earlier **recipe-aware completed-vial plumbing demo**, exact reproduction,
operator recipe/reference contracts, Aleyna's QA review manifest and the physical
input boundary, use [HANDOFF.md](HANDOFF.md). Image detection still runs independently
of evaluation annotations; the intended recipe is a separate explicit input.

## Start the app

From the repository, after the team data bootstrap:

```powershell
pwsh -File setup/run-replay.ps1 -Config replay/examples/bootstrap-vision.json
```

Open **http://127.0.0.1:8765**. For a dataset stored elsewhere:

```powershell
pwsh -File setup/run-replay.ps1 -Data 'D:/data/csi-vial-fill-dataset'
```

On the GB10 or another Linux machine with the data environment:

```bash
export TEAM_MEMBER=cameron  # use your own name on the shared account
bash setup/run-replay.sh --data /path/to/csi-vial-fill-dataset
```

The Linux launcher separates output/reference banks by `TEAM_MEMBER`. Coordinate
ports on a shared host or pass `--port 8766` for a second server.

The default source is `datasets/dataset2/csi-vial-fill-dataset`. Raw data is not
bundled with the app. The server binds to loopback. For remote use, forward port
8765 over SSH and open the forwarded localhost URL. Stop the server with Ctrl+C.

## First useful walkthrough

1. Select Run 5. The first preparation pass decodes the recording and caches tracking/mask
   features; playback can start while preparation finishes.
2. Use Play, speed, the video scrubber or **Jump to fill**. **Jump to completed
   check** moves to the frame where carrier departure confirms completion.
3. Inspect QA before the color. The ledger contains only observations available
   by the current playhead. Backward seeking hides later observations again.
4. Image mode reports fill IDs, visible stream episodes and completed color QA.
   Load `--recipe-plan replay/examples/run5-demo-plan.json` or use **Set intended
   recipe** before a fill starts. Missing/changed context remains unavailable.
   With stable known context, deliberately select a completed QA-valid example as
   a development reference for later fills. Detection IDs are sequential image
   candidates, not supplied vial IDs. See the handoff for provenance and limits.
5. Export events JSONL, observations CSV and the reference bank. Exports stop at
   the playhead. Resetting playback does not make a future reference eligible for
   earlier vials.
6. All runs use the same image-only detector. The full-recording fill selector is
   for replay navigation; current state and exports reveal only events available
   through the playhead. An unfinished fill at EOF remains open.

The two secondary videos follow the primary MP4 time. This is container-time
alignment, not verified physical synchronization. The UI labels this distinction.

## What this version establishes

- All five recordings and all three camera views are accessible in one app.
- **Image detection is the default.** The normal CLI app/analysis does not read
  annotation files. Visible stream contrast uses the center column under the
  nozzle against its same-frame neighboring background. Three samples over
  0.4 MP4 seconds confirm stream on/off. Hysteresis suppresses flicker.
- Stream episodes remain part of one fill through any ingredient pause. Carrier
  motion above 8 for 0.2 MP4 seconds confirms departure. Empty carrier cycles
  do not become fills. A candidate must have at least 0.6 MP4 seconds of stream
  evidence and peak contrast 6; rejected candidates are logged. These are pilot
  thresholds inspected/tuned on Run 5, not held-out learned parameters.
- A pinned weak-label detector proposes vial/liquid locations. Causal short
  tracks and a conservative colored-liquid sampling mask produce candidate
  camera Lab. Exact source frames and actual masks are displayed/exported.
  Missing geometry makes color unavailable; there is no fixed-patch fallback.
- A completed observation uses a 0.5 MP4 second stationary tail before departure,
  at least 0.5 MP4 seconds after the final detected stream stop. It is published
  only at departure confirmation. Three valid samples, stable color and no
  carrier motion in the window are required. The six-frame stride gives 5 Hz
  in these 30 fps MP4 files. Missing evidence fails QA; gaps reset debounce.
- `start_frame` and `end_frame` estimate visible flow, not pump commands.
  `start_available_frame`, stream episode availability, `departure_frame` and
  `confirmed_frame` preserve what the detector knew and when. A mid-flow first
  frame is marked `partial_start_possible`; EOF never fabricates completion.
- Comparison uses Î”E2000 with an explicit reference and an **unvalidated** default
  threshold of 5.0. It is separate from the older drift-normalized/transmittance
  analysis; that analysis's threshold must not be transplanted into this policy.
- Outcomes are `review`, `within_reference_band`, or `unavailable`. No outcome
  authorizes release. Recipe comparisons/hold policies require a recipe source.

The old annotation-guided baseline is retained explicitly:

```powershell
& "$HOME/ai-club-tools/envs/data-windows/Scripts/python.exe" -m replay serve --fill-source annotations
```

In that mode only, supplied boundaries schedule the original 0.5â€“1.0 MP4 second
post-end window, supplied recipe context enables development references, and
first-vial/first-two holds are separate baseline policies. A reference must be
published before the next evaluated fill starts; it never certifies clean product.

A small box model is trained experimentally on provisional Jev labels; no reviewed
segmentation accuracy is established. Current geometry keeps comparisons unavailable
while allowing explicit diagnostic deviations. There is no stage-specific early-warning claim,
carryover-cause classifier, physical concentration estimate or defect accuracy
metric. Development references from real footage are not synthetic training data.

## Batch processing and audit artifacts

Use the interpreter printed by the setup script:

```powershell
& "$HOME/ai-club-tools/envs/data-windows/Scripts/python.exe" -m replay inspect
& "$HOME/ai-club-tools/envs/data-windows/Scripts/python.exe" -m replay analyze --runs all
```

Default outputs are `~/ai-club-tools/results/replay`, outside Git. Each run writes
`manifest.json`, `features.jsonl`, `events.jsonl` and `observations.csv`. The manifest
records input video/annotation hashes, Python/library versions, implementation/config IDs, ROI policy,
sample stride, runtime and limitations. Caches invalidate when the measurement
code, runtime versions, configuration, source recording, labels or optional clock map change.

For isolated experiments, pass `--out /path/to/new/experiment`. Manual development
references persist in that output directory. `--references bank.json` imports a
bank explicitly; `--config replay/config.example.json` selects a reviewable policy.
The full CLI is `python -m replay --help`.

The [executed checks](verification/README.md) report coverage and test scope.
QA rejections are part of the result, not discarded evaluation trials.

## Validate independent detection

```powershell
& "$HOME/ai-club-tools/envs/data-windows/Scripts/python.exe" -m replay validate-detection --runs 5 --out "$HOME/ai-club-tools/results/fill-validation"
```

This command reads labels **only for offline evaluation**, after image detection.
It additionally writes `run5/detection-validation.json`: maximum one-to-one
chronological matches at interval IoU â‰¥ 0.5, misses/extras, every matched boundary,
signed errors and absolute median/p95 errors. It reports departure confirmation
delay separately from estimated flow-end error. The first partial fill is excluded
from start-error aggregates. Missing ground truth produces no precision/recall.

Run 5 development results: **30/30 matches, zero misses/extras**; start absolute
median/p95 18/35.4 frames, end 20/66.1 frames. Completion confirmation lag median
71.5 frames. These are event-detection results against supplied annotations, not
defect accuracy, exact pump timing, physical seconds or held-out generalization.
See [full evidence](verification/fill-detection.md).

## Annotation handoff for the data group

Use a separate reviewed folder, with one CSV per run named `run1.csv` through
`run5.csv`. Do not overwrite the supplied labels. Required columns:

```csv
vial,recipe_id,t100_ml,t200_ml,t400_ml,dispense_order,start_frame,end_frame
```

Vial IDs must be unique, rows chronological and intervals non-overlapping. Frame
indices are zero-based decoded MP4 frame indices, with an inclusive end frame.
Volumes are commanded mL, blank for an unused tank. Additional columns are ignored.
Recipe identity must come from reviewed evidence, not the observed color alone.
Missing runs remain unlabeled; known supplied annotations are the fallback in
annotation-baseline mode and offline validation only. Image mode ignores labels.

```powershell
& "$HOME/ai-club-tools/envs/data-windows/Scripts/python.exe" -m replay validate-detection --labels-dir clean_data/labels
```

For richer handoff tables, add a small adapter into `Catalog`; keep the monitor's
internal `Annotation` contract stable. Defect/acceptability labels are a separate
future evaluation input and must not be fed into reference selection or decisions.

## Time conventions

Frame index and decoded MP4 presentation time are recorded separately. The viewer
uses the verified constant-frame-rate MP4 mapping; samples inconsistent with that
mapping or missing PTS fail QA. A container's 30 fps is not a claim of 30 frames
per physical second. Optional existing OCR maps can be displayed with:

```powershell
& "$HOME/ai-club-tools/envs/data-windows/Scripts/python.exe" -m replay serve --clock-map-dir analysis/cache
```

These maps are labeled **unreviewed estimates**. They do not change decisions or
establish warning time in physical seconds. No wall clock is fabricated when a
map is absent. Cross-camera physical offsets still need the reviewed data handoff.

## Extension points

| Owner/work | Interface |
| --- | --- |
| Data group | Reviewed per-run annotations and clock landmarks; optional `Catalog` adapter |
| Synthetic group | Keep generated RGB and provenance separate; future predictor can provide a recipe/stage expectation without changing transport or event logging |
| Image detector | `FillDetector.update(feature)` returns causal state; `fill_started`, `stream_started`, `stream_stopped`, `fill_completed` events preserve availability timestamps |
| Recipe context | `RecipePlan` and manual UI input; real operator workflow/plan still needs confirmation. Never infer the intended recipe from observed color or evaluation matches |
| Austin: comparison | `assess(observation, references, config)`; stage expectations need a separate interface before comparing during-fill color |
| Evaluation | Join exported observations to independent reviewed quality labels; count unavailable measurements and split by run/transition |
| Cosmos | Consume frozen structured evidence after a decision; keep the deterministic explanation as fallback |

The current fixed ROI and post-fill window reject some usable-looking vials,
particularly when departure is fast. Review overlays and timestamps before
changing thresholds. Threshold tuning and final evaluation need separate trials.

## Tests

```powershell
& "$HOME/ai-club-tools/envs/data-windows/Scripts/python.exe" -m unittest discover -s replay/tests -v
```

Checks cover future-frame independence, observation availability, backward seek,
reference provenance/timing, same-vial leakage, invalid-image handling, byte-range
video delivery and the HTTP reference/export contract. Detector checks add
ingredient pauses, empty cycles, debounce, partial starts, sample gaps, open EOF,
nontrivial label removal/poisoning, one-to-one matching and prefix invariance.
Small temporary video
fixtures are generated during tests; no supplied files are modified.
