# Full liquid pipeline: TensorRT integration and early-fill review

October 20, 2026. Development branch: `austin-work`.

**The full sampled pipeline improves from 89.48 ms to 83.22 ms median: a 6.99%
latency reduction, or 1.075× speedup.** One of 1,581 sample-QA decisions changes,
reproducibly. Keep TensorRT as an opt-in prototype while reviewing that difference.
Deployed defaults, weights and measurement QA are unchanged.

## What actually ran

The original pipeline's YOLO11n input is **640×640**, while the earlier component
benchmark used 480×640. A fresh fixed 640×640 TensorRT FP16 engine preserves the
existing full-pipeline preprocessing and class/NMS options. The earlier 3.13×
component result cannot be substituted for a full-pipeline measurement.

Both modes run the same Faster R-CNN ResNet50-FPN liquid detector, SAM2.1 Hiera
Tiny, causal prompt policy, tracked geometry and unchanged color/sample QA. CPU
processing also computes image-derived fill state, the frozen experimental
background-normalized feature and the recipe evidence gate. Missing recipe,
reference, metering and reviewed geometry keep actual recipe decisions unavailable.
Cosmos is not part of this path.

The benchmark covers all 9,482 native Run 4 frames and analyzes every sixth frame:
**1,581 samples per pass**. Two passes per backend use PyTorch → TensorRT →
TensorRT → PyTorch ordering, for **6,324 measured pipeline samples**. Model loading
and warmup are excluded. YOLO and Faster R-CNN run for every measured sample;
SAM runs only when a fresh eligible prompt exists (918/1,581 per pass). No cached
predictions or masks replace measured inference. The frozen first plan's generic
all-models flag describes this live conditional graph; the source now uses explicit
no-cache and conditional-SAM fields for future runs.

All input RGB hashes/native PTS are checked against the original frozen evidence.
The new PyTorch control reproduces all 1,581 original measurements, raw masks and
fill states exactly. The same full liquid detector output is obtained by both
backends. Three frozen model hashes, the engine ancestry, calibration, prompt
policy and QA implementation are recorded in the experiment plan.

## Measured processing latency

| Metric | PyTorch vial backend | TensorRT FP16 vial backend |
| --- | ---: | ---: |
| Decode + sampled processing p50 | 89.48 ms | 83.22 ms |
| Decode + sampled processing p95 | 101.20 ms | 94.39 ms |
| Processing without decode p50 | 87.70 ms | 81.46 ms |
| Vial component p50 | 8.88 ms | 2.84 ms |
| Liquid component p50 | 18.28 ms | 18.28 ms |
| SAM stage p50 across all samples | 33.11 ms | 32.82 ms |
| Measurement/state/camera stage p50 | 23.59 ms | 23.71 ms |

There are 3,162 measured samples per backend. Stage medians do not sum to the total
median. SAM is conditional, so its median across all samples is different from its
latency conditional on a prompted frame. Prompted processing p50, excluding decode,
is 93.23 → 86.86 ms. Maximum sampled durations exceed 200 ms in both arms; this is
not a hard real-time guarantee. Capture, network latency, disk/video encoding and
the final per-recording window report are outside these measurements. The pipeline
samples at 5 Hz; these timings do not establish full-rate 30 fps processing.

## Output agreement and the actual limitation

The two complete repeats reproduce identical masks, state and QA within each
backend. Between backends, each 1,581-frame comparison has:

| Output | Agreement/result |
| --- | --- |
| Liquid detector predictions | 1,581/1,581 identical |
| Visual track ID/age and image fill state | 1,581/1,581 identical |
| Vial box matching | Zero unmatched boxes; median matched IoU 0.99963 |
| Final sampling masks | 1,531/1,581 exactly identical |
| Raw SAM masks | 1,431/1,581 exactly identical |
| Sample-QA decision | 1,580/1,581 identical |
| Accepted sample count | 849 PyTorch / 848 TensorRT |
| Completed-fill QA windows | 4/6 in both modes |
| Causal phase checkpoints | Nine in both modes |
| First accepted sample in each of six review neighborhoods | Same source frame in both modes |

At **source frame 4308, PTS 143.6 s**, PyTorch accepts 313 sampling pixels;
TensorRT yields a raw proposal that the unchanged containment/geometry checks
reject, so its sample is unavailable. Both use the same integer vial box and
visual track at that frame, but slightly different floating-point prompt geometry
can change SAM's discrete mask. No threshold was loosened to restore agreement.

Across frames with nonempty final masks on both sides, median IoU is 1.0 but the
minimum is **0.70455**: at frame 4236 the small sample changes from 88 to 62 pixels
(26 changed pixels). Thus median agreement alone hides a meaningful small-pool
difference. The verifier reports the lower tail separately. A human mask is needed
to determine which outline is better.

For the 848 frames passing sample QA on both sides, maximum raw camera-color
difference is **0.9201 dE2000**; the normalized-feature maximum is **0.7993**.
The missing sample is excluded from that paired-color denominator, not counted
as zero error. These are backend differences, not absolute color/composition
accuracy. No detector, SAM mask or backend has been promoted automatically.

The first-valid-frame comparisons use provisional image-derived neighborhoods.
They do not measure delay from independently reviewed liquid visibility. That
metric remains null until the data team returns onset labels.

## Media and reviewer handoff

Local root: `C:/Users/Austin/ai-club-tools/results/pipeline-acceleration-20261020`.

- `published-media-v1/accelerated-pipeline-run4-60s.mp4`: one camera view with
  measured TensorRT-pipeline outputs, a liquid close-up and explicit sample/recipe
  status. H.264, 60 s, 300 decoded output frames. No captions or before/after panes.
- `published-media-v1/accelerated-pipeline-six-early-fills.mp4`: all six eight-second
  early neighborhoods, 48 s / 240 decoded frames. These are development examples.
- `published-media-v1/pipeline-scorecard.png`: full-pipeline versus vial-component
  timing, with the distinction explicit.
- `published-media-v1/qa-disagreement-frame004308.png`: both backend outputs at
  the single changed QA sample, retained for inspection.
- `early-fill-review-v1.zip`: six raw source clips and 72 exact lossless PNGs,
  source hashes/native PTS, blank onset/mask return forms, and the existing review
  editor. All cases are unreviewed and unassigned. No reviewer workspace is copied.
- `CSI-VALIDATION-PLAN.md`: an unsent 90-minute commissioning/test-session draft,
  with reference preparation, process logs, clock mapping, physical identity,
  acceptance evidence and routing-deadline requirements.

The [early-fill handoff](../../annotation_review/EARLY-FILL-HANDOFF.md) explains
how Aleyna, Zoe and Reagan can review exact first-visible timing and mask geometry.
The box editor does not implement mask drawing or approve onset automatically;
those decisions use the separate return files and team conventions. Existing
Run 3 assignments and all existing reviewer workspaces remain untouched.

The [CSI plan](CSI-VALIDATION-PLAN.md) makes the next operational test concrete.
Dates, approved recipes/conditioning and feed arrangement need CSI confirmation.
No request was sent and no industrial equipment/network/camera was connected.

## Reproduce and verify

Use `envs/nvidia-pilot-windows/Scripts/python.exe` for GPU work and the existing
vision environment for artifact analysis. Each experiment helper refuses an
existing output directory; preserve partial evidence. Source snapshots and hashes
live under the corresponding experiment folders.

1. `export_pipeline_vial_engine.py`: export the pinned baseline at 640×640.
2. `benchmark_liquid_pipeline.py`: supply source/evidence, frozen liquid experiment,
   baseline, SAM checkpoint, new engine, frozen calibration and a fresh output.
   Default two passes execute the complete paired experiment. `--passes 1
   --limit-samples 140` was used for the separate smoke test.
3. `compare_pipeline_backends.py`: compare completed runs to the original evidence,
   including state/mask/color agreement and timing. This does not load human labels.
4. `annotation_review.early_fill`: consume the public unreviewed Run 4 request;
   extract exact source PNGs/clips and blank reviewer return formats. The packaging
   helper accepts `--readme` for this task-specific portable handoff.
5. `render_accelerated_pipeline.py`: render current measured outputs and retain
   exact source traces. `published-media-v1/encoding.json` records H.264 conversion
   commands and source/output hashes; raw `media-v1` outputs remain intact.
6. `verify_pipeline_acceleration.py`: recompute state/mask/color comparisons and
   timing, independently decode native source RGB/PTS, verify ZIP bytes/CRCs,
   demonstrate no onset/mask approvals and fully decode both raw/published media.

Sixteen annotation-review tests and four focused pipeline-comparison tests pass;
new helpers compile. See [verification](../verification/pipeline-acceleration-20261020.json).
Run 5 is excluded from all model/sampling decisions. The portable editor retains
the existing complete machine-prelabel metadata for compatibility; it contains
only Run 4 imagery for this batch. Large data, weights and videos stay outside Git.
