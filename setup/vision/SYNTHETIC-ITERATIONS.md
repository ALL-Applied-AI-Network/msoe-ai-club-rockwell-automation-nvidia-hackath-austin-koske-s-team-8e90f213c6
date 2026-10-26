# GB10 synthetic transfer iterations — October 22

All new GPU work in this round runs on `msoe-spark` (GB10), using the existing
native vision environment and shared `spark-gpu.lock`. Local work is CPU-only
packaging, review and verification. Check completion markers before relaunching.
No model is promoted, app/QA defaults are unchanged, and no Run 5 input is used.

## Completed mask experiment

Three randomly initialized small U-Nets (16/32/64/128 channels, GroupNorm) share
a binary-mask head and presence head. Inputs are current RGB; signed/absolute
difference from an earlier reference; or difference after background-anchor RGB
gain normalization. Training uses only 144 synthetic renders from 36 paired
families; development uses 48 renders from 12 disjoint families. Sixty epochs,
AdamW 0.001, batch 8, seed 19; synthetic augmentation includes gain, bias, noise
and reference shift up to one pixel. Fixed source ROI `[256,240,368,432]`.
No generic pretraining or real fitting. Targets are geometric liquid-body masks,
not independently reviewed optical visibility.

Checkpoints selected every five epochs by synthetic presence accuracy plus mean
positive geometric IoU, earliest tie. Thresholds stay 0.5 for masks and presence.

| Input | Selected epoch | Synthetic presence | Positive geometric IoU | Real raw QA availability |
| --- | ---: | ---: | ---: | ---: |
| Current RGB | 45 | 47/48 | 0.8763 | 0/264 |
| Temporal difference | 50 | 48/48 | 0.8929 | 101/264 |
| Difference + anchor gain | 40 | 48/48 | 0.8933 | 105/264 |

All checkpoints freeze before real inference. The draft model is selected by
synthetic validation, not by the real QA counts. Actual synthetic YOLO vial
proposals plus learned direct masks enter the unchanged measurement QA. SAM is
not used by these direct-mask candidates. Color measurement always uses raw
source RGB. Presence scores are not calibrated probabilities of correctness.

The single-RGB model makes vial/glass-shaped proposals: all 264 video inputs
are presence-positive, and provisional agreement on 157 Run 4 stills is only
1.92% AP50–95. Temporal variants visually localize growing bottom pools in the
six inspected clips. This is promising transfer evidence, **not real accuracy**.
Their first clip frame is an unreviewed reference, not certified empty or the
same stationary vial. Zero output on self-reference does not demonstrate empty
rejection. Remaining issues include base/glass spill, reference motion, lighting,
clear/pale liquids and unmeasured first-visible latency.

Inputs comprise 240 playback frames at 5 Hz plus 24 extra exact review frames,
264 unique frames total. Six correlated eight-second clips, not 264 independent
physical trials. QA availability is not localization, onset or product-quality
accuracy and is not directly comparable with earlier 111-input probes.

Artifacts under
`C:/Users/Austin/ai-club-tools/results/synthetic-iterations-20261022/`:

- `gb10-results/mask-probe-v1`: three-model 48-second video, exact source/PTS
  records, six PNG comparisons, all predictions and 72 draft review frames.
- `gb10-results/masks-v1`: fitted checkpoint bytes, frozen metadata, curves and
  held-out synthetic results.
- `visible-pool-review-v2`: [offline human review workflow](../../annotation_review/VISIBLE-POOL-REVIEW.md).
  Its draft masks have not been approved. Human review is for evaluation only.
- `review-browser-v1`: executed editing/export/import evidence using isolated,
  explicitly ineligible demonstration annotations. All 21 annotation tests pass,
  including five scoring-integrity checks; no team labels were produced by verification.

Model SHA256:

```text
single_rgb       80dd310e3f6729924377256d2168b14c93b998fe84e79bd62729a556b9b8ae50
paired_diff      d26820ca9f56ae9f51234949c396ea5c3aad385c383e0dfe88eb17f432045754
paired_diff_gain 6714fe32191d8eba230aa87246f38a25c2906c619d60d7ba30c9e60205352eef
```

## Completed optical review

GB10 Isaac produced 36 renders: 12 one-factor/profile settings × empty, early and
established states, PathTracing 16 spp / four subframes. Variants cover glass
profile/roughness, absorption, lighting, pale/clear fluid, paper brightness and
camera offset. Original asset hashes are unchanged. These scenes are marked
`review_only` / `training_eligible=false`; no measured optical calibration or
real-based parameter optimization is claimed. The early pool remains visually
subtle relative to the conspicuous geometric target. More supervision cannot
repair a discrepancy between geometric body and visible fluid on its own.

`gb10-results/optics-grid-v1/optical-controls.png` shows all 36 pinned RGB crops.
Physical calibration needs supervised, controlled empty/known-fill observations
with camera/lighting identity. No equipment or industrial network was connected.

## Completed duration controls

Two predeclared Faster R-CNN ResNet50-FPN fits extend original/paired synthetic
corpora to 15 epochs. Generic COCO initialization, 614 synthetic samples per
epoch, batch 4, AdamW 0.0001, seed 19. Select synthetic AP50–95 at epochs
3/6/9/12/15, earliest tie; retain epoch 3 and last epoch for same-GB10 duration
controls. Real evaluation occurs only after the primary checkpoint freezes.
The original and paired synthetic development sets differ (154/202 images), so
compare duration within each arm rather than interpreting AP between arms as
an isolated causal comparison. Real Run 4 references remain provisional.

Both fits selected epoch 15 using synthetic AP. Matched GB10 duration-control
re-scores show original-corpus real provisional AP **23.40% → 14.50%** from
epoch 3 to 15; paired-corpus AP **23.58% → 18.39%**. Longer fitting improves
synthetic geometry while worsening this real development agreement. It supplies
no improved detector candidate. Primary strict-backend scoring reports
14.54% / 18.35%; default-backend control re-scores have small numerical
differences, retained explicitly. Neither candidate is promoted.

Completed remote suite:
`~/ai-club-tools/experiments/synthetic-iterations-20261022-v2`.
`completed-long.json` marks both fits; `duration-controls-v1/completed.json`
marks post-freeze epoch-3/last-epoch diagnostics. All following stress and timing
jobs also completed; no replacement training is needed.

## Completed temporal stress checks and processing benchmark

The same frozen candidate was tested with eight synthetic conditions and eight
digitally perturbed real conditions. Synthetic tests use 48 held-out renders
per condition (24 geometric positives, 24 negatives including empty/stream-only).
Real tests use the same 264 exact RGB/native-PTS frames and unreviewed references.
Real prediction consistency is not accuracy against human annotations.

| Candidate condition | Synthetic presence detections / 24 | Synthetic alarms / 24 negatives | Real proposal changes / 264 | Mean real mask IoU against clean prediction on 189 clean accepted frames |
| --- | ---: | ---: | ---: | ---: |
| Clean | 24 | 0 | 0 | 1.000 |
| Current image ×0.85 | 24 | 0 | 1 | 0.997 |
| Current image ×0.70 | 24 | 0 | 2 | 0.993 |
| Current image ×1.30 | 22 | 0 | 13 | 0.917 |
| Green channel ×0.85 | 24 | 0 | 1 | 0.997 |
| Reference shift 2px | 24 | 0 | 1 | 0.940 |
| Reference shift 5px | 23 | 2 | 12 | 0.805 |

Synthetic shifts roll the ROI; real shifts translate the full reference image
with reflected borders. They are separate digital nuisance tests. A deliberately
different synthetic empty-family reference produces 11/24 positive detections
and 8/24 negative alarms. A later real-clip reference changes 126/264 proposal
decisions and reduces mask consistency to 0.254. These invalid-reference controls
establish a concrete reference dependency; a suitable capture/reset protocol is
required before physical use. Do not automatically infer emptiness from the
model's absence output. Larger shifts and exposure clipping remain weaknesses.

The gain-corrected variant is substantially more stable than the basic temporal
variant under digital darkening: ×0.70 gives 2 versus 99 proposal changes and
0.993 versus 0.283 clean-positive mask consistency. Both checkpoint choices were
already frozen by synthetic validation. Physical illumination robustness and
real mask/presence/onset accuracy remain unmeasured.

Actual **synthetic YOLO vial → direct temporal mask → unchanged QA** processing
was timed on 264 cached exact RGB frames in two passes (528 measured samples),
after 30 warm-up frames. Same source hashes, native PTS and references checked.

| Stage | Median ms | p95 ms |
| --- | ---: | ---: |
| Vial detector | 4.38 | 4.75 |
| Temporal mask, component and proposal | 3.93 | 5.00 |
| Existing QA | 6.68 | 15.10 |
| Complete CV path | **15.47** | **24.53** |

Both passes reproduce all 264 completed-probe QA decisions, final sampling masks
and measured colors exactly, with 105 valid measurements each. These are cached
processing times, excluding decode, rendering, transport and browser latency;
they do not establish a deployed 30 fps camera-to-dashboard system. The liquid
branch alone in a separate 200-sample warmed probe measured 1.92 ms median /
2.06 ms p95, excluding proposal encoding/vial inference/QA.

Of 189 clean liquid proposals, 84 lack a qualified measurement. Existing flags
identify 80 association failures (`liquid_missing_or_ambiguous_inside_vial`).
This does not independently adjudicate whether a mask, vial box or containment
rule is wrong. Preserve raw masks and inspect paired geometry with human review;
do not relax QA to manufacture coverage. Detection and qualified metrology are
separate states.

All six result roots and 118 report artifacts were hash-verified before and
after transfer. Local outputs:

- `gb10-results/candidate-checks-v1/`: completed raw reports, frozen plans and
  executed evaluation helpers. Checkpoints stay on GB10 with recorded hashes.
- `candidate-summary-v1/`: tables, JSON and three standalone PNG plots.
- `gb10-results/candidate-checks-v1/real-stability/real-stability-comparison.png`:
  clean/darkened/shifted/late-reference masks for all six clips.
- [Development integration contract](TEMPORAL-POOL-CANDIDATE.md): actual model
  inputs, source clocks, fixed ROI and reference dependency for Srijan.

Human-reviewed accuracy remains pending in the existing 72-frame packet.
No app/default change, training on real imagery, equipment connection or push
was performed. No further GPU jobs were submitted after these checks.

The preserved v1 long fit failed at its first validation because a scorer
unnecessarily imported unavailable ONNX Runtime. Three training epochs completed,
but that directory has no completed selected model. V2 uses a frozen NumPy-only
scorer overlay with identical scoring logic; original helper snapshots and shared
packages remain intact. A Windows CPU video export rejected differing decoded
RGB hashes; its partial directory remains. The successful GB10 CPU export uses
the inference decoder and verifies every source RGB hash and native PTS.
