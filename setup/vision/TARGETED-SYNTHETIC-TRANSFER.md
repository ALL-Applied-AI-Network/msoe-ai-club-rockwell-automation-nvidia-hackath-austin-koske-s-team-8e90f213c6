# Targeted synthetic transfer experiment — October 22, 2026

This is an isolated local experiment on `austin-work`. No source USD was saved,
app defaults changed, human labels approved, or real CSI images used for weight
fitting. Generic COCO initialization remains part of the model.

The preceding diagnostic showed partial transfer: the detector could find real
bottom pools while also proposing liquid on glass and paper. It also exposed
inconsistent provisional real target scope. This experiment targets visual
confusion without pretending to repair human annotation quality automatically.

## Frozen design

- 48 nuisance configurations, each with empty, stream-only, early body, and
  established body: 192 new native 640×480 Isaac PathTracing images.
- Replace the vial's clear-plastic opacity approximation with a session-layer
  MDL transmission material, IOR 1.47, provisional absorption depth 0.6 m.
  Vary roughness, lights, paper brightness, camera image-plane translation,
  body color/absorption, pool footprint and height. Original bindings/geometry
  remain on disk; only vial mesh material bindings change in the session.
- Each four-state family shares nuisance settings and stays entirely in one
  split: 36 training families and 12 development families, seed 2719.
- Body only is the positive target. Empty and stream-only scenes are synthetic
  negatives. Clear-liquid cases are outside this pilot. A prespecified paired
  RGB difference screen can exclude invisible positives; it is not optical truth.
- Keep 470 original synthetic training images and add 144 new ones, totaling
  614. Retain the original 154 development images and add 48 grouped cases.
  Both baseline and candidate use three epochs, batch four, 462 optimizer
  steps, identical AdamW settings, generic COCO initialization and fixed final
  epoch 3. Candidate checkpoint freezes before real scoring.
- Real scoring uses the same 157 original Run 4 machine-reference stills, with
  confidence 0.35 and IoU 0.5 operating-point reporting. Run 5 remains excluded.
  Previously inspected Run 1/3/4 examples are illustrative development evidence.
- Reuse the unchanged detector→SAM2.1→measurement QA probe on three fixed real
  windows and cached Isaac frames. QA availability is not segmentation accuracy.

## Interpretation boundary

The renderer changes and data mix are tested together. A favorable result would
support this candidate, not isolate glass, negatives or small pools as the cause.
Synthetic native body masks are geometric proxies; refraction may shift visible
boundaries. Real references are provisional and may include a stream or omit a
visible region. Neither AP nor QA availability establishes reviewed early-onset,
segmentation or recipe/defect accuracy. A human target-scope queue accompanies
the comparison; no original labels are overwritten.

Local artifact root:
`C:/Users/Austin/ai-club-tools/results/targeted-synthetic-20261022/`.
The rendered batch is `paired-render-v1`, fixed-budget checkpoint is
`fasterrcnn-paired-v1`, and comparison/visuals are `review-v1`. The verified combined scorecard and CSV
are in `summary-v2`; `summary-v1` is preserved after a path-separator verification
error, without retraining or rewriting experiment evidence.


## Completed results

Every detector row uses the same 157 original Run 4 stills / 89 provisional
liquid targets. Confidence stays 0.35; matching IoU stays 0.5. AP50–95 uses the
same 101-point scorer. Missing targets are not reviewed empty examples.

| Arm | Real provisional AP50–95 | Matches | Unmatched boxes | Missed targets | Real raw pipeline QA passes /111 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original synthetic Faster R-CNN | 26.50% | 70 | 265 | 19 | 61 |
| New paired-render fit | 23.18% | 68 | 439 | 21 | 20 |
| Lower-vial crop fit (paired corpus) | 8.44% | 48 | 148 | 41 | 23 |
| Original fit + synthetic-calibrated basal association | 25.45% | 61 | 88 | 28 | 59 |

The new paired fit **does improve coverage of the new renderer**: held-out
geometric AP rises from 42.47% to 74.40%, early-case matches from 5/12 to 10/12,
and small-body matches from 0/7 to 5/7. Predictions on synthetic empty/stream-only
frames fall from 20/24 to 1/24. Original synthetic development AP changes from
95.79% to 94.63%. None of this establishes real first-visible-liquid accuracy. The provisional
Run 4 still set contains no liquid targets ≤32px high; first-visible onset
accuracy remains unavailable.

The lower-vial crop was a separate third hypothesis, using the same 614 source
samples, three epochs and COCO initialization as the paired-render fit. It crops
from half the detected vial height down, adds 10% width /2% bottom margin, then
letterboxes onto 640×480. Synthetic labels intersect the crop; 36 development or
training targets are truncated. Inference maps native boxes back into the source
frame. Real teacher boxes stay unchanged. It removes the paper region but often
still calls the lower glass a body; the selected shallow blue example becomes an
oversized box again. Real agreement and downstream QA both worsen.

Basal association freezes its tolerance from **72 new synthetic training body
cases**, excluding development and real labels: maximum observed body-bottom
deficit relative to the synthetic-trained YOLO vial, plus one pixel at median
vial height. The resulting fraction is 0.046456. Existing vial association margins
are retained, then at most one original body box survives per active vial. No box
is invented, expanded or color-calibrated. It removes 177 unmatched real boxes but
also nine matches. It still predicts on 63/68 frames lacking a supplied target,
versus 68/68 before filtering. It cannot establish liquid presence by geometry. Detector-level postprocessing
uses retained ONNX vial proposals from the same synthetic checkpoint export;
the actual pipeline probe uses native PyTorch vial inference.

Actual YOLO vial → Faster R-CNN body → generic SAM2.1 → unchanged measurement QA
ran on all attempts. All arms use the same 111 real samples (three fixed windows)
and 111 cached Isaac frames, native source hashes/PTS, vial/SAM weights and QA
implementation/configuration. The basal probe also completed green-gain and
JPEG-q40 conditions: real QA 54/111 and 31/111 versus baseline 60/111 and 32/111.
There are **1,110 new pipeline attempts**, including repeated domains/conditions,
not independent physical trials. QA counts are availability, not correctness.

**Decision: keep the original model as the development baseline. Neither new fit
nor the basal filter is promoted.** Greater synthetic coverage did not improve
real transfer in these pilots. This does not isolate a cause: renderer optics,
scene diversity, reference semantics and small fixed training budgets remain
confounded. Cropping/filtering cannot resolve all empty-glass confusion.

## Concrete review and calibration work

`Human-Target-Scope-Review.zip` contains seven hash-verified original images,
unchanged machine geometry and questions. It is a portable reference packet,
not an annotation-editor import or an approved correction export. The existing
human review workflow still owns conventions and corrections. Agree body versus
stream scope, mark ambiguous visibility uncertain, and establish reviewed empty
and first-visible-pool examples. No labels were approved automatically.

For the already planned supervised CSI session, add controlled empty/filled
captures of the same vial with fixed camera, exposure and white balance. Have CSI
staff choose safe, measurable small increments through the first visible pool,
recording actual amounts and native frame/PTS landmarks. Include pale and strongly
colored fluid, flowing and stopped states, and matched empty glass. Keep still
capture timestamps separate from video timing unless explicitly mapped.

Use those measurements to check synthetic camera/glass/dye optics, then rerender
matched counterfactuals and hold complete scene families out. Separate these
controls in the next ablation instead of coupling all appearance changes. Keep
new detector weight fitting synthetic-only, freeze selection before real scoring,
and report reviewed onset delay, empty-frame precision and mask quality once
human references exist.

## Evidence and reproducibility

- `paired-render-v1/render-plan.json`, `manifest.jsonl`, `scene-inventory.json`,
  `appearance-controls.json`, `completed-render.json`: exact cases, shader/pose
  controls, RGB/mask hashes and original asset hashes. All 192 cases passed the
  exploratory visibility screen; none was filtered out. Original USDs are unchanged.
- `fasterrcnn-paired-v1/` and `fasterrcnn-roi-v1/`: source plans, synthetic splits,
  frozen executed helper snapshots, pretrained pins, curves, fixed final weights,
  freeze markers and real/synthetic predictions. Real scoring follows each freeze.
- `review-v1/` and `roi-review-v1/`: source-frame overlays and machine-readable
  comparison. ROI synthetic comparisons map boxes back and keep original full-frame
  geometric targets; the real reference labels are also unchanged.
- `basal-v1/`: frozen geometry rule, native vial detections, raw/filtered scores and
  illustrative before/after PNG. `basal-pipeline-v1`, `paired-pipeline-v1`,
  `roi-pipeline-v1`: actual per-frame predictions/SAM masks/QA and completion hashes.
- `summary-v2/comparison.json`, `metrics.csv`, `targeted-transfer-scorecard.png`:
  combined verified results. Source/clock identity, QA config and implementation,
  model hashes, three fixed epochs and 154 optimizer steps/epoch were checked.

Scripts: `target_synthetic_transfer.py`, `vial_basal_policy.py`, `vial_input_view.py`,
`review_targeted_transfer.py`, `summarize_targeted_transfer.py`. Existing training
and robustness helpers gained optional crop/association branches; defaults remain
unchanged. New roots refuse existing outputs. Preserve incomplete directories and
use a fresh path when repeating. Python compilation, ROI coordinate round-trip,
upper-label exclusion, missing-vial abstention, synthetic-family separation and
matched training-budget checks passed. No models/media were committed or pushed.
