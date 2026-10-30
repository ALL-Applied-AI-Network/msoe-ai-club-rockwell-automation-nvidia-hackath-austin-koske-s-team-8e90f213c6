# Dataset 1 five-class / two-fixture extension — October 23

This extends the earlier three-class, G-only pilot toward the
[official Dataset 1 task](https://github.com/MSOE-AI-Club/maic-content/blob/main/unlisted/Rockwell-NVIDIA-Fall-2026-Vial-Inspection/Rockwell-NVIDIA-Fall-2026-Vial-Inspection.md).
The task is standalone overhead inspection of explicit target positions after
placement; it does not require a filling-camera physical-vial identity join.
The models and placement module are opt-in and do not replace the operator app.

## Experiment design

All new GPU work uses the GB10 and its shared GPU lock. Local preparation,
review tooling, tests and media packaging use CPU. Inspect existing jobs and
completion markers before starting anything; the run does not resubmit itself.

Local root: `C:/Users/Austin/ai-club-tools/results/dataset1-fiveclass-20261023`.
Remote root: `~/ai-club-tools/experiments/dataset1-fiveclass-20261023-v1`.

| Item | Frozen design |
| --- | --- |
| Target classes | pass, missing_cap, crooked_cap, damaged_vial, no_vial |
| Target fixtures | G01–G30 and proposed S01–S30 |
| Scenes | 120 independent nuisance families, each with matched golden/inspection RGB |
| Split | 72 train / 24 development / 24 test families; 4,320 / 1,440 / 1,440 targets |
| Per scene | Six examples per class per fixture, randomly assigned |
| Crops | 96×96 source pixels around fixed aperture-plane projections; full scene retained |
| Models | RGB; RGB plus Gaussian target-position channel; current/golden RGB plus target-position channel |
| Architecture | Random-initialized Conv24/48/96, GroupNorm6, adaptive4×4, Linear128/5 |
| Fitting | 35 epochs per arm; AdamW0.001, weight decay0.0001; batch128 |
| Selection | Synthetic development macro recall, earliest tie; every arm frozen before test RGB decoding |
| Defect score | 1−P(pass), not calibrated quality probability |
| Warning threshold | Minimum development score over all three occupied defect classes; frozen before test |
| Real inference | RGB and target-RGB only; review arm selected by synthetic development, simpler RGB wins ties |

Generic pretraining is absent in these fixture models. Real data and Run5 do not
enter fitting, model/threshold selection or render-family split decisions.
The model weights and training-plan hashes are retained with executed helpers.

## Synthetic geometry and labels

The original supplied USD stage and camera configuration are hash-pinned and
never saved over. All changes use a session layer. Both trays use canonical vial
copies. S positions derive from the G aperture layout transformed between the
supplied tray frames; the original few scattered S objects are hidden in the
session. This gives 60 attributed synthetic targets, not approved real numbering.

The golden image has correctly seated caps and unchanged bodies at the same
camera, lighting and per-vial lean as its inspection image. Class geometry:

- `pass`: intact tube and correctly seated cap.
- `missing_cap`: tube present, cap hidden.
- `crooked_cap`: cap-only tilt12–32 degrees plus1–4mm lift relative to its tube.
- `damaged_vial`: a controlled upper-wall chip or dent deformation, without a
  class-specific color/material. These are damage proxies, not a comprehensive
  physical fracture model or validation of real damage visibility.
- `no_vial`: tube and cap hidden at that target.

Body lean0/8/18/30 degrees varies independently across all classes, so a leaning
vial alone does not define a crooked cap. Other nuisance includes light scale
0.65–1.4, cap color variation and camera translation±3 pixels. RGB is label-free;
the optional Gaussian channel indicates the numbered aperture, not its class.
Illustrative target rings in output contact sheets are added after inference.

Every scene and its golden/crops stay in one split. All splits share the supplied
CAD, positions and camera family; held-out performance does not establish new
geometry or real-domain generalization. Synthetic masks/geometry do not replace
independent real optical labels.

## Metrics and interpretation

**October23 cap-demo audit correction:** the fresh720-target cap-state result
is608/608 nominal,36/38 missing,31/38 crooked and36/36 empty. The98.75% aggregate
must not be described as98.75% crooked-cap detection; crooked exact-class recall
is81.58%. Earlier score-only cap-warning metrics exclude empty targets using
evaluation truth; the UI also suppresses predictions of empty. Its actual gated
warning misses are0/38 missing and1/38 crooked. Those are different policies.
The cap classifier emits a class for a numbered fixture crop, not a detected
cap-center location. The first demo drew its fixture-plane centers on the image
as cap markers; that overlay was misleading and is now replaced by explicit
slot input regions/raw crops. See the
[display and metric correction](../../replay/operator/FULL-APP-DEMO.md#cap-display--metric-correction).
Frozen original model files and historical reports are retained unchanged.

The first completed five-class baseline is **not ready for operator use**.
On1,440 held-out targets (288 per class), the results are:

| Model | Exact class accuracy | Missing cap recall | Crooked cap recall | Damaged vial recall | Acceptable-vial false alarms at frozen all-defect-recall threshold |
| --- | ---: | ---: | ---: | ---: | ---: |
| RGB | 59.79% | 84.72% | 47.57% | 30.56% | 288/288 |
| Target RGB | 61.39% | 89.58% | 39.24% | 45.14% | 287/288 |
| Target paired | 74.86% | 93.40% | 82.29% | 42.71% | 288/288 |

Every arm has0/24 entirely correct60-position scenes and0/48 entirely correct
30-position fixtures. The target-paired arm improves exact classification, but
the minimum-score warning policy is unusable: approximately every acceptable
target warns. RGB misses0 synthetic occupied defects at that operating point;
target RGB and target paired each miss one damaged target. High recall alone
does not make this a useful inspection system. These are synthetic results.

The six post-fit placement states expose the same issue. Target RGB is313/360
exact across all states, driven by many empty targets; on the completed normal
G order it raises30 G and5 S review requests. This is a failed normal-state
control, not a successful placement demo. The original video retains those
actual predictions. No model was promoted.

The completed archive SHA256 is
`2a214de9bad8bb2cffec506e0879089222b2001506b2672ed2382e6347eabff6`.
All4,320 original held-out prediction joins and288 output video frames were
independently checked after transfer. The real probe yields960 unreviewed
target-frame proposals per RGB arm; no reviewed accuracy is available.

`gb10-results-v1/fitted-v1/comparison.json` is the source for completed synthetic
metrics. `summary-v1/report.json` independently checks every source/slot/label/
prediction join, reported confusion and threshold/epoch selection rule.

Report exact-class confusion and per-class recall/precision alongside the frozen
occupied-target warning. A damaged vial classified as crooked is a class error
but need not be an escaped defect. Count warning misses separately for each
defect class, and report the acceptable-vial false-alarm rate. Empty slots are
excluded from this occupied-target diagnostic and assessed with progress context.
All30-correct fixture counts and all60-correct scene counts expose correlated
failure that aggregate accuracy can hide. Forward timing excludes preprocessing,
decoding, rendering, transport and dashboard; it is not live station latency.

## Placement progress and rendered rehearsal

`replay/operator/fixture_placement.py` consumes one fixture's 30 predictions and
explicitly attributed order ID, position order and placement count or completion
context. It never uses its own occupancy predictions to invent expected progress.
The supplied order is independent for G and S; real ordering must be confirmed.

- An empty target within the expected pattern requests missed-placement review.
- A trailing empty target remains not-yet-expected.
- Missing progress gives unknown placement status, not an automatic defect.
- Unconfirmed mapping or visibility blocks a qualified target finding.
- Visible defect classes and frozen occupied-target warning scores request review.
- Source changes, rewind and explicit new orders clear stale progress context.
- No cap proposal clears a filling warning, approves a product or commands hardware.

Six separately generated, post-freeze synthetic states exercise actual selected
model predictions plus this logic: empty, first placement, a missing G03, crooked
G07/missing S02, damaged G16, and a completed normal G order. Model input receives
images/target positions only; private synthetic labels are used after inference
for evaluation. Public expected progress is separate from the predictions.

The MP4 in `summary-v1` holds each rendered state for two seconds. It shows actual
model results on six images; it is not a robot-motion simulation, native source
video, live inference loop or camera-to-dashboard latency measurement.

## Real transfer and independent evaluation

The real probe reuses the earlier16 source JPEGs from Runs1–4, preserving original
bytes/capture timestamps. Each supplies60 proposed targets. G-anchor camera fit
projects both trays; S has not been independently calibrated. Both fixture maps
and target visibility remain human-unreviewed. No still/video clock join is made.

A vertical structure visibly obscures part of the proposed rightmost S column
in inspected examples. The full-scene review is essential: a classifier's empty
proposal there does not prove a missing vial. The fixed-parameter S circle
diagnostic is recorded separately and does not approve mapping or visibility.

`real-batch-v1/inference/completed.json` records actual frozen-model proposal
counts, not accuracy or distinct vials. The paired-reference arm is unavailable
on real imagery because there is no qualified capped real golden; none is invented.

The expanded offline review packet uses the existing raw-first editor with960
targets/192 positional priority cases. It accepts all five classes plus uncertain,
retains full-scene/crop/source identity and rejects demo exports as real truth.
The scorer now handles both fixtures,60-target scenes, model-supported classes
and escaped-defect versus class-error metrics at the frozen threshold.
No actual human export has been supplied; reviewed real accuracy remains null.

`reserved-real-evaluation.json` prospectively reserves25 additional Runs1–4
images by filename/hash, excluding the16 diagnostic sources, without decoding
their pixels in this session or generating predictions. Prior exposure elsewhere
in the project is not audited; do not describe this as a proven blind test.
Obtain independent human review before scoring.

### Organizer participant tool and native imagery

Austin supplied the organizer's October23 clarification: the former promise of
a separate labeled Dataset1 package is deprecated. Dataset2 cap_seal imagery is
available for real testing, and synthetic generation is intended. The organizers
allow teams to choose their training mix; Austin's current experiments continue
to use synthetic-only CSI fitting.

`datasets/vial-labeler-participants-v1` is a manual labeling tool, not a neural
autolabeler. All79 files match its package manifest. It contains two1920×1080
real example images, two calibration profiles and a portable manual interface.
Both examples were prepared into `native-review-v1/index.html`, with120 unresolved
positions and no predictions or confirmed labels. Its tool/configuration
fingerprint is called `model_hash`; that field does not identify a trained model.
The package does not attribute reviewer identity, so any eventual evaluation
import must separately retain the real reviewer's identity and immutable template.

These native calibration coordinates cannot be reused on our640×480 stills by
assumed scaling. The view, lens mapping, target crop scope and resolution must be
explicitly matched. The expanded640×480 `review-v1` is a separate960-target,
192-priority packet with unapproved mapping/visibility. Browser save/download/
reopen passed in demonstration mode; that export remains ineligible as truth.
The native manual page also opens with two images and0/120 confirmed positions.

## Integration boundary and remaining work

### October23 follow-up: detail, normal coverage and cap scope

The detail ablation trains two random-initialized branches: a central64×64 crop
upsampled to96 and the full96×96 context. The paired arm adds golden RGB and
absolute RGB difference, plus the target-position channel. Models fit for45
epochs on the original train/development families; every checkpoint and warning
rule freezes before new test inference. A new seed90313 supplies12 test families.
RGB / target RGB / target paired / detail RGB / detail paired exact-class accuracy
is57.50% /58.75% /76.39% /60.69% /78.19%, respectively, on the same720 targets.
Detail paired gives136/144 crooked-cap hits but65/144 damaged-body hits.
Development95%-pooled-defect thresholds still warn on114/144 acceptable targets
for detail paired, versus112/144 for the simpler paired model. More detail alone
does not solve warning quality.

The normal-coverage ablation adds each **training** golden image as an all-pass
current example, and likewise adds **development** goldens only to development.
It doubles training target observations without mixing whole-family splits.
These normal-current/reference inputs share exact original RGB bytes; joint
augmentation preserves that equality. This is an optimistic normal control,
not a simulation of independent camera captures.

New seed502113 supplies six dense and six sparse test families. Each inspection
and its all-good golden-as-current are evaluated:1,440 target observations from
12 correlated families. On this same set, the previous detail-paired model is
78.33% exact /84.43% macro recall; the normal-trained model is93.82% exact /
89.35% macro recall. Dense / sparse / identical-normal exact accuracy changes
84.17% /73.33% /77.92% to80.83% /94.44% /100%. This improves sparse/normal behavior
while worsening dense exact classification, so the aggregate gain needs context.

At each model's independently frozen95%-pooled-development-recall threshold,
acceptable-target false alarms fall867/1,140→90/1,140. Damaged-body misses rise
8/76→15/76. Both have0/76 missing-cap and0/76 crooked-cap warning misses. The
new model's exact damaged-body recall is46/76. Pooled95% development recall is
not a95% per-defect guarantee. No real operating threshold was selected here.

Controlled isolated mesh changes expose the optical issue: in central64×64
crops, twelve capped body-damage proxies change only1–42 pixels above a
two-digital-level floor, versus939–1,229 for four cap controls. Mean absolute
change is0.059–0.308 versus3.994–10.207 digital levels. Repeated unchanged
rendering gives zero measured variation, but may reuse accumulation; no physical
camera noise is modeled. Some damage signal exists. This does not prove that
all physical damage is visible or that every geometric damage label is optically
useful. Full source images and difference boards are retained.

The source-CAD pose audit finds another concrete gap. Its five authored S bodies
have principal axes80.88–83.79 degrees from vertical; the canonical generated
bodies cover only0–30 degrees. These scores do not establish performance on the
authored lying-S geometry. Rebuild/validate S placement and target association
from those assets rather than treating upright S clones as complete coverage.
No original asset was modified and no semantic defect labels were assigned by
this geometric audit.

The dedicated cap-state fit separates nominal cap pose, missing cap, crooked cap
and empty position. In this renderer the damaged-body proxy retains the nominal
cap transform, so it can supply a cap-state target while **body condition remains
unassessed**. This mapping is a synthetic geometry fact, not permission to infer
cap state from real five-class annotation precedence. A nominal-looking cap does
not approve its seal, damaged body, intended recipe or product. The cap model is
not automatically compatible with the old five-class `pass` field; Srijan's
adapter must retain explicit cap scope and unavailable body quality.

Its selected checkpoint is
`05fea80bfcba0f1ec477eaeb5f2e40ad55669a6d093ece442f72a3b2a2da7091`.
It scores1,431/1,440 exact cap states (99.38%);19/24 complete60-target image states
and42/48 complete fixtures are exact. At the frozen99%-development-cap-recall
warning rule it misses0/76 missing caps and2/76 crooked caps, with8/1,216 nominal
cap false warnings. At the separately frozen95% rule those counts are0/76,9/76
and2/1,216. The100%-development rule is zero and warns on every nominal target;
it is retained as a failed diagnostic and must not be promoted as a usable policy.

Reference mismatch is a real engineering constraint. Before correction, reducing
only reference brightness10% gives546/720 normal false warnings at the95% rule;
a one-source-pixel horizontal reference shift gives359/720. These controls do not
qualify a camera alignment tolerance. The existing metadata-based reference gate
can refuse an invalid/unreviewed alignment; an automatic registration-validity
detector for this cap model has not been demonstrated.

`fixture_reference_preprocess.py` estimates one robust current/reference gain per
60-target scene using unclipped pixels outside each central32px target disk.
It uses no labels or predictions, never rescales individual vials independently,
does not align images or approve a reference, and is separate from liquid RGB/
measurement QA. A declared gain range/available-pixel check can return unavailable.
On the same diagnostic sources it restores the tested±10% brightness conditions
to the clean-model result without new fitting or threshold changes. That is a
post-hoc engineering fix; the following fresh confirmation is the separate check.

Seed973133 supplies six new families (three dense/three sparse), their inspection
images and golden-as-current normal controls:720 targets per condition. The model,
gain algorithm, operating rules and seven conditions are frozen before rendering.
Clean/corrected cap-state accuracy is711/720 (98.75%). At the frozen99% rule,
missing/crooked warning misses are0/38 each, with4/608 nominal-cap false warnings.
Reference gains0.83 and1.17 preserve those counts. Reference noise with sigma1/3
digital levels, current noise sigma1, and per-crop JPEG90 controls give98.47–98.75%
exact accuracy; warning misses vary0–2/38 for crooked caps and remain0/38 for
missing caps. JPEG90 gives16/608 nominal-cap warnings, including3/360 all-good
controls. These are correlated digital perturbations, not calibrated physical
sensor noise or real transfer accuracy. Unmodified all-good controls give0/360
warnings. Body condition and authored lying-S coverage remain unassessed.

Classifier forward time for60 crops on GB10 is10.57ms median/10.58ms p95; this
excludes decoding, alignment, gain estimation, cropping, transport and browser.
It is not an end-to-end latency or live-frame-rate claim.

All follow-ups preserve executed helpers and fresh output directories on GB10.
Their source/metric-verified results, reference stress controls, PDF and diagnostic
single-panel MP4 are assembled under
`C:/Users/Austin/ai-club-tools/results/dataset1-followups-20261023/summary-v1`.
The video holds discrete rendered images; it is not a live robot/camera trial.

`export_fiveclass_observations.py` emits the same opt-in `station-evidence/v1`
contract with explicit supported classes, frozen warning score/threshold and
source identity. It retains null product decisions, unreviewed geometry/visibility
and unlinked physical identity. Srijan owns live model integration; Josiah/Hayden
own the dashboard/backend. Their application defaults are unchanged.

Remaining physical gates: review real defects/visibility/slot numbering, establish
a qualified golden if using the paired model, confirm progress/reset/order input
and test the complete operator-mediated pipeline with CSI. Synthetic body damage
coverage and S occlusion remain limitations. No direct sensors/actuators or
industrial network connection is part of this work.

## Reproduction entry points

```text
python setup/vision/dataset1_fiveclass.py prepare --root ROOT
python setup/vision/run_dataset1_fiveclass.py --root ROOT
python setup/vision/finish_dataset1_fiveclass.py --root ROOT
python setup/vision/summarize_dataset1_fiveclass.py --root RETRIEVED --out FRESH_SUMMARY
python -m annotation_review.slot_review build --batch RETRIEVED/real-batch-v1 --out FRESH_REVIEW
python setup/vision/export_fiveclass_observations.py --batch RETRIEVED/real-batch-v1 --out FRESH_EVIDENCE
```

The first three are existing-job-aware GB10 entry points, not instructions to
repeat the completed experiment. Inputs, helpers and smoke artifacts must already
be pinned/prepared. Packaging verifies the complete transferred archive before
extraction; failed/partial directories are preserved rather than overwritten.
