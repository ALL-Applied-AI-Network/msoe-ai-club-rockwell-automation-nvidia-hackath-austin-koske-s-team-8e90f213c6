# Synthetic liquid transfer: inspection before another fit

## Targeted follow-up — paired rendering, basal association and vial crop

[Completed protocol and results](TARGETED-SYNTHETIC-TRANSFER.md). Generated 192
new paired scenes with an isolated refractive-glass material, small pools and
lighting/paper/camera variation; original USDs unchanged. Two new synthetic-only
fits retained 614 training samples/epoch, three fixed epochs and COCO initialization.
Paired-render held-out geometric AP improved 42.47%→74.40%, including small-body
matches 0/7→5/7. Real provisional AP worsened 26.50%→23.18%; a lower-vial crop fit
scored 8.44%. Basal filtering reduced unmatched boxes 265→88 but matches 70→61.
Actual real raw QA availability was 61/111 baseline, 20/111 paired, 23/111 crop,
59/111 basal. These are availability/provisional agreement metrics, not reviewed
accuracy. No candidate is promoted and no Run 5 input was used.

Verified scorecard/CSV/JSON:
`C:/Users/Austin/ai-club-tools/results/targeted-synthetic-20261022/summary-v2/`.
Seven unchanged source images/questions are in `Human-Target-Scope-Review.zip`.
Target-scope approval and controlled empty/known-fill optical calibration remain
needed; neither additional appearance variation nor geometric filtering closed
the real-transfer gap in these pilots.

## October 22 follow-up — synthetic-only competition direction

Austin reaffirmed that new CSI-domain training must use generated imagery. Real
data supports inspection/evaluation, not weight fitting. The real-trained demo
remains a legacy development comparator; it is not the synthetic-trained
competition solution. Generic COCO/SAM pretraining and prior real-informed
camera/prompt development are disclosed separately. Current team assignments
are in [the team handoff](../../team_handoff/README.md#current-assignments--october-22).
The October 19 audit below is preserved as historical evidence.

**October 22 source-image comparison:** five new PNG boards are under
`C:/Users/Austin/ai-club-tools/results/real-vs-synthetic-20261022/review-v1/`.
They show raw real/render RGB at native scale, identical lower-vial crops,
eight fresh frozen-model predictions, unchanged target scopes and native masks.
Pairs are illustrative, not matched recipes/volumes; real Run 1/3/4 images were
not used to train this synthetic-only model. Synthetic train/val status is shown.

The selected shallow-blue real example gets a plausible bottom-pool proposal
at 0.988 **plus three additional proposals** over vial/label regions, versus
one proposal on the rendered example. A real image with no supplied liquid
target gets two high-confidence proposals; its rendered empty counterpart gets
none. An established blue example gets a single visually localized body box in
both domains. This is partial transfer with spurious proposals, not evidence that
the pipeline cannot find any real liquid. No reviewed accuracy is inferred from
these selected examples.

The real source `vial_fill_20260922_171633_379025` has a 20px provisional liquid
target near the holder, not a box spanning the dark lower region visible in RGB.
Its small box is not evidence of a first shallow pool. Other unchanged real boxes
switch between stream-plus-body and body-only scope. Therefore the proxy AP gap
mixes visual transfer and reference-semantic differences; it cannot be attributed
entirely to rendering. The three tiny blue/pale/clear render examples each have
the same 1,158-pixel native body mask despite different RGB visibility. Flag
these target/visibility cases for human review; no labels were replaced.

Priority for the next controlled iteration: agree visible-pool scope and a
reviewed real evaluation set; match empty glass, label placement, lighting and
camera geometry with varied synthetic hard negatives; then add realistic first
pool/optical variation. These are evidence-based hypotheses to test, not an
isolated causal diagnosis or permission to fit on real images.

Local root: `C:/Users/Austin/ai-club-tools/results/sim-real-robustness-20261022`.
The verified combined report and chart are `summary-v1/comparison.json` and
`summary-v1/synthetic-transfer-scorecard.png`. New model roots are
`C:/Users/Austin/ai-club-tools/results/synthetic-liquid-20261022/fasterrcnn-v1`
and `fasterrcnn-aug-v1`. No candidate was promoted and no original labels or
teammate scene assets were edited.

### Completed local work

- **CPU transfer/robustness:** 2,412 fresh detector evaluations: three frozen
  YOLO checkpoints × 157 original Run 4 stills plus 111 cached Isaac samples ×
  raw/green-gain/blur conditions. This is 268 source images repeated across arms,
  not 2,412 independent trials. The legacy operating point reproduces the
  previous Run 4 audit's 69 matches / 12 unmatched boxes / 20 missed targets.
- **New synthetic-only Faster R-CNN fits:** two fixed three-epoch pilots,
  identical COCO initialization and disjoint image-hash split, 614 synthetic
  train / 154 synthetic development images. One uses raw rendered RGB; the
  other samples encoded exposure/color-gain/blur/JPEG variation during training.
  Both freeze the final epoch before opening real Run 4 evaluation images.
- **Actual detector → SAM → QA:** synthetic-trained joint YOLO on six conditions
  (1,332 samples), then synthetic YOLO vial + each new Faster R-CNN liquid model
  + generic SAM2.1 on raw/green/JPEG conditions (666 samples each). All use the
  same original RGB/clock joins and frozen prompt/QA rules. No recorded model
  predictions are substituted for inference. Real input is three predeclared
  37-sample windows; simulation uses all 111 cached frames.

### What the new detector scores say

All real scores below compare with **89 provisional liquid boxes on 157 Run 4
images**, including 68 provisional label-absence images. They are not human
accuracy or confirmed-empty false-alarm rates. AP uses confidence-ordered
one-to-one matching and 101-point precision-envelope averaging; the integration
differs slightly from previous Ultralytics AP, so do not compare the AP decimals
as if the implementations were identical. All rows here use the same new scorer.

| Model / CSI-domain fine-tuning | Run 4 liquid AP50–95 | Precision at .35 | Recall at .35 | Unmatched boxes |
| --- | ---: | ---: | ---: | ---: |
| Legacy real-trained YOLO comparator | 50.21% | 85.19% | 77.53% | 12 |
| Original synthetic-only YOLO, 768 scenes | 26.45% | 55.08% | 73.03% | 53 |
| Prior revised synthetic-only YOLO, 18 scenes | 10.59% | 11.64% | 24.72% | 167 |
| New synthetic-only Faster R-CNN | 26.50% | 20.90% | 78.65% | 265 |
| New synthetic-only Faster R-CNN + RGB variation | 24.96% | 14.10% | 74.16% | 402 |

The new Faster R-CNN raw/augmentation arms score **95.79% / 96.70%** on held-out
synthetic geometry but **26.50% / 24.96%** on provisional real targets. That is
a proxy generalization failure, not a calibrated physical accuracy gap: source
distributions and target semantics differ. The three-epoch budget, one seed and
different older YOLO training budgets also prevent an architecture superiority
claim. This RGB-augmentation recipe did **not** improve real agreement.

### Full-pipeline availability exposes a second transfer problem

| Synthetic-trained detector path → generic SAM2.1 | Real raw QA / 111 | Real green-gain QA / 111 | Real JPEG QA / 111 |
| --- | ---: | ---: | ---: |
| Joint YOLO vial/liquid | 31 | 30 | 22 |
| YOLO vial + new Faster R-CNN liquid | 61 | 60 | 32 |
| YOLO vial + augmented Faster R-CNN liquid | 55 | 56 | 31 |

These are **availability counts, not correctly segmented frames**. More passes
cannot establish improved localization or recipe decisions without reviewed
mask/visibility truth. For the raw Faster R-CNN arm, JPEG retained **31/61**
previously usable real samples versus **72/73** simulated ones. Masks compared
with raw predictions had mean self-IoU **0.599 real / 0.936 simulated**. The
domains are different scenes and trajectories, so this is a diagnostic mismatch
in nuisance response, not a controlled causal estimate of the sim-to-real gap.
It identifies a failure the current simulated scenes do not expose equally.

### What should change in the next generated batch

The byte-verified corpus audit (`synthetic-corpus-audit-v2.json`) checks all 768
original and 18 revised source image/label pairs. Each corpus has **one native
vial-box geometry**. Original positive body boxes start at **36 pixels high**;
none is ≤32 pixels. The revised set has five ≤32-pixel boxes, but only 18 unique
scenes repeated 91–151 times each in its 2,210-sample training epoch. Neither
original fit had a held-out synthetic scene split; its training-set validation
was resubstitution. The new Faster R-CNN pilots fix the split, not the optics.

Prioritize a bounded, reviewed fixture spanning first localized pools, pale/clear
RGB observability, meniscus/refraction, glass/label appearance, camera variation
and real encoding sensitivity. Preserve distinct pool/stream/glass targets and
raw geometric masks; do not upgrade geometric masks to approved visible-pool
truth. Hold scene/seed families out before fitting. Increasing repeated-image
counts or adding generic RGB variation alone is not evidence of better transfer.
No new 3D scenes were rendered in these October 22 experiments.

For a defensible sim-to-real claim, report a synthetic-trained model against
reviewed real onset/mask/defect references, then demonstrate improvement on fixed
development cases and confirm it on a fresh independent physical sequence. Keep
all missed, uncertain and unavailable cases. Real first-visible-pool and recipe
defect accuracy remain unmeasured. Run 5 was not used.

### Dataset 1 extension and operational boundary

The official [Dataset 1 page](https://msoe-maic.com/library?article=Rockwell-NVIDIA-Fall-2026-Vial-Inspection)
describes per-position final-fixture inspection and synthetic training. Our local
tray-inspection USD/camera assets are available, but its real labeled fixture
crops were not located. Dataset 2 cap-seal context is not automatically Dataset 1
evaluation truth. The [extension proposal](../../team_handoff/integration/dataset1-extension-20261022.json)
keeps station evidence separate unless a physical vial-to-slot link is confirmed.
Expected placement progress is needed to distinguish a missed placement from an
unfilled future position. The operator workflow can present both stations without
inventing that link. No industrial connection or control action is implemented.

### Reproduction and preserved failures

Use `experiment_sim_real_cpu.py` for the frozen CPU probe,
`train_synthetic_liquid_pilot.py prepare` then `fit` for new synthetic-only fits,
and `experiment_sim_real_robustness.py --arm synthetic_fasterrcnn
--synthetic-rcnn MODEL_ROOT --conditions raw green_085 jpeg_q40` for the model-to-QA
comparison. Each requires a fresh output. Retained plans/helpers pin the executed
bytes; `summarize_synthetic_transfer.py` verifies completed artifacts, disjoint
synthetic image splits, matched source clocks and legacy fixed-point parity.

The first GPU attempt failed at warm-up near the Windows memory-commit limit.
Austin approved closing existing Isaac/Python PID 91932, which held about 191 GiB
private memory; GPU inference then completed. This does not diagnose the cause of
that process's memory growth. CPU attempt v1 had a reporting error; its replacement
v2 completed. Partial directories are retained and are not scored results. The
verified summary uses only completion-marked runs. No process or UI was deployed
as a production station by these experiments.

---

October 19, 2026. This is a diagnostic of the frozen 768-image GB10 pilot,
not a new model, reviewed label set, optical calibration or renderer change.
The evidence shows **both inconsistent target scope and a visible rendering
gap**. Their individual contributions to transfer failure have not been
isolated experimentally. More random colors or a larger training job alone
would leave these problems in place.

## Local examples

All files are under
`C:/Users/Austin/ai-club-tools/results/real-synthetic-detector-20261019/transfer-diagnosis/`.
They are ordinary PNG/JSON files; no browser or preview server is required.

| File | What to inspect |
| --- | --- |
| `01-scene-gap.png` | Original real and GB10 RGB at identical display scale. The station is approximately aligned, but lighting, glass, paper labels, background and surface appearance differ. Colors/volumes are illustrative, not paired physical measurements. |
| `02-target-contract.png` | Three consecutive Run 1 training stills: the teacher's liquid box covers stream plus pool, then switches to pool only. Synthetic targets consistently separate body and stream. |
| `03-pale-rgb.png` | Real pale low/higher fills versus pale/clear proxies. Real images contain curved surfaces, bubbles and irregular bottom pools. Pale proxies can look milky; clear proxies can barely differ visibly from empty glass. |
| `04-native-masks.png` | The body geometry masks corresponding to the last four examples in sheet 03. They remain positive regardless of whether RGB supports a visible liquid boundary. |
| `05-real-early-pale.png` | Run 5 frames 2730, 2760, 2796, 2820, 2865: 91.00–95.50 MP4 seconds. Thin stream and evolving bottom pool are separate visual signals. The displayed onset is an image estimate, not a reviewed physical event. |
| `06-transfer-errors.png` | Frozen current/sim-only predictions at frames 1410 (47.00s), 2796 (93.20s), 4188 (139.60s). Sim-only produces oversized/duplicate boxes; both miss the pale low pool at 2796. |
| `07-evaluation-label-issue.png` | Run 5 still `vial_fill_20260923_194709_622176`: the teacher spans almost the whole vial, while both models localize a plausible bottom pool. Agreement penalties can reflect target scope. |

Sheets enlarge a fixed source crop solely for review. Inference and saved
targets use original 640x480 full-frame coordinates. Cyan=vial, yellow=liquid
body, pink=stream. Source images, full-frame target overlays and native mask
paths are linked from `audit.json`; selected original video frames are in
`sources/`. No boxes have been corrected by this audit.

## Findings and what they establish

**1. Body and stream are not consistently separated in the real proposals.**
The intended target is the visible accumulated body, including its meniscus
and lower extent, as documented in [BOXES.md](BOXES.md). For example, Run 1
17:05:44.697 and 17:05:46.705 have liquid boxes extending from near the nozzle
to the bottom. At 17:05:48.711 the box instead surrounds only the bottom pool.
These are existing teacher proposals, not reviewed truth.

Across unique originals, 20/354 Runs 1–3 liquid targets and 14/89 Run 4 targets
have tops above y=200. The audit flags them for review; that coordinate rule
does **not** certify that all flagged boxes are wrong or that remaining boxes
are correct. Run 5 has another 35/541 flagged liquid targets. No targets are
dropped and no score is recomputed with a favorable subset.

**2. The synthetic training contract does not teach stream detection.**
The raw batch has stream masks/boxes, but the two-class training copy keeps
only active vial 101 and liquid body 201. All 96 stream-only scenes therefore
have a vial target and **no liquid target**. This is consistent with a body-only
contract; it cannot provide positive supervision for detecting the earliest
falling stream. Stretching the body box to include the stream would recreate
the ambiguity in finding 1. See [VISIBILITY.md](VISIBILITY.md).

**3. Native geometric correctness is not RGB observability.**
The content-label pass hides transparent vial shells and the guard, and makes
liquid opaque. It retains opaque paper/holder/nozzle occlusion. Its mask-derived
boxes and projected geometry passed the earlier native integrity checks, but
those checks do not establish the outline seen through transparent/refracting
materials in RGB. In the original fixture, tiny blue, pale and clear bodies
have the same `[289,381,333,417]` geometric box and the same 1,158-pixel mask,
although their RGB appearance differs markedly. These are geometric proxies,
not reviewed RGB liquid segmentation truth. A bbox can legitimately enclose
occlusion; that still does not make every enclosed pixel a measurement pixel.

The renderer creates constant-radius, flat-ended cylinders with one shared
`UsdPreviewSurface` material for body and stream. Its controls are diffuse
color, surface opacity, roughness and IOR. It contains no modeled meniscus,
wall wetting, bubbles, splash evolution or volume-dependent dye absorption.
Real early-pour examples visibly contain cues missing from these proxies.
Pale/clear are already numerous: 292 pale and 169 clear-proxy randomized
scenes, plus the original fixture. The issue is not simply their absence.

**4. The earliest body states and scene variation are limited.**
Positive synthetic fills begin at 2% of the proxy's height envelope. The
smallest positive body box is 36 pixels tall; original real teacher boxes
include extents near 20 pixels. These sizes are a diagnostic distribution
comparison, not verified equivalent physical volumes. A low cylinder still
has a full-radius projected cap: lowering its height alone will not create a
localized wet patch or irregular first pool. Stream-only fixtures are short
jets below the nozzle, not a time sequence of drop/jet impact and pooling.

All 768 native active-vial boxes are `[279,66,341,420]`, a single pose/geometry.
Image translation/scale augmentation does move training boxes, so this does
not prove position memorization. It does leave paper-label orientation,
camera perspective, glass appearance and physical surface variation largely
unchanged. Prioritize these after target scope and pale RGB are credible.

**5. The transfer errors are not display-crop or ONNX projection errors.**
This trial used original full-frame pixels with ordinary YOLO letterboxing in
all arms. Existing checkpoint/ONNX parity checks passed. The oversized boxes
in sheet 06 are present in saved PyTorch predictions at the unchanged 0.35
confidence / 0.45 NMS settings, before display cropping or tracking. Synthetic
body targets never start above y=217, so the huge real-image boxes are not
literal copies of giant native body targets. They are generalization failures;
this audit does not isolate which appearance cue causes them.

The existing liquid AP50–95 results remain current 49.18%, real-only control
48.31%, mixed 48.40%, sim-only 20.33%. They measure agreement with provisional
teacher boxes, not independently reviewed physical accuracy. The synthetic
training-set score is resubstitution, not transfer validation. Label problems
complicate interpretation but do not excuse the visible sim-only errors.
Detection, completed-vial measurement availability and color QA remain separate.

## Minimum changes before another training run

| Owner | Specific next input | Review gate before scaling/fitting |
| --- | --- | --- |
| Aleyna / Zoe / Reagan | Review the body-vs-stream cases, then a small Runs 1–3 set spanning empty, first visible pool, pale/clear, partial/full and occlusion. Record uncertain/unobservable separately; a missing body proposal does not prove empty. | One consistent active-vial / visible-body definition, separate stream annotations if that class is pursued, plus reviewed examples and visibility status. No blanket conversion of flags to corrections. Review Run 4 independently for development; keep Run 5 evaluation material separate. |
| Cameron / Hayden | Produce a small revised RGB+body/stream-mask fixture using Runs 1–3 references: first drops/jet, localized first pool, curved surface/bubbles, pale/clear optics, representative paper labels. Keep native ID/mask/box integrity checks. | Review RGB alongside geometric masks. A reviewer should be able to identify the intended visible cue or mark it unobservable. Demonstrate empty-vs-pale distinctions and consistent body/stream scope before another 768-image batch. Match appearance using training references, not Run 5 parameter fitting. |
| Austin / replay owner | Keep body boxes for localization and downstream verified pixel sampling. For earliest flow, evaluate a separate `stream` target/cue rather than redefining the accumulated body. Prepare a fixed, reviewed localization benchmark once labels arrive. | Freeze classes, visibility policy, source splits, settings and palette/stage balance. Compare real-only vs mixed vs sim-only with reviewed detection metrics, then assess tracking and color QA separately. Change one major factor at a time so the next trial answers a specific question. |
| Srithej / Josiah | Continue CSI/workflow coordination and obtain approved evidence about pale/clear fluid behavior and the intended operational onset signal. | Physical observations/approved footage can constrain the renderer; this audit neither contacts equipment nor establishes camera/network approval. |

No new training, renderer changes, QA thresholds, model promotion, event-label
adjudication or stage-aware alerts were performed. The current detector remains
the default. No additional compute booking is needed to review these examples.

## Reproduction and machine-readable handoff

From the repository root, with a **fresh output directory**:

```powershell
$trial = "$HOME/ai-club-tools/results/real-synthetic-detector-20261019"
$python = "$HOME/ai-club-tools/envs/data-windows/Scripts/python.exe"
& $python setup/vision/audit_synthetic_transfer.py --experiment $trial --data datasets/dataset2/csi-vial-fill-dataset --teacher "$HOME/ai-club-tools/results/typesafe-box-all-20261018" --out "$trial/transfer-diagnosis-reproduced"
```

The script checks all 878 original real image/label hashes (721 train, 157
development), selected native RGB/mask hashes, seven original Run 5 RGB hashes
against frozen prediction records, and MP4 timestamps. It inventories all 768
native manifest records, reuses frozen predictions, writes seven sheets and
27 selected case records, and refuses to overwrite an output directory.
The existing full-batch native verification remains in `renders/verification.json`.

`audit.json` schema version 1:

| Field | Meaning |
| --- | --- |
| `kind`, `schema_version` | Diagnostic format identity. |
| `no_training`, `no_label_edits`, `no_qa_changes` | True for this handoff. |
| `experiment`, `source_inventory_sha256`, `render_manifest_sha256`, `video_sha256` | Frozen input identities. |
| `statistics` | Per-run/renderer box counts, dimensions and top-y quantiles; palette and level inventory; review-flag counts. Quantiles are min/q25/median/q75/max. No new accuracy scores. |
| `cases[].id`, `origin`, `training_used` | Example identity, real/synthetic/video/evaluation origin, and whether it entered fitting. |
| `source_image` / `source_video`, SHA fields | Absolute source location and encoded-file identity; video cases also carry decoded RGB SHA. Evaluation stills use `image_sha256`. |
| `frame_index`, `mp4_s` | Video-only zero-based index and container-relative time at 30 fps. Stills retain wall time in their source identity; no guessed MP4 mapping. |
| `targets` | Unchanged full-frame half-open XYXY boxes and class IDs; null where no reviewed/teacher target is supplied. Class 2 is raw synthetic stream, excluded from this experiment's training. |
| `predictions` | Where available, frozen `baseline`/`candidate` boxes, classes and confidence. Video records use the original 0.35 filter; the selected evaluation still is filtered for display identically. |
| `parameters`, `native_annotations`, `masks_root` | Synthetic-only controls and per-object native mask, box, hash, pass, visibility provenance. Mask paths are relative to `masks_root`. |
| `exported_image`, `target_overlay` | Absolute local files for pixel/target review where applicable. |
| `target_status`, `limitations` | Proposal/geometric/unknown status and interpretation limits. |

`real-target-review.json` contains 34 unchanged Runs 1–4 proposals flagged for
human scope review; `run5-target-review.json` contains 35 separate evaluation
proposals. Each record includes `source_id`, `source_run`, absolute image/label
paths, both hashes, unchanged `targets`, `reason`, and
`status: needs_human_review_not_adjudicated`. These lists are not corrected
labels, an exhaustive error inventory, or an automatic training admission list.

Run 5 remains excluded from fitting/checkpoint selection. It has been repeatedly
inspected, so it is diagnostic evidence, not a pristine blind test. Examples
are deliberately chosen; pale is an image description, not an inferred recipe.
