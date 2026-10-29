# Synthetic liquid transfer: inspection before another fit

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
