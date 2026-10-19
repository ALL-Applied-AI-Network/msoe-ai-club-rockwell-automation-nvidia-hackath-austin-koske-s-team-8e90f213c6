# Revised 18-render transfer probe

Austin authorized provisional fitting after reviewing the small GB10 batch:
“we should train it and try.” Raw `training_eligible` and appearance-approval
flags remain false. This is explicit research admission into a **new** experiment,
not an optical certification or modification of the previous frozen plans.

The [renderer handoff](../isaac/PALE-EARLY-REVIEW.md) identifies the final 18-image
native batch and its remaining pale/clear appearance gaps. Body and stream stay
separate in its source masks/three-class export. The present probe keeps the
existing two-class detector: active vial 101 and accumulated liquid body 201.
Stream 301 and neighbor vials stay excluded as training targets, consistent with
the previous station-specific experiment. It does not claim to train a stream
detector; incomplete real stream annotations need review first.

## Frozen fitting protocol

- Mixed: the unchanged 1,442 real Runs 1–3 training images (721 originals and
  their existing translations), plus 768 exposures drawn from the **18 unique**
  revised renders. All 18 enter once; remaining draws use seed 19.
- Sim-only: all 18 enter once, then seeded repeats reach 2,210 synthetic training
  exposures. No real CSI pixels/boxes enter fine-tuning or checkpoint selection.
  The renderer itself was qualitatively guided by real Runs 1–3 RGB, and the
  shared COCO initialization also used generic real images. Sim-only does not
  mean the entire pipeline had no real-image design guidance.
- Both: same COCO YOLO11n initialization, 12 epochs, batch 16, size 640, FP32,
  seed 19 and prior augmentation. Reuse the exact prior fitting functions in
  `train_mixed_detector.py` / `train_sim_only.py` without weakening their guards.
- Mixed checkpoint selection uses the unchanged 157 Run 4 teacher-labeled stills.
  Sim-only uses fixed epoch 12 `last.pt`; training-set synthetic validation is
  resubstitution and its `best.pt` is ignored.
- Run 5 is excluded from fitting, augmentation ancestry and checkpoint selection.
  Existing teacher targets are unchanged. Its repeated use makes this diagnostic,
  not a new blind test. No measurement QA, reference/color logic or default model
  changes are made.

`expected_synthetic_images: 768` in the reused mixed trainer describes the
materialized **exposure budget**, not unique native scenes. The new plan records
`unique_synthetic_images: 18`, the repetition policy, source hashes and limits.
This is not a causal renderer-only ablation: revised and previous batches differ
in scene diversity and positive-class proportions as well as geometry/materials.

## Local preparation and reproduction

From the repository root, choose a new output directory:

```powershell
$dataPython = "$HOME/ai-club-tools/envs/data-windows/Scripts/python.exe"
$visionPython = "$HOME/ai-club-tools/envs/vision-windows/Scripts/python.exe"
$previous = "$HOME/ai-club-tools/results/real-synthetic-detector-20261019"
$renders = "$HOME/ai-club-tools/results/pale-early-render-20261019/gb10-review-final"
$trial = "$HOME/ai-club-tools/results/revised-render-detector-reproduced"
& $dataPython setup/vision/train_revised_render.py prepare --previous $previous --renders $renders --out $trial --accept-provisional-render-proxies
& $dataPython setup/vision/train_revised_render.py materialize --previous $previous --out $trial
```

Preparation requires the exact published final batch's manifest/run/verification
hashes, complete pinned native provenance, source preservation and Runs 1–3-only
appearance references. It verifies RGB and native mask-derived active-vial/body
targets, keeps repeat ancestry together, and records frozen plan/inventory/helper
hashes in `preparation.json`. No source labels or masks are rewritten.

Local fitting, sequentially on one GPU:

```powershell
& $visionPython setup/vision/train_mixed_detector.py train --out $trial --arm mixed --pretrained "$previous/pretrained-yolo11n.pt"
& $visionPython setup/vision/train_sim_only.py train --out $trial --pretrained "$previous/pretrained-yolo11n.pt"
```

For Rosie, copy the frozen plans, inventories, `synthetic/` package and exact
training helper snapshots to a private result directory. Materialize from the
existing Rosie previous experiment with standard Python (no GPU work), then use
the existing `train-mixed-rosie.sbatch` / `train-sim-only-rosie.sbatch` launchers.
They request H100/Slurm and the Singularity 3.10 site PyTorch image; no Docker.
Do not submit duplicate jobs into directories containing another fit.

This execution initially reached Rosie, then SSH hostname resolution failed.
The probes continued on the local RTX 4080 Super, Torch 2.11.0+cu128 and
Ultralytics 8.4.182. Prior Rosie models used H100/Torch 2.8. That hardware/library
difference is an additional comparison limit, recorded in `execution-context.json`.
The previously completed real-only control is reused with its original model
and source-plan identities; it is not presented as refitted under the new plan.

## Original-pixel evaluation and local videos

After both frozen models exist, with a fresh evaluation output:

```powershell
& $visionPython setup/vision/evaluate_revised_render.py --experiment $trial --previous $previous --baseline "$HOME/ai-club-tools/results/crop-clahe-detector-20261019/baseline-best.pt" --teacher "$HOME/ai-club-tools/results/typesafe-box-all-20261018" --data datasets/dataset2/csi-vial-fill-dataset --cache "$HOME/ai-club-tools/results/tracked-liquid-20261019/cache/run5-8bdd8cad2977ee74e6a3.json" --cache-sha256 aff504002184d856ac06451f1730677eace2e840b89110c2cb6c0af39a8850d8 --out "$trial/evaluation" --device 0
```

Evaluation verifies checkpoint/ONNX parity on 16 Run 4 frames after selection,
uses original full-frame pixels, AP confidence .001, display confidence .35 and
NMS .45, and binds both arms to the same 816 provisional Run 5 scoring sources
and targets. Original video indices/timestamps/RGB hashes and current-detector
predictions must match across video arms. All four delivered MP4s are completely
decoded to check their frame counts. No browser/server is used.

Under `evaluation/mixed/` and `evaluation/sim-only/`:

- `run5-full-comparison.mp4`: current left, revised candidate right, full recording
  sampled at 5 fps. Detection presence is not accuracy.
- `run5-early-pours.mp4`: eight fixed neighborhoods at native 30 fps, with source
  frame/time. Their starts are image estimates, not adjudicated physical events.
- `teacher-agreement.json`, `scoring-predictions.json`, `video-predictions.jsonl`,
  `onnx-parity.json`: auditable targets, predictions, source identities and parity.

`evaluation/comparison.json` includes both revised models and explicitly cached
previous mixed, sim-only and real-only-control diagnostics, plus input/output
hashes and full MP4 decode checks. Scores measure agreement with unreviewed
teacher boxes, including known scope inconsistencies. Keep detection metrics,
measurement availability and physical recipe/color decisions separate. No model
is automatically promoted, regardless of its score.

## Executed result, October 19

Both local fits completed all 12 epochs and exported checkpoints/ONNX files.
Mixed fitting recorded 310.89 seconds through its last epoch; sim-only recorded
378.61 seconds, excluding final validation/export. Run 5 remained excluded.
The repeated mixed training inventory contains 1,316 liquid boxes; sim-only
contains 1,779. Positive-class proportions are not matched to the old trials.

**Keep the current detector.** The revised mixed probe catches a plausible
pale low pool at source frame 2796 / 93.20 MP4 seconds that the baseline misses,
but still produces an oversized stream-plus-body box at frame 1410 / 47.00s.
Those are selected visual observations, not reviewed localization truth or a
physical onset measurement. Overall provisional liquid-box agreement declines.

All AP values below are percent-scale AP50–95 against the unchanged 816 Run 5
teacher-labeled stills (814 vial targets, 541 liquid targets; 29 unknown stills
excluded). Precision/recall/FP use confidence .35 and IoU .5; NMS remains .45.

| Model | Vial AP | Liquid AP | Liquid precision | Liquid recall | Liquid FP |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current detector | 81.39 | 49.18 | 96.80 | 72.64 | 13 |
| Previous real-only control | 82.68 | 48.31 | 92.27 | 79.48 | 36 |
| Previous 768-render mixed | 84.70 | 48.40 | 86.65 | 78.00 | 65 |
| Previous 768-render sim-only | 71.62 | 20.33 | 44.11 | 45.66 | 313 |
| Revised 18-render mixed | 83.58 | 47.49 | 94.95 | 73.01 | 21 |
| Revised 18-render sim-only | 77.27 | 11.66 | 35.62 | 29.76 | 291 |

These results do not establish that physically better materials hurt transfer:
the small revised batch has much less unique scene diversity, changed class
balance and a different fitting runtime. Real teacher scope inconsistencies
also remain. The evidence does not justify replacing the current liquid detector
or claiming improved completed-vial color/QA coverage. Reviewed real body/stream
targets and more varied RGB-supported scenes remain the next data inputs.

Local output:
`C:/Users/Austin/ai-club-tools/results/revised-render-detector-20261019/`.
Start with `evaluation/mixed/run5-early-pours.mp4` and
`evaluation/sim-only/run5-early-pours.mp4`; both put the current detector left.
Full recordings, source prediction rows, parity reports and machine-readable
comparison remain alongside them. No browser/preview server is required.

Both checkpoint/ONNX parity checks passed. The two video arms agree on all
9,299 sampled source-frame identities, decoded RGB hashes, timestamps and
current-detector predictions. All four MP4s fully decode: 8,491 frames at 5 fps
for each full comparison and 968 frames at 30 fps for each early-pour comparison.
Start/middle/end frames and selected early examples were visually inspected.
`selected-review-cases.json` records frames 1410 and 2796 and their unchanged
predictions as diagnostic observations, without assigning supplied vial IDs.
Published execution evidence is `setup/verification/revised-render-training.json`.
