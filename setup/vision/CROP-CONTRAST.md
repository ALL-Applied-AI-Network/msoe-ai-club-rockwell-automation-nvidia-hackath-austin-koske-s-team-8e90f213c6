# Cropped + contrast-enhanced box detector experiment

**Result: keep the current detector.** The candidate slightly increases liquid
recall at the fixed display threshold, but its box agreement is worse overall.
This is one frozen 12-epoch trial, not a search using Run 5.

| Run 5 provisional-box metric | Current detector | Crop + CLAHE candidate |
| --- | ---: | ---: |
| Vial AP50-95 | 0.8139 | 0.7780 |
| Liquid AP50-95 | 0.4918 | 0.4648 |
| Overall mAP50-95 | 0.6528 | 0.6214 |
| Liquid precision at confidence 0.35 / IoU 0.5 | 0.9680 | 0.9498 |
| Liquid recall at confidence 0.35 / IoU 0.5 | 0.7264 | 0.7338 |

These are agreement with 816 existing **unreviewed teacher-labeled images**,
not independent human accuracy. Twenty-nine unknown/manual cases are excluded.
The teacher had previously seen all recordings. Run 5 remains excluded from
student fitting and checkpoint selection, but has already been inspected as a
diagnostic recording. Both arms are rescored with the same backend and matcher;
these scores supersede neither reviewed ground truth nor other reports using
different NMS/backend settings.

## Frozen protocol

- Source: the exact original detector dataset, provenance SHA256
  `a7756db7360e4cbf142edc38d7584810e290fcb032401add5bd1bea8e35d3b9e`.
- Fit: Runs 1-3, 721 original frames plus their existing translations (1,442
  images). Development: 157 Run 4 images. Per-source/parent checks reject Run 5,
  its exact source duplicates, cross-split descendants and cross-split identical
  processed RGB. Unknown targets are not relabeled or reviewed by this work.
- Fixed source ROI: `[227,22,414,454]`, exclusive upper bounds, derived from
  fitting-box bounds plus 24px horizontal and 16px vertical context. No Run 4/5
  geometry chooses the ROI. Every fitting/development box fits completely;
  preparation does not clip or drop targets.
- Crop first, then CLAHE on OpenCV Lab L only: clip limit 2, 8x8 tiles. Normalize
  YOLO box labels into crop coordinates. The same shared transform runs at
  inference; all 1,599 processed training/development RGB arrays were checked
  byte-for-byte against the local inference implementation.
- YOLO11n detection, COCO initialization reused from the original experiment,
  12 epochs, seed 19, 640 input, batch 16. Run 4 selects the checkpoint; epoch
  12 had the best development fitness (0.67144). No Run 5 checkpoint selection.
- Same exposed augmentation settings as the reproducible baseline trainer;
  no hue/saturation shifts, flips or mosaic. Cropping changes effective pixel
  scale and augmentation extent. The historic original used AMP; this trainer
  uses FP32 and avoids AMP-check downloads. The combined pipeline comparison
  does not isolate CLAHE, cropping or numerical training differences separately.
- Rosie job **285702**, H100 through Singularity 3.10.0 and the site PyTorch SIF,
  completed successfully in **3m49s**. No Docker or login-node GPU training.

## Local outputs

The experiment directory is
`$HOME/ai-club-tools/results/crop-clahe-detector-20261019` on Windows and Rosie.
The candidate model is deliberately outside the ordinary replay configuration.
It **requires crop + CLAHE + source-coordinate projection** and cannot be
substituted directly for a full-frame model in the existing application.

On Windows:

- `evaluation/run5-full-comparison.mp4`: current left, candidate right, all
  8,491 stride-6 frames at 5 fps; original MP4 duration about 28m18s.
- `evaluation/run5-early-pours.mp4`: eight chronological non-partial
  image-estimated starts, from -0.5 to +3.5 seconds, at native 30 fps (32.27s
  total). These estimates choose clips, not reviewed physical onset labels.
- Both videos overlay raw boxes on the **original source image**, using the
  same confidence 0.35 and NMS IoU 0.45. They do not color-enhance display pixels,
  force box continuity, hide unassociated detections or infer physical vial IDs.
- `evaluation/video-predictions.jsonl`: predictions and exact source RGB hashes
  for 9,299 unique frames used across both videos. All 8,491 control-cache input
  hashes match. The current detector reproduces its prior full-video presence
  counts; presence counts alone are not accuracy.
- `evaluation/teacher-agreement.json` and `scoring-predictions.json`: per-class
  AP50/AP50-95, fixed-threshold precision/recall, original scoring identities and
  both arms' predictions. Ground-truth boxes stay in full source coordinates;
  one Run 5 target lies outside the candidate ROI and is still scored, not
  silently removed or clipped. AP uses confidence 0.001 and the same 0.45 NMS.
- `model/best.pt`, `model/detector.onnx`, `model/model.json`, frozen plan,
  preprocessing/data provenance, actual training log, args and epoch CSV.

The ONNX export matches PyTorch CPU on 16 development frames (25 boxes), with
the same classes/counts and maximum score error about 1.79e-7. Integer OpenCV
box coordinates differ by at most about one crop pixel. GPU/CPU convolution
produces small numerical score differences; both video/scoring arms use the
same PyTorch GPU backend. No inference thresholds were changed for the result.
The original model and the replay's measurement/reference QA remain unchanged.

## Reproduction

From the repository, use a **new** experiment directory:

```powershell
$python = "$HOME/ai-club-tools/envs/data-windows/Scripts/python.exe"
$vision = "$HOME/ai-club-tools/envs/vision-windows/Scripts/python.exe"
$trial = "$HOME/ai-club-tools/results/crop-clahe-detector-20261019"
& $python setup/vision/train_crop_contrast.py prepare --source "$HOME/ai-club-tools/results/vial-bootstrap-20261019/data" --teacher "$HOME/ai-club-tools/results/typesafe-box-all-20261018" --out $trial
```

Copy `data/`, `experiment-plan.json`, `preprocessing.json`, `crop_contrast.py`,
`train_crop_contrast.py` and `train-crop-contrast-rosie.sbatch` into the new
experiment folder on Rosie. The existing isolated vision environment and
original `vial-bootstrap-20261019/yolo11n.pt` initialization are reused:

```bash
cd ~/ai-club-tools/results/crop-clahe-detector-20261019
sbatch --output=training-%j.log train-crop-contrast-rosie.sbatch "$PWD"
```

After successful completion, copy `model/model.json`, `model/detector.onnx`,
and `model/train/weights/best.pt` to the corresponding local `model/` folder
(the last file becomes local `model/best.pt`). Copy the original detector's
`model/train/weights/best.pt` to local `baseline-best.pt`. Then:

```powershell
& $vision setup/vision/evaluate_crop_contrast.py --experiment $trial --baseline "$trial/baseline-best.pt" --teacher "$HOME/ai-club-tools/results/typesafe-box-all-20261018" --cache "$HOME/ai-club-tools/results/tracked-liquid-20261019/cache/run5-8bdd8cad2977ee74e6a3.json" --cache-sha256 aff504002184d856ac06451f1730677eace2e840b89110c2cb6c0af39a8850d8 --out "$trial/evaluation" --device 0
```

This comparison needs the isolated PyTorch vision environment; the unchanged
replay still needs no PyTorch. No browser preview, camera or industrial network
connection is required. Full evidence is indexed in the
[verification record](../../replay/verification/crop-contrast-detector-run5.json).
