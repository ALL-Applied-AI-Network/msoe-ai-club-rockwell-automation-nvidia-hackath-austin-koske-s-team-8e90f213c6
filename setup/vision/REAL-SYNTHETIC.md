# Real + synthetic bbox pilot

This compares the current detector with a full-frame YOLO11n trained on real
Runs 1-3 plus a seeded GB10 synthetic batch. Run 4 alone selects mixed/control
checkpoints; the supplemental sim-only arm uses its fixed final epoch.
Run 5 is excluded from fitting, augmentation ancestry and checkpoint selection,
then scored after the candidates are frozen. The replay model and measurement
QA are not replaced or modified.

## Frozen design

* Synthetic fixture: **768 images**, seed 19, no real images/labels read by the
  generator: 96 empty-vial cases, 96 stream-only cases and 576 fills. The first
  12 are the reviewed cases. Other cases vary proxy color, opacity, fill height,
  roughness, illumination and stream radius. Supplied FillCam and CAD stay fixed.
* Real fitting: **721 unique Runs 1-3 images plus their 721 existing translated
  copies**, from the immutable prior bootstrap artifact. These remain
  unreviewed teacher targets; this does not redo the mentees' labeling work.
* Mixed fitting: **1,442 real + 768 synthetic = 2,210 images**.
* Matched real-only control: the same 1,442 real images plus 768 deterministic
  repeats. This matches training size, epochs and optimizer steps without
  adding synthetic data or claiming more unique real images.
* Development: the same **157 real Run 4 frames** in both arms. No synthetic
  development. Same pinned COCO YOLO11n initialization, seed 19, 12 epochs,
  FP32, augmentation and hyperparameters. Best is Run 4 box mAP50-95 only.
* Inference: original full-frame RGB and identical YOLO letterboxing, confidence
  0.35, NMS IoU 0.45. No CLAHE or station crop in either arm.

The raw synthetic export keeps all neighbor-vial and stream labels. The
**training copy** uses only object **101 (fill-position vial)** as class 0 and
object **201 (accumulated liquid body)** as class 1, matching the existing real
label scope. Neighbor vials and streams are excluded as training targets in
this two-class trial; their source labels are never erased. A bbox does not
certify that every enclosed pixel is liquid: holders/labels can occlude it.

## Executed results, October 19

**Keep the current detector for now.** Mixed training improved vial localization,
while liquid AP fell slightly and liquid false positives rose from 13 to 65.
Sim-only fine-tuning transferred vial detection, but liquid localization was
poor on the real footage. No detector promotion or measurement-QA change.

All scores below are percent-scale AP50-95 / precision / recall against the
same **816 provisional Run 5 teacher-labeled frames**. The 29 unknown frames
remain excluded. Precision/recall and false positives use confidence 0.35 and
IoU 0.5 matching; NMS IoU stays 0.45. AP uses confidence 0.001. These are frame
box metrics, separate from event detection, per-fill QA coverage and color.

| Model | mAP50-95 | Vial AP | Liquid AP | Liquid precision | Liquid recall | Liquid FP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| current | 65.28 | 81.39 | 49.18 | 96.80 | 72.64 | 13 |
| real-only | 65.50 | 82.68 | 48.31 | 92.27 | 79.48 | 36 |
| mixed | 66.55 | 84.70 | 48.40 | 86.65 | 78.00 | 65 |
| sim-only | 45.97 | 71.62 | 20.33 | 44.11 | 45.66 | 313 |

The teacher previously saw all five recordings and Run 5 has been inspected
repeatedly. These results are diagnostic agreement, not a new blind test or
independent human accuracy. Review labels and simulated liquid appearance
before relying on further scale-up. The sim-only arm uses generic COCO real-image
pretraining; all CSI domain fine-tuning and checkpoint selection are synthetic.

The **768-image GB10 / Isaac 6.0.0.1** render completed in **2,689.24 seconds**
at effective 32 SPP / one subframe. Remote and local verification passed with
no unavailable required objects or synchronization retries: 2,304 vial,
576 liquid-body and 497 stream boxes. Source assets match the preceding GB10
pilot, and raw proxy eligibility/optical-calibration flags remain unchanged.

Rosie main jobs all completed `0:0`: real-only control **285766** (5:52), mixed
**285785** (6:40), sim-only **285786** (8:08). All fitted 12 epochs using
Singularity and the pinned COCO initialization. Control and mixed select by
Run 4 mAP50-95; sim-only exports its fixed epoch-12 `last.pt` and uses no real
CSI checkpoint selection. The model/provenance/plan hashes match locally.

All four model exports passed PyTorch/ONNX consistency on 16 Run 4 images after
checkpoint freeze. Both paired video runs decoded all 50,942 source frames;
all 9,299 inferred frame identities, source RGB hashes, timestamps and baseline
predictions match between the mixed and sim-only comparisons. Four output
movies were fully decoded and inspected at the start, middle, end and fixed
early cases. No inference crop, CLAHE, threshold change or cached scores.

Execution evidence: [`real-synthetic-execution.json`](../../replay/verification/real-synthetic-execution.json).
Illustrative frame 1410 / MP4 47.0s:
[`real-synthetic-run5-example.png`](../../replay/verification/real-synthetic-run5-example.png).

Local experiment:
`C:/Users/Austin/ai-club-tools/results/real-synthetic-detector-20261019`

* `evaluation/run5-early-pours.mp4`: current left, mixed right; 968 frames at
  native 30 fps, covering the same eight fixed image-estimated neighborhoods.
* `evaluation/run5-full-comparison.mp4`: current left, mixed right; the whole
  recording sampled at 5 fps (8,491 frames, source time preserved).
* `evaluation/sim-only/`: equivalent early/full videos with sim-only right.
* `evaluation/comparison.json`: full metrics, candidate/control metadata and hashes.
* `evaluation/{scoring-predictions.json,video-predictions.jsonl}` plus the
  corresponding `sim-only/` files: source-level diagnostic predictions.
* `evaluation/paired-source-check.json`, parity files and video decode checks:
  model/export and source consistency evidence.
* `control-preview/`: supplementary current-versus-real-only-control videos.
* `renders/contact-boxes.png`: native batch review; `dataset-audit.json`: counts
  and class-mixture limits. Model weights remain outside Git in `models/`.

Exact local reproduction after downloading these artifacts:

```powershell
& "$HOME/ai-club-tools/results/real-synthetic-detector-20261019/reproduce-run5.ps1"
```

This uses the published evaluation code and writes a fresh
`evaluation-reproduced` folder. To reproduce on another machine, use the full
CLI below with matching model/data artifacts and add `--include-sim-only`.

Rosie experiment: `~/ai-club-tools/results/real-synthetic-detector-20261019`.
GB10 batch: `~/ai-club-tools/results/austin/real-synthetic-768-20261019`.
Completed jobs need not be rerun to review the existing results.

## Prepare on Windows

From the repository with its existing data environment (adjust paths):

```powershell
$dataPython = "$HOME/ai-club-tools/envs/data-windows/Scripts/python.exe"
$trialRoot = "$HOME/ai-club-tools/results/real-synthetic-detector-NEW"
$fixturePath = "$HOME/ai-club-tools/results/real-synthetic-fixture-NEW.json"
& $dataPython setup/isaac/make_training_fixture.py --out $fixturePath --count 768 --seed 19
& $dataPython setup/vision/train_mixed_detector.py prepare-real --source "$HOME/ai-club-tools/results/vial-bootstrap-20261019/data" --teacher "$HOME/ai-club-tools/results/typesafe-box-all-20261018" --fixture $fixturePath --out $trialRoot
& $dataPython setup/vision/train_mixed_detector.py prepare-control --out $trialRoot
```

The real inventory hash is pinned to the published bootstrap artifact. The code
rejects Run 5 sources/duplicates, cross-recording augmentations, split duplicates
and synthetic development records. Copy `train_mixed_detector.py` and
`train-mixed-rosie.sbatch` into the experiment before submitting jobs.

## Render on GB10

Deploy current setup without overwriting the mentees' twin sources:

```powershell
pwsh -File setup/deploy.ps1 -Target Spark
scp $fixturePath msoe-spark:ai-club-tools/results/real-synthetic-fixture-NEW.json
```

On GB10, from `~/ai-club-onboarding`:

```bash
export OMNI_KIT_ACCEPT_EULA=YES
export TEAM_MEMBER=austin  # substitute your name
source setup/env.sh
out="$AI_CLUB_HOME/results/$TEAM_MEMBER/synthetic-NEW"
fixture="$AI_CLUB_HOME/results/real-synthetic-fixture-NEW.json"
nohup bash setup/isaac/render_training.sh "$out" "$fixture" 768 > "$AI_CLUB_HOME/logs/synthetic-NEW.log" 2>&1 < /dev/null &
```

The GPU lock prevents competing jobs. Use a fresh folder. Require both
`run.json` (`complete: true`) and `verification.json` (`passed: true`). SPP and
total SPP are explicitly set to 32 **at capture time**, because initialization
can restore defaults. Actual settings are recorded and checked. This reduces
render cost without changing annotation requirements or measurement QA.

Compact mode omits annotation-pass RGB copies but preserves normal RGB, masks,
native label planes and semantic maps. Every row is verified; sheets/full-size
panels are limited to 24 examples to avoid huge review images. Camera/asset
hashes must stay unchanged. Raw exports retain `training_eligible: false` as
uncalibrated proxies. This user-authorized experiment explicitly admits them
into a separate training copy without changing those raw flags.

Copy the completed folder locally, inspect its review sheet, then:

```powershell
& $dataPython setup/vision/train_mixed_detector.py prepare-mixed --out $trialRoot --renders C:/path/to/completed/GB10/batch --accept-provisional-render-proxies
```

Partial batches, mismatched fixtures, missing targets and changed masks are
rejected. Mixed labels derive from verified mask bounds, not model predictions.

## Train on Rosie with Singularity

Upload `experiment-plan.json`, `render-fixture.json`, `data/real`, `data/control`,
`data/mixed`, and the two source files above. The real-only control can start
while GB10 renders. From the experiment directory on Rosie:

```bash
sbatch --output=real-only-%j.log train-mixed-rosie.sbatch "$PWD" real-only
sbatch --output=mixed-%j.log train-mixed-rosie.sbatch "$PWD" mixed
```

These request H100/Slurm using the existing Singularity 3.10 site SIF and
`vision-rosie`; no Docker or login-node GPU work. The existing initialization
at `~/ai-club-tools/results/vial-bootstrap-20261019/yolo11n.pt` must match the
plan's hash. No implicit download. Inspect Slurm completion and logs.

Copy `models/real-only` and `models/mixed` back locally. Each includes `best.pt`,
`detector.onnx`, `model.json` and training outputs. Keep weights/data outside
Git. Provenance records source, plan, dataset, initialization and model hashes,
versions and exclusions. Do not modify the plan after either model starts.

## Evaluate and create local videos

Use the existing Windows vision environment. The baseline is the checkpoint
corresponding to the current published ONNX, not the crop/CLAHE candidate:

```powershell
$visionPython = "$HOME/ai-club-tools/envs/vision-windows/Scripts/python.exe"
& $visionPython setup/vision/evaluate_mixed_detector.py --experiment $trialRoot --baseline "$HOME/ai-club-tools/results/crop-clahe-detector-20261019/baseline-best.pt" --teacher "$HOME/ai-club-tools/results/typesafe-box-all-20261018" --data datasets/dataset2/csi-vial-fill-dataset --cache "$HOME/ai-club-tools/results/tracked-liquid-20261019/cache/run5-8bdd8cad2977ee74e6a3.json" --cache-sha256 aff504002184d856ac06451f1730677eace2e840b89110c2cb6c0af39a8850d8 --out "$trialRoot/evaluation" --device 0
```

This checks identity and CPU ONNX parity on 16 Run 4 frames, then scores frozen
candidates against the same provisional Run 5 teacher boxes. Both inference
arms use the same PyTorch backend and original coordinates. Matching and
thresholds stay unchanged. The real-only control is also scored; no Run 5 result
reselects a checkpoint. Outputs are ordinary local files, with no web server:

* `evaluation/run5-full-comparison.mp4`: current left, mixed right, full
  recording sampled at 5 fps with source timestamps.
* `evaluation/run5-early-pours.mp4`: the same eight fixed neighborhoods at
  native 30 fps. Starts are image estimates, not reviewed physical-flow onsets.
* `evaluation/comparison.json`, `teacher-agreement.json`, `control/`,
  `video-predictions.jsonl`: sources, hashes, predictions and limits.
* `evaluation/onnx-parity.json`: checkpoint/export consistency.

Teacher-box agreement is not independent human accuracy. The teacher previously
saw all five recordings; Run 5 has been inspected repeatedly. Unknown/empty
teacher records stay excluded from AP. Report AP/precision/recall separately
from detection presence, completed-fill measurement coverage, color accuracy
and defect decisions. Nothing here authorizes release or promotes a detector.

## Supplemental sim-only transfer arm

Austin requested this additional arm after the real-only control diagnostic.
It has a separate `sim-only-plan.json` referencing the unchanged base plan.
**Only generated CSI images/labels enter fine-tuning.** All 768 unique renders
are used, with seeded repeats to reach the same 2,210-image training budget,
same COCO YOLO11n initialization, seed 19, 12 epochs and augmentation.
The generic COCO initialization itself includes real-image pretraining; this
is a test of synthetic-only **domain fine-tuning**, not training from scratch.

**The fixed final epoch is selected in advance. No real Run 4 checkpoint
selection is used for this arm.** Ultralytics still performs a final validation;
its source is the synthetic training set. Those resubstitution scores are not
transfer or held-out accuracy, and its `best.pt` is ignored. The exported
`models/sim-only/checkpoint.pt` comes from `train/weights/last.pt`.

Freeze the supplemental plan before fitting, then prepare metadata after the
verified mixed copy is available:

```powershell
& $dataPython setup/vision/train_sim_only.py plan --out $trialRoot
& $dataPython setup/vision/train_sim_only.py prepare --out $trialRoot
```

Preparation reads the mixed inventory's metadata and selects synthetic rows
only. It does not read/copy real pixels or labels. Upload `sim-only-plan.json`,
`data/sim-only/provenance.json`, `train_sim_only.py` and
`train-sim-only-rosie.sbatch` to the existing Rosie experiment. The job copies
only the verified synthetic sources from `data/mixed`, rejects unlisted files
or real rows, and runs with Singularity:

```bash
sbatch --output=sim-only-%j.log train-sim-only-rosie.sbatch "$PWD"
```

Download `models/sim-only`, then add `--include-sim-only` to the evaluation
command above. Additional local outputs are
`evaluation/sim-only/run5-full-comparison.mp4`,
`evaluation/sim-only/run5-early-pours.mp4`, scores, export parity and prediction
records. Current detector is left and sim-only is right. All paired scoring
uses the same fixed Run 5 sources/settings. Run 4 parity images are consumed
only after the checkpoint is frozen to verify its export, without selection.

The fixed render scene always contains vials. Empty-vial cases are liquid
negatives, not absent-vial negatives, so inspect real vial changes in the full
videos. The repeated training inventories contain 1,100 liquid boxes in the
real-only control, 1,284 in mixed and 1,659 in sim-only. Matching optimizer steps
controls the training budget; class-positive proportions and checkpoint
selection protocols still differ. Report transfer results with these limits.
