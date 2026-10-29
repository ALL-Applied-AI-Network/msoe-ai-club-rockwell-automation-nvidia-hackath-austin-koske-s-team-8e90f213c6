# Controlled epoch and augmentation comparison

This experiment tests the liquid-confidence misses seen in the Run 5 walkthrough.
It does not replace the app detector, modify measurement QA, or approve labels.

## October 20 result

Longer plain training recovers some liquid misses. The added photometric recipe
does not improve this trial and sometimes predicts a second, nearly whole-vial
box as liquid. Keep the app's deployed detector while reviewing the longer plain
candidate against corrected labels.

Run 5 **liquid-class agreement with provisional teacher boxes**, 816 scored
frames / 541 liquid targets, identical confidence 0.35 and NMS IoU 0.45:

| Detector | Liquid AP50–95 | Precision | Recall | Unmatched boxes at IoU 0.5 |
| --- | ---: | ---: | ---: | ---: |
| Deployed historical anchor | 49.18% | 96.80% | 72.64% | 13 |
| Fresh 12-epoch control | 47.85% | 95.85% | 72.64% | 17 |
| Longer plain | 48.85% | 96.40% | 79.11% | 16 |
| Longer + contrast/blur/noise | 43.77% | 78.08% | 73.75% | 112 |

These are not reviewed accuracy or measurement-QA coverage. An unmatched box can
reflect a wrong prediction or a flawed target. In the saved augmentation examples,
the extra whole-vial liquid proposals are visibly different from the liquid body;
no target adjudication or corrections were made. The historical anchor has a
different training runtime, so the fresh 12-epoch arm is the matched control for
the training-budget comparison. All fresh arms use the same H100/runtime/FP32.

Slurm array `286232` completed successfully: 12, 31 and 42 epochs respectively.
The selected Run 4 epochs were 12, 21 and 32 (combined AP50–95: 69.551%, 69.692%,
70.693%). Both longer arms stopped under the predefined patience of 10. The
additional transform executed on 60,564 training images and changed 40,600;
training/validation source images and target files passed their original hashes.

The longer plain candidate detects liquid in the four inspected diagnostic frames
3162, 3300, 3306 and 3312. The deployed model misses frames 3162 and 3306 at the
same threshold. This small, already inspected example does not establish general
improvement. Full-recording and early-pour videos are also supplied for review.

The next data input remains the mentees' reviewed **vial / liquid body / stream**
labels, particularly pale/low liquid and the first pour. This experiment retains
the existing two-class contract and does not resolve its semantic ambiguities.
Do not use its results to certify a color measurement or product release.

## Frozen experiment

| Arm | Maximum epochs | Early-stop patience | Additional training augmentation |
| --- | ---: | ---: | --- |
| `plain12` | 12 | 12 | None |
| `plain50` | 50 | 10 | None |
| `augmented50` | 50 | 10 | Contrast, blur, Gaussian noise |

All three start from the same COCO YOLO11n checkpoint, use seed 19, FP32,
640-pixel full-frame inputs and batch 16. They retain the same ordinary YOLO
geometric/brightness augmentation. Fifty epochs also changes the standard
learning-rate schedule horizon: this compares training budgets, not a fixed-rate
update-count experiment. A fresh 12-epoch arm controls for runtime/FP32 differences
from the historical deployed checkpoint.

Training uses the existing provisional Runs 1–3 inventory: 721 original images and
their 721 existing translated variants. Run 4 supplies 157 development images.
Run 5 is excluded from fitting and checkpoint selection. Checkpoint selection uses
Run 4 combined-class AP50–95, as implemented by the pinned Ultralytics trainer.
Human review drafts are excluded; no target boxes are corrected automatically.

Additional augmentation, in order, uses an independent random generator:

- Contrast: probability 0.5, factor 0.85–1.15 around intensity 127.5.
- Gaussian blur: probability 0.15, kernel 3, sigma 0.2–0.7.
- Gaussian pixel noise: probability 0.25, sigma 1–4 on the 8-bit scale.

It runs only on training images, before the ordinary training transforms.
Source files and box geometry remain unchanged. `augmentation-execution.json`
records actual operation counts and the first 12 changed image pairs. The
`augmentation-preview/` files are actual transformed training inputs, not examples
generated separately after fitting. Validation and inference use original images.

## Reproduction

Use the installed vision environment described in this directory's onboarding
instructions. The experiment requires Ultralytics 8.4.182. Rosie runs through
Singularity 3.10 and Slurm; do not run GPU training on its login node.

Prepare a **new**, nonexistent experiment directory with the frozen real inventory:

```powershell
python setup/vision/train_epoch_augmentation.py prepare --source C:/Users/Austin/ai-club-tools/results/real-synthetic-detector-20261019/data/real --out C:/Users/Austin/ai-club-tools/results/NEW-EXPERIMENT
```

On Rosie, prepare the corresponding directory using its local inventory, copy the
four training helpers (`train_epoch_augmentation.py`, `detector_photometric.py`,
`train_mixed_detector.py`, `train-epoch-augmentation-rosie.sbatch`) into it, and
preserve the frozen plan bytes when moving between operating systems. Record
the actual copied helper bytes and plan before submitting (set `EXPERIMENT` to
the absolute prepared directory):

```bash
export EXPERIMENT="$experiment"
python - <<'PY'
import hashlib, json, os
from pathlib import Path
root = Path(os.environ['EXPERIMENT'])
names = ['train_epoch_augmentation.py', 'detector_photometric.py',
         'train_mixed_detector.py', 'train-epoch-augmentation-rosie.sbatch',
         'experiment-plan.json']
hashes = {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}
(root/'preparation.json').write_text(json.dumps(hashes, indent=2)+'\n', encoding='utf-8')
PY
```

Use the prepared vision Python for that command if the login shell has no Python.
Then submit:

```bash
sbatch --output="$experiment/slurm-%A_%a.log" "$experiment/train-epoch-augmentation-rosie.sbatch" "$experiment"
```

Here `$experiment` is an absolute path to the prepared directory. The array runs
three independent arms with private image/label copies and caches. Each completed
arm exports `models/<arm>/best.pt`, `detector.onnx`, `model.json`, training CSV,
arguments and augmentation execution evidence. Jobs survive SSH/VPN disconnection.
Do not resubmit an existing experiment or overwrite its outputs.

For intermittent VPN access, start `collect-epoch-artifacts-rosie.sh` on Rosie
with the experiment path and numeric array job ID. It waits for three successful
Slurm exits, packages only models/logs/evidence into `completed-artifacts.tar.gz`,
and writes its SHA-256. It neither resubmits nor changes the GPU jobs. Run it with
`nohup` and redirected input/output to survive SSH disconnection. Its wait is
bounded to 45 minutes; a failed Slurm job produces no successful-results bundle.

Retrieve those artifacts into the corresponding local `models/<arm>/` folders.
Evaluate once all three are available, using a new output directory:

```powershell
python setup/vision/evaluate_epoch_augmentation.py --experiment C:/Users/Austin/ai-club-tools/results/epoch-augmentation-20261020 --baseline C:/Users/Austin/ai-club-tools/results/crop-clahe-detector-20261019/baseline-best.pt --teacher C:/Users/Austin/ai-club-tools/results/typesafe-box-all-20261018 --data datasets/dataset2/csi-vial-fill-dataset --cache C:/Users/Austin/ai-club-tools/results/tracked-liquid-20261019/cache/run5-8bdd8cad2977ee74e6a3.json --cache-sha256 aff504002184d856ac06451f1730677eace2e840b89110c2cb6c0af39a8850d8 --out C:/Users/Austin/ai-club-tools/results/epoch-augmentation-20261020/evaluation --device 0
```

The evaluator verifies model/data identities, compares each ONNX export with its
checkpoint on Run 4, and runs the historical baseline and candidates through the
same local PyTorch inference backend. Score floor is 0.001 for AP; displayed boxes
and precision/recall use confidence 0.35 and NMS IoU 0.45. Neither threshold is tuned.

`--arms plain50` (or another subset) permits evaluating an already completed arm
while the other jobs continue. Use a separate fresh output directory per call.
`collect_epoch_comparison.py` assembles three completed diagnostic directories
into the normal `evaluation/<arm>/` layout. It verifies shared scoring identities,
model ancestry, training runtime, actual augmentation evidence and source pixels,
fully decodes every delivered video, and creates `selected-frame-comparison.png`.
Its inputs are `--experiment`, `--data`, `--out`, `--plain12`, `--plain50` and
`--augmented50`; the last three point directly to each arm's diagnostic files.

## Local review outputs

Root: `C:/Users/Austin/ai-club-tools/results/epoch-augmentation-20261020/`.
Each arm's `evaluation/<arm>/` contains:

- `run5-full-comparison.mp4`: whole recording sampled at 5 fps, original timing.
- `run5-two-fills.mp4`: two complete fill windows at 5 fps, including the previously
  inspected pale-liquid neighborhood; source frames 1284–2364 and 2670–3762.
- `run5-early-pours.mp4`: eight detector-estimated start neighborhoods at native
  30 fps. These are not reviewed physical flow-onset labels.
- `video-predictions.jsonl`: source frame, MP4 time, RGB hash and both models' boxes.
- `teacher-agreement.json`: 816 Run 5 teacher-labeled frames, including the exact
  scoring inventory and separate vial/liquid metrics.
- `screenshot-neighborhood-probes.json`: diagnostic frames 3162/3300/3306/3312.
- `onnx-parity.json`, decoded video counts/hashes and per-frame scoring predictions.

Top-level `evaluation/comparison.json` collects all three arms. In every video,
the deployed detector is on the left and the candidate is on the right. Video
overlays use original source pixels; no display enhancement is fed to inference.
`evaluation/selected-frame-comparison.png` shows the four inspected frames across
all four models, with a clearly identified display crop. The separate root-level
`augmentation-failure-examples.png` compares unchanged provisional targets with
deployed, longer plain and augmented predictions. Its adjacent JSON records the
selected source IDs. `augmentation-contact-sheet.png` shows actual training pairs.

## Interpretation

Teacher agreement is not human-reviewed detection accuracy. Those provisional
labels can contain wrong boxes, omissions and ambiguous liquid/body boundaries.
The scoring subset excludes 29 unknown/empty records and cannot establish an
absent-vial false-positive rate. Run 5 has already been inspected repeatedly and
is a diagnostic holdout, not a blind final test.

Count of frames with a liquid box measures detection presence, not correct
localization, measurement coverage, successful tracking, or recipe comparison.
The existing measurement QA remains separate and unchanged. A candidate with
more boxes is not automatically better. Aleyna's group's reviewed labels are
still needed for independent evaluation and any subsequent supervised correction.
