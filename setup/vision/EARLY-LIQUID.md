# First-visible liquid: architecture and resolution pilots

Austin's target is to detect liquid as soon as it exists visibly at the bottom of
the active vial. We are iterating on models while preserving the original labels
and app. A visible pool and a falling stream are separate eventual annotation
targets; the existing `colored_liquid` class can combine them.

## Frozen experiment

Local root: `C:/Users/Austin/ai-club-tools/results/early-liquid-20261020`.
Remote root: `$HOME/ai-club-tools/results/early-liquid-20261020`.
Slurm array **286291** completed three independent H100 fits through Singularity 3.10.
Original four-arm experiments and frozen fitting helpers are unchanged.

| Arm | What changes | Initialization / fitting |
| --- | --- | --- |
| `yolo11n-960` | Increase the existing nano model's input to 960 | Existing COCO YOLO11n; batch 16 |
| `yolo11s-640` | Larger YOLO11 architecture at 640 | Official COCO YOLO11s; batch 16 |
| `fasterrcnn-resnet50` | Standard two-stage detector with trainable ResNet stages | Existing COCO Faster R-CNN ResNet50-FPN; new background/liquid predictor; batch 8 |

YOLO arms retain the existing all-images specialist's training settings, source
inventory and seed. The earlier `yolo-all` 640 fit provides the reference.
Faster R-CNN trains ResNet layers 2–4, FPN, RPN and ROI heads; the stem and layer 1
stay frozen. It uses FrozenBatchNorm2d, full original 480×640 geometry, AdamW
1e-4, cosine-50 schedule, weight decay 1e-4, FP32, gradient clipping 10, and the
existing translated images without extra online augmentation. Its standard
five-level COCO anchors and ImageNet normalization remain unchanged. This provides
a trainable CNN alternative to the frozen DINOv3 custom detector. These are family
pilots with different initialization/optimization, not isolated architecture effects.

Sources: [official YOLO11 variants](https://docs.ultralytics.com/models/yolo11/),
[TorchVision Faster R-CNN](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html).
YOLO11s was downloaded from the official Ultralytics v8.4.0 asset release; SHA256
`85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5`.
Other initializations reuse the already pinned local assets.

## What is measured

Runs 1–3 train, with 1,442 samples (708 positive / 734 provisional negatives).
Run 4 has the identical 157 development originals (89 positive / 68 provisional
negatives). All arms use seed 19, max 50 epochs, patience 10, and select **overall
Run 4 liquid AP50–95**. Run 5 remains excluded from fitting, selection, threshold
tuning and development sampling. The frozen plan and copied executed helpers are
hashed in `preparation.json`; the complete 295,575,192-byte input bundle was
SHA256/gzip checked and all frozen inputs validated on Rosie before submission.

The common development audit also reports detection of the **18 supplied Run 4
targets with height <=64 pixels** at the unchanged .35 cutoff. It filters and
rematches predictions at each cutoff in a predefined diagnostic grid; it does not
select a new threshold or change the app. Existing models match 15/18 (deployed),
16/18 (YOLO all), 15/18 (focused), 17/18 (RT-DETR), 17/18 (DINO).

This is a geometry proxy, **not physical onset or reviewed first-visible-liquid
accuracy**. Run 4 has no target <=32 pixels high; training has only four such
targets, including translated variants. More resolution/capacity may help, but
the very first shallow pool lacks adequate supervision/evaluation. Human review
under [LABEL-CONTRACT.md](LABEL-CONTRACT.md) stays separate and source files are
never automatically corrected. Existing detector-start clips cannot measure
detection delay without reviewed visible-pool onset frames.

## Run and retrieve

`train_early_liquid.py prepare` takes `--source` (the completed liquid-specialist
root), `--assets` (folder containing official `yolo11s.pt`) and a fresh `--out`.
It copies the immutable all-images inventory and pins every weight/helper/plan.
Transfer the complete prepared root and validate it before submitting:

```bash
sbatch --output="$experiment/slurm-%A_%a.log" "$experiment/train-early-liquid-rosie.sbatch" "$experiment"
```

The ResNet arm performs an empty-target/two-step/checkpoint smoke before fitting.
The same local GPU smoke passed finite losses, unchanged frozen stem and exact
save/reload predictions. `collect-early-liquid-rosie.sh EXPERIMENT ARRAY_JOB_ID`
waits for three successful exits and packages the models, logs, curves and frozen
helpers. It publishes the archive atomically and writes a checksum, never
resubmits, and stops on failure or its 135-minute limit. Preserve partial runs.
The collector runs detached in the remote root; its log is `collection.log`.
All three fits ultimately completed, exit 0:0, and the remote ResNet smoke passed.
The brief VPN interruption did not interrupt training. See the completed results
below; do not repeat these fits or their completed development evaluations.
The [launch verification record](../verification/early-liquid-20261020.json)
includes frozen hashes and the local adapter smoke, which verified actual 960
inference dimensions and the unchanged vial branch. Pilot completion is not
claimed by that launch record.

Use the existing `vision-alternatives-windows` Python for local development audit:

```powershell
python setup/vision/evaluate_early_liquid.py --experiment C:/Users/Austin/ai-club-tools/results/early-liquid-20261020 --source-experiment C:/Users/Austin/ai-club-tools/results/liquid-specialists-20261020 --previous-audit C:/Users/Austin/ai-club-tools/results/liquid-specialists-20261020/development-run4 --baseline C:/Users/Austin/ai-club-tools/results/crop-clahe-detector-20261019/baseline-best.pt --out C:/Users/Austin/ai-club-tools/results/early-liquid-20261020/development-run4
```

Run after retrieving and hash-verifying models. `--arms` supports completed subsets
with fresh output directories. The evaluator uses each arm's trained input size,
compares identical source/target/baseline predictions, includes earlier development
results, and exports short-target disagreement PNGs. It never reads Run 5 images.
No detector promotion, measurement-QA change or industrial connection is made.

## Completed results and source verification

All results use the same 157 Run 4 images / 89 provisional liquid targets through
the common diagnostic adapter. Precision/recall are rematched at .35; confidence
is not calibrated across model families. Native training-validator selection AP
is recorded separately in each model's metadata.

| Model | Common liquid AP50–95 | Precision .35 | Recall .35 | Short targets matched |
| --- | ---: | ---: | ---: | ---: |
| Deployed | 50.14% | 85.19% | 77.53% | 15/18 |
| Earlier YOLO11n 640 specialist | 51.51% | 82.80% | 86.52% | 16/18 |
| Earlier focused specialist | 48.84% | 91.25% | 82.02% | 15/18 |
| RT-DETR-L | 49.28% | 73.27% | 83.15% | 17/18 |
| DINOv3 custom detector | 51.48% | 68.91% | 92.13% | 17/18 |
| YOLO11n 960 | 53.81% | 81.32% | 83.15% | 15/18 |
| YOLO11s 640 | 53.66% | 88.24% | 84.27% | 15/18 |
| Faster R-CNN ResNet50 | 53.50% | 79.80% | 88.76% | 17/18 |

The three new fits completed 39/37/17 epochs and selected 29/27/7. Higher
resolution and the larger YOLO improve overall provisional AP here but do not
improve the small supplied-target count at the fixed cutoff. ResNet matches 17/18
with fewer unmatched boxes overall than the DINO arm at .35. This small group
and provisional semantics do not establish architectural superiority or physical
first-visible-pool performance. No candidate is promoted.

`development-run4/` holds `comparison.json`, `model-scorecard.png`, per-model
predictions and 12 short-target comparison PNGs. `collect_early_liquid.py`
assembles previously completed subset evaluations without repeating GPU inference;
it verifies source/target identity, checkpoints and metrics against saved predictions.
The [completed evidence](../verification/early-liquid-results-20261020.json) pins
the final report, checkpoints and complete native training curves.

The original management-node collector saw metadata before it became visible,
then NUL-filled reads of an already completed CSV. Its original logs remain
`collection-first-attempt.log` / `collection-retry.log`; the reader never sanitized
the files. A CPU-only Slurm read on the original node, job **286305**, verified
every raw checkpoint/curve and selected epoch, then packaged the original bytes
using `package_early_artifacts.py`. It used no GPU. The resulting verified archive,
`completed-artifacts-node.tar.gz`, has SHA256
`31b86fe156e70d839daa90389030c0f836046682a2ae8d56dfed5dd7ca489a94`.
Its `collection-read.json` records per-file hashes and node identity. The one
incomplete local CSV retrieval remains under `retrieval-first-attempt/` with its
original hash/NUL count in `retrieval-verification.json`; the clean node snapshot
supplies the final local curve. The now-unneeded polling collector was stopped.
No fitting job was repeated and original remote logs/curves remain untouched.

Austin's proposed next pipeline component, [SAM2/SAM3 fluid masks](SAM-LIQUID.md),
now has a separate executed SAM2.1 prompt probe on Run 4. Those raw masks are
neither reviewed labels nor changes to the replay's measurement gates.
