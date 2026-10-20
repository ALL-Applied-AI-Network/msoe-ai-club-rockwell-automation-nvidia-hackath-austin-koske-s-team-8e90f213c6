# Liquid specialists and non-YOLO comparison

Four diagnostic fits test whether concentrating supervision on liquid helps.
The app's detector, tracker, measurement QA and original annotation files remain
unchanged. All targets and negative absences are provisional. Austin explicitly
authorized retaining existing vial-only labels as provisional negatives.

## Frozen design

| Arm | Model | Training sampling |
| --- | --- | --- |
| `yolo-all` | COCO-initialized YOLO11n, one liquid class | Same image inventory as prior real-only fits |
| `yolo-focused` | Same YOLO11n | Positive-heavy, matched 1,442 samples per epoch |
| `rtdetr-all` | COCO-initialized RT-DETR-L, one liquid class | Same as `yolo-all` |
| `dinov3-all` | Frozen DINOv3 ConvNeXt Small + trainable FPN/Faster R-CNN | Same as `yolo-all` |

Original class 1 coordinates are copied into derived class 0 labels; vial targets
are removed only in the derived datasets. Original labels are never edited or
approved. An empty derived file means **no supplied liquid target**, not confirmed
empty glass. Pale/clear liquid missing from the teacher targets can therefore
become a wrong negative; this experiment does not resolve that annotation issue.

The same-images arm has 708 liquid-positive and 734 provisional-negative training
samples. Its 721 originals each have one existing translated variant. The focused
arm retains every one of the 354 positive originals, repeats 222 positive units,
and keeps 145 of 367 negative originals, sampled using seed 19. Including paired
translations, this produces 1,152 positive and 290 negative samples, with 499
unique source images. Both arms have the same number of optimizer batches per
epoch. This changes the sampling distribution, not the underlying labels.

All training uses Runs 1–3. The identical Run 4 development set contains 89
positive and 68 provisional-negative images. Maximum 50 epochs, patience 10,
seed 19; **Run 4 liquid AP50–95** selects checkpoints. Run 5 is excluded from
fitting and selection, including exact source hashes and transformed descendants.
No reviewed mentee annotations or synthetic images enter these fits.

The Ultralytics arms use its pinned 8.4.182 runtime, FP32, image size 640, batch
16 and the earlier plain-training geometry/brightness settings. Additional
contrast/blur/noise is disabled. Exact args and CSV curves accompany checkpoints.
The DINOv3 pilot uses batch 8, AdamW 1e-4, weight decay 1e-4, a 50-epoch cosine
schedule, FP32, gradient clipping 10, and the existing translations without
additional online augmentation. These are family pilots with different native
preprocessing and initialization, not an isolated architecture-only experiment.

## What the DINOv3 model is

This is a custom detector built from cached official DINOv3 **ConvNeXt Small**
weights, revision `421020015aa4266363db803b6049a529999afb1e`. Its four spatial
feature stages feed a trainable 256-channel feature pyramid and standard
TorchVision Faster R-CNN. The RPN and ROI box head initialize from TorchVision's
COCO ResNet50-FPN checkpoint; the two-output background/liquid predictor is new.
The DINOv3 backbone stays frozen and in evaluation mode. A smoke test exercises
empty targets, two training batches, validation and checkpoint save/reload.

It is distinct from Meta's published COCO detector using the 7B ViT backbone.
It is also distinct from DINO/DETR, whose similar name refers to a detector
architecture. Sources: [Meta DINOv3](https://github.com/facebookresearch/dinov3),
[Transformers DINOv3 implementation](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/dinov3),
[TorchVision detection API](https://docs.pytorch.org/vision/stable/models/faster_rcnn.html),
[RT-DETR](https://docs.ultralytics.com/models/rtdetr/).

## Reproduction and environments

Use the existing vision setup for YOLO and RT-DETR. For DINOv3 on Rosie,
`bootstrap-alternatives-rosie.sh` creates an isolated environment, reuses the
established vision packages through a `.pth` entry, and installs Transformers
4.57.1. It leaves the existing environment/site image intact. The Windows
equivalent is `C:/Users/Austin/ai-club-tools/envs/vision-alternatives-windows/`;
its `Lib/site-packages/team_vision.pth` points to the existing vision environment's
`Lib/site-packages`, and Transformers 4.57.1 is installed in the new environment.

All pretrained asset hashes are pinned in `liquid_specialists_data.py` and the
frozen plan. DINOv3 uses already cached authorized weights; this setup does not
request access, accept model terms or substitute an unofficial mirror. Other
initializations are official Ultralytics and TorchVision checkpoints.

From the repository, using the data Python:

```powershell
python setup/vision/train_liquid_specialists.py prepare --source C:/Users/Austin/ai-club-tools/results/epoch-augmentation-20261020/data --assets C:/Users/Austin/ai-club-tools/results/liquid-specialists-assets-20261020 --out C:/Users/Austin/ai-club-tools/results/NEW-LIQUID-EXPERIMENT
```

Copy the training helpers into the prepared root: `train_liquid_specialists.py`,
`liquid_specialists_data.py`, `dino_liquid.py`, `train-liquid-specialists-rosie.sbatch`,
and the reused `train_epoch_augmentation.py`, `train_mixed_detector.py`,
`detector_photometric.py`. Make the batch script LF-terminated and record SHA-256
hashes of those exact helper files plus `experiment-plan.json` in
`preparation.json`, using the procedure in [EPOCH-AUGMENTATION.md](EPOCH-AUGMENTATION.md).
Freeze **after** copying/normalizing bytes. Move the complete prepared directory
to Rosie. Verify the entire transfer before unpacking/submitting.

```bash
sbatch --output="$experiment/slurm-%A_%a.log" "$experiment/train-liquid-specialists-rosie.sbatch" "$experiment"
```

The array requests four independent H100 jobs through Slurm, Singularity 3.10,
the existing `/data/containers/msoe-pytorch-25.05-py3.sif`, 48 GB host memory and
at most two hours per arm. GPU fitting never runs on the login node. The DINOv3
arm runs its GPU smoke test first. `collect-liquid-specialists-rosie.sh` can run
detached with experiment path and array ID to package four successful fits when
complete. It preserves the source data and never resubmits jobs.

For this run, array `286263` was canceled while pending after an incomplete
transfer was detected; no fits from it were used. Array `286264` was submitted
after the full archive passed gzip integrity checks and every frozen helper/plan
hash was verified.

## Local evaluation

Root: `C:/Users/Austin/ai-club-tools/results/liquid-specialists-20261020/`.
Retrieve each arm's `best.pt`, `model.json` and training evidence into `models/`.
The DINOv3 checkpoint stores the trainable detector components and requires the
exact pinned frozen backbone from `pretrained/dinov3-convnext-small/`.

```powershell
python setup/vision/evaluate_liquid_specialists.py --experiment C:/Users/Austin/ai-club-tools/results/liquid-specialists-20261020 --baseline C:/Users/Austin/ai-club-tools/results/crop-clahe-detector-20261019/baseline-best.pt --teacher C:/Users/Austin/ai-club-tools/results/typesafe-box-all-20261018 --data datasets/dataset2/csi-vial-fill-dataset --cache C:/Users/Austin/ai-club-tools/results/tracked-liquid-20261019/cache/run5-8bdd8cad2977ee74e6a3.json --cache-sha256 aff504002184d856ac06451f1730677eace2e840b89110c2cb6c0af39a8850d8 --out C:/Users/Austin/ai-club-tools/results/liquid-specialists-20261020/NEW-EVALUATION --device cuda:0
```

Use `--arms yolo-all yolo-focused` or another subset with separate fresh outputs
to evaluate completed fits while other jobs continue. Each arm exports whole-run
5-fps comparison video, eight early neighborhoods at native 30 fps, two complete
fill windows, per-frame predictions/source hashes and metrics. The deployed
detector is left; the liquid specialist is right. **Blue vial boxes are copied
from the same fixed baseline detector on both sides**. They are not predictions
from the liquid-only models and cannot demonstrate improved vial tracking.

YOLO uses ordinary letterboxing, RT-DETR uses its native square resize, and the
DINOv3 detector uses ImageNet normalization and native 480×640 geometry. There
is no inference contrast enhancement or fixed liquid ROI. Source coordinates
remain unchanged after reversing each model's native processing.

All displayed liquid predictions use confidence 0.35 and final NMS IoU 0.45.
RT-DETR is normally NMS-free; this diagnostic explicitly applies the common final
NMS. DINOv3/Faster R-CNN retains its standard internal proposal filtering.
Confidence scores are **not calibrated across families**. Interpret liquid
AP50–95 alongside precision/recall; do not rank by box presence alone. Combined
class AP includes the copied vial branch and is not the primary model metric.

`provisional-negative-diagnostics.json` counts alarms on scored frames with a
vial target but no liquid target. These are unreviewed absences and cannot
establish the false-alarm rate on genuinely empty vials. Full reviewed empty
periods, pale-liquid boxes and early-flow timing remain inputs from Aleyna's group.
No measurement masks, color comparisons, thresholds, release decisions, renderer
or operator-workflow assignments are changed by these experiments.
