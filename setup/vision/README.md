# Bootstrap vial/liquid localization

The [real + synthetic bbox pilot](REAL-SYNTHETIC.md) has a frozen 768-render
fixture, a size-matched real-only control, and local video evaluation commands.
Its execution status is recorded in that guide; preparation is not a training
result. Run 5 stays outside fitting and checkpoint selection.

**Current next milestone: [box detection and tracking](BOXES.md).** Establish
vial/liquid localization on reviewed targets before further segmentation or
color decisions. The existing two-class detector is the starting baseline;
the color-QA coverage counts below do not score localization accuracy.
The [crop + CLAHE retraining trial](CROP-CONTRAST.md) is complete, with local
Run 5 comparison videos. Keep the original detector: the candidate has slightly
higher provisional liquid recall but worse box precision/overlap overall.

The follow-up [crop segmentation experiment](SEGMENTATION.md) completed on Rosie.
It initially passed 7/30 Run 5 windows; the [post-hoc adapter correction](SEGMENTATION-DEBUG.md)
recovers fill 30 for 8/30 versus this model's 13/30 baseline under unchanged
QA, so the baseline remains recommended. The existing vial detector was held
fixed for cropping/tracking; it does not need another retrain for this comparison.

Austin authorized treating the existing Jev draft boxes as provisional training
targets on October 19. This is an application-side bootstrap experiment. It does
not approve labels, adjudicate timing/windows, or replace Cameron/Hayden's renderer.

The supplied recipe/timing tables have no object shapes. The separate Jev export
does contain two-class boxes (`fill_position_vial`, `colored_liquid`). We train
YOLO11n **detection**, not segmentation, on those boxes. The app then derives a
conservative colored-liquid sampling mask inside the proposals. Rectangles must
not be relabeled as reviewed segmentation masks.

## Prepare an immutable experiment

Use the team data Python; no PyTorch needed for preparation:

```powershell
$python = "$HOME/ai-club-tools/envs/data-windows/Scripts/python.exe"
$source = "$HOME/ai-club-tools/results/typesafe-box-all-20261018"
$experiment = "$HOME/ai-club-tools/results/vial-bootstrap-20261019"
& $python setup/vision/bootstrap_detector.py prepare --source $source --out "$experiment/data"
```

Preparation requires a new output directory. It checks source file hashes,
records label hashes, excludes empty/manual/API-failure cases, deduplicates source
files, and excludes Run 5 file hashes everywhere. Runs 1–3 train, Run 4 develops;
Run 5 is excluded from both. Each eligible training image gets one exact box
translation if it stays inside the frame. No held-out augmentation is performed.
`provenance.json` records every source/parent/split. Partial/missing teacher boxes
can still be wrong; these are deliberately weak labels.

## Train on Rosie without Docker

Copy `data/`, `bootstrap_detector.py`, and `train-rosie.sbatch` to a new experiment
directory on Rosie. Rewrite the first `path:` entry in `data/data.yaml` to its
absolute Rosie `data/` path. From the repository on Rosie:

```bash
bash setup/vision/bootstrap-rosie.sh
cd ~/ai-club-tools/results/vial-bootstrap-20261019
sbatch --output=training-%j.log train-rosie.sbatch "$PWD"
```

Slurm allocates one H100, 8 CPU cores, 32 GB and at most 30 minutes. Singularity
uses `/data/containers/msoe-pytorch-25.05-py3.sif`. A separate `vision-rosie`
environment uses its PyTorch; the headless OpenCV overlay avoids the missing
`libX11.so.6` in the site image. GPU learning never runs on a login node.

The script trains 12 epochs with a fixed seed, selects `best.pt` using Run 4,
and exports `model/detector.onnx`, `model/model.json`, training args/results and
plots. Current scripts download pretrained weights into the experiment itself
and disable automatic AMP-check downloads. The executed first experiment used
Ultralytics' AMP check; its original script/args are retained with the evidence.
Package versions and input/model hashes are recorded. Exact numerical retraining
across different hardware/library builds is not guaranteed.

The replay uses OpenCV DNN on CPU with the pinned ONNX model; it needs no PyTorch
or GPU. The explicit config enables the model. Missing/mismatched model files
raise an error; omitting a model makes color unavailable. No fixed-patch fallback.

## Harvest candidate labels and synthetic augmentations

After the model is frozen:

```powershell
& $python setup/vision/harvest_candidates.py --data datasets/dataset2/csi-vial-fill-dataset --config replay/examples/bootstrap-vision.json --out "$experiment/candidates"
```

This reads only Runs 1–3 and samples at most 20 mask-check-passing frames per run,
with at least 90-frame spacing. It exports the source and a translated version,
two-class YOLO box labels, **heuristic liquid sampling masks**, and provenance.
These are synthetic image augmentations, not new physical fluid simulations or
reviewed masks. They remain outside evaluation and are not automatically fed
back into training. Review/correct them before another training round.

The next supervised milestone is corrected vial/liquid boxes and video tracking
as described in [BOXES.md](BOXES.md). Learned segmentation is deferred. Split by
recording, keep source and descendants together, and evaluate localization
separately from mask/color decisions. Clear/neutral liquid needs its own evidence;
the current `colored_liquid` class cannot establish its presence.

For a separate frozen-model agreement check against the existing Run 5 teacher
boxes, install Ultralytics 8.4.182/ONNX Runtime in an isolated vision environment:

```powershell
& "$HOME/ai-club-tools/envs/vision-windows/Scripts/python.exe" setup/vision/evaluate_boxes.py --teacher $source --weights replay/models/vial-bootstrap.onnx --out "$experiment/run5-box-agreement"
```

This never trains, excludes unknown/manual/empty cases, and records every scoring
target and hash. Its box mAP measures agreement with **unreviewed teacher boxes**,
not human truth, mask accuracy or absent-vial false positives. The Jev teacher
was previously developed on examples from all five recordings; only student
fitting/checkpoint selection excludes Run 5. Main replay verification does not
depend on this optional evaluation environment.

Official task interfaces: [detection](https://docs.ultralytics.com/tasks/detect/),
[segmentation](https://docs.ultralytics.com/tasks/segment/).
