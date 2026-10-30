# Liquid vision experiments

**Current architecture — October 24:** [direct temporal U-Net, SAM-free](CURRENT-ARCHITECTURE.md).
YOLO supplies vial geometry; paired U-Net directly supplies the liquid mask and
presence. SAM2.1 is optional experimental refinement, DINOv3 is a comparator,
and Faster R-CNN is not required by the selected liquid path. This records the
team integration direction; existing application defaults are unchanged.

**October24 SAM-free DINOv3 mask pilot:** [architecture, protocol and matched results](DINOV3-DIRECT-MASK.md).
Two heads trained only on144 synthetic images over a frozen generic ConvNeXt Small.
Paired DINO supplies180/264 real development QA samples versus U-Net105/264,
at29.43ms versus16.37ms median CV processing. RGB-only proposes liquid on all264
frames. Visual masks still include glass/stream; all77 paired QA gains coincide
with U-Net containment rejection, so this is not proven segmentation improvement.
No reviewed mask/onset accuracy; app defaults unchanged.

**October23 eight-fit fixture investigation:**
[five-class failures, normal-data improvement and cap-state candidate](DATASET1-FIVECLASS.md).
Normal-current coverage improves matched synthetic five-class exact accuracy
78.33%→93.82%, with body-defect misses still unresolved. A separately scoped
cap-state model reaches98.75% on a fresh720-target confirmation, preserving
performance under corrected brightness controls. Real accuracy remains unknown;
reference misalignment and authored lying-S/body-damage coverage remain blockers.
The three-page PDF and actual single-panel cap-proposal MP4 are under
`results/dataset1-followups-20261023/summary-v1`; no app/default/research-main change.
See [verification](../verification/dataset1-fiveclass-20261023.json).

**Dataset1 frozen RGB real-camera probe:** [protocol/media/review gate](DATASET1-REAL-TRANSFER.md).
480 unreviewed slot-frame predictions on16 Run1–4 stills;41 capped-looking/439 empty.
Real accuracy is unavailable until source-matched human review. Proposed geometric
registration and the96-priority/480-total offline review packet are prepared; no
qualified real golden exists for the paired arm. App defaults remain unchanged.

**Dataset 1 first synthetic baseline completed:** [fixture pilot](DATASET1-PILOT.md).
GB10 rendered 60 paired scene families and trained RGB/reference CNNs from random
initialization. Each scored360/360 targets and12/12 whole scenes on the same held-out
synthetic test, covering capped/uncapped/empty only. No paired-model advantage or
real-transfer accuracy established. Local summary includes an actual 12-scene
fixture demo, confusion matrix and unlinked experimental observations; no app change.

**Matched SAM/temporal comparison:** [same-frame results and refinement](SAM-TEMPORAL-COMPARISON.md).
Direct temporal is ~6× faster; temporal-guided SAM recovers80 additional
QA-eligible samples (105→185/264) without R-CNN. Availability is not accuracy;
human review and reference qualification remain pending. App defaults unchanged.

**GB10 follow-up:** [synthetic temporal masks, optical controls and duration fits](SYNTHETIC-ITERATIONS.md).
Temporal candidates visually localize growing bottom pools in six Run 4 clips;
real mask/onset accuracy awaits [human review](../../annotation_review/VISIBLE-POOL-REVIEW.md).
All new GPU work runs on GB10; no candidate is promoted.

**October 22 update:** [synthetic-only Dataset 2 training and transfer results](SYNTHETIC-TRANSFER-AUDIT.md#october-22-follow-up--synthetic-only-competition-direction).
Two new synthetic-trained Faster R-CNN pilots and three detector/SAM/QA paths
completed locally. High held-out synthetic AP did not establish real transfer;
generic RGB augmentation did not improve real box agreement. Current assignment
and synthetic-only training rules are in [the team handoff](../../team_handoff/README.md).
Legacy real-trained weights below remain development comparators; no defaults
were promoted. The older dated sections are preserved as experiment history.

## Historical state — October 21, 2026

This section records the October 21 twin study and camera-correction replay;
the October 24 architecture decision above governs new integration work.
Continue research on `austin-work`. The data handoff and separately authorized
frontend/API starter are published on `main`; research remains on `austin-work`.

| Area | Current evidence / next work |
| --- | --- |
| Student data | [Downloads and exact YOLO label paths](../../team_handoff/README.md); Aleyna, Zoe and Reagan are working on vial detection and review |
| Offline pipeline | YOLO11n vial detection → Faster R-CNN ResNet50-FPN liquid localization → SAM2.1 Hiera Tiny → sampling QA and image-derived state |
| Live digital twin | [Isaac camera pilot](LIVE-TWIN.md): four complete sequences / 148 verified frames; provisional warning on the different-residue case; nominal repeat and scripted-flush case did not warn. One trajectory per condition; operational recipe decisions unavailable. |
| Latest twin study | [21-case experiments](TWIN-EXPERIMENTS.md): 803 verified live frames; adaptive corrected-RGB replay gives 0/10 nominal and 5/6 residue warnings, with all 16 cases assessable. Simulated development only; half-mL-equivalent case missed. |
| Appearance-transfer candidate | [Completed Cosmos Transfer trial](COSMOS-TRANSFER-TRIAL.md): 93 generated frames verified, but empty-frame liquid/hue hallucinations block inherited labels. Source/generated QA availability 24/29 versus 17/29. No training. |
| Acceleration | [Full comparison](PIPELINE-ACCELERATION.md): decode + sampled processing p50 89.48 → 83.22 ms, 6.99% reduction; 6,324 measured samples across four passes |
| Backend limitation | One changed sample-QA decision / 1,581 frames; worst paired nonempty final-mask IoU 0.70455; TensorRT remains opt-in |
| Human evaluation | [Early-fill review](../../annotation_review/EARLY-FILL-HANDOFF.md): six raw Run 4 clips, 72 PNGs, no approved onset/mask truth incorporated |
| Recipe-aware actions | [Stage 2 refinement](STAGE2-REFINEMENT.md): offline/simulated advisory experiments; real quality remains unavailable without commissioned references and aligned process evidence |
| NVIDIA reasoning | [Pilot](NVIDIA-PILOT.md): Cosmos3-Edge failed structured-output suitability; absent from the current decision pipeline |
| Real-video correction | [Original Run 4 paired replay](REAL-CAMERA-PREPROCESSING.md): 849 QA passes per arm; no aggregate gain; 12 differences retained; raw baseline preserved |
| Physical validation | [CSI visit packet](CSI-SESSION-PACKET.md): four-page PDF, eight forms and rehearsed native-clock inspector; October 26–28 proposed, unsent/unbooked |

The current single-view 60-second video and six-early-fill montage are under
`C:/Users/Austin/ai-club-tools/results/pipeline-acceleration-20261020/published-media-v1/`.
See [the full handoff](../../HANDOFF.md) for artifact paths and priorities.
Reviewed onset delay, mask accuracy and real defect precision/recall remain
unestablished. The deployed detector, app defaults and measurement QA are preserved.

## Completed detector experiments and original bootstrap procedure

The [consolidated handoff](../../HANDOFF.md) records the current
[liquid-specialist / non-YOLO comparison](LIQUID-SPECIALISTS.md): all four fits,
evaluations and assembly are complete, with 12 decoded local videos. DINO improves
provisional recall but produces more unmatched boxes. The next
[early-liquid architecture pilots](EARLY-LIQUID.md) have also completed: overall
Run 4 AP improved, while early/small-pool evidence remains limited. A separate
[SAM2/SAM3 mask-pipeline probe](SAM-LIQUID.md) tests prompt-conditioned fluid
outlines and exposes the need for a separate presence decision. No candidate has
replaced the app detector.

The [controlled epoch and augmentation trial](EPOCH-AUGMENTATION.md) compares a
fresh 12-epoch control with longer training and training-only contrast/blur/noise.
It preserves provisional labels, excludes Run 5 from fitting/selection, and does
not promote a detector or change measurement QA. All three Rosie jobs completed.
Longer plain training improves provisional liquid recall; added contrast/blur/noise
worsens agreement and adds whole-vial liquid proposals. Local videos and limits
are documented there.

The [real + synthetic bbox pilot](REAL-SYNTHETIC.md) includes a verified
768-image GB10 batch, completed real-only/mixed/sim-only Rosie runs, Run 5
results and local comparison videos. Run 5 stays outside fitting and checkpoint
selection. Current detector and measurement QA remain unchanged.

**Human-label milestone: [box detection and tracking](BOXES.md).** Establish
vial/liquid localization on reviewed targets alongside the experimental mask
pipeline. The existing two-class detector remains the deployed baseline;
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

The supervised milestone is corrected vial/liquid boxes and video tracking
as described in [BOXES.md](BOXES.md). The crop-segmentation trial above is complete
and was not promoted; the newer SAM pipeline is a separate development pilot. Split by
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
