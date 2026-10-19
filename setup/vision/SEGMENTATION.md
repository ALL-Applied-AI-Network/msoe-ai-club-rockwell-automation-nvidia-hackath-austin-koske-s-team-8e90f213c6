# Provisional sampling-interior segmentation experiment

**Result: keep the 13/30 baseline.** The frozen candidate passed 7/30 Run 5
windows with unchanged QA. It gained detected fills 8 and 9, but lost 4, 5, 6,
12, 13, 18, 29 and 30. All fill events, vial proposals/tracks and window frames
were identical. The model remains an opt-in diagnostic experiment.

Three lost windows produced tiny base/edge proposals that failed the interior
gate; five produced multiple disconnected components and failed the single
liquid-region gate. These are observed failure modes, not adjudicated liquid
labels. The model target is already an inset sampling interior; retaining all
proposal-border exclusions can shrink small predictions further. Full liquid
outlines, sampling interiors and visible reflections must be distinguished in
the next reviewed target definition. No thresholds/checkpoints were changed
after observing this Run 5 result.

The existing YOLO11n detector already supplies vial boxes. Keep it frozen for
cropping/tracking while evaluating the new liquid mask. Training another vial
detector would change a second variable; a future vial-only retrain should use
corrected boxes and difficult absent/moving/occluded cases.

This model predicts a **conservative liquid sampling interior**, not the full
liquid outline or vial silhouette. The provisional targets have holes and
exclude glass/base/glare; binary PNG supervision preserves those holes. A tiny
binary U-Net on vial crops uses 118,273 parameters and exports to CPU OpenCV ONNX.
Training needs PyTorch only on Rosie; ordinary app use needs neither PyTorch
nor a GPU.

## Data and training

From the repository, using the existing data interpreter:

```powershell
$python = "$HOME/ai-club-tools/envs/data-windows/Scripts/python.exe"
$experiment = "$HOME/ai-club-tools/results/sampling-segmentation-20261019"
& $python setup/vision/prepare_sampling_segmentation.py --candidates "$HOME/ai-club-tools/results/vial-bootstrap-20261019/candidates" --teacher "$HOME/ai-club-tools/results/typesafe-box-all-20261018" --config replay/examples/bootstrap-vision.json --out "$experiment/data"
```

Use a new output directory. The training pack contains 104 source/translated
examples from **52 unique source frames and 52 unique crops**, all Runs 1–3.
Translation of the image and vial box together yields the same crop, so those
duplicates do not constitute new scenes. Training adds small matched image/mask
translations/scales and image brightness changes; no hue changes or rendering.

Development has 78 Run 4 stills with masks proposed by the frozen baseline.
Still-image motion/timing flags are retained as provenance; they are not mask
truth. Unknown/empty targets are excluded, not assumed negative. No Run 5 image,
mask or interval is used for fitting or checkpoint selection. The earlier Jev
teacher had already been developed using all five recordings, so these targets
are not an independent human benchmark.

Copy `data/`, `tiny_unet.py`, `train_sampling_segmentation.py` and
`train-segmentation-rosie.sbatch` to an experiment folder on Rosie. Reuse the
isolated environment provisioned by `bash setup/vision/bootstrap-rosie.sh`:

```bash
cd ~/ai-club-tools/results/sampling-segmentation-20261019
sbatch --output=training-%j.log train-segmentation-rosie.sbatch "$PWD"
```

This uses one Slurm H100 through Singularity and the existing site SIF. No Docker
or login-node GPU learning. The fixed protocol is 40 epochs, seed 19, batch 16,
AdamW, weighted BCE + Dice loss and a 128×384 RGB crop. The checkpoint is selected
only by Run 4 provisional mask IoU. The sigmoid threshold stays **0.5**, fixed
before Run 5. Crop padding is 10% horizontally and 3% vertically; preparation and
inference share the exact helper.

Executed Slurm job **285668** completed in 19 seconds, including 9.0 seconds
inside the trainer. Best epoch: **22**; Run 4 teacher-mask IoU: **0.65796**.
This measures agreement with provisional masks. All 78 exported ONNX binary
masks matched PyTorch exactly; maximum logit error was about 1.91e-5.
The original executed training script and checkpoint remain in the external
experiment folder; later split validation also checks every provenance row.

## Controlled Run 5 reproduction

Reuse the original image-only detector cache, which is pinned by SHA256. The
cache adapter consumes only detector proposals from the correct model,
recording/frame and exact source RGB hash. Cached timing, tracks, masks, colors,
recipes and annotation fields are ignored. Each replay recomputes tracking,
mask/color QA and completed windows through the same `MonitorSession` path.
Missing/wrong source pixels stop processing; no fallback is allowed.

```powershell
$sourceCache = "$HOME/ai-club-tools/results/tracked-liquid-20261019/cache/run5-8bdd8cad2977ee74e6a3.json"
& $python setup/vision/compare_sampling_segmentation.py --baseline-cache $sourceCache --recipe-plan replay/examples/run5-demo-plan.json --out $experiment
```

If that original cache is unavailable, first generate one with
`python -m replay analyze --runs 5 --config replay/examples/bootstrap-vision.json
--out <new-source-directory>` and use its `cache/run5-*.json`. Supply the normal
`--data` override to either command for data stored elsewhere.

`comparison.json` reports each detected fill's baseline/candidate QA result,
reasons and color, plus gained/lost windows. It asserts exact baseline
reproduction, identical vial detections/tracks, events and window frames, and
the frozen QA configuration. Both model outputs have full v2 `run5/qa-review/`
manifests with actual source PNGs, overlays and exact sampling masks. Recipes
come only from the explicit illustrative plan; they are never inferred.

The registered protocol is
[`segmentation-experiment-plan.json`](../../replay/verification/segmentation-experiment-plan.json).
The candidate replaces the liquid box proposal with connected components of the
learned binary mask. It then intersects with the **unchanged** vial containment,
wall/base/holder exclusions, HSV gate, opening/erosion, 60-pixel floor and 0.35
coherent-area floor. Stable tracks, motion/holder QA, clipping, erosion
sensitivity, three samples, temporal stability and measurement windows retain
their original values. No union/fallback to a baseline liquid mask.

## Run the candidate app

```powershell
pwsh -File setup/run-replay.ps1 -Config replay/examples/segmentation-vision.json -RecipePlan replay/examples/run5-demo-plan.json -Out "$HOME/ai-club-tools/results/segmentation-replay" -Port 8766
```

This portable config performs fresh inference for both models. For the audited
accelerated replay, use the generated `$experiment/segmentation-reuse.json` and
`$experiment/segmentation` output instead. That cache is explicitly tied to
Run 5; selecting another recording rejects it. The portable config supports
ordinary recordings without the optional detector cache.

Existing `bootstrap-vision.json` remains the baseline. Model/config/code hashes
make references from different mask profiles incompatible. All learned masks
remain unreviewed, and trusted recipe comparisons stay unavailable. Increased
window coverage alone cannot establish segmentation precision or color validity.

The next data work is reviewed sampling masks or full outlines with their target
type clearly identified, plus empty/clear/reflective hard negatives. Keep all
descendants of a source recording in its split. Aleyna/Zoe/Reagan retain label
and window review; Cameron/Hayden retain physical rendering; Srithej/Josiah
retain operator workflow and CSI coordination.
