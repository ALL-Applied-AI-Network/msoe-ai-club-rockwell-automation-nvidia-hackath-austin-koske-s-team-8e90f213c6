# Experimental vial/liquid box model

`vial-bootstrap.onnx` is the frozen YOLO11n two-class detector from Rosie Slurm
job **285667**, October 19, 2026. The successful job used one H100 through
Singularity and completed in 3m40s. `vial-bootstrap.json` records the model hash,
runtime versions and source-provenance hash. Input is RGB float32 `[1,3,640,640]`
with centered 114 padding; output is `[1,6,8400]`. Classes are 0=vial and
1=visibly colored liquid. The app reverses padding/scaling and applies classwise
NMS. These are target-position proposals, not an exhaustive inventory of vials.

Austin authorized the existing **unreviewed Jev boxes** as provisional training
targets. Runs 1–3 contribute 721 source images plus 721 translated copies.
157 Run 4 images select the checkpoint. Run 5 images/file hashes are excluded
from student fitting and checkpoint selection. The teacher itself was previously
developed using examples from all five runs; this is not a fully blind benchmark.
Manual/empty/API-failure cases were excluded rather than assumed negative.

Training used COCO-pretrained YOLO11n, 12 epochs, 640 px, batch 16, seed 19,
deterministic mode, 0 workers, AMP enabled, no mosaic/flips/hue/saturation changes,
small translations/scale/rotation/value augmentation. The executed original script
and complete args are retained in Austin's external experiment directory. Current
training scripts disable AMP's auxiliary download and use an absolute pretrained
path to avoid the account-wide Ultralytics weights setting.

Run 4 teacher-box agreement at the final epoch: precision 0.91049, recall
0.91573, mAP50 0.92004, mAP50–95 0.69717. These are **agreement with provisional
teacher boxes**, not human-verified localization accuracy, liquid-mask IoU,
color correctness or defect performance. See
[training evidence](../verification/bootstrap-training.json) and
[epoch results](../verification/bootstrap-training-results.csv).

The model is deliberately a **box detector**, not a segmentation model.
`replay/geometry.py` derives the conservative sampling mask and rejects uncertain
regions. All current masks remain unreviewed. Their colors/deviations are
diagnostic; the app keeps recipe assessments unavailable pending reviewed geometry.

This small ONNX artifact is committed so teammates can reproduce the app with
ordinary CPU OpenCV. Training data, candidate masks, PyTorch checkpoints and logs
stay outside Git. Reproduction/candidate generation:
[setup/vision](../../setup/vision/README.md). Deployment and mask-review contract:
[VISION.md](../VISION.md).
