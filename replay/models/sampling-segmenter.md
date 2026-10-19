# Experimental liquid sampling-interior segmenter

**Not promoted:** the original adapter passed 7/30 Run 5 windows. A post-hoc
[adapter correction](../../setup/vision/SEGMENTATION-DEBUG.md) recovers fill 30
for **8/30**, versus the baseline's **13/30**: two gained, seven lost. Model
weights and all QA thresholds stay fixed. Existing vial detections, tracks,
fill events and measurement windows are identical. Coverage is not mask accuracy.

This 118,273-parameter binary U-Net consumes RGB `[1,3,384,128]` vial crops and
returns `[1,1,384,128]` logits. Foreground threshold is logit > 0 (sigmoid > 0.5),
fixed before Run 5. Runtime uses CPU OpenCV ONNX. Crop preprocessing is shared
with preparation: 10% horizontal/3% vertical padding, RGB / 255, linear resizing.

The 104 training rows are 52 unique source frames/crops from Runs 1–3 plus
translated copies. Targets are provisional sampling interiors, not complete
liquid outlines. No verified empty/clear negatives are included. Development
uses 78 Run 4 teacher masks. Run 5 is excluded from fitting/checkpoint selection;
the earlier Jev teacher had already seen examples from all five runs.

Rosie Slurm job 285668 used an H100 through Singularity and completed in 19s.
40 epochs were trained from scratch with seed 19; best epoch 22 was selected
on Run 4 provisional mask IoU 0.65796. The model metadata and ONNX hashes are
in `sampling-segmenter.json`. All 78 development binary masks match PyTorch
exactly after ONNX export. No QA tuning, fallback or mask unions were applied.

The corrected adapter supplies one vial-conditioned binary semantic proposal;
its disconnected fragments are not separate instances. It still passes the
original containment, wall/base exclusions, HSV, morphology,
minimum-area, motion, temporal and completed-window checks. Tiny base proposals
and sparse/fragmented foreground account for the remaining lost windows. The already-inset target and
the retained border exclusions can further shrink small regions. More complete
target definitions and reviewed masks/negatives are needed before another study.

All masks remain unreviewed and trusted color assessments stay unavailable.
The object detector `vial-bootstrap.onnx` remains unchanged and supplies cropping
and tracking. Use the optional `segmentation-vision.json` for this experiment;
`bootstrap-vision.json` remains the baseline. Reproduction and exact data splits:
[SEGMENTATION.md](../../setup/vision/SEGMENTATION.md).
