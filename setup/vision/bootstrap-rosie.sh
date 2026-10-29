#!/usr/bin/env bash
# Separate writable environment; the site image and Cosmos environment stay intact.
set -euo pipefail
image="${ROSIE_PYTORCH_SIF:-/data/containers/msoe-pytorch-25.05-py3.sif}"
venv="$HOME/ai-club-tools/envs/vision-rosie"
singularity exec --cleanenv "$image" python -m venv --system-site-packages "$venv"
singularity exec --cleanenv --env 'PYTHONNOUSERSITE=1' "$image" "$venv/bin/python" -m pip install \
  ultralytics==8.4.182 onnx==1.17.0 onnxruntime==1.26.0
# The GUI OpenCV wheel needs libX11, absent in this site SIF. Both wheels own
# cv2, so remove the GUI wheel before installing the matching headless wheel.
singularity exec --cleanenv --env 'PYTHONNOUSERSITE=1' "$image" "$venv/bin/python" -m pip uninstall -y opencv-python
singularity exec --cleanenv --env 'PYTHONNOUSERSITE=1' "$image" "$venv/bin/python" -m pip install --no-deps opencv-python-headless==4.11.0.86
