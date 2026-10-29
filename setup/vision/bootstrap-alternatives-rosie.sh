#!/usr/bin/env bash
# Keep the established detector environment and site image intact.
set -euo pipefail
image=/data/containers/msoe-pytorch-25.05-py3.sif
base="$HOME/ai-club-tools/envs/vision-rosie"
venv="$HOME/ai-club-tools/envs/vision-alternatives-rosie"
[[ -x "$base/bin/python" ]] || { echo 'Run bootstrap-rosie.sh first'; exit 2; }
[[ ! -e "$venv" ]] || { echo 'Environment exists; inspect/reuse it instead of overwriting'; exit 2; }
singularity exec --cleanenv "$image" python -m venv --system-site-packages "$venv"
printf '%s\n' "$base/lib/python3.12/site-packages" > "$venv/lib/python3.12/site-packages/team_vision.pth"
singularity exec --cleanenv --env PYTHONNOUSERSITE=1 "$image" "$venv/bin/python" -m pip install transformers==4.57.1
echo 'DINOv3 requires separately obtained, authorized weights; this script does not request access or accept terms.'
