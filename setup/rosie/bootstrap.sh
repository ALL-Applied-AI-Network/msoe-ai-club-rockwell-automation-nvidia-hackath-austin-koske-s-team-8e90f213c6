#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/../env.sh"
command -v singularity >/dev/null || { echo 'Load the site Singularity 3.10 module first.' >&2; exit 2; }
[[ -r "$ROSIE_PYTORCH_SIF" ]] || { echo "Missing site image: $ROSIE_PYTORCH_SIF" >&2; exit 2; }
venv="$AI_CLUB_HOME/envs/cosmos-rosie"
if [[ ! -x "$venv/bin/python" ]]; then
  singularity exec --cleanenv "$ROSIE_PYTORCH_SIF" python -m venv --system-site-packages "$venv"
fi
# Preserve the site CUDA/PyTorch build. Never install Docker or modify the SIF.
cosmos_requirements="$SETUP_DIR/requirements/cosmos-linux-x86_64-rosie.lock.txt"
[[ -f "$cosmos_requirements" ]] || cosmos_requirements="$SETUP_DIR/requirements/cosmos.in"
singularity exec --cleanenv --bind "$AI_CLUB_HOME:$AI_CLUB_HOME" --bind "$REPO_ROOT:$REPO_ROOT" \
  "$ROSIE_PYTORCH_SIF" "$venv/bin/python" -m pip install --no-compile --disable-pip-version-check \
  -r "$cosmos_requirements"
# The site image constrains NumPy to 1.26.4. Keep the modern data stack in a
# separate, isolated environment rather than overriding the CUDA image's NumPy.
data_venv="$AI_CLUB_HOME/envs/data-rosie"
if [[ ! -x "$data_venv/bin/python" ]]; then
  singularity exec --cleanenv "$ROSIE_PYTORCH_SIF" python -m venv "$data_venv"
fi
data_requirements="$SETUP_DIR/requirements/data-linux-x86_64-rosie.lock.txt"
[[ -f "$data_requirements" ]] || data_requirements="$SETUP_DIR/requirements/data.in"
singularity exec --cleanenv --bind "$AI_CLUB_HOME:$AI_CLUB_HOME" --bind "$REPO_ROOT:$REPO_ROOT" \
  "$ROSIE_PYTORCH_SIF" env -u PIP_CONSTRAINT -u PYTHONPATH "$data_venv/bin/python" \
  -m pip install --no-compile --disable-pip-version-check -r "$data_requirements"
singularity exec --cleanenv --bind "$AI_CLUB_HOME:$AI_CLUB_HOME" --bind "$REPO_ROOT:$REPO_ROOT" \
  "$ROSIE_PYTORCH_SIF" "$venv/bin/python" "$SETUP_DIR/doctor.py" --profile cosmos
singularity exec --cleanenv --bind "$AI_CLUB_HOME:$AI_CLUB_HOME" --bind "$REPO_ROOT:$REPO_ROOT" \
  "$ROSIE_PYTORCH_SIF" env -u PYTHONPATH "$data_venv/bin/python" "$SETUP_DIR/doctor.py" --profile data
