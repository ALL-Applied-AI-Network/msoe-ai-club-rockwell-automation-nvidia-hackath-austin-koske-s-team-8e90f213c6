#!/usr/bin/env bash
#SBATCH --job-name=carryover-estimator
#SBATCH --partition=dgxh100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-2
# Three seeds of the carryover estimator on the synthetic renders (one array task per seed).
#   cd REPO && sbatch setup/vision/train-carryover-rosie.sbatch OUT_ROOT RENDER_DIR...  (submit from the repo root)
set -euo pipefail
[[ $# -ge 2 ]] || { echo 'Pass an output root and render batch folders'; exit 2; }
out_root=$(realpath -m "$1"); shift
image="${ROSIE_PYTORCH_SIF:-/data/containers/msoe-pytorch-25.05-py3.sif}"
venv="$HOME/ai-club-tools/envs/vision-rosie"
seed=$((29 + SLURM_ARRAY_TASK_ID))
repo="$SLURM_SUBMIT_DIR"   # Slurm runs a spooled copy of this script, so $0 is not in the repo
singularity exec --cleanenv --nv --bind "$HOME:$HOME" \
  --env 'PYTHONNOUSERSITE=1' --env 'MPLBACKEND=Agg' --env 'OMP_NUM_THREADS=4' \
  "$image" "$venv/bin/python" "$repo/setup/vision/train_carryover_estimator.py" "$@" \
  --out "$out_root/seed-$seed" --seed "$seed"
