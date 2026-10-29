#!/usr/bin/env bash
#SBATCH --job-name=vial-sim-only
#SBATCH --partition=dgxh100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
set -euo pipefail
[[ $# == 1 ]] || { echo 'Pass experiment directory'; exit 2; }
experiment=$(realpath "$1")
image="${ROSIE_PYTORCH_SIF:-/data/containers/msoe-pytorch-25.05-py3.sif}"
venv="$HOME/ai-club-tools/envs/vision-rosie"
pretrained="$HOME/ai-club-tools/results/vial-bootstrap-20261019/yolo11n.pt"
cd "$experiment"
singularity exec --cleanenv --nv --bind "$HOME:$HOME" \
  --env 'PYTHONNOUSERSITE=1' --env 'MPLBACKEND=Agg' --env 'OMP_NUM_THREADS=4' \
  "$image" "$venv/bin/python" train_sim_only.py train --out "$experiment" --pretrained "$pretrained"
