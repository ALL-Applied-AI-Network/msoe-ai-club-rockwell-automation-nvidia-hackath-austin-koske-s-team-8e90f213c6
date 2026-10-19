#!/usr/bin/env bash
#SBATCH --job-name=vial-real-synth
#SBATCH --partition=dgxh100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
set -euo pipefail
[[ $# == 2 ]] || { echo 'Pass experiment directory and real-only|mixed'; exit 2; }
experiment=$(realpath "$1")
arm="$2"
[[ "$arm" == real-only || "$arm" == mixed ]] || exit 2
image="${ROSIE_PYTORCH_SIF:-/data/containers/msoe-pytorch-25.05-py3.sif}"
venv="$HOME/ai-club-tools/envs/vision-rosie"
pretrained="$HOME/ai-club-tools/results/vial-bootstrap-20261019/yolo11n.pt"
export OMP_NUM_THREADS=4
cd "$experiment"
singularity exec --cleanenv --nv --bind "$HOME:$HOME" \
  --env 'PYTHONNOUSERSITE=1' --env 'MPLBACKEND=Agg' --env 'OMP_NUM_THREADS=4' \
  "$image" "$venv/bin/python" train_mixed_detector.py train --out "$experiment" --arm "$arm" --pretrained "$pretrained"
