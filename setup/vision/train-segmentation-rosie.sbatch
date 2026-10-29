#!/usr/bin/env bash
#SBATCH --job-name=liquid-segment
#SBATCH --partition=dgxh100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
set -euo pipefail
[[ $# == 1 ]] || { echo 'Pass the experiment directory'; exit 2; }
experiment=$(realpath "$1")
cd "$experiment"
singularity exec --cleanenv --nv --bind "$HOME:$HOME" \
  --env 'PYTHONNOUSERSITE=1' --env 'OMP_NUM_THREADS=4' --env 'MPLBACKEND=Agg' \
  "${ROSIE_PYTORCH_SIF:-/data/containers/msoe-pytorch-25.05-py3.sif}" \
  "$HOME/ai-club-tools/envs/vision-rosie/bin/python" train_sampling_segmentation.py \
  --data "$experiment/data" --out "$experiment/model" --epochs 40
