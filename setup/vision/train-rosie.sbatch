#!/usr/bin/env bash
#SBATCH --job-name=vial-bootstrap
#SBATCH --partition=dgxh100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
set -euo pipefail
[[ $# == 1 ]] || { echo 'Pass an experiment directory containing data/ and bootstrap_detector.py'; exit 2; }
experiment=$(realpath "$1")
venv="$HOME/ai-club-tools/envs/vision-rosie"
image="${ROSIE_PYTORCH_SIF:-/data/containers/msoe-pytorch-25.05-py3.sif}"
export OMP_NUM_THREADS=4
cd "$experiment"
singularity exec --cleanenv --nv --bind "$HOME:$HOME" \
  --env 'PYTHONNOUSERSITE=1' --env 'MPLBACKEND=Agg' --env 'OMP_NUM_THREADS=4' \
  "$image" "$venv/bin/python" bootstrap_detector.py train \
  --source "$experiment/data" --out "$experiment/model" --epochs 12 --device 0
