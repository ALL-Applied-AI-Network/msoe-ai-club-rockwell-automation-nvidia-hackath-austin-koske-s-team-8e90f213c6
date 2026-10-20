#!/usr/bin/env bash
#SBATCH --job-name=vial-epochs
#SBATCH --partition=dgxh100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --array=0-2
set -euo pipefail
[[ $# == 1 ]] || { echo 'Pass the frozen experiment directory'; exit 2; }
experiment=$(realpath "$1")
arms=(plain12 plain50 augmented50)
arm=${arms[$SLURM_ARRAY_TASK_ID]}
image=/data/containers/msoe-pytorch-25.05-py3.sif
venv="$HOME/ai-club-tools/envs/vision-rosie"
pretrained="$HOME/ai-club-tools/results/vial-bootstrap-20261019/yolo11n.pt"
export OMP_NUM_THREADS=4
cd "$experiment"
singularity exec --cleanenv --nv --bind "$HOME:$HOME" \
  --env PYTHONNOUSERSITE=1 --env MPLBACKEND=Agg --env OMP_NUM_THREADS=4 \
  "$image" "$venv/bin/python" train_epoch_augmentation.py train \
  --out "$experiment" --arm "$arm" --pretrained "$pretrained"
