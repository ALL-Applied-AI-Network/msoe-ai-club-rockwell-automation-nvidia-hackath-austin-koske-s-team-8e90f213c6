#!/usr/bin/env bash
#SBATCH --job-name=liquid-specialists
#SBATCH --partition=dgxh100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --array=0-3
set -euo pipefail
[[ $# == 1 ]] || { echo 'Pass the frozen experiment path'; exit 2; }
experiment=$(realpath "$1")
arms=(yolo-all yolo-focused rtdetr-all dinov3-all)
arm=${arms[$SLURM_ARRAY_TASK_ID]}
venv="$HOME/ai-club-tools/envs/vision-rosie"
[[ "$arm" != dinov3-all ]] || venv="$HOME/ai-club-tools/envs/vision-alternatives-rosie"
cd "$experiment"
if [[ "$arm" == dinov3-all ]]; then
  singularity exec --cleanenv --nv --bind "$HOME:$HOME" \
    --env PYTHONNOUSERSITE=1 --env MPLBACKEND=Agg --env OMP_NUM_THREADS=4 \
    --env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    /data/containers/msoe-pytorch-25.05-py3.sif "$venv/bin/python" \
    train_liquid_specialists.py smoke-dino --out "$experiment"
fi
singularity exec --cleanenv --nv --bind "$HOME:$HOME" \
  --env PYTHONNOUSERSITE=1 --env MPLBACKEND=Agg --env OMP_NUM_THREADS=4 \
  --env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /data/containers/msoe-pytorch-25.05-py3.sif "$venv/bin/python" \
  train_liquid_specialists.py train --out "$experiment" --arm "$arm"
