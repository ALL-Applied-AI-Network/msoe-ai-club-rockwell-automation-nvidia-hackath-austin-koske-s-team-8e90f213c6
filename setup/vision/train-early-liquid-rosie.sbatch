#!/usr/bin/env bash
#SBATCH --job-name=early-liquid
#SBATCH --partition=dgxh100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --array=0-2
set -euo pipefail
[[ $# == 1 ]] || { echo 'Pass the frozen experiment root'; exit 2; }
experiment=$(realpath "$1")
arms=(yolo11n-960 yolo11s-640 fasterrcnn-resnet50)
arm=${arms[$SLURM_ARRAY_TASK_ID]}
cd "$experiment"
runner=(singularity exec --cleanenv --nv --bind "$HOME:$HOME"
  --env PYTHONNOUSERSITE=1 --env MPLBACKEND=Agg --env OMP_NUM_THREADS=4
  --env CUBLAS_WORKSPACE_CONFIG=:4096:8
  /data/containers/msoe-pytorch-25.05-py3.sif "$HOME/ai-club-tools/envs/vision-rosie/bin/python")
if [[ "$arm" == fasterrcnn-resnet50 ]]; then
  "${runner[@]}" train_early_liquid.py smoke-resnet --out "$experiment"
fi
"${runner[@]}" train_early_liquid.py train --out "$experiment" --arm "$arm"
