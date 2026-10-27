#!/usr/bin/env bash
#SBATCH --job-name=vial-gpu-check
#SBATCH --partition=dgxh100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:05:00
set -euo pipefail
source setup/env.sh
bash setup/rosie/run.sh setup/doctor.py --profile cosmos --gpu \
  --out "$AI_CLUB_HOME/results/$TEAM_MEMBER/rosie-gpu-$SLURM_JOB_ID.json"
