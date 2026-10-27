#!/usr/bin/env bash
#SBATCH --job-name=vial-python
#SBATCH --partition=dgxh100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
set -euo pipefail
source setup/env.sh
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
[[ $# -gt 0 ]] || { echo 'Pass a Python script and arguments.' >&2; exit 2; }
bash setup/rosie/run.sh "$@"
