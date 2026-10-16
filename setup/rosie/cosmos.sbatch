#!/usr/bin/env bash
#SBATCH --job-name=vial-cosmos
#SBATCH --partition=dgxh100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
set -euo pipefail
source setup/env.sh
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
bash setup/rosie/run.sh setup/cosmos/reason.py "$@"
