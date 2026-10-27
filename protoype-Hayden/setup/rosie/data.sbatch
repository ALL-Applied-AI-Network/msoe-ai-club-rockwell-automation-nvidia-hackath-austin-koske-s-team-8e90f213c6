#!/usr/bin/env bash
#SBATCH --job-name=vial-data
#SBATCH --partition=batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
set -euo pipefail
source setup/env.sh
if [[ $# -eq 0 ]]; then set -- setup/doctor.py --profile data; fi
bash setup/rosie/run.sh --data "$@"
