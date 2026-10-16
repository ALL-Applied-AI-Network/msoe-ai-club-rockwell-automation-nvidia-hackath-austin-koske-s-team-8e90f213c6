#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/../env.sh"
gpu_flags=(--nv)
venv="$AI_CLUB_HOME/envs/cosmos-rosie"
if [[ "${1:-}" == --cpu ]]; then gpu_flags=(); shift; fi
if [[ "${1:-}" == --data ]]; then gpu_flags=(); venv="$AI_CLUB_HOME/envs/data-rosie"; shift; fi
if [[ ${#gpu_flags[@]} -gt 0 && -z "${SLURM_JOB_ID:-}" ]]; then
  echo 'GPU work needs a Slurm allocation. Use setup/rosie/submit.sh or srun.' >&2; exit 2
fi
[[ -x "$venv/bin/python" ]] || { echo 'Run bash setup/bootstrap.sh rosie first.' >&2; exit 2; }
export SINGULARITYENV_COSMOS_MODEL="$COSMOS_MODEL"
if [[ -n "${COSMOS_REVISION:-}" ]]; then export SINGULARITYENV_COSMOS_REVISION="$COSMOS_REVISION"; fi
if [[ -n "${HF_TOKEN:-}" ]]; then export SINGULARITYENV_HF_TOKEN="$HF_TOKEN"; fi
if [[ -n "${NVIDIA_API_KEY:-}" ]]; then export SINGULARITYENV_NVIDIA_API_KEY="$NVIDIA_API_KEY"; fi
exec singularity exec --cleanenv "${gpu_flags[@]}" \
  --bind "$AI_CLUB_HOME:$AI_CLUB_HOME" --bind "$REPO_ROOT:$REPO_ROOT" \
  --env "HF_HOME=$HF_HOME" --env 'PYTHONPATH=' --env 'PYTHONNOUSERSITE=1' --env 'MPLBACKEND=Agg' \
  "$ROSIE_PYTORCH_SIF" "$venv/bin/python" "$@"
