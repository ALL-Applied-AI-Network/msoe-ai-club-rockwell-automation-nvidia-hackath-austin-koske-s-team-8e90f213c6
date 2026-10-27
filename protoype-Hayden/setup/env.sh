#!/usr/bin/env bash
# Common paths; safe to source repeatedly. Does not change shell startup files.
SETUP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SETUP_DIR/.." && pwd)"
if [[ -f "$SETUP_DIR/config.env" ]]; then source "$SETUP_DIR/config.env"; fi
export AI_CLUB_HOME="${AI_CLUB_HOME:-$HOME/ai-club-tools}"
member_default="${USER:-member}"
member_default="${member_default%%@*}"
export TEAM_MEMBER="${TEAM_MEMBER:-$member_default}"
if [[ ! "$TEAM_MEMBER" =~ ^[a-zA-Z0-9_-]+$ ]]; then
  echo 'TEAM_MEMBER must contain only letters, numbers, underscores or hyphens.' >&2
  return 2 2>/dev/null || exit 2
fi
export ROSIE_PYTORCH_SIF="${ROSIE_PYTORCH_SIF:-/data/containers/msoe-pytorch-25.05-py3.sif}"
export COSMOS_MODEL="${COSMOS_MODEL:-nvidia/Cosmos3-Edge}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$AI_CLUB_HOME/cache/uv}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export MPLBACKEND=Agg
export PYTHONNOUSERSITE=1
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$AI_CLUB_HOME/envs" "$AI_CLUB_HOME/logs" "$AI_CLUB_HOME/results/$TEAM_MEMBER" "$AI_CLUB_HOME/cache"
