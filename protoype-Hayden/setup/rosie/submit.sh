#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/../env.sh"
kind="${1:-smoke}"
if [[ $# -gt 0 ]]; then shift; fi
case "$kind" in smoke|cosmos|job|data) ;; *) echo 'Usage: submit.sh smoke|cosmos|job|data [script args]' >&2; exit 2;; esac
exec sbatch --chdir="$REPO_ROOT" --output="$AI_CLUB_HOME/logs/%x-%j.log" \
  "$SETUP_DIR/rosie/$kind.sbatch" "$@"
