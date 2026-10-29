#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/env.sh"
cd "$REPO_ROOT"
exec bash "$SETUP_DIR/spark/run.sh" data -m replay serve --out "$AI_CLUB_HOME/results/$TEAM_MEMBER/replay" "$@"
