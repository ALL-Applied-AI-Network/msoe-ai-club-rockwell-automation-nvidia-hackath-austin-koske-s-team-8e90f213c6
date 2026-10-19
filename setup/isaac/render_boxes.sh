#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/../env.sh"
out="${1:-$AI_CLUB_HOME/results/$TEAM_MEMBER/camera-boxes-$(date +%Y%m%d-%H%M%S)}"
limit="${2:-12}"
bash "$SETUP_DIR/spark/run.sh" isaac "$SETUP_DIR/isaac/render_box_batch.py" --out "$out" --limit "$limit"
bash "$SETUP_DIR/spark/run.sh" data "$SETUP_DIR/isaac/review_box_batch.py" "$out"
echo "Local PNG review sheets and verification.json: $out"
