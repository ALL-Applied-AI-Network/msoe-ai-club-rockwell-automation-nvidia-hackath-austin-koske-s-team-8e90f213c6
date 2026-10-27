#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/../env.sh"
out="${1:-$AI_CLUB_HOME/results/$TEAM_MEMBER/vial-render-$(date +%Y%m%d-%H%M%S)}"
limit="${2:-1}"
samples="${3:-$SETUP_DIR/fixtures/render_samples.jsonl}"
project="$REPO_ROOT/datasets/mixing_twin"
[[ -f "$project/scripts/render_isaac.py" ]] || { echo 'Deploy the existing twin with: pwsh -File setup/deploy.ps1 -Target Spark -IncludeTwin' >&2; exit 2; }
[[ -f "$samples" ]] || { echo "Missing sample JSONL: $samples" >&2; exit 2; }
echo "Saving renders to $out"
bash "$SETUP_DIR/spark/run.sh" isaac "$SETUP_DIR/isaac/render_project.py" \
  --asset-root "$project/assets/csi_digital_twin_v1" --samples "$samples" \
  --out "$out" --limit "$limit" --subframes 8 --fixed-light-scale 1.0
bash "$SETUP_DIR/spark/run.sh" data "$SETUP_DIR/isaac/validate_render.py" "$out"
