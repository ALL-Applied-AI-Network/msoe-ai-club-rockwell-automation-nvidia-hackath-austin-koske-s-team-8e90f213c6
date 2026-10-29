#!/usr/bin/env bash
set -euo pipefail
[[ $# -ge 2 && $# -le 3 ]] || { echo 'Usage: render_training.sh FRESH_OUTPUT FIXTURE_JSON [COUNT=768]' >&2; exit 2; }
source "$(dirname -- "$0")/../env.sh"
out="$1"
fixture="$2"
count="${3:-768}"
bash "$SETUP_DIR/spark/run.sh" isaac "$SETUP_DIR/isaac/render_box_batch.py" \
  --fixture "$fixture" --out "$out" --limit "$count" --subframes 1 --pt-spp 32 --compact
bash "$SETUP_DIR/spark/run.sh" data "$SETUP_DIR/isaac/review_box_batch.py" "$out"
echo "Verified synthetic training pilot: $out"
