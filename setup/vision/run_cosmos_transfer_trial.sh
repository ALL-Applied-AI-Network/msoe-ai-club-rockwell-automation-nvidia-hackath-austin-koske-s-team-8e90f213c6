#!/usr/bin/env bash
# One opt-in appearance sample on the shared GB10, with its normal GPU lock.
set -euo pipefail
root=$(realpath "${1:?Pass the completed experiment root}")
variant="${2:-neutral}"
case "$variant" in neutral|cool-light|warm-light) ;; *) exit 2 ;; esac
attempt="${3:-v1}"
[[ "$attempt" =~ ^v[0-9]+$ ]] || exit 2
out="$root/outputs-$variant-$attempt"
[[ ! -e "$out" ]] || { echo 'Preserve existing output; inspect it before retrying.'; exit 2; }
[[ -f "$root/downloads-completed.json" && -f "$root/environment-freeze.txt" ]]
exec 9>"$HOME/ai-club-tools/spark-gpu.lock"
flock -n 9 || { echo 'Another team GPU command is running'; exit 75; }
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="$root/model-cache"
runtime_lib="$root/sysroot/usr/lib/aarch64-linux-gnu"
export LD_LIBRARY_PATH="$runtime_lib:$runtime_lib/blas:$runtime_lib/lapack${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
cd "$root/source"
"$root/environment/bin/python" examples/inference.py \
  -i "$root/inputs-v1/transfer-$variant.json" -o "$out" --offload-guardrail-models
