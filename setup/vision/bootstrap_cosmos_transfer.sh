#!/usr/bin/env bash
# Isolated upstream environment. No shared environment or driver changes.
set -euo pipefail
root="${1:?Usage: bootstrap_cosmos_transfer.sh EXPERIMENT_ROOT cu130|cu128}"
variant="${2:?Specify the CUDA variant}"
case "$variant" in cu130|cu128) ;; *) exit 2 ;; esac
cd "$root/source"
export UV_PROJECT_ENVIRONMENT="$root/environment"
export UV_CACHE_DIR="$HOME/ai-club-tools/cache/uv"
export UV_LINK_MODE=copy
if [[ -d "$root/sysroot/usr/include" ]]; then
  export CPATH="$root/sysroot/usr/include${CPATH:+:$CPATH}"
fi
python_version=$(cat .python-version)
"$HOME/.local/bin/uv" sync --locked --extra "$variant" --no-default-groups --python "$python_version"
"$HOME/.local/bin/uv" pip freeze --python "$UV_PROJECT_ENVIRONMENT/bin/python" > "$root/environment-freeze.txt"
"$UV_PROJECT_ENVIRONMENT/bin/python" -c 'import torch,sys; print(sys.version); print(torch.__version__); print(torch.version.cuda)'
