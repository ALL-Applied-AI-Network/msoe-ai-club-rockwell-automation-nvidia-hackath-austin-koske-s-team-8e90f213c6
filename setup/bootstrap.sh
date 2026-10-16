#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/env.sh"
profile="${1:-data}"
if [[ "$profile" == rosie ]]; then exec bash "$SETUP_DIR/rosie/bootstrap.sh"; fi
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
fi
case "$profile" in
  data) env_name=data; req=data ;;
  isaac) env_name=isaac-6.0.0.1; req=isaac ;;
  cosmos) env_name=cosmos; req=cosmos ;;
  *) echo 'Usage: bash setup/bootstrap.sh data|rosie|isaac|cosmos' >&2; exit 2 ;;
esac
env_path="$AI_CLUB_HOME/envs/$env_name"
# Managed CPython includes headers required by ARM source dependencies.
if [[ ! -x "$env_path/bin/python" ]]; then
  uv python install 3.12
  uv venv --managed-python --python 3.12 "$env_path"
fi
if [[ "$req" == isaac ]]; then
  if [[ "$(uname -m)" != aarch64 && "$(uname -m)" != x86_64 ]]; then
    echo 'Isaac requires a supported Linux x86_64 RTX system or aarch64 GB10.' >&2; exit 2
  fi
  lock="$SETUP_DIR/requirements/isaac-linux-$(uname -m).lock.txt"
  if [[ -f "$lock" ]]; then
    uv pip install --python "$env_path/bin/python" -r "$lock" \
      --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match
  else
    uv pip install --python "$env_path/bin/python" 'isaacsim[all,extscache]==6.0.0.1' \
      --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match
    uv pip install --python "$env_path/bin/python" pillow
  fi
else
  lock="$SETUP_DIR/requirements/$req-linux-$(uname -m).lock.txt"
  if [[ "$req" == cosmos ]]; then
    if ! "$env_path/bin/python" -c 'from importlib.metadata import version; assert version("torch") == "2.11.0+cu130" and version("torchvision") == "0.26.0+cu130"' 2>/dev/null; then
      torch_options=()
      # The full lock supplies dependencies; avoid downloading them twice from
      # different indexes on a fresh environment.
      [[ ! -f "$lock" ]] || torch_options+=(--no-deps)
      uv pip install --python "$env_path/bin/python" --index-url https://download.pytorch.org/whl/cu130 \
        "${torch_options[@]}" 'torch==2.11.0' 'torchvision==0.26.0'
    fi
  fi
  if [[ -f "$lock" ]]; then input="$lock"; else input="$SETUP_DIR/requirements/$req.in"; fi
  uv pip install --python "$env_path/bin/python" -r "$input"
fi
printf '\nReady: %s\n' "$env_path"
if [[ "$req" == data ]]; then "$env_path/bin/python" "$SETUP_DIR/doctor.py" --profile data; fi
