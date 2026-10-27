#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/../env.sh"
export PYTHONUNBUFFERED=1
profile="${1:-data}"
if [[ $# -gt 0 ]]; then shift; fi
case "$profile" in
  data|cosmos) env_name="$profile" ;;
  isaac) env_name=isaac-6.0.0.1
    if [[ "$(uname -m)" == aarch64 && -f /lib/aarch64-linux-gnu/libgomp.so.1 ]]; then
      export LD_PRELOAD="/lib/aarch64-linux-gnu/libgomp.so.1${LD_PRELOAD:+:$LD_PRELOAD}"
    fi ;;
  *) echo 'Usage: bash setup/spark/run.sh data|cosmos|isaac script.py [args]' >&2; exit 2 ;;
esac
python_bin="$AI_CLUB_HOME/envs/$env_name/bin/python"
if [[ ! -x "$python_bin" ]]; then echo "Run bash setup/bootstrap.sh $profile first." >&2; exit 2; fi
if [[ "$profile" == isaac || "$profile" == cosmos ]]; then
  exec 9>"$AI_CLUB_HOME/spark-gpu.lock"
  flock -n 9 || { echo 'Another team GPU command is running. Try again after its handoff.' >&2; exit 75; }
fi
if [[ $# -eq 0 && "$profile" == isaac ]]; then
  exec "$python_bin" -c 'import importlib.metadata; print("Isaac Sim", importlib.metadata.version("isaacsim"))'
fi
if [[ $# -eq 0 ]]; then exec "$python_bin" "$SETUP_DIR/doctor.py" --profile "$profile"; fi
if [[ "$profile" == isaac && "$(uname -m)" == aarch64 && "${ISAAC_DISABLE_ASLR:-1}" == 1 ]]; then
  # This GB10/Kit build can deadlock before logging with randomized addresses.
  # Apply the tested workaround to this process only, never system settings.
  command -v setarch >/dev/null || { echo 'Install util-linux for the tested GB10 setarch launcher.' >&2; exit 2; }
  exec setarch aarch64 -R "$python_bin" "$@"
fi
exec "$python_bin" "$@"
