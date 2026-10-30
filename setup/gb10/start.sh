#!/usr/bin/env bash
# Native Linewatch launcher for an NVIDIA GB10 (aarch64 Linux). No Docker, Node, root or uv.
#
#   bash START.sh                      # install if needed, then serve http://127.0.0.1:8788
#   bash START.sh --check              # report machine readiness only
#   bash START.sh --selftest           # install, then verify outputs match the reference run
#
# Options: --device auto|cuda|cpu  --port N  --host ADDR  --assets DIR|ZIP  --data DIR
#          --wheels DIR (offline; a kit's wheels/ is used automatically)  --install-only
#          --station FILE (live cameras for the LINE page; default setup/csi-station.json)  --no-cameras
#          --mock-fill VIDEO [--mock-tray VIDEO] [--mock-schedule CSV]: recorded 640x480 videos offered
#          as mock cameras next to the real ones (LINE page: WEBCAMS | MOCK), played at real time
#          through the camera input with their own framing (Run 5's), not the CSI vial box and slots.
# Installs exact, hash-locked packages (setup/gb10/lock-*.txt) into .venv-gb10-<device>.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HERE="$ROOT/setup/gb10"
ASSETS="${LINEWATCH_ASSETS_DIR:-}"
DEVICE=auto
PORT=8788
HOST=127.0.0.1
WHEELS=""
DATA="${LINEWATCH_DATA_DIR:-$HOME/linewatch-data}"
MODE=serve
STATION="${LINEWATCH_STATION:-}"
CAMERAS=1
MOCK_FILL="" MOCK_TRAY="" MOCK_SCHEDULE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --assets) ASSETS="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --wheels) WHEELS="$2"; shift 2 ;;
    --data) DATA="$2"; shift 2 ;;
    --install-only) MODE=install; shift ;;
    --station) STATION="$2"; shift 2 ;;
    --no-cameras) CAMERAS=0; shift ;;
    --mock-fill) MOCK_FILL="$2"; shift 2 ;;
    --mock-tray) MOCK_TRAY="$2"; shift 2 ;;
    --mock-schedule) MOCK_SCHEDULE="$2"; shift 2 ;;
    --check) MODE=check; shift ;;
    --selftest) MODE=selftest; shift ;;
    -h|--help) sed -n '2,13p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
case "$DEVICE" in auto|cuda|cpu) ;; *) echo "--device must be auto, cuda or cpu" >&2; exit 2 ;; esac
[ -n "$ASSETS" ] || ASSETS="$ROOT/models/temporal-cnn-assets-v1"
[ -z "$WHEELS" ] && [ -d "$ROOT/wheels" ] && WHEELS="$ROOT/wheels"
say() { printf '\n== %s\n' "$*"; }
fail() { echo "ERROR: $*" >&2; exit 1; }

port_free() { ! (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }

# Python 3.12: bundled kit interpreter first, then the system.
PYTHON=""
for candidate in "$ROOT/python/bin/python3" python3.12 python3; do
  if command -v "$candidate" >/dev/null 2>&1 && \
     "$candidate" -c 'import sys,venv,ensurepip; sys.exit(sys.version_info[:2]!=(3,12))' 2>/dev/null; then
    PYTHON="$(command -v "$candidate")"; break
  fi
done

if [ "$MODE" = check ]; then
  say "Machine"
  echo "host $(hostname)  arch $(uname -m)  $(grep -s PRETTY_NAME /etc/os-release | cut -d= -f2)"
  [ "$(uname -m)" = aarch64 ] || echo "note: not aarch64; lock files target the GB10"
  echo "python: ${PYTHON:-NOT FOUND (needs Python 3.12 with venv, or a kit with python/)}"
  echo "disk free here: $(df -h "$ROOT" | awk 'NR==2{print $4}')  (CUDA install needs ~8 GB)"
  say "GPU"
  if command -v nvidia-smi >/dev/null; then nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
  else echo "nvidia-smi not found: CPU inference only"; fi
  say "Packages"
  if [ -n "$WHEELS" ]; then echo "offline wheelhouse: $WHEELS ($(ls "$WHEELS" | wc -l) files)"
  elif "${PYTHON:-python3}" - <<'EOF' 2>/dev/null
import urllib.request; urllib.request.urlopen("https://pypi.org/simple/pip/", timeout=8)
EOF
  then echo "internet: PyPI reachable"; else echo "internet: PyPI NOT reachable and no wheelhouse -> use the offline kit"; fi
  say "Cameras (browser LIVE CAMERA needs one visible to a local browser)"
  ls /dev/video* 2>/dev/null || echo "no /dev/video* devices"
  say "Ports"
  port_free "$PORT" && echo "port $PORT free" || echo "port $PORT IN USE -> pass --port"
  say "Docker (not required)"
  docker info >/dev/null 2>&1 && echo "usable" || echo "not usable by this account (fine)"
  exit 0
fi

[ -n "$PYTHON" ] || fail "Python 3.12 with venv not found. Use the kit (bundles python/) or install python3.12-venv."

# Models: accept the release ZIP or folder; verify the released checkpoint pins.
declare -A PINS=(
  [models/vial-best.pt]=0bfac07a256eaf190416b632ad1f094a6b534309df1bb11a8c175e9d811b56bc
  [models/liquid-best.pt]=6714fe32191d8eba230aa87246f38a25c2906c619d60d7ba30c9e60205352eef
)
if [ -f "$ASSETS" ] && [[ "$ASSETS" == *.zip ]]; then
  target="${ASSETS%.zip}"
  [ -d "$target" ] || "$PYTHON" -c 'import sys,zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])' "$ASSETS" "$(dirname "$target")"
  ASSETS="$target"
fi
[ -d "$ASSETS" ] || fail "Model assets not found: $ASSETS (pass --assets)"
ASSETS="$(cd "$ASSETS" && pwd)"
for file in "${!PINS[@]}"; do
  actual="$(sha256sum "$ASSETS/$file" 2>/dev/null | cut -d' ' -f1 || true)"
  [ "$actual" = "${PINS[$file]}" ] || fail "Model hash mismatch or missing: $ASSETS/$file"
done
DIST="$ROOT/client/dist"
[ -f "$DIST/index.html" ] && [ -d "$DIST/assets" ] || \
  fail "Prebuilt UI missing at $DIST. Use the kit, or run 'npm ci && npm run build' in client/ with Node."

install_env() {  # $1 = cuda|cpu; prints the venv path on success
  local dev="$1" venv="$ROOT/.venv-gb10-$1" lock="$HERE/lock-$1.txt"
  local want; want="$(sha256sum "$lock" | cut -d' ' -f1)"
  if [ ! -f "$venv/.linewatch-lock" ] || [ "$(cat "$venv/.linewatch-lock")" != "$want" ]; then
    say "Installing $dev packages (exact versions, hash-checked)" >&2
    rm -rf "$venv"
    "$PYTHON" -m venv "$venv" >&2
    local pip=("$venv/bin/python" -m pip install --disable-pip-version-check --no-deps --require-hashes -r "$lock")
    [ -n "$WHEELS" ] && pip+=(--no-index --find-links "$WHEELS")
    "${pip[@]}" >&2 || { rm -rf "$venv"; return 1; }
    echo "$want" > "$venv/.linewatch-lock"
  fi
  if [ "$dev" = cuda ]; then
    "$venv/bin/python" -c 'import torch,sys; ok=torch.cuda.is_available(); print("CUDA:", ok, torch.cuda.get_device_name(0) if ok else "", file=sys.stderr); sys.exit(0 if ok else 1)' || return 1
  fi
  echo "$venv"
}

VENV=""
if [ "$DEVICE" != cpu ]; then
  if VENV="$(install_env cuda)"; then DEVICE=cuda
  elif [ "$DEVICE" = cuda ]; then fail "CUDA install or GPU check failed; rerun with --device cpu"
  else echo "CUDA unavailable; falling back to CPU inference." >&2; VENV=""; fi
fi
if [ -z "$VENV" ]; then VENV="$(install_env cpu)" || fail "CPU install failed"; DEVICE=cpu; fi
[ "$MODE" = install ] && { echo "Ready: $VENV (device $DEVICE)"; exit 0; }

export PYTHONPATH="$ROOT:$ROOT/api" PYTHONUNBUFFERED=1
export APP__PIPELINE_MODE=model APP__MODEL_ASSETS="$ASSETS" APP__CLIENT_DIST="$DIST"
export APP__DEVICE="$([ "$DEVICE" = cuda ] && echo cuda:0 || echo cpu)"
export APP__REFERENCE_IMAGE="${LINEWATCH_REFERENCE_IMAGE:-$ASSETS/examples/run4-fill1-f000264.png}"

if [ "$MODE" = selftest ]; then
  FRAMES="$ROOT/selftest/frames"; [ -d "$FRAMES" ] || FRAMES="$HERE/selftest-frames"
  [ -d "$FRAMES" ] || fail "Self-test frames missing at $FRAMES (included in the kit)"
  TEST_PORT=18788; while ! port_free "$TEST_PORT"; do TEST_PORT=$((TEST_PORT + 1)); done
  STAMP="$(date +%Y%m%dT%H%M%S)"
  TEST_DATA="$DATA/selftest-$STAMP"; mkdir -p "$TEST_DATA"
  export APP__DATA_DIR="$TEST_DATA" YOLO_CONFIG_DIR="$TEST_DATA/.ultralytics"
  (cd "$ROOT/api" && exec "$VENV/bin/python" -m uvicorn main:app --host 127.0.0.1 --port "$TEST_PORT") \
    > "$TEST_DATA/server.log" 2>&1 &
  SERVER=$!; trap 'kill $SERVER 2>/dev/null || true' EXIT
  for _ in $(seq 120); do
    "$VENV/bin/python" -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:$TEST_PORT/health', timeout=2)" 2>/dev/null && break
    kill -0 $SERVER 2>/dev/null || { cat "$TEST_DATA/server.log" >&2; fail "Server exited during self-test"; }
    sleep 1
  done
  say "Self-test on $APP__DEVICE (data in $TEST_DATA)"
  "$VENV/bin/python" "$HERE/selftest.py" --base-url "http://127.0.0.1:$TEST_PORT" --frames "$FRAMES" \
    --expected "$HERE/selftest-expected.json" --report "$TEST_DATA/selftest-report.json"
  exit $?
fi

port_free "$PORT" || fail "Port $PORT is in use; pass --port <other>"
mkdir -p "$DATA"
export APP__DATA_DIR="$DATA" YOLO_CONFIG_DIR="$DATA/.ultralytics"
# Live cameras for the LINE page (the landing page). Close the Linewatch recorder first:
# a camera can be open in only one program.
[ -n "$STATION" ] || STATION="$ROOT/setup/csi-station.json"
if [ -n "$MOCK_FILL" ]; then
  # The station plus a "mock" entry: the LINE page's WEBCAMS | MOCK switch picks between them.
  BASE="$STATION"; STATION="$DATA/station-with-mock.json"
  "$VENV/bin/python" - "$BASE" "$STATION" "$ROOT/client/public/data/run5/recipe-library.json" \
    "$MOCK_FILL" "$MOCK_TRAY" "$MOCK_SCHEDULE" <<'EOF'
import json, os, sys
base, out, library, fill, tray, schedule = sys.argv[1:]
for path in (fill, tray, schedule):
    if path and not os.path.isfile(path):
        sys.exit(f"ERROR: mock file not found: {path}")
station = json.load(open(base)) if os.path.isfile(base) else {"title": "No station cameras"}
for key in ("library", "schedule"):  # relative to the original station file
    if isinstance(station.get(key), str):
        station[key] = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(base)), station[key]))
station.setdefault("library", library)
mock = {"title": "Mock cameras (recorded video)", "fill_camera": "file:" + os.path.abspath(fill)}
if tray:
    mock["tray_camera"] = "file:" + os.path.abspath(tray)
if schedule:
    mock["schedule"] = os.path.abspath(schedule)
station["mock"] = mock
json.dump(station, open(out, "w"), indent=1)
EOF
fi
if [ "$CAMERAS" = 1 ] && [ -f "$STATION" ]; then
  STATION="$(cd "$(dirname "$STATION")" && pwd)/$(basename "$STATION")"
  export APP__LINE_STATION="$STATION" APP__LINE_RECORDINGS="$DATA/line-recordings"
  say "Cameras from $STATION"
  for key in fill_camera tray_camera; do
    dev="$("$VENV/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1])).get(sys.argv[2]) or "")' "$STATION" "$key")"
    [ -z "$dev" ] && continue
    if [[ "$dev" == file:* ]]; then echo "  $key: recorded video ${dev#file:} (mock camera)"
    elif [ -e "$dev" ]; then echo "  $key: $dev -> $(readlink -f "$dev")"
    else echo "  $key: $dev NOT FOUND (unplugged? ls -la /dev/v4l/by-id/; pass --station with the right names)"; fi
  done
  [ -n "$MOCK_FILL" ] && echo "  mock: $MOCK_FILL${MOCK_TRAY:+ + $MOCK_TRAY} (LINE page: WEBCAMS | MOCK)"
fi
echo "Linewatch on http://$HOST:$PORT  (device $APP__DEVICE, data $DATA). Ctrl+C stops."
cd "$ROOT/api"
exec "$VENV/bin/python" -m uvicorn main:app --host "$HOST" --port "$PORT"
