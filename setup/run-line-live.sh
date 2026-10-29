#!/usr/bin/env bash
# Live line view on the GB10 (Linux): the API with the models on the GPU, the station's USB cameras, and the UI.
#   setup/run-line-live.sh [--station setup/csi-station.json] [--assets DIR] [--port 8794]
# Uses the kit environment that setup/gb10/start.sh builds (.venv-gb10-cuda); set PYTHON to use another.
# Close the Linewatch recorder first: a camera can be open in only one program. See docs/line-view.md.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATION="$REPO/setup/csi-station.json"
ASSETS="${LINEWATCH_ASSETS_DIR:-$REPO/models/temporal-cnn-assets-v1}"
DATA="${LINEWATCH_DATA_DIR:-$HOME/linewatch-live/data}"
DEVICE="cuda:0"
PORT=8794
while [ $# -gt 0 ]; do
  case "$1" in
    --station) STATION="$2"; shift 2 ;;
    --assets) ASSETS="$2"; shift 2 ;;
    --data) DATA="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
PYTHON="${PYTHON:-$REPO/.venv-gb10-cuda/bin/python}"
[ -x "$PYTHON" ] || { echo "No Python at $PYTHON: run setup/gb10/start.sh once, or set PYTHON" >&2; exit 1; }
[ -d "$ASSETS/models" ] || { echo "No model release at $ASSETS (pass --assets)" >&2; exit 1; }
[ -f "$REPO/client/dist/index.html" ] || { echo "Build the UI first: cd client && npm ci && npm run build" >&2; exit 1; }
[ -f "$STATION" ] || { echo "No station file at $STATION" >&2; exit 1; }
# H.264 recordings (browser playable) need imageio-ffmpeg, which ships its own aarch64 ffmpeg.
# It goes in a side folder so the locked kit environment is not changed. Without it: OpenCV mp4v.
SITE="$HOME/linewatch-live/site"
if [ ! -d "$SITE/imageio_ffmpeg" ]; then
  "$PYTHON" -m pip install --quiet --target "$SITE" imageio-ffmpeg \
    || echo "imageio-ffmpeg not installed: recordings fall back to OpenCV mp4v" >&2
fi
mkdir -p "$DATA"
for camera in fill_camera tray_camera; do
  dev="$("$PYTHON" -c "import json,sys; print(json.load(open(sys.argv[1])).get(sys.argv[2]) or '')" "$STATION" "$camera")"
  case "$dev" in
    /dev/*) [ -e "$dev" ] && echo "$camera: $dev -> $(readlink -f "$dev")" \
              || echo "WARNING $camera: $dev not found (ls -la /dev/v4l/by-id/)" >&2 ;;
  esac
done
export PYTHONPATH="$SITE:$REPO/api:$REPO"
export APP__MODEL_ASSETS="$ASSETS" APP__DEVICE="$DEVICE" APP__DATA_DIR="$DATA"
export APP__CLIENT_DIST="$REPO/client/dist" APP__LINE_STATION="$(readlink -f "$STATION")"
export YOLO_CONFIG_DIR="$DATA/yolo"
echo "Linewatch live line view on http://127.0.0.1:$PORT  (LINE page; device $DEVICE, data $DATA). Ctrl+C stops."
cd "$REPO/api"
exec "$PYTHON" -m uvicorn main:app --host 127.0.0.1 --port "$PORT"
