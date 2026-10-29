#!/usr/bin/env bash
# One-command GB10 onboarding from a GitHub checkout:
#
#   git clone -b austin-work <repo-url> linewatch && cd linewatch
#   bash setup/gb10/onboard.sh                 # models from the GitHub release, Docker GPU
#
# Options: --assets ZIP|DIR (e.g. from USB)  --port N  --cpu (skip GPU)  --native (skip Docker)
#          --docker (use Docker even with cameras attached)  --no-selftest
# Order: Docker+GPU -> Docker CPU -> native launcher (setup/gb10/start.sh). With cameras attached
# (/dev/video*) the native launcher runs the app, because the containers cannot open them and
# the LINE page's live cameras are the main view; Docker then only builds the UI if needed. Every step, the
# machine facts and the self-test report are saved under ~/linewatch-onboarding/<timestamp>/.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HERE="$ROOT/setup/gb10"
RELEASE_TAG=temporal-cnn-v1.0.0
ASSET_ZIP=temporal-cnn-assets-v1.zip
ASSETS=""; PORT=8788; WANT_GPU=1; USE_DOCKER=1; SELFTEST=1; FORCE_DOCKER=0
PROJECT="${LINEWATCH_PROJECT:-linewatch-demo}"
while [ $# -gt 0 ]; do
  case "$1" in
    --assets) ASSETS="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --cpu) WANT_GPU=0; shift ;;
    --native) USE_DOCKER=0; shift ;;
    --docker) FORCE_DOCKER=1; shift ;;
    --no-selftest) SELFTEST=0; shift ;;
    -h|--help) sed -n '2,10p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

REPORT="$HOME/linewatch-onboarding/$(date +%Y%m%dT%H%M%S)"
mkdir -p "$REPORT"
exec > >(tee -a "$REPORT/onboard.log") 2>&1
step() { printf '\n== %s\n' "$*"; }
fail() { echo "ONBOARDING FAILED: $*"; echo "Report: $REPORT"; echo "Backup: USB kit -> bash START.sh (setup/gb10/README.md)"; exit 1; }
MODE=""; RESULT=""

step "Machine facts -> $REPORT/facts.txt"
{
  echo "date: $(date -Is)"; echo "host: $(hostname)  arch: $(uname -m)  kernel: $(uname -r)"
  grep -s PRETTY_NAME /etc/os-release
  echo "repo: $(git -C "$ROOT" rev-parse HEAD 2>/dev/null) $(git -C "$ROOT" status --porcelain 2>/dev/null | grep -q . && echo '(modified)')"
  echo "--- nvidia-smi"; nvidia-smi 2>&1 | head -15
  echo "--- docker"; docker version --format 'client {{.Client.Version}} server {{.Server.Version}}' 2>&1
  docker compose version 2>&1; docker info --format 'runtimes: {{range $k, $v := .Runtimes}}{{$k}} {{end}}' 2>&1 | head -3
  echo "--- cameras"; ls -l /dev/video* 2>&1
  echo "--- disk"; df -h "$ROOT" "$HOME" 2>&1
} > "$REPORT/facts.txt" 2>&1
cat "$REPORT/facts.txt"
[ "$(uname -m)" = aarch64 ] || echo "note: not aarch64 (GB10 is aarch64); continuing"

step "Model assets"
if [ -z "$ASSETS" ] && [ -d "$ROOT/models/temporal-cnn-assets-v1" ]; then ASSETS="$ROOT/models/temporal-cnn-assets-v1"; fi
if [ -z "$ASSETS" ]; then
  mkdir -p "$ROOT/models"
  if command -v gh >/dev/null && gh auth status >/dev/null 2>&1; then
    gh release download "$RELEASE_TAG" -R "$(git -C "$ROOT" remote get-url origin)" -p "$ASSET_ZIP" -p SHA256SUMS.txt \
      -D "$ROOT/models" --clobber || fail "gh release download failed"
  elif [ -n "${GITHUB_TOKEN:-}" ]; then
    slug="$(git -C "$ROOT" remote get-url origin | sed -E 's#(git@github.com:|https://github.com/)##; s#\.git$##')"
    for name in "$ASSET_ZIP" SHA256SUMS.txt; do
      id="$(curl -fsSL -H "Authorization: Bearer $GITHUB_TOKEN" "https://api.github.com/repos/$slug/releases/tags/$RELEASE_TAG" |
        python3 -c "import json,sys; print(next(a['id'] for a in json.load(sys.stdin)['assets'] if a['name']==sys.argv[1]))" "$name")" \
        || fail "release asset $name not found (token needs repo read access)"
      curl -fsSL -H "Authorization: Bearer $GITHUB_TOKEN" -H "Accept: application/octet-stream" \
        "https://api.github.com/repos/$slug/releases/assets/$id" -o "$ROOT/models/$name" || fail "download of $name failed"
    done
  else
    fail "No model assets. Run 'gh auth login', export GITHUB_TOKEN, or pass --assets <USB path to $ASSET_ZIP>"
  fi
  (cd "$ROOT/models" && grep " $ASSET_ZIP\$" SHA256SUMS.txt | sha256sum -c -) || fail "release ZIP checksum mismatch"
  ASSETS="$ROOT/models/$ASSET_ZIP"
fi
if [ -f "$ASSETS" ]; then
  target="$(dirname "$ASSETS")/$(basename "$ASSETS" .zip)"
  [ -d "$target" ] || python3 -c 'import sys,zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])' "$ASSETS" "$(dirname "$target")" \
    || fail "could not extract $ASSETS"
  ASSETS="$target"
fi
ASSETS="$(cd "$ASSETS" && pwd)" || fail "assets folder missing"
for pin in "models/vial-best.pt 0bfac07a256eaf190416b632ad1f094a6b534309df1bb11a8c175e9d811b56bc" \
           "models/liquid-best.pt 6714fe32191d8eba230aa87246f38a25c2906c619d60d7ba30c9e60205352eef"; do
  set -- $pin
  [ "$(sha256sum "$ASSETS/$1" 2>/dev/null | cut -d' ' -f1)" = "$2" ] || fail "checkpoint hash mismatch: $ASSETS/$1"
done
echo "assets: $ASSETS (checkpoint hashes verified)"

port_free() { ! (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }
healthy() { curl -fsS --max-time 3 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"pipeline_mode":"model"'; }

compose_up() {  # $1 = gpu|cpu
  local files=(-f "$ROOT/compose.yaml"); [ "$1" = gpu ] && files+=(-f "$ROOT/compose.gpu.yaml")
  step "Docker build and start ($1) -> $REPORT/docker-$1.log"
  LINEWATCH_ASSETS_DIR="$ASSETS" LINEWATCH_PORT="$PORT" \
    docker compose -p "$PROJECT" "${files[@]}" up -d --build --wait --wait-timeout 300 > "$REPORT/docker-$1.log" 2>&1
  local status=$?; tail -5 "$REPORT/docker-$1.log"
  [ $status -eq 0 ] && healthy || return 1
  if [ "$1" = gpu ]; then
    docker compose -p "$PROJECT" exec -T api /app/api/.venv/bin/python -c \
      'import torch,sys; ok=torch.cuda.is_available(); print("container CUDA:", ok, torch.cuda.get_device_name(0) if ok else ""); sys.exit(0 if ok else 1)' \
      || return 1
  fi
}

if [ "$USE_DOCKER" = 1 ] && [ "$FORCE_DOCKER" = 0 ] && ls /dev/video* >/dev/null 2>&1; then
  step "Cameras attached: running natively so the LINE page can open them (--docker to override)"
  USE_DOCKER=0
  if [ ! -f "$ROOT/client/dist/index.html" ] && docker info >/dev/null 2>&1; then
    echo "Building the UI with Docker -> $REPORT/docker-ui.log"
    docker compose -p "$PROJECT" -f "$ROOT/compose.yaml" build client > "$REPORT/docker-ui.log" 2>&1 && \
      cid="$(docker create "$PROJECT-client")" && mkdir -p "$ROOT/client/dist" && \
      docker cp "$cid:/usr/share/nginx/html/." "$ROOT/client/dist/" && docker rm "$cid" >/dev/null
  fi
fi
if [ "$USE_DOCKER" = 1 ]; then
  docker info >/dev/null 2>&1 || { echo "Docker not usable by $(id -un) (try: sudo usermod -aG docker \$USER, then log in again)"; USE_DOCKER=0; }
fi
if [ "$USE_DOCKER" = 1 ]; then
  if ! port_free "$PORT" && ! docker compose -p "$PROJECT" ps --status running 2>/dev/null | grep -q client; then
    fail "port $PORT is used by something else; pass --port <other>"
  fi
  gpu_ready=0
  if [ "$WANT_GPU" = 1 ] && command -v nvidia-smi >/dev/null && \
     { docker info 2>/dev/null | grep -qi nvidia || command -v nvidia-ctk >/dev/null; }; then gpu_ready=1; fi
  [ "$WANT_GPU" = 1 ] && [ $gpu_ready = 0 ] && echo "No NVIDIA container runtime detected; using CPU in Docker."
  if [ $gpu_ready = 1 ] && compose_up gpu; then MODE="docker-gpu"
  else
    [ $gpu_ready = 1 ] && echo "GPU container path failed (see docker-gpu.log); falling back to CPU container."
    if compose_up cpu; then MODE="docker-cpu"; else echo "Docker CPU path failed (see docker-cpu.log)."; fi
  fi
fi

if [ -z "$MODE" ]; then
  step "Native fallback (setup/gb10/start.sh)"
  if [ ! -f "$ROOT/client/dist/index.html" ] && docker info >/dev/null 2>&1; then
    # Reuse the UI bundle from the built client image; no Node needed on the host.
    image="$(docker compose -p "$PROJECT" -f "$ROOT/compose.yaml" images -q client 2>/dev/null | head -1)"
    if [ -n "$image" ]; then cid="$(docker create "$image")" && docker cp "$cid:/usr/share/nginx/html/." "$ROOT/client/dist/" && docker rm "$cid" >/dev/null; fi
  fi
  [ -f "$ROOT/client/dist/index.html" ] || fail "No Docker and no prebuilt UI. Use the USB kit: bash START.sh"
  docker compose -p "$PROJECT" -f "$ROOT/compose.yaml" stop >/dev/null 2>&1 || true
  port_free "$PORT" || fail "port $PORT in use; pass --port <other>"
  device=auto; [ "$WANT_GPU" = 0 ] && device=cpu
  if [ "$SELFTEST" = 1 ]; then
    bash "$HERE/start.sh" --selftest --device "$device" --assets "$ASSETS" --data "$HOME/linewatch-data" \
      > "$REPORT/native-selftest.log" 2>&1; RESULT="$(grep -E 'Self-test (PASSED|FAILED)' "$REPORT/native-selftest.log")"
    cp "$(grep -o '/[^ ]*selftest-report.json' "$REPORT/native-selftest.log" | tail -1)" "$REPORT/" 2>/dev/null || true
    SELFTEST=0
  fi
  nohup bash "$HERE/start.sh" --device "$device" --assets "$ASSETS" --port "$PORT" > "$REPORT/native-server.log" 2>&1 &
  for _ in $(seq 600); do healthy && break; sleep 1; done
  healthy || fail "native server did not become healthy (see native-server.log)"
  MODE="native-$(grep -o 'device [a-z:0-9]*' "$REPORT/native-server.log" | head -1 | cut -d' ' -f2)"
fi

if [ "$SELFTEST" = 1 ]; then
  step "Reproducibility self-test (40 reference frames vs. frozen outputs)"
  python3 -I "$HERE/selftest.py" --base-url "http://127.0.0.1:$PORT" --frames "$HERE/selftest-frames" \
    --expected "$HERE/selftest-expected.json" --report "$REPORT/selftest-report.json" | tee "$REPORT/selftest.txt"
  RESULT="$(grep -E 'Self-test (PASSED|FAILED)' "$REPORT/selftest.txt")"
fi

python3 - "$REPORT" "$MODE" "$PORT" "$RESULT" <<'EOF'
import json, sys
report, mode, port, result = sys.argv[1:]
json.dump({"mode": mode, "url": f"http://127.0.0.1:{port}", "selftest": result or "not run",
           "report_dir": report}, open(f"{report}/summary.json", "w"), indent=2)
EOF
step "Ready"
echo "Mode:      $MODE"
echo "Open:      http://127.0.0.1:$PORT  (browser on this GB10)"
echo "Self-test: ${RESULT:-not run}"
echo "Report:    $REPORT"
case "$MODE" in docker-*) echo "Stop: docker compose -p "$PROJECT" stop   (never 'down -v'; it deletes history)";;
                *) echo "Stop: kill the uvicorn process (see native-server.log)";; esac
