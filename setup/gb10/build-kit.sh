#!/usr/bin/env bash
# Build a self-contained GB10 kit for USB or release-asset transfer.
#
#   bash setup/gb10/build-kit.sh --assets DIR --frames DIR --out linewatch-gb10.tar.gz [--offline]
#
# The kit holds the same runtime files as api/Dockerfile, the prebuilt UI, the released models,
# a pinned standalone CPython 3.12 (aarch64), hash-locked requirements and the self-test frames.
# --offline adds an aarch64 wheelhouse for both CUDA and CPU (~3.5 GB) so no internet is needed.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ASSETS=""; FRAMES=""; OUT=""; OFFLINE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --assets) ASSETS="$2"; shift 2 ;;
    --frames) FRAMES="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --offline) OFFLINE=1; shift ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
[ -d "$ASSETS" ] && [ -d "$FRAMES" ] && [ -n "$OUT" ] || { echo "Need --assets DIR --frames DIR --out FILE" >&2; exit 2; }
[ -f "$ROOT/client/dist/index.html" ] || { echo "Run 'npm run build' in client/ first" >&2; exit 1; }

# python-build-standalone release 20261009 (CPython 3.12.15), aarch64 glibc build.
PY_URL="https://github.com/astral-sh/python-build-standalone/releases/download/20261009/cpython-3.12.15%2B20261009-aarch64-unknown-linux-gnu-install_only.tar.gz"
PY_SHA256=b342ccfb577fda349d0237c98b24848234ca767c10f034628a5ff9fc858d662f
CACHE="${LINEWATCH_KIT_CACHE:-$HOME/.cache/linewatch-gb10-kit}"
mkdir -p "$CACHE"
PY_TGZ="$CACHE/cpython-3.12.15-aarch64.tar.gz"
[ -f "$PY_TGZ" ] || curl -fsSL "$PY_URL" -o "$PY_TGZ"
echo "$PY_SHA256  $PY_TGZ" | sha256sum -c --quiet

STAGE="$(mktemp -d)"; trap 'rm -rf "$STAGE"' EXIT
K="$STAGE/linewatch-gb10"
mkdir -p "$K/replay" "$K/integration" "$K/setup/vision" "$K/setup/gb10" "$K/client" "$K/models" "$K/selftest"
(cd "$ROOT" && tar --exclude=__pycache__ --exclude='.venv*' --exclude=.pytest_cache --exclude=data --exclude=.env --exclude=tests -cf - api) | tar -xf - -C "$K"
cp "$ROOT"/replay/{__init__,config,geometry,measurement,diagnostics}.py "$K/replay/"
cp "$ROOT"/integration/{runtime,measurement_qa}.py "$K/integration/"
cp "$ROOT"/setup/vision/{temporal_pool,train_pair_masks}.py "$K/setup/vision/"
cp "$ROOT"/setup/gb10/{start.sh,README.md,selftest.py,selftest-expected.json,lock-cuda.txt,lock-cpu.txt} "$K/setup/gb10/"
cp -r "$ROOT/client/dist" "$K/client/dist"
cp -r "$ASSETS" "$K/models/temporal-cnn-assets-v1"
cp -r "$FRAMES" "$K/selftest/frames"
find "$K" -name __pycache__ -prune -exec rm -rf {} +
tar -xzf "$PY_TGZ" -C "$K"   # -> python/
printf '#!/usr/bin/env bash\nexec bash "$(dirname "$0")/setup/gb10/start.sh" "$@"\n' > "$K/START.sh"
chmod +x "$K/START.sh" "$K/setup/gb10/start.sh"

if [ "$OFFLINE" = 1 ]; then
  # pip matches --platform tags exactly: list every glibc tag the GB10 (glibc 2.39) accepts.
  platforms=(--platform manylinux2014_aarch64 --platform linux_aarch64)
  for minor in $(seq 17 39); do platforms+=(--platform "manylinux_2_${minor}_aarch64"); done
  for lock in "$ROOT"/setup/gb10/lock-{cuda,cpu}.txt; do
    python3 -m pip download --disable-pip-version-check --no-deps --require-hashes --only-binary=:all: \
      "${platforms[@]}" \
      --python-version 3.12 --implementation cp --abi cp312 --abi abi3 --abi none \
      -d "$K/wheels" -r "$lock"
  done
fi

{ git -C "$ROOT" rev-parse HEAD; git -C "$ROOT" status --porcelain | grep -q . && echo "plus uncommitted working-tree changes"; } > "$K/SOURCE_COMMIT" || true
(cd "$K" && find . -type f ! -name MANIFEST.sha256 ! -path './python/*' -print0 | sort -z | xargs -0 sha256sum > MANIFEST.sha256)
tar -czf "$OUT" -C "$STAGE" linewatch-gb10
sha256sum "$OUT"
