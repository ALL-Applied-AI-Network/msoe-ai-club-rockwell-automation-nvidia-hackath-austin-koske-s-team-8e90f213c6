#!/usr/bin/env bash
# Bulk carryover renders on a Brev Isaac Sim instance (see README.md, "Carryover bulk render").
#   bash setup/brev/render_carryover.sh OUT ASSET_ROOT [FAMILIES=400] [CHANGEOVERS=8] [SHARD=400]
# ISAAC_PY selects Isaac's Python (default /isaac-sim/python.sh); FIXTURE_ARGS adds fixture options
# (e.g. "--fractions full --first-family 400 --seed 31"). Shards run round-robin on every
# visible GPU; each shard is a separate renderer launch with its own output folder, so a failed
# shard is rerun alone. timing.jsonl records GPU, start and end per shard for GPU-hour accounting.
set -euo pipefail
[[ $# -ge 2 && $# -le 5 ]] || { echo 'Usage: render_carryover.sh OUT ASSET_ROOT [FAMILIES] [CHANGEOVERS] [SHARD]' >&2; exit 2; }
out="$(realpath -m "$1")"; assets="$(realpath "$2")"
families="${3:-400}"; changeovers="${4:-8}"; shard="${5:-400}"
isaac_py="${ISAAC_PY:-/isaac-sim/python.sh}"
helper_py="${HELPER_PY:-$(command -v python3 || echo "$isaac_py")}"   # the Isaac container has no python3
isaac_dir="$(cd -- "$(dirname -- "$0")/../isaac" && pwd)"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:?Accept the Omniverse license, then export OMNI_KIT_ACCEPT_EULA=YES}"
[[ -x "$isaac_py" ]] || { echo "No Isaac Python at $isaac_py; set ISAAC_PY" >&2; exit 2; }
[[ -f "$assets/stations/fill_station/fill_station.usda" ]] || { echo "No twin assets at $assets" >&2; exit 2; }
mkdir -p "$out"

if [[ ! -f "$out/fixture.json" ]]; then
  (cd "$isaac_dir" && "$helper_py" make_carryover_fixture.py --out "$out/fixture.json" \
    --families "$families" --changeovers "$changeovers" ${FIXTURE_ARGS:-})
fi
# Split into shard fixtures once; reruns reuse them.
"$helper_py" - "$out" "$shard" <<'EOF'
import json, sys
from pathlib import Path
out, size = Path(sys.argv[1]), int(sys.argv[2])
fixture = json.loads((out / "fixture.json").read_text())
cases = fixture["cases"]
for i in range(0, len(cases), size):
    path = out / "shards" / f"shard-{i // size:03d}.json"
    if not path.exists():
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(dict(fixture, cases=cases[i:i + size]), indent=1))
EOF

gpus=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
echo "Rendering $(ls "$out"/shards/*.json | wc -l) shards on $gpus GPU(s) into $out"

run_shard() {  # gpu fixture
  local gpu="$1" fixture="$2" name; name="$(basename "$2" .json)"
  local dest="$out/renders/$name"
  if [[ -f "$dest/verification.json" ]] && grep -q '"passed": true' "$dest/verification.json"; then return 0; fi
  rm -rf "$dest"; mkdir -p "$out/logs"
  local start; start=$(date +%s)
  local count; count=$("$helper_py" -c "import json,sys; print(len(json.load(open(sys.argv[1]))['cases']))" "$fixture")
  CUDA_VISIBLE_DEVICES="$gpu" "$isaac_py" "$isaac_dir/render_box_batch.py" --asset-root "$assets" \
    --fixture "$fixture" --out "$dest" --limit "$count" --subframes 4 --pt-spp 32 --compact \
    > "$out/logs/$name.log" 2>&1 \
    && "$isaac_py" "$isaac_dir/review_box_batch.py" "$dest" >> "$out/logs/$name.log" 2>&1
  local status=$?
  printf '{"shard":"%s","gpu":%s,"gpu_name":"%s","cases":%s,"start":%s,"end":%s,"exit":%s}\n' \
    "$name" "$gpu" "$(nvidia-smi -i "$gpu" --query-gpu=name --format=csv,noheader)" "$count" \
    "$start" "$(date +%s)" "$status" >> "$out/timing.jsonl"
  return $status
}

worker() {  # gpu: every gpus-th shard starting at this index
  local gpu="$1" i=0 failed=0
  for fixture in "$out"/shards/*.json; do
    if (( i % gpus == gpu )); then run_shard "$gpu" "$fixture" || failed=1; fi
    i=$((i + 1))
  done
  return $failed
}

pids=()
for ((g = 0; g < gpus; g++)); do worker "$g" & pids+=($!); done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
"$helper_py" - "$out" <<'EOF'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
rows = [json.loads(l) for l in (out / "timing.jsonl").read_text().splitlines()]
ok = [r for r in rows if r["exit"] == 0]
hours = sum(r["end"] - r["start"] for r in rows) / 3600
print(f"{len(ok)}/{len(rows)} shard runs passed; {sum(r['cases'] for r in ok)} renders; "
      f"{hours:.2f} GPU-hours on {sorted({r['gpu_name'] for r in rows})}")
EOF
exit $status
