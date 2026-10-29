#!/usr/bin/env bash
# Read-only scheduler polling, then atomic packaging; never resubmits jobs.
set -euo pipefail
[[ $# == 2 && "$2" =~ ^[0-9]+$ ]] || { echo 'Pass EXPERIMENT ARRAY_JOB_ID'; exit 2; }
cd "$(realpath "$1")"
early_job_id=$2
[[ ! -e completed-artifacts.tar.gz ]] || { echo 'Bundle already exists'; exit 2; }
for attempt in $(seq 1 540); do
  sacct -X -j "$early_job_id" -n -P -o JobID,State,ExitCode,Elapsed,NodeList > scheduler-status.txt
  if grep -Eq '\|(FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED)' scheduler-status.txt; then
    cat scheduler-status.txt
    echo 'A fit did not succeed; no successful-results bundle made.'
    exit 1
  fi
  completed=$(grep -c '|COMPLETED|0:0|' scheduler-status.txt || true)
  if [[ "$completed" == 3 ]]; then
    # Scheduler completion can precede visibility of shared-filesystem writes.
    # Re-read originals; never sanitize evidence or resubmit a completed fit.
    if ! singularity exec --cleanenv --bind "$HOME:$HOME" \
      /data/containers/msoe-pytorch-25.05-py3.sif python - "$PWD" <<'PY'
import csv, hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1])
try:
    plan = hashlib.sha256((root/'experiment-plan.json').read_bytes()).hexdigest()
    for arm in ('yolo11n-960', 'yolo11s-640', 'fasterrcnn-resnet50'):
        folder = root/'models'/arm
        model = json.loads((folder/'model.json').read_text())
        if model['arm'] != arm or model['experiment_plan_sha256'] != plan:
            raise ValueError('Model ancestry mismatch')
        if hashlib.sha256((folder/'best.pt').read_bytes()).hexdigest() != model['checkpoint_sha256']:
            raise ValueError('Checkpoint not fully visible')
        path = folder/('training.csv' if arm == 'fasterrcnn-resnet50' else 'train/results.csv')
        raw = path.read_bytes()
        if b'\0' in raw:
            raise ValueError('Training CSV contains NUL bytes')
        if len(list(csv.DictReader(raw.decode().splitlines()))) != model['epochs_completed']:
            raise ValueError('Training CSV incomplete')
except (OSError, ValueError, KeyError) as exc:
    print('Completed artifacts not ready:', str(exc), flush=True)
    sys.exit(1)
PY
    then
      sleep 15
      continue
    fi
    files=(experiment-plan.json preparation.json scheduler-status.txt job-id.txt
      train_early_liquid.py train-early-liquid-rosie.sbatch liquid_specialists_data.py
      train_liquid_specialists.py train_epoch_augmentation.py train_mixed_detector.py
      detector_photometric.py dino_liquid.py resnet-smoke/model-check.json resnet-smoke/training.csv)
    for arm in yolo11n-960 yolo11s-640 fasterrcnn-resnet50; do
      for name in best.pt model.json; do
        [[ -s "models/$arm/$name" ]] || { echo "Missing models/$arm/$name"; exit 1; }
        files+=("models/$arm/$name")
      done
      if [[ "$arm" == fasterrcnn-resnet50 ]]; then
        files+=("models/$arm/training.csv" "models/$arm/model-check.json")
      else
        files+=("models/$arm/train/results.csv" "models/$arm/train/args.yaml")
      fi
    done
    for name in slurm-"$early_job_id"_*.log; do [[ ! -f "$name" ]] || files+=("$name"); done
    tar -czf completed-artifacts.tar.gz.tmp "${files[@]}"
    mv completed-artifacts.tar.gz.tmp completed-artifacts.tar.gz
    sha256sum completed-artifacts.tar.gz > completed-artifacts.tar.gz.sha256
    echo 'All three fits completed; bundle and checksum ready.'
    exit 0
  fi
  sleep 15
done
echo 'Collector reached 135-minute limit; GPU jobs are unchanged.'
exit 1
