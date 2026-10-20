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
