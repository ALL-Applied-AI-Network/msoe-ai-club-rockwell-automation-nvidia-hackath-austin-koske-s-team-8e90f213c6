#!/usr/bin/env bash
set -euo pipefail
[[ $# == 2 && "$2" =~ ^[0-9]+$ ]] || { echo 'Pass EXPERIMENT ARRAY_JOB_ID'; exit 2; }
cd "$(realpath "$1")"
job=$2
[[ ! -e completed-artifacts.tar.gz ]] || { echo 'Bundle exists'; exit 2; }
for attempt in $(seq 1 540); do
  sacct -X -j "$job" -n -P -o JobID,State,ExitCode,Elapsed,NodeList > scheduler-status.txt
  if grep -Eq '\|(FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL)' scheduler-status.txt; then
    cat scheduler-status.txt;echo 'A job failed; inspect logs. No successful-results bundle made.';exit 1
  fi
  completed=$(grep -c '|COMPLETED|0:0|' scheduler-status.txt || true)
  if [[ "$completed" == 4 ]]; then
    files=(experiment-plan.json preparation.json scheduler-status.txt
      train_liquid_specialists.py liquid_specialists_data.py dino_liquid.py train-liquid-specialists-rosie.sbatch
      train_epoch_augmentation.py train_mixed_detector.py detector_photometric.py
      dino-smoke/training.csv dino-smoke/frozen-backbone-check.json)
    for arm in yolo-all yolo-focused rtdetr-all dinov3-all; do
      for name in best.pt model.json; do
        [[ -f "models/$arm/$name" ]] || { echo "Missing models/$arm/$name";exit 1; }
        files+=("models/$arm/$name")
      done
      if [[ "$arm" == dinov3-all ]]; then
        files+=("models/$arm/training.csv" "models/$arm/frozen-backbone-check.json")
      else
        files+=("models/$arm/train/results.csv" "models/$arm/train/args.yaml")
      fi
    done
    for name in slurm-"$job"_*.log; do [[ ! -f "$name" ]] || files+=("$name");done
    tar -czf completed-artifacts.tar.gz "${files[@]}"
    sha256sum completed-artifacts.tar.gz > completed-artifacts.tar.gz.sha256
    echo 'All four fits completed; bundle ready.';exit 0
  fi
  sleep 15
done
echo 'Collector reached 135-minute limit; GPU jobs were not changed.'
exit 1
