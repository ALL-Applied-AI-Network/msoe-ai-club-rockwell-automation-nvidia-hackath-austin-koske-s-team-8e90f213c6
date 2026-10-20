#!/usr/bin/env bash
# Small CPU-only collector; the GPU jobs remain under Slurm.
set -euo pipefail
[[ $# == 2 && "$2" =~ ^[0-9]+$ ]] || { echo 'Usage: collect-epoch-artifacts-rosie.sh EXPERIMENT ARRAY_JOB_ID'; exit 2; }
experiment=$(realpath "$1")
job=$2
cd "$experiment"
[[ ! -e completed-artifacts.tar.gz ]] || { echo 'Artifact bundle already exists'; exit 2; }
for attempt in $(seq 1 180); do
  sacct -X -j "$job" -n -P -o JobID,State,ExitCode,Elapsed,NodeList > scheduler-status.txt
  if grep -Eq '\|(FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL)' scheduler-status.txt; then
    cat scheduler-status.txt
    echo 'A training job failed; inspect its log. No successful-results bundle made.'
    exit 1
  fi
  completed=$(grep -c '|COMPLETED|0:0|' scheduler-status.txt || true)
  if [[ "$completed" == 3 ]]; then
    files=(experiment-plan.json preparation.json scheduler-status.txt
      train_epoch_augmentation.py detector_photometric.py train_mixed_detector.py
      train-epoch-augmentation-rosie.sbatch)
    for arm in plain12 plain50 augmented50; do
      for name in best.pt detector.onnx model.json augmentation-execution.json train/results.csv train/args.yaml; do
        [[ -f "models/$arm/$name" ]] || { echo "Missing models/$arm/$name"; exit 1; }
        files+=("models/$arm/$name")
      done
    done
    files+=(models/augmented50/augmentation-preview)
    for name in slurm-"$job"_*.log; do [[ ! -f "$name" ]] || files+=("$name"); done
    tar -czf completed-artifacts.tar.gz "${files[@]}"
    sha256sum completed-artifacts.tar.gz > completed-artifacts.tar.gz.sha256
    echo 'All three jobs completed; artifact bundle ready.'
    exit 0
  fi
  sleep 15
done
echo 'Collector reached its 45-minute limit; training jobs were not altered.'
exit 1
