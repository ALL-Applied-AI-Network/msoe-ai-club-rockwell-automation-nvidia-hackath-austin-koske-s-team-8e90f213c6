"""Finish the inspected render recovery, then fit once; no resubmission loop."""
import argparse
import os
from pathlib import Path
import subprocess
import time

from dataset1_pilot import read, sha, write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve()
    if (root/'fitted').exists() or (root/'recovery-running.json').exists():
        raise ValueError('Inspect existing fit/recovery; do not duplicate')
    write(root/'recovery-running.json',dict(pid=os.getpid(),started=time.time(),helper_sha256=sha(__file__),
        reason='Initial Kit startup stalled before any image; direct renderer recovery already inspected.',
        failed_launch_record_sha256=sha(root/'failed.json'),training_repeated=False))
    deadline=time.monotonic()+600
    while not (root/'rendered/completed.json').exists():
        if (root/'rendered/failed.json').exists():
            raise RuntimeError('Render recovery failed; inspect retained log')
        if time.monotonic()>deadline:
            raise TimeoutError('No complete render after waiting; no training submitted')
        time.sleep(2)
    python=Path.home()/'ai-club-tools/experiments/gb10-station-benchmark-20261022/vision-env/bin/python'
    with (root/'train-recovery-v1.log').open('x') as log:
        result=subprocess.run([str(python),'-u',str(root/'train_dataset1_pilot.py'),'--root',str(root)],
            cwd=root,stdout=log,stderr=subprocess.STDOUT,timeout=600)
    if result.returncode:
        raise RuntimeError('Fit failed; no repeat attempted')
    write(root/'completed.json',dict(completed=time.time(),gpu='GB10',real_data_used=False,
        comparison_sha256=sha(root/'fitted/comparison.json'),render_manifest_sha256=sha(root/'rendered/manifest.jsonl'),
        prior_startup_failure_preserved=True,recovery_complete=True))
    print('COMPLETED Dataset1 synthetic pilot',flush=True)


if __name__=='__main__':
    main()
