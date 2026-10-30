"""Verify a complete transfer, unpack once, then run one detached GB10 fit suite."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import time


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve()
    if (root/'launch.json').exists() or (root/'run').exists():raise ValueError('Inspect existing run; duplicate refused')
    transfer=json.loads((root/'transfer.json').read_text());archive=root/'input.tar.gz'
    if archive.stat().st_size!=transfer['bytes'] or hashlib.sha256(archive.read_bytes()).hexdigest()!=transfer['archive_sha256']:raise ValueError('Input transfer incomplete or corrupt')
    run=root/'run';run.mkdir()
    with tarfile.open(archive) as tar:tar.extractall(run,filter='data')
    if hashlib.sha256((run/'experiment-plan.json').read_bytes()).hexdigest()!=transfer['plan_sha256']:raise ValueError('Extracted plan differs')
    python=Path.home()/'ai-club-tools/experiments/gb10-station-benchmark-20261022/vision-env/bin/python'
    with (root/'train.log').open('x') as log:
        process=subprocess.Popen([str(python),'-u',str(run/'train_dinov3_liquid_mask.py'),'--root',str(run)],cwd=run,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    result=dict(pid=process.pid,launched=time.time(),transfer_sha256=transfer['archive_sha256'],GPU='GB10',fits_requested=2,training_real_runs=[])
    (root/'launch.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
