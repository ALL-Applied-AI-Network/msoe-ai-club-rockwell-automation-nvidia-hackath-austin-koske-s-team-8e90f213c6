"""Serialize the normal-state follow-up behind existing GB10 visibility work."""
import argparse
from pathlib import Path
import os
import time

from dataset1_fiveclass import sha,write
from experiment_dataset1_normals import run


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','parent','dependency'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if (a.root/'queued.json').exists():raise ValueError('Existing queue entry; inspect rather than repeat')
    write(a.root/'queued.json',dict(pid=os.getpid(),utc=time.time(),dependency=str(a.dependency),helper_sha256=sha(__file__)))
    deadline=time.monotonic()+1800
    while not (a.dependency/'inference-v1/completed.json').exists():
        if (a.dependency/'failed.json').exists():raise RuntimeError('Visibility run failed; GPU follow-up not started')
        if time.monotonic()>deadline:raise TimeoutError('Queue expired; inspect existing work')
        time.sleep(3)
    run(a.root.resolve(),a.parent.resolve())
