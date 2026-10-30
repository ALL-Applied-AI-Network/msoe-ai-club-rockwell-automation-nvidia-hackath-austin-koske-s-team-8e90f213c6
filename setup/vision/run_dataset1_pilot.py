"""Detached GB10-only render/train runner; preserves failures and refuses repeats."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from dataset1_pilot import read, sha, write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve()
    if (root/'job-running.json').exists() or (root/'rendered').exists() or (root/'fitted').exists():
        raise ValueError('Inspect prior process/artifacts; do not duplicate this experiment')
    plan=read(root/'experiment-plan.json')
    if sha(root/'dataset1_pilot.py')!=plan['helper_sha256']:
        raise ValueError('Pinned rendering helper changed')
    for name,expected in read(root/'input/input-manifest.json').items():
        if sha(root/'input'/name)!=expected:
            raise ValueError('Original input mismatch: '+name)
    write(root/'job-running.json',dict(pid=os.getpid(),started=time.time(),host=os.uname().nodename,
        plan_sha256=sha(root/'experiment-plan.json'),renderer_sha256=sha(root/'dataset1_pilot.py'),
        trainer_sha256=sha(root/'train_dataset1_pilot.py'),runner_sha256=sha(__file__),local_gpu_used=False))
    env=dict(os.environ,LD_PRELOAD='/lib/aarch64-linux-gnu/libgomp.so.1',MPLBACKEND='Agg',OMNI_KIT_ACCEPT_EULA='YES')
    isaac=Path.home()/'ai-club-tools/envs/isaac-6.0.0.1/bin/python'
    vision=Path.home()/'ai-club-tools/experiments/gb10-station-benchmark-20261022/vision-env/bin/python'
    stages=[('render',['setarch','aarch64','-R',str(isaac),'-u',str(root/'dataset1_pilot.py'),'render','--root',str(root),'--no-ros-env'],900),
            ('train',[str(vision),'-u',str(root/'train_dataset1_pilot.py'),'--root',str(root)],600)]
    try:
        for name,command,timeout in stages:
            print('START',name,flush=True)
            with (root/(name+'.log')).open('x') as log:
                result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=env,timeout=timeout,cwd=root)
            if result.returncode:
                raise RuntimeError(f'{name} failed with exit {result.returncode}; original log retained')
            print('COMPLETE',name,flush=True)
        write(root/'completed.json',dict(completed=time.time(),gpu='GB10',real_data_used=False,
            comparison_sha256=sha(root/'fitted/comparison.json'),render_manifest_sha256=sha(root/'rendered/manifest.jsonl')))
    except BaseException as error:
        write(root/'failed.json',dict(type=type(error).__name__,error=str(error),finished=time.time()))
        raise


if __name__=='__main__':
    main()
