"""Wait for the existing detail job, then run one isolated visibility experiment."""
import argparse
import os
from pathlib import Path
import subprocess
import time

from dataset1_fiveclass import sha,write


def run(root,dependency):
    if any((root/n).exists() for n in ('running.json','rendered-v1','completed.json')):raise ValueError('Inspect existing job; duplicate refused')
    write(root/'running.json',dict(pid=os.getpid(),started=time.time(),dependency=str(dependency),helper_sha256=sha(__file__)))
    try:
        deadline=time.monotonic()+1800
        while not (dependency/'completed.json').exists():
            if (dependency/'failed.json').exists():raise RuntimeError('Dependency failed; no additional GPU work started')
            if time.monotonic()>deadline:raise TimeoutError('Dependency wait expired')
            time.sleep(5)
        isaac=Path.home()/'ai-club-tools/envs/isaac-6.0.0.1/bin/python'
        env=dict(os.environ,LD_PRELOAD='/lib/aarch64-linux-gnu/libgomp.so.1',MPLBACKEND='Agg',OMNI_KIT_ACCEPT_EULA='YES')
        with (root/'render.log').open('x') as log:
            result=subprocess.run([str(isaac),'-u',str(root/'visibility_renderer.py'),'render','--root',str(root),'--no-ros-env'],
                cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=900)
        if result.returncode:raise RuntimeError('Visibility rendering failed; inspect preserved log')
        write(root/'completed.json',dict(finished=time.time(),gpu='GB10',render_manifest_sha256=sha(root/'rendered-v1/manifest.jsonl'),
            plan_sha256=sha(root/'experiment-plan.json'),classifiers_fitted=0))
    except BaseException as error:
        write(root/'failed.json',dict(type=type(error).__name__,error=str(error),utc=time.time()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--dependency',type=Path,required=True)
    a=p.parse_args();run(a.root.resolve(),a.dependency.resolve())
