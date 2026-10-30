"""One detached GB10 render/fit run; no retries, app changes or real fitting."""
import argparse
import os
from pathlib import Path
import subprocess
import time

from dataset1_fiveclass import read,sha,write


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve()
    if any((root/name).exists() for name in ('job-running.json','rendered-v1','fitted-v1')):raise ValueError('Inspect existing process/artifacts; no duplicate run')
    smoke=read(root/'smoke-v1/completed.json');plan=read(root/'experiment-plan.json')
    if smoke['plan_sha256']!=sha(root/'experiment-plan.json') or plan['renderer_sha256']!=sha(root/'dataset1_fiveclass.py'):raise ValueError('Smoke/frozen helper differs')
    write(root/'job-running.json',dict(pid=os.getpid(),started=time.time(),plan_sha256=sha(root/'experiment-plan.json'),
        runner_sha256=sha(__file__),trainer_sha256=sha(root/'train_dataset1_fiveclass.py'),gpu='GB10',training_real_runs=[]))
    isaac=Path.home()/'ai-club-tools/envs/isaac-6.0.0.1/bin/python';vision=Path.home()/'ai-club-tools/experiments/gb10-station-benchmark-20261022/vision-env/bin/python'
    env=dict(os.environ,LD_PRELOAD='/lib/aarch64-linux-gnu/libgomp.so.1',MPLBACKEND='Agg',OMNI_KIT_ACCEPT_EULA='YES')
    try:
        for name,command,timeout in [('render',[str(isaac),'-u',str(root/'dataset1_fiveclass.py'),'render','--root',str(root),'--no-ros-env'],1800),
            ('train',[str(vision),'-u',str(root/'train_dataset1_fiveclass.py'),'--root',str(root)],1200)]:
            print('START',name,flush=True)
            with (root/(name+'-v1.log')).open('x') as log:
                result=subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
            if result.returncode:raise RuntimeError(f'{name} exit {result.returncode}; inspect preserved log')
            print('COMPLETE',name,flush=True)
        write(root/'completed.json',dict(completed=time.time(),gpu='GB10',training_real_runs=[],run5_used=False,
            plan_sha256=sha(root/'experiment-plan.json'),comparison_sha256=sha(root/'fitted-v1/comparison.json')))
    except BaseException as error:write(root/'failed.json',dict(type=type(error).__name__,error=str(error),finished=time.time()));raise
