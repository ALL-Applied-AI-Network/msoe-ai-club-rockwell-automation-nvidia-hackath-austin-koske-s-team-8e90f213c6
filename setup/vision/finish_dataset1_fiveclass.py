"""Wait for the existing fit, then evaluate once and package exact evidence."""
import argparse
import os
from pathlib import Path
import subprocess
import tarfile
import time
from dataset1_fiveclass import read,sha,write


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve()
    if (root/'followup-running.json').exists():raise ValueError('Inspect existing follow-up before doing anything')
    write(root/'followup-running.json',dict(pid=os.getpid(),started=time.time(),helper_sha256=sha(__file__),training_resubmitted=False))
    isaac=Path.home()/'ai-club-tools/envs/isaac-6.0.0.1/bin/python';vision=Path.home()/'ai-club-tools/experiments/gb10-station-benchmark-20261022/vision-env/bin/python'
    try:
        deadline=time.monotonic()+4200
        while not (root/'completed.json').exists():
            if (root/'failed.json').exists():raise RuntimeError('Parent run failed; preserve and inspect it, no replacement submitted')
            if time.monotonic()>deadline:raise TimeoutError('Parent still incomplete; no job resubmitted')
            time.sleep(5)
        completed=read(root/'completed.json')
        if completed['comparison_sha256']!=sha(root/'fitted-v1/comparison.json'):raise ValueError('Frozen result differs')
        env=dict(os.environ,LD_PRELOAD='/lib/aarch64-linux-gnu/libgomp.so.1',MPLBACKEND='Agg',OMNI_KIT_ACCEPT_EULA='YES')
        scenes=root/'placement-scenes-v1'
        stages=[('real-inference',[str(vision),'-u',str(root/'dataset1_fiveclass_real.py'),'infer','--root',str(root),'--batch',str(root/'real-batch-v1'),'--out',str(root/'real-batch-v1/inference')],300),
            ('placement-render',[str(isaac),'-u',str(scenes/'dataset1_fiveclass.py'),'render','--root',str(scenes),'--no-ros-env'],300),
            ('placement-inference',[str(vision),'-u',str(root/'evaluate_fixture_placement_scenes.py'),'--parent',str(root),'--scenes',str(scenes),'--out',str(root/'placement-inference-v1')],300)]
        for name,command,timeout in stages:
            print('START',name,flush=True)
            with (root/(name+'.log')).open('x') as stream:result=subprocess.run(command,cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=timeout)
            if result.returncode:raise RuntimeError(f'{name} failed exit {result.returncode}; no retry')
            print('COMPLETE',name,flush=True)
        files=[p for p in root.iterdir() if p.is_file() and p.suffix in ('.py','.json','.log')]
        for relative in ('smoke-v1','rendered-v1','fitted-v1','real-batch-v1','placement-inference-v1'):
            files.extend(p for p in (root/relative).rglob('*') if p.is_file())
        files.extend(p for p in scenes.rglob('*') if p.is_file() and 'input' not in p.relative_to(scenes).parts)
        write(root/'artifact-manifest.json',{p.relative_to(root).as_posix():sha(p) for p in files})
        files.append(root/'artifact-manifest.json');archive=root/'completed-artifacts-v1.tar.gz'
        with tarfile.open(archive,'x:gz') as tar:
            for path in files:tar.add(path,arcname=path.relative_to(root).as_posix(),recursive=False)
        (root/'completed-artifacts-v1.tar.gz.sha256').write_text(sha(archive)+'  completed-artifacts-v1.tar.gz\n')
        write(root/'followup-completed.json',dict(completed=time.time(),archive_sha256=sha(archive),artifact_manifest_sha256=sha(root/'artifact-manifest.json'),training_resubmitted=False))
        print('COMPLETED five-class/two-fixture experiment and evaluations',flush=True)
    except BaseException as error:write(root/'followup-failed.json',dict(type=type(error).__name__,error=str(error),finished=time.time()));raise
