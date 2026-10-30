"""Launch isolated Windows inference and Isaac processes for a local live pilot."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from twin_contract import default_plan,validate_plan


def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def main(a):
    if os.name!='nt':raise RuntimeError('Run with the existing native Windows vision Python, not WSL Python')
    if a.out.exists():raise ValueError('Use a fresh run directory')
    if not a.isaac_python.is_file():raise FileNotFoundError(a.isaac_python)
    if not 1<=a.spp<=32 or not 1<=a.subframes<=16 or (a.limit_frames is not None and a.limit_frames<1):
        raise ValueError('Use spp 1..32, subframes 1..16 and a positive frame limit')
    a.out.mkdir(parents=True)
    here=Path(__file__).resolve().parent;repo=here.parents[1]
    tools=Path.home()/'ai-club-tools'
    plan=json.loads(a.plan.read_text(encoding='utf-8')) if a.plan else default_plan()
    validate_plan(plan)
    write(a.out/'plan.json',plan)
    (a.out/'helpers').mkdir()
    helpers=['twin_contract.py','live_twin_observer.py','live_twin_isaac.py','run_live_twin.py','twin_diagnostics.py']
    for name in helpers:shutil.copy2(here/name,a.out/'helpers'/name)
    dependencies=[here/n for n in ['benchmark_liquid_pipeline.py','train_early_liquid.py',
        'compare_pool_prompts.py','pool_prompt_policy.py','render_liquid_pipeline.py',
        'stage2_camera_reference.py','stage2_guard.py','stage2_policy.py']]
    dependencies += list((repo/'replay').glob('*.py'))
    dependencies += [repo/'setup/isaac'/n for n in ['liquid_geometry.py','box_annotations.py']]
    dependency_hashes={str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies}
    for path in dependencies:
        target=a.out/'helpers/dependencies'/path.relative_to(repo)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    write(a.out/'launch.json',dict(mode='live_lockstep_software_in_loop',
        executable=sys.executable,isaac_launcher=str(a.isaac_python),spp=a.spp,subframes=a.subframes,
        limit_frames=a.limit_frames,gui=a.gui,hardware_actions=0,
        helpers_sha256={n:hashlib.sha256((here/n).read_bytes()).hexdigest() for n in helpers},
        dependency_sha256=dependency_hashes))
    observer=[sys.executable,'-u',str(here/'live_twin_observer.py'),'--out',str(a.out/'observer'),
        '--baseline',str(tools/'results/crop-clahe-detector-20261019/baseline-best.pt'),
        '--experiment',str(tools/'results/early-liquid-20261020'),
        '--sam',str(tools/'results/prompted-liquid-assets-20261020/sam2.1_t.pt'),
        '--calibration',str(tools/'results/stage2-refinement-20261020/camera-reference-v2/calibration.json')]
    if plan.get('diagnostic_suite'):observer+=['--diagnostic-suite']
    env=dict(os.environ,PYTHONUNBUFFERED='1')
    with (a.out/'observer.log').open('w') as observerlog, (a.out/'isaac.log').open('w') as isaaclog:
        child=subprocess.Popen(observer,stdout=observerlog,stderr=subprocess.STDOUT,cwd=repo,env=env)
        simulator=None
        try:
            start=time.monotonic()
            while not (a.out/'observer/ready.json').exists():
                if child.poll() is not None:raise RuntimeError('Observer startup failed; inspect observer.log')
                if time.monotonic()-start>180:raise TimeoutError('Observer model loading timed out')
                time.sleep(.25)
            simulation=[str(a.isaac_python),'-u',str(here/'live_twin_isaac.py'),
                '--out',str(a.out/'simulation'),'--plan',str(a.out/'plan.json'),
                '--assets',str(repo/'datasets/mixing_twin/assets/csi_digital_twin_v1'),
                '--endpoint',str(a.out/'observer/ready.json'),'--spp',str(a.spp),'--subframes',str(a.subframes),'--no-ros-env']
            if a.limit_frames:simulation+=['--limit-frames',str(a.limit_frames)]
            if a.gui:simulation+=['--gui']
            simulator=subprocess.Popen(['cmd.exe','/d','/c',*simulation],stdout=isaaclog,stderr=subprocess.STDOUT,cwd=repo,env=env)
            write(a.out/'pids.json',dict(observer=child.pid,isaac_launcher=simulator.pid))
            result=simulator.wait(timeout=a.timeout)
            if result:raise RuntimeError(f'Isaac exited {result}; inspect isaac.log')
            if (a.out/'simulation/failed.json').exists():raise RuntimeError('Isaac recorded a simulation failure; inspect simulation/failed.json')
            if child.wait(timeout=30):raise RuntimeError('Observer failed; inspect observer.log')
            if not (a.out/'simulation/completed.json').exists() or not (a.out/'observer/completed.json').exists():
                raise RuntimeError('Child completion marker missing')
            write(a.out/'completed.json',dict(status='completed',mode='live_lockstep',
                source_assets_unchanged=True,hardware_actions=0,main_branch_changed=False))
        except Exception as error:
            write(a.out/'failed.json',dict(type=type(error).__name__,message=str(error)))
            # Terminate only descendants started by this launcher; preserve logs.
            for process in [simulator,child]:
                if process is not None and process.poll() is None:
                    subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],capture_output=True)
            raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--plan',type=Path,help='Optional versioned synthetic experiment plan')
    p.add_argument('--isaac-python',type=Path,default=Path('C:/Apps/IsaacSim/isaacsim/_build/windows-x86_64/release/python.bat'))
    p.add_argument('--limit-frames',type=int)
    p.add_argument('--spp',type=int,default=8);p.add_argument('--subframes',type=int,default=2)
    p.add_argument('--timeout',type=float,default=3600.)
    p.add_argument('--gui',action='store_true')
    main(p.parse_args())
