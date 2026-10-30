"""GB10 queued one-factor optical review; no real fitting or automatic approval."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')


def main(a):
    import fcntl
    import os
    root=a.out;root.mkdir(parents=True,exist_ok=False);(root/'helpers').mkdir()
    base=Path.home()/'ai-club-tools/experiments/gb10-station-benchmark-20261022/bundle-v1'
    vision=a.bundle/'repo/setup/vision';isaac=base/'repo/setup/isaac'
    sources=[a.renderer,vision/'live_twin_isaac.py',vision/'twin_contract.py',isaac/'liquid_geometry.py',isaac/'box_annotations.py']
    for p in sources:shutil.copy2(p,root/'helpers'/p.name)
    controls=dict(light_scale=1.,glass_roughness=.06,label_brightness=1.,camera_dx_px=0.,camera_dy_px=0.,
        color=[.12,.55,.82],absorption_depth_m=.025,phase=0.)
    variations=[('control',{}),('original-glass',dict(glass_profile='original')),
        ('roughness-025',dict(glass_roughness=.025)),('roughness-12',dict(glass_roughness=.12)),
        ('absorption-012',dict(absorption_depth_m=.012)),('absorption-05',dict(absorption_depth_m=.05)),
        ('light-07',dict(light_scale=.7)),('light-13',dict(light_scale=1.3)),
        ('pale',dict(color=[.8,.86,.52])),('clear-review-only',dict(color=[.98,.995,.97])),
        ('paper-07',dict(label_brightness=.7)),('camera-x5',dict(camera_dx_px=5.))]
    cases=[]
    for group,(label,changes) in enumerate(variations):
        nuisance=dict(controls,**changes)
        for state,level,radius in [('empty',0.,.997),('early',.004,.55),('established',.30,.997)]:
            cases.append(dict(id=f'{group:02d}-{label}-{state}',group=group,split='review_only',state=state,
                level=level,stream='off',pool_radius_fraction=radius,nuisance=nuisance,axis=label))
    plan=dict(schema_version=1,cases=cases,assets=str(base/'repo/datasets/mixing_twin/assets/csi_digital_twin_v1'),
        spp=16,subframes=4,helpers={p.name:sha(p) for p in sources},
        purpose='One factor at a time against identical control. Review only; no parameter optimized on real evaluation.',
        physical_calibration=False,training_real_runs=[],selection_real_runs=[],run5_used=False)
    write(root/'render-plan.json',plan);write(root/'preparation.json',dict(plan_sha256=sha(root/'render-plan.json'),helpers=plan['helpers']))
    write(root/'queued.json',dict(pid=os.getpid(),queued=time.time(),GPU='GB10'))
    lock=(Path.home()/'ai-club-tools/spark-gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX)
    write(root/'running.json',dict(pid=os.getpid(),started=time.time()))
    env=dict(os.environ,LD_PRELOAD='/lib/aarch64-linux-gnu/libgomp.so.1',MPLBACKEND='Agg',OMNI_KIT_ACCEPT_EULA='YES')
    python=Path.home()/'ai-club-tools/envs/isaac-6.0.0.1/bin/python'
    command=['setarch','aarch64','-R',str(python),'-u',str(root/'helpers'/a.renderer.name),'render','--out',str(root),'--no-ros-env']
    with (root/'isaac.log').open('w') as log:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=env,timeout=900)
    if result.returncode:raise RuntimeError('Optical review renderer failed: '+str(result.returncode))
    write(root/'completed.json',dict(images=36,physical_calibration=False,review_only=True,run5_used=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['bundle','renderer','out']:p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
