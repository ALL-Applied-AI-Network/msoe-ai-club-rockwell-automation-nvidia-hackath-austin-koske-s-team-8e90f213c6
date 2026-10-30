"""Fresh-scene confirmation of frozen cap weights and gain correction; no fitting."""
import argparse
import copy
import fcntl
from io import BytesIO
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import time

from dataset1_fiveclass import families,read,sha,write
from experiment_dataset1_cap_state import cap_data,cap_model,evaluate,metrics
from fixture_reference_preprocess import estimate_scene_gain


def run(root,parent,modelroot):
    root.mkdir(exist_ok=False);(root/'input').symlink_to((parent/'input').resolve(),target_is_directory=True)
    shutil.copy2(parent/'dataset1_fiveclass.py',root/'dataset1_fiveclass.py')
    frozen=read(modelroot/'fitted-v1/frozen.json');plan=copy.deepcopy(read(parent/'experiment-plan.json'))
    plan['families']=families(973133,(0,0,6));rng=random.Random(883219)
    for i,row in enumerate(plan['families']):
        row.update(id=f'confirmation-{i:03d}',group=40000+i,density='dense' if i<3 else 'sparse')
        if i>=3:
            row['labels']={slot:'pass' for slot in row['labels']}
            for fixture in ('G','S'):row['labels'][fixture+f'{rng.randint(1,30):02d}']=['missing_cap','crooked_cap','damaged_vial'][i%3]
    plan.update(groups=dict(train=0,val=0,test=6),seed=973133,purpose='Fresh post-freeze cap/gain confirmation; no fitting')
    write(root/'experiment-plan.json',plan)
    cases=['unchanged','reference_gain_0.83','reference_gain_1.17','reference_noise_1dl','reference_noise_3dl','current_noise_1dl','reference_jpeg90']
    write(root/'confirmation-plan.json',dict(cases=cases,noise_seed=11931,noise='Independent zero-mean digital Gaussian; not calibrated to a physical sensor',
        checkpoint_sha256=frozen['checkpoint_sha256'],thresholds=frozen['thresholds'],gain_preprocessor_sha256=sha(Path(__file__).with_name('fixture_reference_preprocess.py')),
        helper_sha256=sha(__file__),test_families=6,training=False,created_before_test_render=time.time(),body_condition='unassessed',
        source_geometry_limit='Canonical0–30deg body poses; original lying-S geometry not covered'))
    isaac=Path.home()/'ai-club-tools/envs/isaac-6.0.0.1/bin/python'
    env=dict(os.environ,LD_PRELOAD='/lib/aarch64-linux-gnu/libgomp.so.1',MPLBACKEND='Agg',OMNI_KIT_ACCEPT_EULA='YES')
    with (root/'render.log').open('x') as log:
        result=subprocess.run([str(isaac),'-u',str(root/'dataset1_fiveclass.py'),'render','--root',str(root),'--no-ros-env'],env=env,cwd=root,stdout=log,stderr=subprocess.STDOUT,timeout=600)
    if result.returncode:raise RuntimeError('Confirmation render failed; preserve log')
    import numpy as np
    import torch
    from PIL import Image
    lock=(Path.home()/'ai-club-tools/spark-gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(4);folder=root/'rendered-v1';rows=[json.loads(s) for s in (folder/'manifest.jsonl').read_text().splitlines()]
    data=cap_data(rows,'test',folder);model=cap_model().cuda();checkpoint=modelroot/'fitted-v1/best.pt'
    if sha(checkpoint)!=frozen['checkpoint_sha256']:raise ValueError('Frozen checkpoint differs')
    model.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=True)['state_dict']);results={}
    for name in cases:
        variant=dict(data);variant['current']=data['current'].clone();variant['golden']=data['golden'].clone();torch.manual_seed(11931)
        if name.startswith('reference_gain'):variant['golden']=(variant['golden']*float(name.rsplit('_',1)[1])).clamp(0,1)
        if 'noise' in name:
            key='golden' if name.startswith('reference') else 'current';sigma=(3 if '3dl' in name else 1)/255
            variant[key]=(variant[key]+torch.randn_like(variant[key])*sigma).clamp(0,1)
        if name=='reference_jpeg90':
            for i in range(len(variant['golden'])):
                rgb=(variant['golden'][i].cpu().numpy().transpose(1,2,0)*255).round().astype('uint8');buffer=BytesIO()
                Image.fromarray(rgb).save(buffer,format='JPEG',quality=90);buffer.seek(0)
                decoded=np.asarray(Image.open(buffer).convert('RGB'),dtype=np.float32)/255
                variant['golden'][i]=torch.from_numpy(decoded.transpose(2,0,1).copy()).cuda()
        estimates=[]
        for start in range(0,len(data['identifiers']),60):
            estimate=estimate_scene_gain(variant['current'][start:start+60].cpu().numpy(),variant['golden'][start:start+60].cpu().numpy())
            if not estimate['available']:raise ValueError('Gain correction unavailable; no success substituted')
            variant['golden'][start:start+60]=(variant['golden'][start:start+60]*estimate['gain']).clamp(0,1);estimates.append(estimate)
        records=evaluate(model,variant);normal=[r for r in records if r['variant']=='golden_as_current']
        results[name]=dict(metrics=metrics(records),normal_exact=metrics(normal),gain_estimates=estimates,
            policies={key:metrics(records,t)['cap_warning'] for key,t in frozen['thresholds'].items()},
            normal_policies={key:metrics(normal,t)['cap_warning'] for key,t in frozen['thresholds'].items()})
        write(root/(name+'-predictions.json'),records)
    write(root/'results.json',dict(cases=results,targets_per_case=len(data['identifiers']),source_families=6,
        model_sha256=frozen['checkpoint_sha256'],real_accuracy=None,training=False,body_condition='unassessed',
        alignment_robustness_established=False,source_geometry_limit=read(root/'confirmation-plan.json')['source_geometry_limit']))
    write(root/'completed.json',dict(finished=time.time(),results_sha256=sha(root/'results.json'),gpu='GB10',training=False,plan_sha256=sha(root/'confirmation-plan.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','parent','modelroot'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();run(a.root.resolve(),a.parent.resolve(),a.modelroot.resolve())
