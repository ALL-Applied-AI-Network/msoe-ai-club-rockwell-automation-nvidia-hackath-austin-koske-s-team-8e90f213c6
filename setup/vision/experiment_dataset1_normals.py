"""Normal-state coverage follow-up to the detail/reference ablation, synthetic-only; GB10 execution only.

Old test errors motivated this exploratory experiment. Its new test families are
rendered only after every new checkpoint and threshold is frozen.
"""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from dataset1_fiveclass import CLASSES, families, read, sha, write
from train_dataset1_fiveclass import load, summarize, validate


def load_with_golden_hashes(rows,split,render):
    data=load(rows,split,render);by_scene={r['id']:r for r in rows}
    for record in data['identifiers']:record['golden_source_sha256']=by_scene[record['scene']]['golden_sha256']
    return data


def with_normals(data):
    import torch
    result=dict(data)
    result['current']=torch.cat([data['current'],data['golden']],0)
    result['golden']=torch.cat([data['golden'],data['golden']],0)
    result['labels']=torch.cat([data['labels'],torch.zeros_like(data['labels'])],0)
    result['identifiers']=[dict(r,variant='inspection') for r in data['identifiers']]+[dict(r,scene=r['scene']+'-normal',source_sha256=r['golden_source_sha256'],variant='golden_as_current') for r in data['identifiers']]
    return result


def model_for(channels):
    import torch
    from torch import nn
    import torch.nn.functional as F

    def branch(pool):
        return nn.Sequential(nn.Conv2d(channels,24,3,1,1),nn.GroupNorm(6,24),nn.SiLU(),
            nn.Conv2d(24,48,3,2,1),nn.GroupNorm(6,48),nn.SiLU(),
            nn.Conv2d(48,96,3,2,1),nn.GroupNorm(6,96),nn.SiLU(),
            nn.AdaptiveAvgPool2d(pool),nn.Flatten())

    class DetailNet(nn.Module):
        def __init__(self):
            super().__init__();self.local=branch((8,8));self.context=branch((4,4))
            self.head=nn.Sequential(nn.Linear(96*80,256),nn.SiLU(),nn.Dropout(.15),nn.Linear(256,5))

        def forward(self,x):
            detail=F.interpolate(x[:,:,16:80,16:80],size=(96,96),mode='bilinear',align_corners=False)
            return self.head(torch.cat([self.local(detail),self.context(x)],1))
    return DetailNet()


def tensor_for(data,mode,indices,augment=False):
    import torch
    current=data['current'][indices];golden=data['golden'][indices]
    heat=data['target'].expand(len(indices),-1,-1,-1)
    if augment:
        gain=torch.empty(len(indices),1,1,1,device='cuda').uniform_(.85,1.15)
        bias=torch.empty(len(indices),1,1,1,device='cuda').uniform_(-.025,.025)
        current=(current*gain+bias).clamp(0,1);golden=(golden*gain+bias).clamp(0,1)
    channels=[current,heat] if mode=='detail_rgb' else [current,golden,(current-golden).abs(),heat]
    x=torch.cat(channels,1)
    if augment:
        dx,dy=torch.randint(-2,3,(2,)).tolist();x=torch.roll(x,(dx,dy),(-1,-2))
    return x


def evaluate(model,data,mode):
    import torch
    model.eval();records=[]
    with torch.inference_mode():
        for start in range(0,len(data['identifiers']),128):
            ix=torch.arange(start,min(start+128,len(data['identifiers'])),device='cuda')
            probabilities=model(tensor_for(data,mode,ix)).softmax(1).cpu().tolist()
            truth=data['labels'][ix].cpu().tolist()
            for i,p in enumerate(probabilities):
                records.append(dict(**data['identifiers'][start+i],truth=truth[i],prediction=max(range(5),key=lambda j:p[j]),probabilities=p))
    return records


def train(root,parent):
    import torch
    import torch.nn.functional as F
    lock=(Path.home()/'ai-club-tools/spark-gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out=root/'fitted-v1';out.mkdir(exist_ok=False)
    render=parent/'rendered-v1';rows=[json.loads(s) for s in (render/'manifest.jsonl').read_text().splitlines()]
    validate(rows)
    plan=dict(schema='dataset1-detail/v1',parent_manifest_sha256=sha(render/'manifest.jsonl'),
        trainer_sha256=sha(__file__),epochs=45,batch_size=128,seed=19,modes=['detail_rgb','detail_paired'],
        architecture='Two branches: center64 upsampled96 pool8x8, full96 pool4x4; Conv24/48/96, Linear256/5',
        paired_channels='currentRGB, matched goldenRGB, abs RGB difference, aperture Gaussian',
        checkpoint_selection='Maximum original synthetic development macro recall, earliest tie',
        threshold_selection='Minimum development occupied-defect 1-P(pass)',
        fresh_test_seed=502113,fresh_test_families=12,old_test_used_for_hypothesis=True,
        normal_current_training='Every train golden reused as an all-pass current input; development goldens added only to development; same family split',
        real_training_runs=[],generic_pretraining=False,run5_used=False)
    write(root/'detail-plan.json',plan)
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    train_data=with_normals(load_with_golden_hashes(rows,'train',render));dev=with_normals(load_with_golden_hashes(rows,'val',render));frozen={}
    for mode in plan['modes']:
        arm=out/mode;arm.mkdir();channels=4 if mode=='detail_rgb' else 10
        torch.manual_seed(19);model=model_for(channels).cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        torch.manual_seed(2719);curve=[];best=-1
        for epoch in range(1,46):
            model.train();order=torch.randperm(len(train_data['identifiers']),device='cuda');losses=[]
            for start in range(0,len(order),128):
                ix=order[start:start+128];x=tensor_for(train_data,mode,ix,True)
                loss=F.cross_entropy(model(x),train_data['labels'][ix])
                if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                optimizer.zero_grad();loss.backward();optimizer.step();losses.append(float(loss.detach()))
            records=evaluate(model,dev,mode);metrics=summarize(records)
            curve.append(dict(epoch=epoch,loss=sum(losses)/len(losses),macro_recall=metrics['macro_recall']))
            if metrics['macro_recall']>best:
                best=metrics['macro_recall'];torch.save(dict(state_dict=model.state_dict(),channels=channels,mode=mode,epoch=epoch),arm/'best.pt')
            print(mode,epoch,round(metrics['macro_recall'],5),flush=True)
        saved=torch.load(arm/'best.pt',map_location='cuda',weights_only=True);model.load_state_dict(saved['state_dict'])
        records=evaluate(model,dev,mode);threshold=min(1-r['probabilities'][0] for r in records if r['truth'] in (1,2,3))
        metadata=dict(mode=mode,channels=channels,selected_epoch=saved['epoch'],checkpoint_sha256=sha(arm/'best.pt'),threshold=threshold,
            development=summarize(records,threshold),test_used_for_selection=False)
        write(arm/'curve.json',curve);write(arm/'development-predictions.json',records);write(arm/'frozen.json',metadata);frozen[mode]=metadata
    write(out/'all-checkpoints-frozen.json',dict(utc=time.time(),models=frozen,plan_sha256=sha(root/'detail-plan.json')))


def test(root,parent):
    import torch
    from train_dataset1_fiveclass import build_model, evaluate as old_evaluate
    lock=(Path.home()/'ai-club-tools/spark-gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    frozen=read(root/'fitted-v1/all-checkpoints-frozen.json')['models']
    render=root/'rendered-v1';rows=[json.loads(s) for s in (render/'manifest.jsonl').read_text().splitlines()]
    validate(rows);data=with_normals(load_with_golden_hashes(rows,'test',render));comparison={}
    for mode,meta in frozen.items():
        arm=root/'fitted-v1'/mode
        if sha(arm/'best.pt')!=meta['checkpoint_sha256']:raise ValueError('Frozen model changed')
        model=model_for(meta['channels']).cuda();model.load_state_dict(torch.load(arm/'best.pt',map_location='cuda',weights_only=True)['state_dict'])
        records=evaluate(model,data,mode);metrics=summarize(records,meta['threshold'])
        write(arm/'fresh-test-predictions.json',records);write(arm/'fresh-test-metrics.json',metrics);comparison[mode]=metrics
    for mode in ('rgb','target_rgb','target_paired'):
        arm=parent/'fitted-v1'/mode;meta=read(arm/'frozen.json')
        if sha(arm/'best.pt')!=meta['checkpoint_sha256']:raise ValueError('Old frozen model changed')
        model=build_model(meta['channels']).cuda();model.load_state_dict(torch.load(arm/'best.pt',map_location='cuda',weights_only=True)['state_dict'])
        records=old_evaluate(model,data,mode);metrics=summarize(records,meta['threshold'])
        oldout=root/'baseline-fresh-v1'/mode;oldout.mkdir(parents=True,exist_ok=False)
        write(oldout/'predictions.json',records);write(oldout/'metrics.json',metrics);comparison[mode]=metrics
    previous=parent.parent/'dataset1-detail-20261023-v1'
    for mode in ('detail_rgb','detail_paired'):
        arm=previous/'fitted-v1'/mode;meta=read(arm/'frozen.json')
        if sha(arm/'best.pt')!=meta['checkpoint_sha256']:raise ValueError('Previous detail model changed')
        model=model_for(meta['channels']).cuda();model.load_state_dict(torch.load(arm/'best.pt',map_location='cuda',weights_only=True)['state_dict'])
        records=evaluate(model,data,mode);metrics=summarize(records,meta['threshold']);key='previous_'+mode
        oldout=root/'baseline-fresh-v1'/key;oldout.mkdir(parents=True,exist_ok=False)
        write(oldout/'predictions.json',records);write(oldout/'metrics.json',metrics);comparison[key]=metrics
    write(root/'comparison.json',dict(arms=comparison,fresh_families=12,targets=1440,real_accuracy=None,
        training_real_runs=[],fresh_seed=502113,all_new_models_frozen_before_fresh_render=True))


def run(root,parent):
    root.mkdir(exist_ok=True)
    if any((root/n).exists() for n in ('running.json','fitted-v1','rendered-v1')):raise ValueError('Inspect existing run; refusing duplicate')
    write(root/'running.json',dict(pid=os.getpid(),started=time.time(),parent=str(parent),helper_sha256=sha(__file__)))
    try:
        train(root,parent)
        plan=copy.deepcopy(read(parent/'experiment-plan.json'));plan['families']=families(502113,(0,0,12))
        for i,row in enumerate(plan['families']):row['id']=f'fresh-{i:03d}';row['group']=30000+i
        import random
        rng=random.Random(55312)
        for i,row in enumerate(plan['families']):
            row['density']='dense' if i<6 else 'sparse'
            if i>=6:
                row['labels']={slot:'pass' for slot in row['labels']}
                for fixture in ('G','S'):
                    slot=fixture+f'{rng.randint(1,30):02d}';row['labels'][slot]=['missing_cap','crooked_cap','damaged_vial'][i%3]
        plan['groups']=dict(train=0,val=0,test=12);plan['seed']=502113;plan['purpose']='Fresh post-freeze test: six dense, six sparse families; each golden also scored as normal current'
        write(root/'experiment-plan.json',plan)
        isaac=Path.home()/'ai-club-tools/envs/isaac-6.0.0.1/bin/python'
        env=dict(os.environ,LD_PRELOAD='/lib/aarch64-linux-gnu/libgomp.so.1',MPLBACKEND='Agg',OMNI_KIT_ACCEPT_EULA='YES')
        with (root/'render.log').open('x') as log:
            result=subprocess.run([str(isaac),'-u',str(root/'dataset1_fiveclass.py'),'render','--root',str(root),'--no-ros-env'],
                cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=900)
        if result.returncode:raise RuntimeError('Fresh render failed; preserve log')
        test(root,parent)
        write(root/'completed.json',dict(finished=time.time(),comparison_sha256=sha(root/'comparison.json'),plan_sha256=sha(root/'detail-plan.json'),gpu='GB10'))
    except BaseException as error:
        write(root/'failed.json',dict(error=str(error),type=type(error).__name__,utc=time.time()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True,type=Path);p.add_argument('--parent',required=True,type=Path)
    a=p.parse_args();run(a.root.resolve(),a.parent.resolve())
