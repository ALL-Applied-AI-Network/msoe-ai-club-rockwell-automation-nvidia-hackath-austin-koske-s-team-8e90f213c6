"""Freeze three synthetic-only five-class models before decoding test scenes."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

from dataset1_fiveclass import CLASSES,SLOTS,read,sha,write


def validate(rows):
    groups={};images={}
    for row in rows:
        if set(row['labels'])!=set(SLOTS) or set(row['boxes'])!=set(SLOTS):raise ValueError('Expected60 explicit target positions')
        if row['split'] not in ('train','val','test') or any(c not in CLASSES for c in row['labels'].values()):raise ValueError('Unknown class/split')
        for key,registry in [(row['group'],groups),(row['image_sha256'],images),(row['golden_sha256'],images)]:
            if key in registry and registry[key]!=row['split']:raise ValueError('Scene/image/golden crosses split boundary')
            registry[key]=row['split']
    return dict(Counter(r['split'] for r in rows))


def summarize(records,threshold=None):
    matrix=[[0]*5 for _ in range(5)];scenes={};fixtures={}
    for row in records:
        matrix[row['truth']][row['prediction']]+=1
        scenes.setdefault(row['scene'],[]).append(row['truth']==row['prediction'])
        fixtures.setdefault((row['scene'],row['slot'][0]),[]).append(row['truth']==row['prediction'])
    per_class={}
    for i,c in enumerate(CLASSES):
        total=sum(matrix[i]);proposed=sum(r[i] for r in matrix)
        per_class[c]=dict(targets=total,correct=matrix[i][i],missed=total-matrix[i][i],
            recall=matrix[i][i]/total if total else None,precision=matrix[i][i]/proposed if proposed else None)
    recalls=[r['recall'] for r in per_class.values() if r['recall'] is not None]
    result=dict(targets=len(records),accuracy=sum(matrix[i][i] for i in range(5))/len(records),
        macro_recall=sum(recalls)/len(recalls),classes=CLASSES,confusion_rows_truth_columns_prediction=matrix,
        per_class=per_class,complete_scenes=len(scenes),all_60_correct_scenes=sum(len(v)==60 and all(v) for v in scenes.values()),
        complete_fixtures=len(fixtures),all_30_correct_fixtures=sum(len(v)==30 and all(v) for v in fixtures.values()))
    if threshold is not None:
        negatives=[r for r in records if r['truth']==0];by_class={}
        for i in (1,2,3):
            positives=[r for r in records if r['truth']==i]
            by_class[CLASSES[i]]=dict(targets=len(positives),missed=sum(1-r['probabilities'][0]<threshold for r in positives))
        result['occupied_defect_diagnostic']=dict(score='1-P(pass)',threshold=threshold,per_class=by_class,
            acceptable_targets=len(negatives),false_alarms=sum(1-r['probabilities'][0]>=threshold for r in negatives),
            false_alarm_rate=sum(1-r['probabilities'][0]>=threshold for r in negatives)/len(negatives) if negatives else None,
            no_vial_excluded=True,counts_are_synthetic=True,quality_release_authorized=False)
    return result


def build_model(channels):
    from torch import nn
    return nn.Sequential(nn.Conv2d(channels,24,3,2,1),nn.GroupNorm(6,24),nn.SiLU(),
        nn.Conv2d(24,48,3,2,1),nn.GroupNorm(6,48),nn.SiLU(),
        nn.Conv2d(48,96,3,2,1),nn.GroupNorm(6,96),nn.SiLU(),
        nn.AdaptiveAvgPool2d((4,4)),nn.Flatten(),nn.Linear(96*4*4,128),nn.SiLU(),nn.Linear(128,5))


def load(rows,split,folder):
    import numpy as np
    from PIL import Image
    import torch
    current=[];golden=[];ids=[];labels=[]
    for row in rows:
        if row['split']!=split:continue
        for key in ('image','golden'):
            if sha(folder/row[key])!=row[key+'_sha256']:raise ValueError('Synthetic RGB bytes differ')
        with Image.open(folder/row['image']) as im,Image.open(folder/row['golden']) as ref:
            for slot in SLOTS:
                for image,collection in ((im,current),(ref,golden)):
                    rgb=np.array(image.convert('RGB').crop(row['boxes'][slot]),dtype=np.float32)/255
                    if rgb.shape!=(96,96,3):raise ValueError('Wrong native crop size')
                    collection.append(torch.from_numpy(rgb.transpose(2,0,1).copy()))
                labels.append(CLASSES.index(row['labels'][slot]));ids.append(dict(scene=row['id'],group=row['group'],slot=slot,
                    box_xyxy=row['boxes'][slot],source_sha256=row['image_sha256']))
    yy,xx=np.mgrid[:96,:96];target=np.exp(-((xx-48)**2+(yy-48)**2)/(2*9**2)).astype(np.float32)
    return dict(current=torch.stack(current).cuda(),golden=torch.stack(golden).cuda(),
        target=torch.from_numpy(target)[None,None].cuda(),labels=torch.tensor(labels,device='cuda'),identifiers=ids)


def inputs(data,mode,indices):
    import torch
    current=data['current'][indices];reference=data['golden'][indices]
    heat=data['target'].expand(len(indices),-1,-1,-1)
    if mode=='rgb':return current
    if mode=='target_rgb':return torch.cat([current,heat],1)
    return torch.cat([current,reference,heat],1)


def evaluate(model,data,mode):
    import torch
    model.eval();records=[]
    with torch.inference_mode():
        for start in range(0,len(data['identifiers']),128):
            ix=torch.arange(start,min(start+128,len(data['identifiers'])),device='cuda')
            probs=model(inputs(data,mode,ix)).softmax(1).cpu().tolist()
            truth=data['labels'][ix].cpu().tolist()
            for i,p in enumerate(probs):records.append(dict(**data['identifiers'][start+i],truth=truth[i],prediction=max(range(5),key=lambda j:p[j]),probabilities=p))
    return records


def main(root):
    import fcntl
    lock=(Path.home()/'ai-club-tools/spark-gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out=root/'fitted-v1';out.mkdir(exist_ok=False)
    plan=read(root/'experiment-plan.json');render=root/'rendered-v1';completion=read(render/'completed.json')
    if completion['manifest_sha256']!=sha(render/'manifest.jsonl') or completion['plan_sha256']!=sha(root/'experiment-plan.json'):raise ValueError('Render provenance differs')
    if sha(root/'dataset1_fiveclass.py')!=plan['renderer_sha256']:raise ValueError('Frozen helper differs')
    rows=[json.loads(s) for s in (render/'manifest.jsonl').read_text().splitlines()];counts=validate(rows)
    if counts!=plan['groups']:raise ValueError('Synthetic family counts differ')
    write(out/'training-plan.json',dict(parent_plan_sha256=sha(root/'experiment-plan.json'),trainer_sha256=sha(__file__),
        render_manifest_sha256=sha(render/'manifest.jsonl'),architecture='Conv24/48/96 + GroupNorm6 + adaptive4x4 + Linear128/5',
        models=plan['models'],groups=counts,epochs=plan['epochs'],generic_pretraining=False,training_real_runs=[],run5_used=False,
        target_conditioning='Gaussian on fixed aperture center, extra input channel, never a class marker',
        selection=plan['checkpoint_selection'],threshold=plan['threshold_selection']))
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    train=load(rows,'train',render);val=load(rows,'val',render);frozen={}
    try:
        for mode in plan['models']:
            arm=out/mode;arm.mkdir();channels={'rgb':3,'target_rgb':4,'target_paired':7}[mode]
            torch.manual_seed(19);model=build_model(channels).cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
            torch.manual_seed(2719);curve=[];best=-1.;selected=None
            for epoch in range(1,plan['epochs']+1):
                model.train();order=torch.randperm(len(train['identifiers']),device='cuda');losses=[]
                for start in range(0,len(order),plan['batch_size']):
                    ix=order[start:start+plan['batch_size']];x=inputs(train,mode,ix).clone();n=len(ix);rgb_channels=6 if mode=='target_paired' else 3
                    gain=torch.empty(n,1,1,1,device='cuda').uniform_(.85,1.15);bias=torch.empty(n,1,1,1,device='cuda').uniform_(-.025,.025)
                    x[:,:rgb_channels]=(x[:,:rgb_channels]*gain+bias).clamp(0,1)
                    # Translate RGB and target-conditioning together, keeping attribution.
                    dx,dy=torch.randint(-2,3,(2,)).tolist();x=torch.roll(x,(dx,dy),(-1,-2))
                    loss=F.cross_entropy(model(x),train['labels'][ix])
                    if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                    optimizer.zero_grad();loss.backward();optimizer.step();losses.append(float(loss.detach()))
                records=evaluate(model,val,mode);metrics=summarize(records)
                curve.append(dict(epoch=epoch,loss=sum(losses)/len(losses),validation_macro_recall=metrics['macro_recall'],per_class=metrics['per_class']))
                if metrics['macro_recall']>best:
                    best=metrics['macro_recall'];selected=epoch
                    torch.save(dict(state_dict=model.state_dict(),epoch=epoch,mode=mode,channels=channels,classes=CLASSES),arm/'best.pt')
                print(mode,epoch,'val_macro_recall',round(metrics['macro_recall'],5),flush=True)
            model.load_state_dict(torch.load(arm/'best.pt',map_location='cuda',weights_only=True)['state_dict'])
            records=evaluate(model,val,mode);threshold=min(1-r['probabilities'][0] for r in records if r['truth'] in (1,2,3))
            metadata=dict(mode=mode,channels=channels,selected_epoch=selected,checkpoint_sha256=sha(arm/'best.pt'),
                classes=CLASSES,threshold=threshold,development=summarize(records,threshold),training_real_runs=[],test_read_for_selection=False)
            write(arm/'curve.json',curve);write(arm/'development-predictions.json',records);write(arm/'frozen.json',metadata);frozen[mode]=metadata
        write(out/'all-checkpoints-frozen.json',dict(utc=time.time(),models=frozen))
        # Test RGB and labels first enter model evaluation after every arm freezes.
        test=load(rows,'test',render);comparison={}
        for mode,metadata in frozen.items():
            arm=out/mode
            if sha(arm/'best.pt')!=metadata['checkpoint_sha256']:raise ValueError('Frozen checkpoint differs')
            model=build_model(metadata['channels']).cuda();model.load_state_dict(torch.load(arm/'best.pt',map_location='cuda',weights_only=True)['state_dict'])
            records=evaluate(model,test,mode);metrics=summarize(records,metadata['threshold'])
            with torch.inference_mode():
                ix=torch.arange(60,device='cuda');x=inputs(test,mode,ix)
                for _ in range(30):model(x)
                torch.cuda.synchronize();times=[]
                for _ in range(100):
                    start=time.perf_counter();model(x);torch.cuda.synchronize();times.append((time.perf_counter()-start)*1000)
            times.sort();metrics['classifier_60_crop_batch_ms']=dict(median=(times[49]+times[50])/2,p95=times[94],excludes='render/decode/crop/transport/browser')
            write(arm/'test-predictions.json',records);write(arm/'test-metrics.json',metrics);comparison[mode]=metrics
        write(out/'comparison.json',dict(classes=CLASSES,arms=comparison,counts_are_synthetic=True,real_accuracy=None,selection_frozen_before_test=True))
        write(out/'completed.json',dict(gpu=torch.cuda.get_device_name(0),completed=time.time(),models=frozen,
            comparison_sha256=sha(out/'comparison.json'),training_plan_sha256=sha(out/'training-plan.json'),local_gpu_used=False))
    except BaseException as error:write(out/'failed.json',dict(type=type(error).__name__,error=str(error)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.root.resolve())
