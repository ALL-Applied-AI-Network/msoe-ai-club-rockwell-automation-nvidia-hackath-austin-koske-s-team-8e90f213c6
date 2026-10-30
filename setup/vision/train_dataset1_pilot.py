"""Fit two synthetic Dataset 1 baselines, freeze on development, then score test.

Not TAO Visual ChangeNet: paired_reference is a small custom concatenated-RGB CNN.
Real data never enters this experiment. Empty occupancy is not a placement error.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from dataset1_pilot import CLASSES, read, sha, write


def validate_records(rows):
    groups, images = {}, {}
    for row in rows:
        if set(row['labels']) != {f'G{i:02d}' for i in range(1,31)}:
            raise ValueError('Expected exactly30 attributed G targets per scene')
        if row['split'] not in ('train','val','test') or any(v not in CLASSES for v in row['labels'].values()):
            raise ValueError('Unknown split or geometry class')
        for key, registry in [(row['group'], groups), (row['image_sha256'], images), (row['golden_sha256'], images)]:
            if key in registry and registry[key] != row['split']:
                raise ValueError('Scene or image bytes cross the split boundary')
            registry[key] = row['split']
        if set(row['boxes']) != set(row['labels']):
            raise ValueError('Missing target/crop association')
    return {split:sum(r['split']==split for r in rows) for split in ('train','val','test')}


def summarize(records, threshold=None):
    confusion = [[0]*3 for _ in range(3)]
    scenes = {}
    for row in records:
        confusion[row['truth']][row['prediction']] += 1
        scenes.setdefault(row['scene'], []).append(row['truth']==row['prediction'])
    metrics = {}
    for i, label in enumerate(CLASSES):
        total = sum(confusion[i]); proposed = sum(row[i] for row in confusion)
        metrics[label] = dict(targets=total, correct=confusion[i][i], missed=total-confusion[i][i],
            recall=confusion[i][i]/total if total else None,
            precision=confusion[i][i]/proposed if proposed else None)
    recalls = [v['recall'] for v in metrics.values() if v['recall'] is not None]
    result = dict(samples=len(records), confusion_rows_truth_columns_prediction=confusion, per_class=metrics,
        accuracy=sum(confusion[i][i] for i in range(3))/len(records), macro_recall=sum(recalls)/len(recalls),
        complete_scenes=len(scenes), all_30_correct_scenes=sum(len(v)==30 and all(v) for v in scenes.values()))
    if threshold is not None:
        # Binary score is diagnostic on occupied targets. Empty-slot action needs context.
        positives = [r for r in records if r['truth']==1]
        negatives = [r for r in records if r['truth']==0]
        result['occupied_cap_diagnostic'] = dict(score='1-P(pass)', threshold=threshold,
            missing_cap_cases=len(positives), missing_cap_missed=sum(1-r['probabilities'][0]<threshold for r in positives),
            acceptable_cases=len(negatives), acceptable_false_alarms=sum(1-r['probabilities'][0]>=threshold for r in negatives),
            no_vial_excluded=True, policy_action='none; occupancy context/real validation not implemented')
    return result


def build_model(channels):
    import torch
    from torch import nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            layers = []
            current_channels = channels
            for width in (32,64,128):
                layers += [nn.Conv2d(current_channels,width,3,stride=2,padding=1),nn.GroupNorm(8,width),nn.SiLU()]
                current_channels = width
            self.features = nn.Sequential(*layers)
            self.head = nn.Sequential(nn.Linear(128*4*4,128),nn.SiLU(),nn.Linear(128,3))

        def forward(self,x):
            return self.head(nn.functional.adaptive_avg_pool2d(self.features(x),(4,4)).flatten(1))

    return Net()


def load_split(rows, split, rendered):
    import numpy as np
    from PIL import Image
    import torch
    current, golden, targets, identifiers = [], [], [], []
    for row in rows:
        if row['split'] != split:
            continue
        for key in ('image','golden'):
            if sha(rendered/row[key]) != row[key+'_sha256']:
                raise ValueError('Rendered input bytes differ')
        with Image.open(rendered/row['image']) as source, Image.open(rendered/row['golden']) as reference:
            for slot in sorted(row['labels']):
                box = row['boxes'][slot]
                for image, collection in ((source,current),(reference,golden)):
                    rgb = np.asarray(image.convert('RGB').crop(box).resize((64,64)),dtype=np.float32)/255
                    collection.append(torch.from_numpy(rgb.transpose(2,0,1).copy()))
                targets.append(CLASSES.index(row['labels'][slot]))
                identifiers.append(dict(scene=row['id'],group=row['group'],slot=slot,box=box,source_sha256=row['image_sha256']))
    return dict(current=torch.stack(current).cuda(), golden=torch.stack(golden).cuda(),
                targets=torch.tensor(targets,device='cuda'), identifiers=identifiers)


def inputs(data, mode, indices=None):
    import torch
    current = data['current'] if indices is None else data['current'][indices]
    golden = data['golden'] if indices is None else data['golden'][indices]
    return current if mode=='rgb' else torch.cat([current,golden],1)


def evaluate(model, data, mode):
    import torch
    model.eval()
    records=[]
    with torch.inference_mode():
        for start in range(0,len(data['identifiers']),64):
            indices = torch.arange(start,min(start+64,len(data['identifiers'])),device='cuda')
            probabilities = model(inputs(data,mode,indices)).softmax(1).cpu().tolist()
            for offset, probs in enumerate(probabilities):
                i=start+offset
                records.append(dict(**data['identifiers'][i],truth=int(data['targets'][i]),
                    prediction=max(range(3),key=lambda c:probs[c]), probabilities=probs))
    return records


def main():
    import fcntl
    import os
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    root=args.root.resolve()
    out=root/'fitted'
    out.mkdir(exist_ok=False)
    lock=(Path.home()/'ai-club-tools/spark-gpu.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    plan=read(root/'experiment-plan.json')
    if sha(root/'dataset1_pilot.py')!=plan['helper_sha256']:
        raise ValueError('Frozen experiment helper changed')
    rendered=root/'rendered'
    completion=read(rendered/'completed.json')
    if completion['manifest_sha256']!=sha(rendered/'manifest.jsonl') or completion['plan_sha256']!=sha(root/'experiment-plan.json'):
        raise ValueError('Render evidence differs')
    rows=[json.loads(line) for line in (rendered/'manifest.jsonl').read_text().splitlines()]
    counts=validate_records(rows)
    if counts!={'train':36,'val':12,'test':12}:
        raise ValueError('Unexpected split counts')
    write(out/'training-plan.json',dict(parent_plan_sha256=sha(root/'experiment-plan.json'),trainer_sha256=sha(__file__),
        render_manifest_sha256=sha(rendered/'manifest.jsonl'),split_counts=counts,training_real_runs=[],
        architecture='Conv stride2 32/64/128 + GroupNorm8 + adaptive4x4 + Linear128/3',initialization='random',
        augmentation='Shared crop translation <=1px, shared gain .85-1.15 and bias +/-.025; input is encoded RGB.',
        epochs=30,batch_size=64,optimizer='AdamW .001 weight_decay.0001',
        selection='Synthetic development macro recall; earliest tie. No synthetic test decoding until both arms freeze.',
        threshold='Minimum development 1-P(pass) across missing_cap; inclusive comparison catches all development missing_cap cases.',
        real_metrics_available=False,run5_used=False))
    train=load_split(rows,'train',rendered);val=load_split(rows,'val',rendered)
    frozen={}
    try:
        for mode in plan['models']:
            arm=out/mode;arm.mkdir()
            torch.manual_seed(19)
            model=build_model(3 if mode=='rgb' else 6).cuda()
            optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
            torch.manual_seed(2719)
            curve=[];best=-1.
            for epoch in range(1,31):
                model.train();losses=[]
                order=torch.randperm(len(train['identifiers']),device='cuda')
                for start in range(0,len(order),64):
                    indices=order[start:start+64]
                    x=inputs(train,mode,indices).clone()
                    count=len(indices)
                    gain=torch.empty(count,1,1,1,device='cuda').uniform_(.85,1.15)
                    bias=torch.empty(count,1,1,1,device='cuda').uniform_(-.025,.025)
                    x=(x*gain+bias).clamp(0,1)
                    # Apply the same small translation to paired current/golden.
                    dx,dy=torch.randint(-1,2,(2,)).tolist()
                    x=torch.roll(x,(dx,dy),(-1,-2))
                    logits=model(x);loss=F.cross_entropy(logits,train['targets'][indices])
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite training loss')
                    optimizer.zero_grad();loss.backward();optimizer.step();losses.append(float(loss.detach()))
                records=evaluate(model,val,mode)
                metrics=summarize(records)
                curve.append(dict(epoch=epoch,loss=sum(losses)/len(losses),val=metrics))
                if metrics['macro_recall']>best:
                    best=metrics['macro_recall'];selected=epoch
                    torch.save(dict(state_dict=model.state_dict(),mode=mode,epoch=epoch,classes=CLASSES),arm/'best.pt')
                print(mode,epoch,'val_macro_recall',round(metrics['macro_recall'],4),flush=True)
            checkpoint=torch.load(arm/'best.pt',map_location='cuda',weights_only=True)
            model.load_state_dict(checkpoint['state_dict'])
            records=evaluate(model,val,mode)
            threshold=min(1-r['probabilities'][0] for r in records if r['truth']==1)
            metadata=dict(mode=mode,selected_epoch=selected,fit_epochs=30,checkpoint_sha256=sha(arm/'best.pt'),
                development=summarize(records,threshold),threshold=threshold,classes=CLASSES,random_initialization=True,
                training_real_runs=[],selection_real_runs=[],test_read_for_selection=False)
            write(arm/'curve.json',curve);write(arm/'development-predictions.json',records);write(arm/'frozen.json',metadata)
            frozen[mode]=metadata
        write(out/'both-checkpoints-frozen.json',dict(frozen_utc=time.time(),models=frozen))
        # Actual test image decoding begins only after both frozen manifests exist.
        test=load_split(rows,'test',rendered)
        result={}
        for mode in plan['models']:
            arm=out/mode
            assert sha(arm/'best.pt')==frozen[mode]['checkpoint_sha256']
            model=build_model(3 if mode=='rgb' else 6).cuda()
            model.load_state_dict(torch.load(arm/'best.pt',map_location='cuda',weights_only=True)['state_dict'])
            records=evaluate(model,test,mode)
            metrics=summarize(records,frozen[mode]['threshold'])
            model.eval();x=inputs(test,mode)[:30]
            with torch.inference_mode():
                for _ in range(30):model(x)
                torch.cuda.synchronize();times=[]
                for _ in range(200):
                    started=time.perf_counter();model(x);torch.cuda.synchronize()
                    times.append((time.perf_counter()-started)*1000)
            times.sort()
            metrics['classifier_30_crop_batch_ms']=dict(median=(times[99]+times[100])/2,p95=times[189],
                excludes='render, source decode, crop extraction, transport, operator UI')
            result[mode]=metrics
            write(arm/'test-predictions.json',records);write(arm/'test-metrics.json',metrics)
        write(out/'comparison.json',dict(classes=CLASSES,test_scenes=12,test_targets=360,arms=result,
            counts_are_synthetic=True,real_accuracy=None,physical_identity_link=False,selection_complete_before_test=True))
        write(out/'completed.json',dict(completed=time.time(),models=frozen,comparison_sha256=sha(out/'comparison.json'),
            train_plan_sha256=sha(out/'training-plan.json'),gpu=torch.cuda.get_device_name(0),local_gpu_used=False))
    except BaseException as error:
        write(out/'failed.json',dict(type=type(error).__name__,error=str(error)))
        raise


if __name__=='__main__':
    main()
