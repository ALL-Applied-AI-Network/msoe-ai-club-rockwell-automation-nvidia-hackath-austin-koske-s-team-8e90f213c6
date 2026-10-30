"""Dedicated cap-state experiment; never infer whole-vial acceptance from a cap.

The existing synthetic damaged-body proxy leaves its cap correctly seated. This
geometry fact supports a cap-state label, while body condition remains unassessed.
Real annotation precedence is not used to infer cap truth.
"""
import argparse
import fcntl
import json
import math
from pathlib import Path
import time

from dataset1_fiveclass import read,sha,write
from experiment_dataset1_normals import load_with_golden_hashes,model_for,tensor_for,with_normals

CAP_CLASSES=['seated_cap','missing_cap','crooked_cap','no_vial']
CAP_LABEL=[0,1,2,0,3]


def metrics(records,threshold=None):
    matrix=[[0]*4 for _ in range(4)];scenes={};fixtures={}
    for r in records:
        matrix[r['truth']][r['prediction']]+=1
        scenes.setdefault(r['scene'],[]).append(r['truth']==r['prediction'])
        fixtures.setdefault((r['scene'],r['slot'][0]),[]).append(r['truth']==r['prediction'])
    per={c:dict(targets=sum(matrix[i]),correct=matrix[i][i],recall=matrix[i][i]/sum(matrix[i]) if sum(matrix[i]) else None) for i,c in enumerate(CAP_CLASSES)}
    result=dict(targets=len(records),accuracy=sum(matrix[i][i] for i in range(4))/len(records),
        macro_recall=sum(r['recall'] for r in per.values() if r['recall'] is not None)/sum(r['recall'] is not None for r in per.values()),
        classes=CAP_CLASSES,confusion=matrix,per_class=per,
        complete_scenes=len(scenes),all_60_correct_scenes=sum(len(v)==60 and all(v) for v in scenes.values()),
        complete_fixtures=len(fixtures),all_30_correct_fixtures=sum(len(v)==30 and all(v) for v in fixtures.values()),
        body_condition='unassessed',quality_release_authorized=False)
    if threshold is not None:
        normal=[r for r in records if r['truth']==0]
        result['cap_warning']=dict(threshold=threshold,seated_targets=len(normal),false_alarms=sum(1-r['probabilities'][0]>=threshold for r in normal),
            per_class={CAP_CLASSES[i]:dict(targets=sum(r['truth']==i for r in records),missed=sum(r['truth']==i and 1-r['probabilities'][0]<threshold for r in records)) for i in (1,2)},
            empty_requires_progress_context=True,body_damage_not_assessed=True)
    return result


def cap_data(rows,split,folder):
    import torch
    data=with_normals(load_with_golden_hashes(rows,split,folder));data['source_labels']=data['labels'].clone()
    data['labels']=torch.tensor(CAP_LABEL,device='cuda')[data['labels']]
    return data


def cap_model():
    import torch
    model=model_for(10);model.head[-1]=torch.nn.Linear(256,4);return model


def evaluate(model,data):
    import torch
    model.eval();records=[]
    with torch.inference_mode():
        for start in range(0,len(data['identifiers']),128):
            ix=torch.arange(start,min(start+128,len(data['identifiers'])),device='cuda')
            probabilities=model(tensor_for(data,'detail_paired',ix)).softmax(1).cpu().tolist()
            truth=data['labels'][ix].cpu().tolist();source=data['source_labels'][ix].cpu().tolist()
            for i,p in enumerate(probabilities):
                records.append(dict(**data['identifiers'][start+i],source_truth=source[i],truth=truth[i],prediction=max(range(4),key=lambda j:p[j]),probabilities=p))
    return records


def run(root,parent,testroot):
    import torch
    import torch.nn.functional as F
    root.mkdir(exist_ok=True)
    if (root/'running.json').exists():raise ValueError('Inspect existing run; duplicate refused')
    write(root/'running.json',dict(started=time.time(),helper_sha256=sha(__file__),parent=str(parent),testroot=str(testroot)))
    deadline=time.monotonic()+1800
    while not (testroot/'completed.json').exists():
        if (testroot/'failed.json').exists():raise RuntimeError('Existing dependent job failed')
        if time.monotonic()>deadline:raise TimeoutError('Dependency wait expired')
        time.sleep(3)
    lock=(Path.home()/'ai-club-tools/spark-gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    out=root/'fitted-v1';out.mkdir(exist_ok=False);render=parent/'rendered-v1'
    rows=[json.loads(s) for s in (render/'manifest.jsonl').read_text().splitlines()]
    write(root/'experiment-plan.json',dict(classes=CAP_CLASSES,source_class_to_cap_index=CAP_LABEL,
        label_basis='Renderer geometry: damaged-body proxy has correctly seated cap. Body condition explicitly unassessed.',
        epochs=45,seed=19,architecture='Same detail-paired dual branch with four-output head',
        normal_current_training=True,parent_manifest_sha256=sha(render/'manifest.jsonl'),helper_sha256=sha(__file__),
        selection='Maximum original synthetic development cap macro recall, earliest tie',
        threshold='Minimum / 99% / 95% development recall across missing and crooked caps only',
        fresh_test_source='Normals experiment seed502113; frozen checkpoint before this process decodes test images',
        test_already_rendered_by_separate_experiment=True,real_training_runs=[],generic_pretraining=False,quality_release_authorized=False))
    train=cap_data(rows,'train',render);dev=cap_data(rows,'val',render)
    torch.manual_seed(19);model=cap_model().cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    torch.manual_seed(2719);curve=[];best=-1
    for epoch in range(1,46):
        model.train();order=torch.randperm(len(train['identifiers']),device='cuda');losses=[]
        for start in range(0,len(order),128):
            ix=order[start:start+128];loss=F.cross_entropy(model(tensor_for(train,'detail_paired',ix,True)),train['labels'][ix])
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            optimizer.zero_grad();loss.backward();optimizer.step();losses.append(float(loss.detach()))
        records=evaluate(model,dev);m=metrics(records);curve.append(dict(epoch=epoch,macro_recall=m['macro_recall'],loss=sum(losses)/len(losses)))
        if m['macro_recall']>best:
            best=m['macro_recall'];torch.save(dict(state_dict=model.state_dict(),epoch=epoch,classes=CAP_CLASSES),out/'best.pt')
        print('cap_state',epoch,round(m['macro_recall'],5),flush=True)
    saved=torch.load(out/'best.pt',map_location='cuda',weights_only=True);model.load_state_dict(saved['state_dict'])
    records=evaluate(model,dev);scores=sorted(1-r['probabilities'][0] for r in records if r['truth'] in (1,2))
    thresholds={str(recall):scores[math.floor((1-recall)*len(scores)+1e-8)] for recall in (1.,.99,.95)}
    frozen=dict(frozen_utc=time.time(),checkpoint_sha256=sha(out/'best.pt'),selected_epoch=saved['epoch'],thresholds=thresholds,
        development={key:metrics(records,t) for key,t in thresholds.items()})
    write(out/'curve.json',curve);write(out/'development-predictions.json',records);write(out/'frozen.json',frozen)
    # First test manifest/pixel decode in this process occurs only after freeze.
    testfolder=testroot/'rendered-v1';testrows=[json.loads(s) for s in (testfolder/'manifest.jsonl').read_text().splitlines()]
    test=cap_data(testrows,'test',testfolder);records=evaluate(model,test)
    write(out/'test-predictions.json',records);result=dict(metrics=metrics(records),policies={key:metrics(records,t)['cap_warning'] for key,t in thresholds.items()},
        strata={name:metrics([r for r in records if (r['variant']=='golden_as_current' if name=='normal' else r['variant']=='inspection' and next(s for s in testrows if s['id']==r['scene'])['nuisance']['density']==name)]) for name in ('dense','sparse','normal')},
        real_accuracy=None,body_damage_recall=None,qualified_real_reference_available=False,full_dataset1_task_solved=False)
    write(out/'results.json',result);write(root/'completed.json',dict(finished=time.time(),gpu='GB10',results_sha256=sha(out/'results.json'),checkpoint_sha256=frozen['checkpoint_sha256']))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','parent','testroot'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.root.resolve(),a.parent.resolve(),a.testroot.resolve())
