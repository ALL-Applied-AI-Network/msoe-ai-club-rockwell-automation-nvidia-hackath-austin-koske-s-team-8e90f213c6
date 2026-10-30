"""Frozen-model predictions on isolated changes and genuinely all-pass renders."""
import argparse
import copy
import fcntl
import json
from pathlib import Path

from dataset1_fiveclass import CLASSES,read,sha,write
from experiment_dataset1_detail import evaluate,model_for
from train_dataset1_fiveclass import build_model,evaluate as baseline_evaluate,load,summarize


def run(root,parent,detail):
    import torch
    lock=(Path.home()/'ai-club-tools/spark-gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out=root/'inference-v1';out.mkdir(exist_ok=False);torch.set_num_threads(4)
    render=root/'rendered-v1';rows=[json.loads(s) for s in (render/'manifest.jsonl').read_text().splitlines()]
    allpass=copy.deepcopy(rows)
    for row in allpass:
        row['image']=row['golden'];row['image_sha256']=row['golden_sha256'];row['labels']={s:'pass' for s in row['labels']}
    changed=load(rows,'test',render);normal=load(allpass,'test',render);results={}
    policies=read(detail/'operating-points-frozen.json')['modes']
    for mode in ('rgb','target_rgb','target_paired','detail_rgb','detail_paired'):
        arm=(detail if mode.startswith('detail_') else parent)/'fitted-v1'/mode;meta=read(arm/'frozen.json')
        if sha(arm/'best.pt')!=meta['checkpoint_sha256']:raise ValueError('Frozen checkpoint differs')
        model=(model_for if mode.startswith('detail_') else build_model)(meta['channels']).cuda()
        model.load_state_dict(torch.load(arm/'best.pt',map_location='cuda',weights_only=True)['state_dict'])
        evaluator=evaluate if mode.startswith('detail_') else baseline_evaluate
        predicted=evaluator(model,changed,mode);golden=evaluator(model,normal,mode)
        selected=[]
        for row in rows:
            slot=row['nuisance']['isolated_case']['slot']
            selected.append(next(r for r in predicted if r['scene']==row['id'] and r['slot']==slot))
        results[mode]=dict(isolated_changed_targets=selected,allpass_targets=len(golden),allpass_exact_correct=sum(r['prediction']==0 for r in golden),
            allpass_warning={key:summarize(golden,t)['occupied_defect_diagnostic'] for key,t in policies[mode]['thresholds'].items()},
            checkpoint_sha256=meta['checkpoint_sha256'])
        write(out/(mode+'-changed-predictions.json'),predicted);write(out/(mode+'-allpass-predictions.json'),golden)
    write(out/'completed.json',dict(models=results,real_accuracy=None,training=False,helper_sha256=sha(__file__)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','parent','detail'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.root,a.parent,a.detail)
