"""Freeze development-only warning policies before fresh test inference."""
import argparse
import math
from pathlib import Path
import time

from dataset1_fiveclass import read,sha,write
from train_dataset1_fiveclass import summarize


def freeze(root,parent):
    out=root/'operating-points-frozen.json'
    if out.exists():raise ValueError('Policy already frozen')
    deadline=time.monotonic()+1200
    while not (root/'fitted-v1/all-checkpoints-frozen.json').exists():
        if (root/'failed.json').exists():raise RuntimeError('Existing experiment failed')
        if time.monotonic()>deadline:raise TimeoutError('Checkpoint wait expired')
        time.sleep(2)
    if (root/'comparison.json').exists():raise ValueError('Fresh test already scored; cannot claim prospective freeze')
    policies={}
    for name,arm in [(n,parent/'fitted-v1'/n) for n in ('rgb','target_rgb','target_paired')]+[(n,root/'fitted-v1'/n) for n in ('detail_rgb','detail_paired')]:
        records=read(arm/'development-predictions.json');scores=sorted(1-r['probabilities'][0] for r in records if r['truth'] in (1,2,3))
        thresholds={str(recall):scores[math.floor((1-recall)*len(scores)+1e-8)] for recall in (1.0,.99,.95,.90)}
        policies[name]=dict(checkpoint_sha256=sha(arm/'best.pt'),development_predictions_sha256=sha(arm/'development-predictions.json'),
            thresholds=thresholds,development={key:summarize(records,t)['occupied_defect_diagnostic'] for key,t in thresholds.items()})
    write(out,dict(frozen_utc=time.time(),selection='Occupied-defect development score quantiles only; no fresh test input',
        modes=policies,helper_sha256=sha(__file__),real_data_used=False))


def score(root):
    policy=read(root/'operating-points-frozen.json');results={};target_counts=set()
    for mode,meta in policy['modes'].items():
        path=root/('fitted-v1/'+mode+'/fresh-test-predictions.json' if mode.startswith('detail_') else 'baseline-fresh-v1/'+mode+'/predictions.json')
        if path.stat().st_mtime<=policy['frozen_utc']:raise ValueError('Policy freeze was not before test prediction write')
        records=read(path);target_counts.add(len(records));positives=[1-r['probabilities'][0] for r in records if r['truth'] in (1,2,3)]
        negatives=[1-r['probabilities'][0] for r in records if r['truth']==0]
        auc=sum((p>n)+.5*(p==n) for p in positives for n in negatives)/(len(positives)*len(negatives))
        results[mode]=dict(occupied_defect_auroc=auc,policies={key:summarize(records,t)['occupied_defect_diagnostic'] for key,t in meta['thresholds'].items()})
    if len(target_counts)!=1:raise ValueError('Models did not score the same target count')
    write(root/'operating-points-results.json',dict(fresh_targets=next(iter(target_counts)),policy_sha256=sha(root/'operating-points-frozen.json'),
        real_accuracy=None,results=results,thresholds_frozen_before_fresh_prediction_writes=True))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['freeze','score']);p.add_argument('--root',type=Path,required=True);p.add_argument('--parent',type=Path)
    a=p.parse_args();freeze(a.root,a.parent) if a.action=='freeze' else score(a.root)
