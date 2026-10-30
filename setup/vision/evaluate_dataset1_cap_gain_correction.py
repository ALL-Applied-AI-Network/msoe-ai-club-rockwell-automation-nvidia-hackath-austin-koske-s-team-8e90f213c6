"""Frozen cap model plus label-free scene gain correction, digital diagnostic only."""
import argparse
import fcntl
import json
from pathlib import Path

from dataset1_fiveclass import read,sha,write
from experiment_dataset1_cap_state import cap_data,cap_model,evaluate,metrics
from fixture_reference_preprocess import estimate_scene_gain


def run(root,testroot):
    import torch
    lock=(Path.home()/'ai-club-tools/spark-gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out=root/'gain-correction-v1';out.mkdir(exist_ok=False);torch.set_num_threads(4)
    frozen=read(root/'fitted-v1/frozen.json');checkpoint=root/'fitted-v1/best.pt'
    if sha(checkpoint)!=frozen['checkpoint_sha256']:raise ValueError('Checkpoint differs')
    folder=testroot/'rendered-v1';rows=[json.loads(s) for s in (folder/'manifest.jsonl').read_text().splitlines()]
    data=cap_data(rows,'test',folder);model=cap_model().cuda();model.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=True)['state_dict'])
    results={}
    for name in ('unchanged','reference_gain_0.9','reference_gain_1.1','current_gain_0.9'):
        variant=dict(data);variant['current']=data['current'].clone();variant['golden']=data['golden'].clone()
        if name.startswith('reference_gain'):variant['golden']=(variant['golden']*float(name.rsplit('_',1)[1])).clamp(0,1)
        if name=='current_gain_0.9':variant['current']=(variant['current']*.9).clamp(0,1)
        estimates=[]
        for start in range(0,len(data['identifiers']),60):
            ids=data['identifiers'][start:start+60]
            if len({r['scene'] for r in ids})!=1:raise ValueError('Scene gain cannot mix sources')
            estimate=estimate_scene_gain(variant['current'][start:start+60].cpu().numpy(),variant['golden'][start:start+60].cpu().numpy())
            if not estimate['available']:raise ValueError('Correction unavailable; do not substitute a successful finding')
            variant['golden'][start:start+60]=(variant['golden'][start:start+60]*estimate['gain']).clamp(0,1)
            estimates.append(dict(scene=ids[0]['scene'],**estimate))
        records=evaluate(model,variant);normal=[r for r in records if r['variant']=='golden_as_current']
        results[name]=dict(metrics=metrics(records),normal_exact=metrics(normal),estimates=estimates,
            policies={key:metrics(records,t)['cap_warning'] for key,t in frozen['thresholds'].items()},
            normal_policies={key:metrics(normal,t)['cap_warning'] for key,t in frozen['thresholds'].items()})
        write(out/(name+'-predictions.json'),records)
    write(out/'completed.json',dict(checkpoint_sha256=frozen['checkpoint_sha256'],cases=results,training=False,thresholds_changed=False,
        helper_sha256=sha(__file__),preprocessor_sha256=sha(Path(__file__).with_name('fixture_reference_preprocess.py')),
        control_selection='Post-hoc digital diagnostic motivated by earlier synthetic stress failure; same stress images reused, no new held-out proof',
        alignment_corrected=False,physical_reference_qualified=False,real_accuracy=None,app_defaults_changed=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','testroot'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.root,a.testroot)
