"""Frozen cap model: reference/current photometry and registration stress, no tuning."""
import argparse
import fcntl
import json
from pathlib import Path
import time

from dataset1_fiveclass import read,sha,write
from experiment_dataset1_cap_state import cap_data,cap_model,evaluate,metrics
from experiment_dataset1_normals import tensor_for


def run(root,testroot):
    import torch
    import torch.nn.functional as F
    lock=(Path.home()/'ai-club-tools/spark-gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out=root/'reference-stress-v1';out.mkdir(exist_ok=False);torch.set_num_threads(4)
    frozen=read(root/'fitted-v1/frozen.json');checkpoint=root/'fitted-v1/best.pt'
    if sha(checkpoint)!=frozen['checkpoint_sha256']:raise ValueError('Frozen cap model differs')
    folder=testroot/'rendered-v1';rows=[json.loads(s) for s in (folder/'manifest.jsonl').read_text().splitlines()]
    data=cap_data(rows,'test',folder);model=cap_model().cuda();model.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=True)['state_dict'])
    cases={}
    for name in ('unchanged','reference_gain_0.9','reference_gain_1.1','reference_shift_1px','reference_shift_3px','current_gain_0.9'):
        variant=dict(data)
        if name.startswith('reference_gain'):variant['golden']=(data['golden']*float(name.rsplit('_',1)[1])).clamp(0,1)
        if name.startswith('reference_shift'):
            pixels=int(name.rsplit('_',1)[1].removesuffix('px'))
            variant['golden']=F.pad(data['golden'],(pixels,0,0,0),mode='replicate')[:,:,:,:96]
        if name=='current_gain_0.9':variant['current']=(data['current']*.9).clamp(0,1)
        records=evaluate(model,variant);normal=[r for r in records if r['variant']=='golden_as_current']
        cases[name]=dict(metrics=metrics(records),normal_exact=metrics(normal),
            policies={key:metrics(records,t)['cap_warning'] for key,t in frozen['thresholds'].items()},
            normal_policies={key:metrics(normal,t)['cap_warning'] for key,t in frozen['thresholds'].items()})
        write(out/(name+'-predictions.json'),records)
    with torch.inference_mode():
        ix=torch.arange(60,device='cuda');x=tensor_for(data,'detail_paired',ix)
        for _ in range(30):model(x)
        torch.cuda.synchronize();times=[]
        for _ in range(100):
            started=time.perf_counter();model(x);torch.cuda.synchronize();times.append((time.perf_counter()-started)*1000)
    times.sort()
    write(out/'completed.json',dict(checkpoint_sha256=frozen['checkpoint_sha256'],cases=cases,training=False,thresholds_changed=False,
        helper_sha256=sha(__file__),real_accuracy=None,source_pixel_dimensions=[640,480],
        classifier_60_crop_ms=dict(median=(times[49]+times[50])/2,p95=times[94],excludes='Decode/crop/alignment/transport/browser/reference capture'),
        interpretation='Digital held-out synthetic perturbations. Registration mismatch should block qualified operation, not be interpreted as physical defects.'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','testroot'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.root,a.testroot)
