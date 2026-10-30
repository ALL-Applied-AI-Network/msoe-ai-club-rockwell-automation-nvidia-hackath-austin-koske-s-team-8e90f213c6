"""Run frozen models on rendered placement states; emit actual source-bound findings."""
import argparse
import importlib.util
from pathlib import Path
import json
from collections import Counter
from dataset1_fiveclass import CLASSES,SLOTS,read,sha,write


def evaluate(parent,scenes,out):
    import fcntl
    lock=(Path.home()/'ai-club-tools/spark-gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out.mkdir(exist_ok=False)
    from train_dataset1_fiveclass import build_model
    from PIL import Image
    import numpy as np
    import torch
    spec=importlib.util.spec_from_file_location('fixture_placement',parent/'fixture_placement.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    frozen=read(parent/'fitted-v1/completed.json')['models']
    mode=max(('rgb','target_rgb'),key=lambda m:frozen[m]['development']['macro_recall']);meta=frozen[mode]
    checkpoint=parent/'fitted-v1'/mode/'best.pt'
    if sha(checkpoint)!=meta['checkpoint_sha256']:raise ValueError('Frozen checkpoint differs')
    renderer=read(scenes/'rendered-v1/completed.json')
    if renderer['manifest_sha256']!=sha(scenes/'rendered-v1/manifest.jsonl'):raise ValueError('Rendered evidence differs')
    torch.set_num_threads(4);model=build_model(meta['channels']).cuda();model.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=True)['state_dict']);model.eval()
    yy,xx=np.mgrid[:96,:96];heat=torch.from_numpy(np.exp(-((xx-48)**2+(yy-48)**2)/(2*9**2)).astype(np.float32))[None,None].cuda()
    records=[];reports=[]
    with torch.inference_mode():
        for row in [json.loads(s) for s in (scenes/'rendered-v1/manifest.jsonl').read_text().splitlines()]:
            path=scenes/'rendered-v1'/row['image']
            if sha(path)!=row['image_sha256']:raise ValueError('Source RGB differs')
            with Image.open(path) as image:
                x=torch.stack([torch.from_numpy((np.array(image.convert('RGB').crop(row['boxes'][slot]),np.float32)/255).transpose(2,0,1).copy()) for slot in SLOTS]).cuda()
            if mode=='target_rgb':x=torch.cat([x,heat.expand(60,-1,-1,-1)],1)
            probs=model(x).softmax(1).cpu().tolist();predictions=[]
            for slot,p in zip(SLOTS,probs):
                predicted=CLASSES[max(range(5),key=lambda i:p[i])]
                record=dict(scene=row['id'],slot=slot,proposed_class=predicted,probabilities=p,box_xyxy=row['boxes'][slot],source_sha256=row['image_sha256'],
                    truth=row['labels'][slot],mapping_confirmed=True,visibility='confirmed',qualification_basis='Known authored synthetic targets; not a real reference approval')
                record.update(defect_score=1-p[0],defect_threshold=meta['threshold'],occupied_defect_warning=1-p[0]>=meta['threshold'])
                predictions.append(record);records.append(record)
            assessments={}
            for fixture in ('G','S'):
                context=dict(fixture=fixture,order_id='SYNTHETIC-REHEARSAL',position_order=[f'{fixture}{i:02d}' for i in range(1,31)],
                    placement_count=row['nuisance']['placement_count'][fixture],order_complete=row['nuisance']['order_complete'][fixture],
                    reviewer='SIMULATOR CONTEXT',evidence_id='synthetic-plan:'+row['id'])
                # Input to assessment is prediction + public progress context only.
                observations=[{k:p[k] for k in ('slot','proposed_class','mapping_confirmed','visibility','occupied_defect_warning')} for p in predictions if p['slot'][0]==fixture]
                assessments[fixture]=module.assess(observations,context)
            reports.append(dict(scene=row['id'],source_sha256=row['image_sha256'],assessments=assessments,
                predictions_correct=sum(p['proposed_class']==p['truth'] for p in predictions),targets=60,
                truth_only_for_evaluation=True))
    write(out/'predictions.json',records);write(out/'assessments.json',reports)
    write(out/'completed.json',dict(model=mode,checkpoint_sha256=meta['checkpoint_sha256'],scenes=len(reports),targets=len(records),
        correct=sum(p['proposed_class']==p['truth'] for p in records),per_scene_correct={r['scene']:r['predictions_correct'] for r in reports},
        proposed_counts=dict(Counter(r['proposed_class'] for r in records)),helper_sha256=sha(__file__),
        placement_helper_sha256=sha(parent/'fixture_placement.py'),predictions_sha256=sha(out/'predictions.json'),
        assessments_sha256=sha(out/'assessments.json'),counts_are_synthetic=True,rendered_states_not_robot_motion=True,
        real_accuracy=None,quality_release_authorized=False,hardware_commands=0))
    print(json.dumps(read(out/'completed.json')),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent',type=Path,required=True);p.add_argument('--scenes',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();evaluate(a.parent.resolve(),a.scenes.resolve(),a.out.resolve())
