"""CPU-only synthetic-detector transfer and input-robustness development probe.

Reuses pinned ONNX exports, not predictions. No Torch/CUDA, training, labels,
renderer changes or operational decision changes. All Run 4 references remain
machine proposals; real accuracy and visible-pool onset are unestablished.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import cv2
import numpy as np
import onnxruntime as ort

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiment_sim_real_robustness import transform

CONDITIONS=['raw','green_085','blur_sigma12']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def overlaps(box,boxes):
    if not len(boxes):
        return np.empty(0)
    boxes=np.asarray(boxes,dtype=np.float64).reshape(-1,4);box=np.asarray(box,dtype=np.float64)
    xy=np.maximum(0,np.minimum(box[2:],boxes[:,2:])-np.maximum(box[:2],boxes[:,:2]))
    intersection=xy[:,0]*xy[:,1]
    return intersection/np.maximum(1e-12,np.prod(box[2:]-box[:2])+np.prod(boxes[:,2:]-boxes[:,:2],axis=1)-intersection)


def predict(session,rgb):
    if rgb.shape!=(480,640,3):
        raise ValueError('Expected native 640x480 RGB')
    canvas=np.full((640,640,3),114,np.uint8);canvas[80:560]=rgb
    blob=np.ascontiguousarray(canvas.transpose(2,0,1)[None],dtype=np.float32)/255.
    raw=session.run(None,{session.get_inputs()[0].name:blob})[0]
    if raw.shape!=(1,6,8400):
        raise ValueError('Expected two-class YOLO11n output, got '+str(raw.shape))
    raw=raw[0].T
    classes=np.argmax(raw[:,4:],axis=1);scores=raw[np.arange(len(raw)),4+classes]
    indices=np.flatnonzero(scores>=.001)
    boxes=np.column_stack((raw[:,0]-raw[:,2]/2,raw[:,1]-raw[:,3]/2,raw[:,0]+raw[:,2]/2,raw[:,1]+raw[:,3]/2))
    selected=[]
    for cls in (0,1):
        ordered=indices[classes[indices]==cls]
        ordered=ordered[np.argsort(-scores[ordered],kind='stable')]
        while len(ordered):
            current=ordered[0];selected.append(current)
            if len(ordered)==1:
                break
            ordered=ordered[1:][overlaps(boxes[current],boxes[ordered[1:]])<=.45]
    selected=sorted(selected,key=lambda i:-float(scores[i]))[:300]
    result=[]
    for i in selected:
        box=boxes[i].astype(float);box[[1,3]]-=80
        box[[0,2]]=np.clip(box[[0,2]],0,640);box[[1,3]]=np.clip(box[[1,3]],0,480)
        if box[2]>box[0] and box[3]>box[1]:
            result.append(dict(class_id=int(classes[i]),confidence=float(scores[i]),box=box.tolist()))
    return result


def score(rows,cls):
    targets={r['source_id']:[t['box'] for t in r['targets'] if t['class_id']==cls] for r in rows}
    count=sum(map(len,targets.values()))
    predictions=sorted([(r['source_id'],p) for r in rows for p in r['predictions'] if p['class_id']==cls],key=lambda pair:-pair[1]['confidence'])
    correct=np.zeros((len(predictions),10),bool)
    for j,threshold in enumerate(np.linspace(.5,.95,10)):
        claimed=defaultdict(set)
        for i,(sid,p) in enumerate(predictions):
            ious=overlaps(p['box'],targets[sid])
            eligible=[k for k in range(len(ious)) if ious[k]>=threshold and k not in claimed[sid]]
            if eligible:
                k=max(eligible,key=lambda k:ious[k]);claimed[sid].add(k);correct[i,j]=True
    aps=[]
    for j in range(10):
        tp=np.cumsum(correct[:,j]);fp=np.cumsum(~correct[:,j]);recall=tp/max(1,count);precision=tp/np.maximum(1,tp+fp)
        envelope=np.maximum.accumulate(precision[::-1])[::-1]
        aps.append(float(np.mean([max(envelope[recall>=r],default=0.) for r in np.linspace(0,1,101)])))
    operating=np.array([p['confidence']>=.35 for _,p in predictions],bool)
    tp=int(correct[operating,0].sum());number=int(operating.sum())
    strata={name:dict(targets=0,detected=0) for name in ['height_le32','height_32_to64','height_gt64']}
    for r in rows:
        if cls!=1:
            break
        ts=[t['box'] for t in r['targets'] if t['class_id']==cls]
        if len(ts)>1:
            raise ValueError('Expected at most one provisional liquid target')
        if ts:
            height=ts[0][3]-ts[0][1];key='height_le32' if height<=32 else 'height_32_to64' if height<=64 else 'height_gt64'
            strata[key]['targets']+=1
            ps=[p for p in r['predictions'] if p['class_id']==cls and p['confidence']>=.35]
            strata[key]['detected']+=int(any(overlaps(p['box'],ts)[0]>=.5 for p in ps))
    return dict(targets=count,ap50_101point=aps[0],ap50_95_101point=float(np.mean(aps)),
        confidence=.35,iou=.5,tp=tp,unmatched_boxes=number-tp,missed_targets=count-tp,
        precision=tp/number if number else 0.,recall=tp/count if count else None,
        provisional_negative_frames_with_predictions=sum(not any(t['class_id']==cls for t in r['targets']) and any(p['class_id']==cls and p['confidence']>=.35 for p in r['predictions']) for r in rows),
        target_height_strata=strata if cls==1 else None)


def metrics_self_check():
    target=dict(class_id=1,box=[0,0,10,10])
    pred=dict(**target,confidence=.9)
    r=score([dict(source_id='a',targets=[target],predictions=[pred,dict(**target,confidence=.8)]),dict(source_id='b',targets=[target],predictions=[])],1)
    assert (r['tp'],r['unmatched_boxes'],r['missed_targets'])==(1,1,1)
    assert r['precision']==r['recall']==.5
    assert .50<r['ap50_101point']<.51


def main(a):
    metrics_self_check()
    if a.out.exists():
        raise ValueError('Use a fresh output directory')
    a.out.mkdir(parents=True)
    data=a.tools/'results/real-synthetic-detector-20261019/data/real'
    provenance=read(data/'provenance.json');sources=[r for r in provenance['samples'] if r['split']=='val']
    if len(sources)!=157 or any(r['source_run']!=4 or r.get('parent_id') for r in sources):
        raise ValueError('Expected 157 original Run 4 images')
    held=set(provenance['held_out_source_hashes'])
    for r in sources:
        if r['source_sha256'] in held or sha(data/r['image'])!=r['image_sha256'] or sha(data/r['labels'])!=r['actual_label_sha256']:
            raise ValueError('Real source identity or split differs')
    models={
        'legacy_real_baseline':ROOT/'replay/models/vial-bootstrap.onnx',
        'original_sim_only':a.tools/'results/real-synthetic-detector-20261019/models/sim-only/detector.onnx',
        'revised_sim_only':a.tools/'results/revised-render-detector-20261019/models/sim-only/detector.onnx'}
    metadata={}
    for name,path in models.items():
        if name=='legacy_real_baseline':
            if sha(path)!='89ed11f7046171e5340dc45a9d528010de1cefbba21e163b23b7098151e0880e':
                raise ValueError('Frozen baseline differs')
            metadata[name]=dict(model_sha256=sha(path),training_status='Legacy real-data-trained comparator; not a synthetic-only submission model.')
        else:
            m=read(path.with_name('model.json'))
            if sha(path)!=m['model_sha256'] or m['train_runs'] or m['development_runs'] or m['excluded_real_runs']!=[1,2,3,4,5]:
                raise ValueError('Synthetic-only model boundary differs')
            metadata[name]=m
    cache=a.tools/'results/operator-prerender-20261021/cache-v1';sim=read(cache/'manifest.json')
    if sha(cache/sim['video'])!=sim['video_sha256']:
        raise ValueError('Isaac input identity differs')
    samples=[]
    for r in sources:
        targets=[]
        for line in (data/r['labels']).read_text().splitlines():
            c,x,y,w,h=map(float,line.split());targets.append(dict(class_id=int(c),box=[(x-w/2)*640,(y-h/2)*480,(x+w/2)*640,(y+h/2)*480]))
        samples.append(dict(domain='real_run4',source_id=r['source_id'],path=str(data/r['image']),image_sha256=r['image_sha256'],targets=targets))
    for r in sim['samples']:
        if sha(cache/r['png'])!=r['png_sha256']:
            raise ValueError('Isaac PNG differs')
        samples.append(dict(domain='isaac_cached',source_id=f'isaac-{r["video_frame"]:06}',path=str(cache/r['png']),image_sha256=r['png_sha256'],targets=None))
    write(a.out/'plan.json',dict(schema_version=1,kind='frozen_cpu_sim_transfer_probe',conditions=CONDITIONS,models=metadata,
        metric='Confidence-ordered one-to-one matching; 101-point precision-envelope AP, 10 IoU thresholds .50:.05:.95. Not numerically identical to prior Ultralytics AP integration.',
        preprocessing='Native RGB, 80-pixel top/bottom pad to 640x640, /255 float32; best class per anchor; confidence .001; per-class NMS .45; max300; fixed operating confidence .35.',
        real_source_provenance_sha256=sha(data/'provenance.json'),isaac_manifest_sha256=sha(cache/'manifest.json'),
        samples=samples,runner_sha256=sha(__file__),transform_helper_sha256=sha(Path(__file__).with_name('experiment_sim_real_robustness.py')),
        backend='ONNX Runtime CPUExecutionProvider; 4 intra-op threads, 1 inter-op thread',training=False,run5_used=False,
        note='Raw Dataset 1 unavailable. This probe uses Dataset 2 Run 4 and cached Isaac filling footage; no final-fixture accuracy claimed.'))
    all_rows=[];summary={}
    for name,path in models.items():
        options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
        options.enable_cpu_mem_arena=False
        session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
        predict(session,np.zeros((480,640,3),np.uint8))
        summary[name]={}
        for condition in CONDITIONS:
            rows=[];started=time.perf_counter()
            for sample in samples:
                rgb=cv2.cvtColor(cv2.imread(sample['path']),cv2.COLOR_BGR2RGB)
                predictions=predict(session,transform(rgb,condition))
                row=dict(**sample,model=name,condition=condition,predictions=predictions)
                rows.append(row);all_rows.append(row)
            real=[r for r in rows if r['domain']=='real_run4'];simulated=[r for r in rows if r['domain']=='isaac_cached']
            stats=dict(real_frames=len(real),isaac_frames=len(simulated),elapsed_s=time.perf_counter()-started,
                real_vial=score(real,0),real_liquid=score(real,1),
                isaac_liquid_proposal_frames=sum(any(p['class_id']==1 and p['confidence']>=.35 for p in r['predictions']) for r in simulated))
            summary[name][condition]=stats
            write(a.out/f'{name}--{condition}.json',rows)
            print(json.dumps(dict(model=name,condition=condition,**stats)),flush=True)
        del session;gc.collect()
    for name in models:
        baseline={r['source_id']:r for r in all_rows if r['model']==name and r['condition']=='raw'}
        for condition in CONDITIONS:
            for domain in ('real_run4','isaac_cached'):
                rows=[r for r in all_rows if r['model']==name and r['condition']==condition and r['domain']==domain]
                raw_positive=retained=0
                for r in rows:
                    raw_has=any(p['class_id']==1 and p['confidence']>=.35 for p in baseline[r['source_id']]['predictions'])
                    current_has=any(p['class_id']==1 and p['confidence']>=.35 for p in r['predictions'])
                    raw_positive+=raw_has;retained+=raw_has and current_has
                summary[name][condition][domain+'_proposal_retention']=dict(raw_positive_frames=raw_positive,retained=retained,fraction=retained/raw_positive if raw_positive else None)
    with (a.out/'predictions.jsonl').open('x',encoding='utf-8') as f:
        for r in all_rows:
            f.write(json.dumps(r,allow_nan=False)+'\n')
    result=dict(schema_version=1,kind='frozen_cpu_sim_transfer_probe',attempts=len(all_rows),summary=summary,
        training=False,run5_used=False,labels_changed=False,promoted=False,hardware_actions=0,
        reviewed_real_accuracy=None,sim_to_real_gap_closed=False,
        limitations=['Real references are provisional machine boxes; missing liquid labels are not human-confirmed absence.',
            'Legacy real liquid target scope can include stream; synthetic targets represent accumulated body, confounding attribution to appearance.',
            'Simulation proposal availability is not synthetic accuracy; no geometric proxy masks are treated as visible-liquid truth.',
            'Sources are already inspected development data, and adjacent frames are correlated.',
            'Synthetic-only describes CSI domain fine-tuning; all models inherit generic COCO pretraining.',
            'The legacy real-data-trained model is a comparator, not proof of an eligible synthetic-only solution.',
            'These are detector results, not full Faster R-CNN/SAM/QA or physical recipe-defect metrics.'])
    write(a.out/'comparison.json',result)
    # No Torch needed to plot or inspect the outputs.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=list(models);fig,axes=plt.subplots(1,2,figsize=(13,5))
    for j,condition in enumerate(CONDITIONS):
        axes[0].bar(np.arange(3)+(j-1)*.24,[summary[n][condition]['real_liquid']['ap50_95_101point']*100 for n in names],width=.24,label=condition)
        axes[1].bar(np.arange(3)+(j-1)*.24,[summary[n][condition]['real_liquid']['recall']*100 for n in names],width=.24,label=condition)
    for ax,title in zip(axes,['Run 4 liquid AP50–95 / 101-point interpolation','Run 4 liquid recall / confidence .35, IoU .5']):
        ax.set_xticks(range(3),['Legacy real comparator','Original sim-only','Revised sim-only']);ax.set_title(title,fontsize=11);ax.set_ylim(0,100);ax.set_ylabel('Percent');ax.legend(fontsize=8)
    fig.suptitle('Frozen synthetic-detector transfer | 157 real development images, 89 provisional liquid targets')
    fig.text(.5,.02,'Actual CPU inference; no retraining. Machine-reference agreement, not reviewed accuracy. Target scope differs.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.92));fig.savefig(a.out/'transfer-scorecard.png',dpi=170);plt.close(fig)
    # Three predeclared short-target source IDs, chosen by source order, not errors.
    positive=[s for s in samples if s['domain']=='real_run4' and any(t['class_id']==1 and t['box'][3]-t['box'][1]<=64 for t in s['targets'])][:3]
    sheets=[]
    for s in positive:
        panels=[]
        for name in ['Provisional target',*names]:
            im=cv2.imread(s['path']);row=next(r for r in all_rows if r['source_id']==s['source_id'] and r['condition']=='raw')
            ps=s['targets'] if name=='Provisional target' else next(r['predictions'] for r in all_rows if r['source_id']==s['source_id'] and r['condition']=='raw' and r['model']==name)
            for p in ps:
                if p['class_id']!=1 or p.get('confidence',1)<.35:
                    continue
                x,y,right,bottom=np.rint(p['box']).astype(int);cv2.rectangle(im,(x,y),(right,bottom),(50,235,90),2)
            top=np.full((35,640,3),25,np.uint8);cv2.putText(top,name,(10,24),cv2.FONT_HERSHEY_SIMPLEX,.6,(245,245,245),1)
            panels.append(cv2.resize(np.vstack([top,im]),(480,386)))
        sheets.append(np.hstack(panels))
    if sheets:
        cv2.imwrite(str(a.out/'short-pool-examples.png'),np.vstack(sheets))
    shutil.copy2(__file__,a.out/Path(__file__).name)
    write(a.out/'completed.json',dict(attempts=len(all_rows),real_sources=157,isaac_sources=111,models=3,conditions=3,
        metrics_self_check=True,artifacts={p.name:sha(p) for p in a.out.iterdir() if p.is_file()}))
    print('COMPLETED',a.out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tools',type=Path,default=Path.home()/'ai-club-tools')
    p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
