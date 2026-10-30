"""Exploratory SAM refinement of the frozen temporal model's same-frame proposals.

Reuses completed matched vial/temporal inference, runs actual SAM and unchanged
QA. Reports incremental refinement timing, never estimated full-pipeline FPS.
"""
import argparse
import copy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
import time


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def main(a):
    import fcntl
    import os
    a.out.mkdir(parents=True,exist_ok=False)
    parent=read(a.parent/'plan.json');assert (a.parent/'completed.json').exists()
    for name,digest in read(a.parent/'completed.json')['artifacts'].items():assert sha(a.parent/name)==digest
    write(a.out/'plan.json',dict(parent_plan_sha256=sha(a.parent/'plan.json'),helper_sha256=sha(__file__),
        temporal_model_sha256=parent['models']['temporal']['sha256'],sam_sha256=parent['models']['sam']['sha256'],
        source_video_sha256=parent['source_video_sha256'],qa_config=parent['qa_config'],source_run=4,run5_used=False,
        adaptive_exploration=True,purpose='Test SAM masks using temporal proposals; no R-CNN input to this branch.',
        cached_inference='Exact same-frame vial and temporal proposals from completed parent comparison.',
        prompt_policy='Unchanged pool_prompts plus CausalPromptBridge; same-frame raw RGB and past boxes only.',
        warmup=264,passes=2,timing='Incremental prompt/SAM/QA only, excludes cached YOLO and temporal inference.',
        training=False,selection=False,app_changed=False))
    write(a.out/'queued.json',dict(pid=os.getpid(),time=time.time()))
    lock=(Path.home()/'ai-club-tools/spark-gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX)
    import av
    import cv2
    import numpy as np
    import torch
    from ultralytics import SAM
    sys.path.insert(0,str(a.parent/'helpers'));sys.path.insert(0,str(a.v1/'bundle/repo'))
    from pool_prompt_policy import pool_prompts,CausalPromptBridge
    from replay.geometry import encode_mask,decode_mask
    from replay.config import load_config
    from replay.measurement import Extractor
    torch.set_num_threads(4);cv2.setNumThreads(4)
    for name,digest in parent['qa_code_sha256'].items():assert sha(a.v1/'bundle/repo/replay'/name)==digest
    cfg=Path.home()/'ai-club-tools/experiments/gb10-station-benchmark-20261022/bundle-v1/repo/replay/examples/bootstrap-vision.json'
    config=load_config(cfg);assert config.to_dict()==parent['qa_config']
    source=a.v1/'bundle/evaluation/vial_fill_video_4.mp4';assert sha(source)==parent['source_video_sha256']
    rows=[json.loads(s)['arms']['temporal'] for s in (a.parent/'pass-1.jsonl').read_text().splitlines()]
    wanted={r['source_frame']:r for r in rows};frames={}
    with av.open(str(source)) as container:
        for i,frame in enumerate(container.decode(video=0)):
            if i>max(wanted):break
            if i not in wanted:continue
            rgb=frame.to_ndarray(format='rgb24');r=wanted[i]
            assert hashlib.sha256(rgb.tobytes()).hexdigest()==r['source_rgb_sha256']
            assert frame.pts==r['pts'] and str(frame.time_base)==r['time_base'];frames[i]=rgb
    assert len(frames)==264
    sam_path=parent['models']['sam']['file'];assert sha(sam_path)==parent['models']['sam']['sha256'];sam=SAM(sam_path)
    class Adapter:
        def set(self,rgb,preds,proposals):
            self.digest=hashlib.sha256(rgb.tobytes()).hexdigest();self.preds=copy.deepcopy(preds);self.proposals=copy.deepcopy(proposals)
            for p in self.preds:p['box']=[int(v) for v in p['box']]
        def check(self,rgb):
            if hashlib.sha256(rgb.tobytes()).hexdigest()!=self.digest:raise ValueError('Stale proposals')
        def detect(self,rgb):self.check(rgb);return copy.deepcopy(self.preds)
        def propose(self,rgb,vial):self.check(rgb);return copy.deepcopy(self.proposals)
    records=[];first=[]
    for repeat in [-1,0,1]:
        current=[];case=None
        with (a.out/('warmup.jsonl' if repeat<0 else f'pass-{repeat+1}.jsonl')).open('w') as log:
            for row in rows:
                if case!=row['case_id']:
                    adapter=Adapter();extractor=Extractor(config,detector=adapter);extractor.geometry.segmenter=adapter
                    bridge=CausalPromptBridge();case=row['case_id']
                rgb=frames[row['source_frame']];preds=row['vials']+row['liquid_predictions']
                torch.cuda.synchronize();start=time.perf_counter()
                policy=pool_prompts(rgb,preds)
                b=bridge.update(policy,row['source_frame'],float(row['pts']*Fraction(row['time_base'])),row['measurement']['motion'])
                masks=[];proposals=[];called=bool(b['boxes'] or b['points'])
                if called:
                    result=sam.predict(rgb[:,:,::-1].copy(),bboxes=b['boxes'] or None,points=b['points'] or None,labels=b['labels'] or None,
                        device='cuda:0',conf=0.,imgsz=1024,verbose=False,save=False)[0]
                    masks=[] if result.masks is None else [m.astype(bool) for m in result.masks.data.cpu().numpy()]
                    qualities=[] if result.masks is None else result.boxes.conf.cpu().tolist()
                    assert len(masks)==len(b['boxes'])
                    for mask,quality in zip(masks,qualities):
                        yy,xx=np.where(mask)
                        if len(xx):proposals.append(dict(class_id=1,box=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)],
                            confidence=row['presence_score'],candidate_mask=encode_mask(mask),sam_mask_quality=quality))
                torch.cuda.synchronize();sam_end=time.perf_counter()
                adapter.set(rgb,preds,proposals);m=extractor.measure(rgb);end=time.perf_counter()
                m['geometry'].update(model_sha256=parent['models']['temporal']['sha256'],vial_model_sha256=parent['models']['vial']['sha256'],
                    segmentation_model_sha256=parent['models']['sam']['sha256'],liquid_proposal_method='temporal_guided_sam')
                union=np.zeros((480,640),dtype=np.uint8)
                for mask in masks:union|=mask.astype(np.uint8)
                r=dict(case_id=case,source_frame=row['source_frame'],pts=row['pts'],time_base=row['time_base'],source_rgb_sha256=row['source_rgb_sha256'],
                    temporal_proposal_available=row['proposal_available'],proposal_available=bool(proposals),sam_called=called,
                    raw_masks=[encode_mask(mask) for mask in masks],raw_union=encode_mask(union),prompt_policy=policy,bridge=b,measurement=m,
                    timing_ms=dict(prompt_sam=(sam_end-start)*1000,qa=(end-sam_end)*1000,incremental_total=(end-start)*1000))
                log.write(json.dumps(r,allow_nan=False)+'\n');current.append(r)
        if repeat>=0:records+=current
        if repeat==0:first=current
        print(json.dumps(dict(pass_id=repeat+1,frames=len(current),qa_pass=sum(r['measurement']['measurement_valid'] for r in current))),flush=True)
    repeat_diffs=dict(raw_masks=sum(x['raw_masks']!=y['raw_masks'] for x,y in zip(records[:264],records[264:])),
        qa=sum(x['measurement']['measurement_valid']!=y['measurement']['measurement_valid'] for x,y in zip(records[:264],records[264:])),
        final_masks=sum(x['measurement']['geometry']['mask']!=y['measurement']['geometry']['mask'] for x,y in zip(records[:264],records[264:])),
        colors=sum(x['measurement']['lab']!=y['measurement']['lab'] for x,y in zip(records[:264],records[264:])))
    report=dict(source_frames=264,measured_samples=528,temporal_proposal_frames=sum(r['proposal_available'] for r in rows),
        sam_invoked_frames=sum(r['sam_called'] for r in first),refined_proposal_frames=sum(r['proposal_available'] for r in first),
        qa_pass=sum(r['measurement']['measurement_valid'] for r in first),original_temporal_qa=sum(r['measurement']['measurement_valid'] for r in rows),
        qa_gains=sum(r['measurement']['measurement_valid'] and not old['measurement']['measurement_valid'] for r,old in zip(first,rows)),
        qa_losses=sum(not r['measurement']['measurement_valid'] and old['measurement']['measurement_valid'] for r,old in zip(first,rows)),
        repeat_differences=repeat_diffs,
        incremental_latency_ms={k:dict(median=float(np.median([r['timing_ms'][k] for r in records])),p95=float(np.percentile([r['timing_ms'][k] for r in records],95))) for k in ['prompt_sam','qa','incremental_total']},
        accuracy=None,onset_accuracy=None,run5_used=False,app_changed=False,
        limits=['Exploratory arm added after inspecting matched comparison; no new training/threshold selection.',
            'Cached detector/temporal outputs; timing is incremental refinement, not full pipeline.',
            'Availability gains do not establish mask or recipe accuracy; reviewed outlines/onset remain pending.'])
    write(a.out/'comparison.json',report)
    video=a.out/'temporal-vs-sam-refinement.mp4';writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),5,(1280,560))
    assert writer.isOpened();count=0
    try:
        for row,r in zip(rows,first):
            cycle=next(c for c in parent['cycles'] if c['case_id']==r['case_id'])
            if r['source_frame']>=cycle['source_end_frame_exclusive'] or (r['source_frame']-cycle['source_start_frame'])%6:continue
            canvas=np.full((560,1280,3),22,np.uint8)
            for col,(title,entry) in enumerate([('Temporal direct',row),('Temporal prompt + actual SAM2.1',r)]):
                rgb=frames[r['source_frame']].copy();mask=decode_mask(entry['raw_union']).astype(bool)
                rgb[mask]=(.6*rgb[mask]+.4*np.array([45,255,150])).astype(np.uint8)
                canvas[65:545,col*640:(col+1)*640]=rgb[:,:,::-1]
                cv2.putText(canvas,title+' | QA '+str(entry['measurement']['measurement_valid']),(col*640+8,24),0,.55,(240,240,240),1)
                cv2.putText(canvas,f'{r["case_id"]} native f{r["source_frame"]} | UNREVIEWED',(col*640+8,48),0,.47,(215,215,215),1)
            writer.write(canvas);count+=1
    finally:writer.release()
    with av.open(str(video)) as container:decoded=sum(1 for _ in container.decode(video=0))
    assert decoded==count==240
    write(a.out/'completed.json',dict(artifacts={p.name:sha(p) for p in a.out.iterdir() if p.is_file()},
        source_frames=264,video_frames=decoded,local_gpu_used=False,inference='Actual SAM; cached vial/temporal inference.',app_changed=False))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['parent','v1','out']:p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
