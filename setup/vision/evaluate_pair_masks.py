"""Frozen GB10 mask probes and unapproved exact-frame annotation drafts."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

from train_pair_masks import ROI,ANCHOR,MODES,build_model,features
from liquid_agreement_metrics import score


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def infer(model,mode,rgb,reference):
    import numpy as np
    import torch
    x,y,r,b=ROI;a,c,d,e=ANCHOR
    cur=torch.from_numpy(rgb[y:b,x:r].transpose(2,0,1).copy()).float().cuda()[None]/255
    ref=torch.from_numpy(reference[y:b,x:r].transpose(2,0,1).copy()).float().cuda()[None]/255
    ca=torch.tensor(np.median(rgb[c:e,a:d],axis=(0,1)),dtype=torch.float32,device='cuda')[None]/255
    ra=torch.tensor(np.median(reference[c:e,a:d],axis=(0,1)),dtype=torch.float32,device='cuda')[None]/255
    with torch.inference_mode():logits,p=model(features(cur,ref,ca,ra,mode))
    mask=np.zeros((480,640),dtype=np.uint8);mask[y:b,x:r]=(logits[0].sigmoid().cpu().numpy()>=.5).astype(np.uint8)
    return mask,float(p.sigmoid()[0])


def largest(mask):
    import cv2
    import numpy as np
    n,labels,stats,_=cv2.connectedComponentsWithStats(mask)
    if n==1:return mask.copy(),None
    k=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]));x,y,w,h,_=stats[k]
    return (labels==k).astype(np.uint8),[int(x),int(y),int(x+w),int(y+h)]


def main(a):
    import fcntl
    import os
    a.out.mkdir(parents=True,exist_ok=False)
    write(a.out/'queued.json',dict(pid=os.getpid(),queued=time.time()))
    lock=(Path.home()/'ai-club-tools/spark-gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX)
    import av
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from ultralytics import YOLO
    sys.path.insert(0,str(a.bundle/'repo'))
    from replay.geometry import encode_mask
    from replay.config import load_config
    from replay.measurement import Extractor
    torch.set_num_threads(4);cv2.setNumThreads(4)
    models={};metadata={}
    for mode in MODES:
        root=a.models/mode;meta=read(root/'frozen-model.json')
        if not (root/'completed.json').exists() or sha(root/'best.pt')!=meta['checkpoint_sha256']:
            raise ValueError('Completed frozen mask model required')
        assert not meta['training_real_runs'] and not meta['selection_real_runs'] and not meta['run5_used']
        model=build_model(3 if mode=='single_rgb' else 6)
        model.load_state_dict(torch.load(root/'best.pt',map_location='cpu',weights_only=True)['model_state']);models[mode]=model.cuda().eval();metadata[mode]=meta
    primary=max(MODES,key=lambda mode:read(a.models/mode/'synthetic-development.json')['raw']['selection_score'])
    video=read(a.bundle/'specs/real-video.json');source=a.bundle/video['file']
    assert sha(source)==video['sha256']
    cycles=read(a.bundle/'specs/onset-review.json')['cycles'];assert len(cycles)==6
    write(a.out/'plan.json',dict(modes=MODES,models=metadata,primary_draft_mode=primary,
        primary_selection='Highest synthetic validation score, chosen before real image access.',
        source_video_sha256=video['sha256'],cycles=cycles,source_run=4,run5_used=False,
        reference='First decoded source frame of each predeclared review clip; NOT certified empty or same stationary vial.',
        mask_threshold=.5,presence_threshold=.5,roi=ROI,training=False,approved_labels=False,
        helper_sha256=sha(__file__),mask_helper_sha256=sha(Path(__file__).with_name('train_pair_masks.py')),
        limits=['Real presence, mask and onset accuracy unavailable until human review.',
            'Static teacher boxes are provisional; temporal models are not scored on stills with invented references.',
            'Model masks/labels here are editable drafts, never evaluation truth.']))
    # Single-frame mask model on the unchanged 157 original real references.
    real=a.bundle/'tools/results/real-synthetic-detector-20261019/data/real';stills=[]
    for row in read(real/'provenance.json')['samples']:
        assert row['source_run']==4 and row['split']=='val'
        image=real/row['image'];label=real/row['labels']
        if sha(image)!=row['image_sha256'] or sha(label)!=row['actual_label_sha256']:raise ValueError('Real still source differs')
        rgb=np.asarray(Image.open(image).convert('RGB')).copy();mask,p=infer(models['single_rgb'],'single_rgb',rgb,rgb)
        _,box=largest(mask);targets=[]
        for line in label.read_text().splitlines():
            cls,x,y,w,h=map(float,line.split())
            if cls==1:targets.append(dict(class_id=1,box=[(x-w/2)*640,(y-h/2)*480,(x+w/2)*640,(y+h/2)*480]))
        stills.append(dict(source_id=row['source_id'],targets=targets,predictions=[] if box is None else [dict(class_id=1,box=box,confidence=p)]))
    write(a.out/'single-rgb-real-still-predictions.json',stills);write(a.out/'single-rgb-real-still-agreement.json',score(stills,1))
    vial_meta=read(a.vial.with_suffix('.json'))
    assert sha(a.vial)==vial_meta['checkpoint_sha256'] and not vial_meta['train_runs'] and not vial_meta['development_runs']
    vial_model=YOLO(str(a.vial))
    config_path=Path.home()/'ai-club-tools/experiments/gb10-station-benchmark-20261022/bundle-v1/repo/replay/examples/bootstrap-vision.json'
    config=load_config(config_path)
    write(a.out/'runtime.json',dict(device=torch.cuda.get_device_name(),torch=torch.__version__,vial_model_sha256=sha(a.vial),
        qa_config=config.to_dict(),legacy_config_detector_not_loaded=True,
        qa_code_sha256={name:sha(a.bundle/'repo/replay'/name) for name in ['geometry.py','measurement.py','config.py']}))
    class Adapter:
        def set(self,rgb,predictions,proposals):
            self.digest=hashlib.sha256(rgb.tobytes()).hexdigest();self.predictions=copy.deepcopy(predictions);self.proposals=copy.deepcopy(proposals)
            for p in self.predictions:p['box']=[int(v) for v in p['box']]
        def check(self,rgb):
            if hashlib.sha256(rgb.tobytes()).hexdigest()!=self.digest:raise ValueError('Stale frame proposals')
        def detect(self,rgb):self.check(rgb);return copy.deepcopy(self.predictions)
        def propose(self,rgb,vial):self.check(rgb);return copy.deepcopy(self.proposals)
    offsets=[-60,-30,0,15,30,45,60,75,90,120,150,180]
    wanted={};drafts=set()
    for cycle in cycles:
        anchor=cycle['provisional_anchor_frame'];start=cycle['source_start_frame'];end=cycle['source_end_frame_exclusive']
        selected=set(range(start,end,6))|{anchor+offset for offset in offsets}
        for index in selected:wanted[index]=cycle
        drafts.update(anchor+offset for offset in offsets)
    for name in ['frames','draft-masks','review-clips']:(a.out/name).mkdir()
    output=a.out/'pair-mask-real-comparison.mp4';writer=cv2.VideoWriter(str(output),cv2.VideoWriter_fourcc(*'mp4v'),5.,(1920,560))
    if not writer.isOpened():raise RuntimeError('Video writer unavailable')
    records=[];tasks=[];reference=None;current=None;adapters={};extractors={};shown=0
    started=time.perf_counter()
    try:
        with av.open(str(source)) as container,(a.out/'frames.jsonl').open('w') as log:
            for index,frame in enumerate(container.decode(video=0)):
                if index>max(wanted):break
                if index not in wanted:continue
                cycle=wanted[index];rgb=frame.to_ndarray(format='rgb24')
                if cycle['case_id']!=current:
                    assert index==cycle['source_start_frame'];current=cycle['case_id'];reference=rgb.copy();reference_index=index
                    reference_sha=hashlib.sha256(reference.tobytes()).hexdigest()
                    for mode in MODES:
                        adapters[mode]=Adapter();extractors[mode]=Extractor(config,detector=adapters[mode]);extractors[mode].geometry.segmenter=adapters[mode]
                native=vial_model.predict([rgb[:,:,::-1].copy()],imgsz=640,rect=False,conf=.35,iou=.45,max_det=300,device='cuda:0',verbose=False,save=False)[0]
                vials=[dict(class_id=0,box=p[:4].tolist(),confidence=float(p[4])) for p in native.boxes.data.cpu().numpy() if int(p[5])==0]
                canvas=np.full((560,1920,3),22,np.uint8);results={};task_drafts=[]
                for col,mode in enumerate(MODES):
                    mask,p=infer(models[mode],mode,rgb,reference);component,box=largest(mask)
                    accepted=box is not None and p>=.5
                    proposal=[] if not accepted else [dict(class_id=1,box=box,confidence=p,candidate_mask=encode_mask(component))]
                    predictions=list(vials)+([] if not accepted else [dict(class_id=1,box=box,confidence=p)])
                    adapters[mode].set(rgb,predictions,proposal);measured=extractors[mode].measure(rgb)
                    measured['geometry'].update(liquid_proposal_method=mode,segmentation_model_sha256=metadata[mode]['checkpoint_sha256'],
                        model_sha256=metadata[mode]['checkpoint_sha256'],vial_model_sha256=sha(a.vial),mask_probability_threshold=.5)
                    results[mode]=dict(presence=p,raw_mask=encode_mask(mask),component_mask=encode_mask(component),presence_gate_passed=accepted,
                        measurement=measured,raw_mask_pixels=int(mask.sum()))
                    display=rgb.copy()
                    if accepted:display[component.astype(bool)]=(.55*display[component.astype(bool)]+.45*np.array([240,80,210])).astype(np.uint8)
                    canvas[64:544,col*640:(col+1)*640]=cv2.cvtColor(display,cv2.COLOR_RGB2BGR)
                    cv2.putText(canvas,f'{mode} | p {p:.2f} | QA {measured["measurement_valid"]}',(col*640+12,25),0,.55,(240,240,240),1)
                    cv2.putText(canvas,f'{current} | source f{index} | ref f{reference_index} UNREVIEWED',(col*640+12,49),0,.45,(200,200,200),1)
                    if index in drafts:
                        path=f'draft-masks/{current}-f{index:06d}-{mode}.png';Image.fromarray(component*255).save(a.out/path)
                        task_drafts.append(dict(mode=mode,model_sha256=metadata[mode]['checkpoint_sha256'],presence_probability=p,
                            mask=path,mask_sha256=sha(a.out/path),status='machine_draft_not_approved'))
                row=dict(case_id=current,source_frame=index,pts=frame.pts,time_base=str(frame.time_base),
                    source_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),reference_frame=reference_index,
                    reference_rgb_sha256=reference_sha,reference_status='unreviewed_not_certified_empty',modes=results)
                log.write(json.dumps(row,allow_nan=False)+'\n');records.append(row)
                if (index-cycle['source_start_frame'])%6==0 and index<cycle['source_end_frame_exclusive']:
                    writer.write(canvas);shown+=1
                if index in drafts:
                    path=f'frames/{current}-f{index:06d}.png';Image.fromarray(rgb).save(a.out/path)
                    tasks.append(dict(sample_id=f'{current}-f{index:06d}',case_id=current,frame_index=index,
                        pts=frame.pts,time_base=str(frame.time_base),image=path,image_sha256=sha(a.out/path),source_rgb_sha256=row['source_rgb_sha256'],
                        source_video_sha256=video['sha256'],drafts=task_drafts,review_status='unreviewed',reviewer=None))
                if index==cycle['provisional_anchor_frame']+30:cv2.imwrite(str(a.out/(current+'-comparison.png')),canvas)
    finally:writer.release()
    assert len(records)==264 and len(tasks)==72 and shown==240
    with av.open(str(output)) as c:decoded=sum(1 for _ in c.decode(video=0))
    assert decoded==shown
    write(a.out/'review-drafts.json',dict(schema='visible-pool-mask-drafts/v1',tasks=tasks,primary_mode=primary,
        primary_selection='Synthetic validation only',approved_labels=False,source_video_sha256=video['sha256'],
        review_status='unreviewed',presence_meaning='Model proposal; absence of a proposed mask never confirms an empty vial.'))
    summary={mode:dict(frames=len(records),presence_positive=sum(r['modes'][mode]['presence']>=.5 for r in records),
        qa_pass=sum(r['modes'][mode]['measurement']['measurement_valid'] for r in records)) for mode in MODES}
    write(a.out/'comparison.json',dict(summary=summary,single_rgb_real_still_provisional_agreement=score(stills,1),
        video_frames=shown,video_fps=5,decoded_video_frames=decoded,draft_frames=72,wall_s=time.perf_counter()-started,
        real_mask_accuracy=None,real_presence_accuracy=None,onset_accuracy=None,approved_labels=False,run5_used=False,app_changed=False))
    write(a.out/'completed.json',dict(artifacts={p.name:sha(p) for p in a.out.iterdir() if p.is_file()},training=False,run5_used=False))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['bundle','models','vial','out']:p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
