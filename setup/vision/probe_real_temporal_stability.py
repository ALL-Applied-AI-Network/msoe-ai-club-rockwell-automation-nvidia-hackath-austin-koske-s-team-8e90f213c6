"""GB10 frozen temporal-mask stability, native-clock and branch-latency probes.

Real perturbation agreement measures sensitivity, not accuracy. No fitting,
threshold search, checkpoint selection, app change, or human approval occurs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def write(p,v): Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def main(a):
    import fcntl
    import os
    a.out.mkdir(parents=True,exist_ok=False)
    conditions=['raw','current_exposure085','current_exposure070','current_exposure130',
                'current_green085','reference_shift2','reference_shift5','late_reference']
    write(a.out/'plan.json',dict(conditions=conditions,modes=['paired_diff','paired_diff_gain'],
        source_run=4,run5_used=False,helper_sha256=sha(__file__),training=False,selection=False,
        mask_threshold=.5,presence_threshold=.5,primary_mode='paired_diff_gain',
        primary_reason='Previously frozen synthetic-validation selection; real perturbations do not select models.',
        interpretation='Prediction consistency against clean model output, not human accuracy or confirmed false alarms.',
        late_reference='Anchor+150 native frame; not certified filled or empty. Deliberately invalid reference control.',
        shift='Full-image reference translation in X with reflected borders; not crop roll.',
        benchmark='Mask branch only: RGB preprocessing, CPU/GPU transfer, features, forward, component mask; excludes vial detector and QA.'))
    write(a.out/'queued.json',dict(pid=os.getpid(),queued=time.time()))
    lock=(Path.home()/'ai-club-tools/spark-gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX)
    import av
    import cv2
    import numpy as np
    import torch
    from PIL import Image,ImageDraw
    sys.path.insert(0,str(a.helper.parent))
    from train_pair_masks import build_model
    from evaluate_pair_masks import infer,largest
    torch.set_num_threads(4);cv2.setNumThreads(4)
    probe_plan=read(a.probe/'plan.json')
    for name,digest in read(a.probe/'completed.json')['artifacts'].items():
        if sha(a.probe/name)!=digest:raise ValueError('Completed probe differs: '+name)
    assert sha(a.video)==probe_plan['source_video_sha256']
    rows=[json.loads(s) for s in (a.probe/'frames.jsonl').read_text().splitlines()]
    expected={r['source_frame']:r for r in rows};frames={}
    with av.open(str(a.video)) as container:
        for index,frame in enumerate(container.decode(video=0)):
            if index>max(expected):break
            if index not in expected:continue
            rgb=frame.to_ndarray(format='rgb24');row=expected[index]
            assert hashlib.sha256(rgb.tobytes()).hexdigest()==row['source_rgb_sha256']
            assert frame.pts==row['pts'] and str(frame.time_base)==row['time_base']
            frames[index]=rgb
    assert len(frames)==264
    cycles={c['case_id']:c for c in probe_plan['cycles']}
    write(a.out/'runtime.json',dict(torch=torch.__version__,device=torch.cuda.get_device_name(),
        gpu_lock=True,local_gpu_used=False,source_video_sha256=sha(a.video),
        inference_helper_sha256=sha(a.helper.parent/'evaluate_pair_masks.py'),
        mask_helper_sha256=sha(a.helper),source_frames=264,native_rgb_and_pts_verified=True))
    summary={};all_results={}
    for mode in ['paired_diff','paired_diff_gain']:
        meta=read(a.models/mode/'frozen-model.json')
        assert sha(a.models/mode/'best.pt')==meta['checkpoint_sha256']
        assert sha(a.helper)==meta['helper_sha256']
        assert not meta['training_real_runs'] and not meta['selection_real_runs'] and not meta['run5_used']
        model=build_model(6).cuda().eval()
        model.load_state_dict(torch.load(a.models/mode/'best.pt',map_location='cpu',weights_only=True)['model_state'])
        def predict(rgb,ref):
            mask,p=infer(model,mode,rgb,ref);component,box=largest(mask)
            accepted=box is not None and p>=.5
            if not accepted:component.fill(0)
            return component,p,accepted
        clean={};summary[mode]={};all_results[mode]={}
        for condition in conditions:
            records=[];shown={}
            for row in rows:
                rgb=frames[row['source_frame']];ref=frames[row['reference_frame']]
                if condition.startswith('current_exposure'):
                    rgb=np.clip(rgb.astype(np.float32)*(int(condition[-3:])/100),0,255).round().astype(np.uint8)
                elif condition=='current_green085':
                    rgb=rgb.copy();rgb[:,:,1]=np.round(rgb[:,:,1].astype(np.float32)*.85).astype(np.uint8)
                elif condition.startswith('reference_shift'):
                    shift=int(condition[-1]);ref=cv2.warpAffine(ref,np.float32([[1,0,shift],[0,1,0]]),(640,480),borderMode=cv2.BORDER_REFLECT_101)
                elif condition=='late_reference':ref=frames[cycles[row['case_id']]['provisional_anchor_frame']+150]
                mask,p,accepted=predict(rgb,ref)
                if condition=='raw':
                    clean[row['source_frame']]=(mask,p,accepted)
                    assert abs(p-row['modes'][mode]['presence'])<1e-4,'Raw presence did not reproduce'
                baseline,bp,ba=clean[row['source_frame']]
                union=int(((mask>0)|(baseline>0)).sum());inter=int(((mask>0)&(baseline>0)).sum())
                records.append(dict(case_id=row['case_id'],source_frame=row['source_frame'],pts=row['pts'],time_base=row['time_base'],
                    presence=p,accepted=accepted,clean_accepted=ba,clean_presence=bp,
                    mask_pixels=int(mask.sum()),prediction_iou=inter/union if union else 1.,
                    self_reference=row['source_frame']==row['reference_frame']))
                if mode=='paired_diff_gain' and condition in ['raw','current_exposure070','reference_shift5','late_reference'] and row['source_frame']==cycles[row['case_id']]['provisional_anchor_frame']+30:
                    display=rgb.copy();display[mask>0]=(.6*display[mask>0]+.4*np.array([40,255,150])).astype(np.uint8)
                    shown[row['case_id']]=Image.fromarray(display).crop((240,230,400,440))
            positives=[r for r in records if r['clean_accepted']]
            negatives=[r for r in records if not r['clean_accepted']]
            report=dict(frames=len(records),clean_positive_frames=len(positives),clean_negative_frames=len(negatives),
                accepted=sum(r['accepted'] for r in records),presence_decision_flips=sum(r['accepted']!=r['clean_accepted'] for r in records),
                clean_negative_to_positive=sum(r['accepted'] for r in negatives),
                clean_positive_to_negative=sum(not r['accepted'] for r in positives),
                clean_positive_mean_prediction_iou=sum(r['prediction_iou'] for r in positives)/len(positives),
                self_reference_accepted=sum(r['accepted'] for r in records if r['self_reference']),
                self_reference_frames=sum(r['self_reference'] for r in records))
            summary[mode][condition]=report;all_results[mode][condition]=records
            write(a.out/(mode+'-'+condition+'.json'),dict(summary=report,records=records))
            for cid,im in shown.items():im.save(a.out/(cid+'-'+condition+'.png'))
        # Warmed serialized branch timing on actual source inputs, never full-pipeline FPS.
        for row in rows[:20]:predict(frames[row['source_frame']],frames[row['reference_frame']])
        times=[]
        for row in rows[:200]:
            torch.cuda.synchronize();start=time.perf_counter()
            predict(frames[row['source_frame']],frames[row['reference_frame']]);torch.cuda.synchronize()
            times.append((time.perf_counter()-start)*1000)
        summary[mode]['branch_latency_ms']=dict(samples=len(times),warmup=20,median=float(np.median(times)),p95=float(np.percentile(times,95)),
            includes='RGB preprocessing, transfer, features, forward, component mask',excludes='decode, vial detection, QA, transport, browser')
        del model;torch.cuda.empty_cache()
    board=Image.new('RGB',(4*240,6*280),'#101925');d=ImageDraw.Draw(board)
    for y,cid in enumerate(cycles):
        for x,condition in enumerate(['raw','current_exposure070','reference_shift5','late_reference']):
            im=Image.open(a.out/(cid+'-'+condition+'.png'));board.paste(im,(x*240+40,y*280+38))
            d.text((x*240+5,y*280+5),cid+' / '+condition,fill='white')
            d.text((x*240+5,y*280+255),'Unreviewed candidate predictions',fill='#b0c4d0')
    board.save(a.out/'real-stability-comparison.png')
    write(a.out/'comparison.json',dict(models=summary,accuracy=None,onset_accuracy=None,run5_used=False,training=False,
        limits=['Agreement with clean output is not accuracy. No reviewed presence/mask/onset labels.',
        'Digital nuisance changes are not physical lighting/camera validation.',
        'Six inspected development clips with unreviewed references. No threshold or checkpoint changes.']))
    write(a.out/'completed.json',dict(artifacts={p.name:sha(p) for p in a.out.iterdir() if p.is_file()},
        source_frames=264,predictions=264*len(conditions)*2,run5_used=False,local_gpu_used=False,approved_labels=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['models','helper','probe','video','out']:p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
