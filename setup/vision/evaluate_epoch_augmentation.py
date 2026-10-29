"""Compare three frozen checkpoints to deployed detector; local MP4 diagnostics."""
import argparse
import csv
from fractions import Fraction
import json
from pathlib import Path

import av
import cv2
import numpy as np

from evaluate_mixed_detector import FullFramePair
from evaluate_crop_contrast import score,videos
from train_epoch_augmentation import ARMS,BASELINE,REAL,sha,read,write


def parity(root,output,checkpoint,exported):
    from ultralytics import YOLO
    p=read(root/'data/provenance.json')
    rows=[r for r in p['samples'] if r['split']=='val']
    rows=[rows[int(i)] for i in np.linspace(0,len(rows)-1,16,dtype=int)]
    images=[]
    for r in rows:
        path=root/'data'/r['image']
        if sha(path)!=r['image_sha256']:raise ValueError('Development parity input changed')
        images.append(cv2.imread(str(path)))
    outputs=[YOLO(str(path)).predict(images,imgsz=640,rect=False,conf=.35,iou=.45,max_det=300,
        device='cpu',half=False,augment=False,agnostic_nms=False,verbose=False) for path in (checkpoint,exported)]
    box_error=score_error=0.;count=0
    for a,b in zip(*outputs):
        left=sorted(a.boxes.data.cpu().numpy(),key=lambda r:(r[5],-r[4]));right=sorted(b.boxes.data.cpu().numpy(),key=lambda r:(r[5],-r[4]))
        if len(left)!=len(right):raise ValueError('Export count mismatch')
        for x,y in zip(left,right):
            if x[5]!=y[5]:raise ValueError('Export class mismatch')
            box_error=max(box_error,float(np.abs(x[:4]-y[:4]).max()));score_error=max(score_error,float(abs(x[4]-y[4])));count+=1
    if box_error>1.1 or score_error>1e-4:raise ValueError('Export parity tolerance exceeded')
    result=dict(passed=True,run=4,frames=16,boxes=count,max_pixel_error=box_error,max_score_error=score_error,
        checkpoint_sha256=sha(checkpoint),onnx_sha256=sha(exported),scope='Post-selection CPU export consistency only')
    write(output,result);return result


def fill_clips(output):
    rows={r['frame_index']:r for r in map(json.loads,(output/'video-predictions.jsonl').read_text().splitlines())}
    windows=[(1284,2364),(2670,3762)]
    # Existing paired full MP4 contains exact sampled frames in frame-index order.
    full=output/'run5-full-comparison.mp4';target=output/'run5-two-fills.mp4';count=0
    with av.open(str(full)) as source,av.open(str(target),'w') as movie:
        stream=movie.add_stream('libx264',rate=5);stream.width=1280;stream.height=560;stream.pix_fmt='yuv420p'
        stream.options={'crf':'21','preset':'veryfast'}
        for i,f in enumerate(source.decode(video=0)):
            frame=i*6
            if any(start<=frame<=end for start,end in windows):
                vf=av.VideoFrame.from_ndarray(f.to_ndarray(format='rgb24'),format='rgb24');vf.pts=count;vf.time_base=Fraction(1,5)
                for packet in stream.encode(vf):movie.mux(packet)
                count+=1
        for packet in stream.encode():movie.mux(packet)
    if count!=364:raise ValueError('Incomplete full-fill comparison')
    # These probes are explicitly diagnostic: none selects checkpoints/thresholds.
    probes=[3162,3300,3306,3312]
    write(output/'screenshot-neighborhood-probes.json',dict(frames=[rows[f] for f in probes],
        purpose='Previously inspected misses/flicker; no fitting or checkpoint selection'))
    return dict(file=target.name,frames=count,fps=5,windows=windows,sha256=sha(target))


def training_evidence(path,arm,model,prepared):
    if model['arm']!=arm or model['extra_augmentation']!=ARMS[arm]['extra_augmentation']:
        raise ValueError('Training arm identity mismatch')
    for field,source in [('trainer_sha256','train_epoch_augmentation.py'),('photo_sha256','detector_photometric.py')]:
        if model[field]!=prepared[source]:raise ValueError('Executed training helper changed')
    rows=[{k.strip():v for k,v in row.items()} for row in csv.DictReader((path/'train/results.csv').read_text().splitlines())]
    if len(rows)!=model['epochs_completed'] or not rows or len(rows)>ARMS[arm]['epochs']:
        raise ValueError('Training curve and recorded epoch count disagree')
    best=max(rows,key=lambda row:float(row['metrics/mAP50-95(B)']))
    augmentation=read(path/'augmentation-execution.json');calls=augmentation['counts'].get('calls',0)
    if augmentation['label_geometry_changed'] or augmentation['validation_augmented']:
        raise ValueError('Augmentation crossed target/validation boundary')
    if ARMS[arm]['extra_augmentation']:
        if not calls or not augmentation['counts'].get('changed_images',0):
            raise ValueError('No evidence of actual additional augmentation')
    elif calls:raise ValueError('Plain control received additional augmentation')
    return dict(epochs_completed=len(rows),epoch_budget=ARMS[arm]['epochs'],
        best_run4_epoch=int(float(best['epoch'])),best_run4_map50_95=float(best['metrics/mAP50-95(B)']),
        augmentation=augmentation,training_csv_sha256=sha(path/'train/results.csv'),
        training_args_sha256=sha(path/'train/args.yaml'))


def main(a):
    if a.out.exists():raise ValueError('Use a fresh evaluation output')
    root=a.experiment;plan=read(root/'experiment-plan.json')
    if plan['excluded_from_fit_and_selection']!=[5] or plan['real_provenance_sha256']!=REAL:
        raise ValueError('Training split/ancestry changed')
    baseline_export=Path(__file__).resolve().parents[2]/'replay/models/vial-bootstrap.onnx'
    if sha(baseline_export)!=BASELINE or sha(a.baseline)!='10856c18a8a5c485e832adcc6b841b739e726a969dc043fe8db64c49163bb361':
        raise ValueError('Deployed baseline changed')
    a.out.mkdir(parents=True)
    runtime=set();reports={};identities=None;evidence={};prepared=read(root/'preparation.json')
    titles={'plain12':'MATCHED: 12 epochs','plain50':'LONGER: up to 50 epochs','augmented50':'50 epochs + contrast / blur / noise'}
    for arm in a.arms:
        path=root/'models'/arm;m=read(path/'model.json')
        if (m['experiment_plan_sha256']!=sha(root/'experiment-plan.json') or m['data_provenance_sha256']!=REAL or m['excluded_runs']!=[5]
                or sha(path/'best.pt')!=m['checkpoint_sha256'] or sha(path/'detector.onnx')!=m['model_sha256']):
            raise ValueError('Frozen checkpoint/fit provenance mismatch')
        runtime.add((m['torch'],m['ultralytics'],m['gpu']))
        evidence[arm]=training_evidence(path,arm,m,prepared)
    if len(runtime)!=1:raise ValueError('Training arms used different runtime/device types')
    baseline_parity=parity(root,a.out/'baseline-onnx-parity.json',a.baseline,baseline_export)
    for arm in a.arms:
        print('Evaluating '+arm,flush=True)
        path=root/'models'/arm;out=a.out/arm;out.mkdir()
        exported=parity(root,out/'onnx-parity.json',path/'best.pt',path/'detector.onnx')
        pair=FullFramePair(a.baseline,path/'best.pt',[0,0,640,480],a.device);pair.candidate_title=titles[arm]
        metrics=score(pair,a.teacher,out)
        if identities is None:identities=metrics['sources']
        elif identities!=metrics['sources']:raise ValueError('Scoring inventory changed between arms')
        video=videos(pair,a.data,a.cache,a.cache_sha256,BASELINE,out)
        fill=fill_clips(out);checks=[]
        for name in ('run5-full-comparison.mp4','run5-early-pours.mp4','run5-two-fills.mp4'):
            with av.open(str(out/name)) as movie:
                frames=sum(1 for _ in movie.decode(video=0))
            expected=video['full_video_frames'] if name=='run5-full-comparison.mp4' else video['early_video_frames'] if name=='run5-early-pours.mp4' else 364
            if frames!=expected:raise ValueError('MP4 decode failed')
            checks.append(dict(file=name,frames=frames,sha256=sha(out/name)))
        reports[arm]=dict(model=read(path/'model.json'),training=evidence[arm],parity=exported,teacher_agreement={k:v for k,v in metrics.items() if k!='sources'},
            video=video,full_fill_video=fill,decoded_videos=checks)
        write(a.out/'progress.json',dict(completed_arms=list(reports)))
    write(a.out/'comparison.json',dict(schema_version=1,kind='matched_epoch_and_photometric_diagnostics',
        experiment_plan_sha256=sha(root/'experiment-plan.json'),training_runtime=list(runtime)[0],baseline_parity=baseline_parity,arms=reports,
        detector_promoted=False,measurement_qa_changed=False,run5_fit_or_checkpoint_selection=False,
        requested_arms=a.arms,all_arms_same_scoring_sources=True,limits=plan['limits']))
    print(json.dumps({arm:report['teacher_agreement']['candidate'] for arm,report in reports.items()}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--teacher',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--cache-sha256',required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='0')
    p.add_argument('--arms',nargs='+',choices=list(ARMS),default=list(ARMS),help='Optional subset, each with a fresh output directory')
    main(p.parse_args())
