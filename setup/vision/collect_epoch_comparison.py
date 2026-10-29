"""Assemble independently exported arm diagnostics, verifying inputs and videos."""
import argparse
import hashlib
from pathlib import Path
import shutil

import av
import cv2
import numpy as np

from evaluate_epoch_augmentation import training_evidence
from train_epoch_augmentation import ARMS,BASELINE,REAL,read,write,sha


def collect(a):
    if a.out.exists():raise ValueError('Use a fresh collection directory')
    prepared=read(a.experiment/'preparation.json');plan=read(a.experiment/'experiment-plan.json')
    reports={};sources=None;runtimes=set();probes={};video_sha=None
    for arm in ARMS:
        folder=getattr(a,arm);model_path=a.experiment/'models'/arm;m=read(model_path/'model.json')
        if (m['experiment_plan_sha256']!=sha(a.experiment/'experiment-plan.json') or m['excluded_runs']!=[5]
                or m['data_provenance_sha256']!=REAL or sha(model_path/'best.pt')!=m['checkpoint_sha256']
                or sha(model_path/'detector.onnx')!=m['model_sha256']):raise ValueError('Model ancestry changed')
        runtimes.add((m['torch'],m['ultralytics'],m['gpu']))
        training=training_evidence(model_path,arm,m,prepared)
        metric=read(folder/'teacher-agreement.json');manifest=read(folder/'video-manifest.json')
        if sources is None:sources=metric['sources'];baseline=metric['baseline'];video_sha=manifest['source_video_sha256']
        if sources!=metric['sources'] or baseline!=metric['baseline'] or video_sha!=manifest['source_video_sha256']:
            raise ValueError('Arms used different scoring/video sources or baseline')
        parity=read(folder/'onnx-parity.json')
        if not parity['passed'] or parity['checkpoint_sha256']!=m['checkpoint_sha256'] or parity['onnx_sha256']!=m['model_sha256']:
            raise ValueError('Export parity identity mismatch')
        checks=[]
        for name,expected in [('run5-full-comparison.mp4',manifest['full_video_frames']),
                ('run5-early-pours.mp4',manifest['early_video_frames']),('run5-two-fills.mp4',364)]:
            with av.open(str(folder/name)) as movie:count=sum(1 for _ in movie.decode(video=0))
            if count!=expected:raise ValueError('Incomplete or undecodable video')
            checks.append(dict(file=name,frames=count,sha256=sha(folder/name)))
        reports[arm]=dict(model=m,training=training,parity=parity,
            teacher_agreement={k:v for k,v in metric.items() if k!='sources'},video=manifest,decoded_videos=checks,
            full_fill_video=dict(file='run5-two-fills.mp4',frames=364,fps=5,windows=[[1284,2364],[2670,3762]],sha256=checks[-1]['sha256']))
        probes[arm]={r['frame_index']:r for r in read(folder/'screenshot-neighborhood-probes.json')['frames']}
    if len(runtimes)!=1:raise ValueError('Training runtime/device differs between arms')
    if plan['excluded_from_fit_and_selection']!=[5] or plan['baseline_onnx_sha256']!=BASELINE:
        raise ValueError('Frozen split/baseline differs')
    a.out.mkdir(parents=True)
    for arm in ARMS:shutil.copytree(getattr(a,arm),a.out/arm)
    # Display crops only. Predictions were computed on full, unchanged source frames.
    source=a.data/'vial_fill/videos/vial_fill_video_5.mp4'
    if sha(source)!=video_sha:raise ValueError('Source recording changed')
    selected=[3162,3300,3306,3312];canvas=np.zeros((4*340+40,4*300,3),np.uint8)
    cv2.putText(canvas,'Same 0.35 gate | display crop only | blue vial, green liquid | provisional labels',(8,26),0,.53,(255,255,255),1)
    titles=['Deployed','Fresh 12','Longer (best Run 4)','Augmented (best Run 4)']
    with av.open(str(source)) as movie:
        for index,frame in enumerate(movie.decode(video=0)):
            if index>max(selected):break
            if index not in selected:continue
            rgb=frame.to_ndarray(format='rgb24');row=selected.index(index)
            if any(hashlib.sha256(rgb.tobytes()).hexdigest()!=probes[arm][index]['source_rgb_sha256'] for arm in ARMS):
                raise ValueError('Probe source pixels differ')
            predictions=[probes['plain12'][index]['baseline']]+[probes[arm][index]['candidate'] for arm in ARMS]
            for col,boxes in enumerate(predictions):
                image=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
                for p in boxes:
                    b=np.rint(p['box']).astype(int);color=(255,180,70) if p['class_id']==0 else (80,240,100)
                    cv2.rectangle(image,tuple(b[:2]),tuple(b[2:]),color,1)
                crop=cv2.resize(image[260:460,250:365],(161,280))
                y=40+row*340;x=col*300;canvas[y+48:y+328,x+65:x+226]=crop
                cv2.putText(canvas,titles[col],(x+8,y+18),0,.49,(255,255,255),1)
                scores=[p['confidence'] for p in boxes if p['class_id']==1]
                label='liquid '+','.join(f'{v:.2f}' for v in scores) if scores else 'no liquid box'
                cv2.putText(canvas,f'f{index} | {label}',(x+8,y+39),0,.45,(90,230,110),1)
    if not cv2.imwrite(str(a.out/'selected-frame-comparison.png'),canvas):raise IOError('Image export failed')
    write(a.out/'comparison.json',dict(schema_version=1,kind='matched_epoch_and_photometric_diagnostics',
        experiment_plan_sha256=sha(a.experiment/'experiment-plan.json'),training_runtime=list(runtimes)[0],arms=reports,
        requested_arms=list(ARMS),all_arms_same_scoring_sources=True,detector_promoted=False,measurement_qa_changed=False,
        run5_fit_or_checkpoint_selection=False,limits=plan['limits'],assembly='Independently exported diagnostics; all videos decoded and source identities compared'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('experiment','data','out',*ARMS):p.add_argument('--'+name,type=Path,required=True)
    collect(p.parse_args())
