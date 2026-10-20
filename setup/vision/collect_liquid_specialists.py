"""Collect completed diagnostics and draw local PR curves/problem-frame crops."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import av
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from evaluate_crop_contrast import matched
from liquid_specialists_data import ARMS,read,write,sha


def curve(rows,key):
    true=[];scores=[];targets=0
    for row in rows:
        p=row[key];targets+=sum(t['class_id']==1 for t in row['targets'])
        tp=matched(p,row['targets'])
        for i,v in enumerate(p):
            if v['class_id']==1:true.append(tp[i,0]);scores.append(v['confidence'])
    order=np.argsort(-np.asarray(scores),kind='stable');scores=np.asarray(scores)[order]
    tp=np.cumsum(np.asarray(true)[order]);recall=tp/targets;precision=tp/np.arange(1,len(tp)+1)
    selected=np.flatnonzero(scores>=.35)
    point=(float(recall[selected[-1]]),float(precision[selected[-1]])) if len(selected) else (0.,0.)
    return recall,precision,point


def main(a):
    if a.out.exists():raise ValueError('Use a fresh collection output')
    reports={};folders={};inventory=None;runtime=set();plan_hash=sha(a.experiment/'experiment-plan.json')
    for folder in a.inputs:
        report=read(folder/'completed-arm.json');m=report['model'];arm=m['arm']
        if arm not in ARMS or arm in reports:raise ValueError('Duplicate/unexpected arm')
        if (m['experiment_plan_sha256']!=plan_hash or m['excluded_runs']!=[5] or
                m['preparation_sha256']!=sha(a.experiment/'preparation.json') or
                sha(a.experiment/'models'/arm/'best.pt')!=m['checkpoint_sha256']):raise ValueError('Model provenance changed')
        source=read(folder/'teacher-agreement.json')['sources']
        if inventory is None:inventory=source
        if source!=inventory:raise ValueError('Scoring inventory differs')
        for video in report['decoded_videos']:
            if sha(folder/video['file'])!=video['sha256']:raise ValueError('Video changed after validation')
        runtime.add((m['torch'],m['ultralytics'],m['gpu']));reports[arm]=report;folders[arm]=folder
    if set(reports)!=set(ARMS) or len(runtime)!=1:raise ValueError('Missing arms or different training runtime/device')
    a.out.mkdir(parents=True)
    for arm in ARMS:shutil.copytree(folders[arm],a.out/arm)
    first=read(folders['yolo-all']/'scoring-predictions.json')
    curves={'Deployed':curve(first,'baseline')};reference=None
    if a.two_class_reference:
        folder=a.two_class_reference/'evaluation/plain50'
        if read(folder/'teacher-agreement.json')['sources']!=inventory:raise ValueError('Reference scoring inventory differs')
        reference=read(a.two_class_reference/'evaluation/comparison.json')['arms']['plain50']
        if reference['model']['excluded_runs']!=[5]:raise ValueError('Reference fit exclusion differs')
        curves['Two-class longer']=curve(read(folder/'scoring-predictions.json'),'candidate')
    for arm in ARMS:curves[arm]=curve(read(folders[arm]/'scoring-predictions.json'),'candidate')
    fig,ax=plt.subplots(figsize=(8,6))
    for name,(r,p,point) in curves.items():
        line,=ax.plot(r,p,label=name,linewidth=1.3);ax.scatter(*point,s=30,color=line.get_color())
    ax.set(xlim=(0,1),ylim=(0,1.02),xlabel='Recall against provisional liquid boxes',ylabel='Precision against provisional liquid boxes',
        title='Run 5 liquid precision–recall at IoU 0.5\nDots: unchanged confidence 0.35; scores not calibrated across families')
    ax.grid(alpha=.2);ax.legend(loc='lower left',fontsize=8);fig.tight_layout();fig.savefig(a.out/'liquid-precision-recall.png',dpi=170);plt.close(fig)
    frames=[1320,2706,3162,3306];predictions={};all_folders=dict(folders)
    if a.two_class_reference:all_folders={'Two-class longer':a.two_class_reference/'evaluation/plain50',**all_folders}
    for name,folder in all_folders.items():
        with (folder/'video-predictions.jsonl').open() as f:
            predictions[name]={r['frame_index']:r for r in map(json.loads,f) if r['frame_index'] in frames}
        if set(predictions[name])!=set(frames):raise ValueError('Problem-frame inventory incomplete')
    columns=['Deployed',*all_folders];canvas=np.zeros((40+len(frames)*340,len(columns)*300,3),np.uint8)
    cv2.putText(canvas,'Same 0.35 confidence | display crops only | fixed blue vial branch, green liquid | provisional labels',(8,27),0,.55,(255,255,255),1)
    video=a.data/'vial_fill/videos/vial_fill_video_5.mp4'
    if sha(video)!=reports['yolo-all']['video']['source_video_sha256']:raise ValueError('Source recording changed')
    with av.open(str(video)) as movie:
        for index,f in enumerate(movie.decode(video=0)):
            if index>max(frames):break
            if index not in frames:continue
            rgb=f.to_ndarray(format='rgb24');digest=hashlib.sha256(rgb.tobytes()).hexdigest()
            if any(predictions[name][index]['source_rgb_sha256']!=digest for name in all_folders):raise ValueError('Frame pixels differ')
            row=frames.index(index)
            for col,name in enumerate(columns):
                boxes=predictions['yolo-all'][index]['baseline'] if name=='Deployed' else predictions[name][index]['candidate']
                image=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
                for p in boxes:
                    b=np.rint(p['box']).astype(int);color=(255,180,70) if p['class_id']==0 else (80,240,100)
                    cv2.rectangle(image,tuple(b[:2]),tuple(b[2:]),color,1)
                x=col*300;y=40+row*340;canvas[y+48:y+328,x+70:x+230]=cv2.resize(image[240:460,250:365],(160,280))
                cv2.putText(canvas,name,(x+8,y+18),0,.49,(255,255,255),1)
                scores=[p['confidence'] for p in boxes if p['class_id']==1]
                label='liquid '+','.join(f'{v:.2f}' for v in scores) if scores else 'no liquid box'
                cv2.putText(canvas,f'f{index} | {label}',(x+8,y+39),0,.43,(90,230,110),1)
    if not cv2.imwrite(str(a.out/'problem-frame-comparison.png'),canvas):raise IOError('Image export failed')
    write(a.out/'comparison.json',dict(schema_version=1,kind='liquid_specialist_diagnostics',arms=reports,
        two_class_long_reference=reference,experiment_plan_sha256=plan_hash,training_runtime=list(runtime)[0],
        detector_promoted=False,measurement_qa_changed=False,run5_fit_or_selection=False,
        fixed_vial_detector=True,confidence_calibrated_across_families=False,
        scope='Teacher agreement only. Dots at 0.35 are diagnostic; primary box metric is liquid AP50-95.',
        reference_limit='Previous two-class longer anchor selected combined Run4 AP; liquid specialists selected liquid Run4 AP.',
        image_sha256={name:sha(a.out/name) for name in ('liquid-precision-recall.png','problem-frame-comparison.png')}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('experiment','data','out','two-class-reference'):p.add_argument('--'+name,type=Path,required=name!='two-class-reference')
    p.add_argument('--inputs',nargs='+',type=Path,required=True);main(p.parse_args())
