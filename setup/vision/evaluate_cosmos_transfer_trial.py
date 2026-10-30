"""Apply frozen vision models to source/generated RGB without treating either as labels."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys

import av
import cv2
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from live_twin_observer import NativeModels,Pipeline
from liquid_specialists_data import read,write,sha
from replay.config import load_config
from replay.detection import FillDetector
from replay.geometry import decode_mask
from replay.measurement import delta_e


def main(a):
    if a.out.exists():raise ValueError('Use a fresh evaluation output')
    prep=read(a.inputs/'preparation.json')
    for rel,digest in prep['artifacts'].items():
        if sha(a.inputs/rel)!=digest:raise ValueError('Prepared input changed')
    pins=read(a.study/'observer/models.json')['models'];tools=Path.home()/'ai-club-tools'
    baseline=tools/'results/crop-clahe-detector-20261019/baseline-best.pt'
    experiment=tools/'results/early-liquid-20261020'
    sam=tools/'results/prompted-liquid-assets-20261020/sam2.1_t.pt'
    for name,path in [('vial',baseline),('liquid',experiment/'models/fasterrcnn-resnet50/best.pt'),('sam',sam)]:
        if sha(path)!=pins[name]:raise ValueError('Frozen model changed')
    a.out.mkdir(parents=True);shutil.copy2(Path(__file__),a.out/Path(__file__).name)
    # Exactly one inference for each distinct held source frame. The same sample
    # indices/decoded timestamps are used in both videos, without event selection.
    mapping=read(a.inputs/'frame-map.json');wanted=[];previous=None
    for i,item in enumerate(mapping):
        if item['source_frame']!=previous:wanted.append(i);previous=item['source_frame']
    config=load_config(Path(__file__).resolve().parents[2]/'replay/examples/bootstrap-vision.json')
    calibration=read(tools/'results/stage2-refinement-20261020/camera-reference-v2/calibration.json')
    write(a.out/'plan.json',dict(models=pins,samples=wanted,resize=[640,480],
        resize_method='OpenCV INTER_AREA, identical for source and generated',
        input_sha256=sha(a.inputs/'source-nominal.mp4'),generated_sha256=sha(a.generated),
        source_plan_sha256=sha(a.inputs/'preparation.json'),training=False,run5_used=False,
        generated_labels_assumed_correct=False,helper_sha256=sha(Path(__file__))))
    models=NativeModels(baseline,experiment,sam);summary={};all_rows={}
    for name,path in [('source',a.inputs/'source-nominal.mp4'),('generated',a.generated)]:
        pipeline=Pipeline(models,'pytorch',config,pins,calibration)
        pipeline.fill=FillDetector(config,16,'transfer-'+name)
        counts=Counter();rows=[];writer=None;last=None;decoded=0
        if name=='generated':
            writer=cv2.VideoWriter(str(a.out/'generated-inspection-raw.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),16.,(960,720))
            if not writer.isOpened():raise RuntimeError('Video writer failed')
        with av.open(str(path)) as c,(a.out/(name+'-frames.jsonl')).open('w') as log:
            if c.streams.video[0].average_rate!=16:raise ValueError('Unexpected input clock')
            for i,f in enumerate(c.decode(video=0)):
                rgb=f.to_ndarray(format='rgb24')
                if rgb.shape!=(720,960,3):raise ValueError('Unexpected image geometry')
                if i in wanted:
                    small=cv2.resize(rgb,(640,480),interpolation=cv2.INTER_AREA)
                    row=pipeline.process(small,i,f.pts,f.time_base)
                    row.pop('recipe_decision')
                    row['source_mapping']=mapping[i]
                    row['operational_recipe_decision']=dict(status='unavailable',reason='uncommissioned_generated_or_synthetic_source')
                    log.write(json.dumps(row,allow_nan=False)+'\n');rows.append(row);last=row
                    counts['samples']+=1;counts['qa_pass']+=int(row['measurement']['measurement_valid'])
                    counts['vial_present']+=int(any(p['class_id']==0 for p in row['predictions']))
                    counts['liquid_present']+=int(any(p['class_id']==1 for p in row['predictions']))
                    counts['nonempty_final_mask']+=int(row['measurement']['geometry']['mask_pixels']>0)
                if writer:
                    image=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
                    mask=decode_mask(last['measurement']['geometry']['mask'],(480,640)).astype('uint8')
                    mask=cv2.resize(mask,(960,720),interpolation=cv2.INTER_NEAREST).astype(bool)
                    image[mask]=(.5*image[mask]+.5*np.array([80,230,240])).astype('uint8')
                    for p in last['predictions']:
                        x,y,r,b=[round(v*1.5) for v in p['box']]
                        cv2.rectangle(image,(x,y),(r,b),(240,170,55) if p['class_id']==0 else (70,230,80),2)
                    cv2.rectangle(image,(0,0),(959,64),(20,25,34),-1)
                    cv2.putText(image,'GENERATED TRANSFER CLIP / ACTUAL FROZEN INSPECTION',(12,25),cv2.FONT_HERSHEY_SIMPLEX,.57,(240,240,240),1,cv2.LINE_AA)
                    text=f'{i/16:.2f}s | sample QA: {"passes" if last["measurement"]["measurement_valid"] else "unavailable"} | labels unreviewed'
                    cv2.putText(image,text,(12,50),cv2.FONT_HERSHEY_SIMPLEX,.52,(240,240,240),1,cv2.LINE_AA)
                    writer.write(image)
                decoded+=1
        if writer:writer.release()
        if decoded!=93 or len(rows)!=len(wanted):raise ValueError('Incomplete paired evaluation')
        summary[name]=dict(counts)
        summary[name]['first_model_liquid_clip_s']=next((r['seconds'] for r in rows if any(p['class_id']==1 for p in r['predictions'])),None)
        summary[name]['first_qa_valid_clip_s']=next((r['seconds'] for r in rows if r['measurement']['measurement_valid']),None)
        all_rows[name]=rows
    paired=[]
    for x,y in zip(all_rows['source'],all_rows['generated']):
        mx=x['measurement'];my=y['measurement']
        a_mask=decode_mask(mx['geometry']['mask'],(480,640)).astype(bool)
        b_mask=decode_mask(my['geometry']['mask'],(480,640)).astype(bool)
        union=(a_mask|b_mask).sum()
        paired.append(dict(clip_s=x['seconds'],source_frame=x['source_frame'],
            final_sampling_mask_iou=float((a_mask&b_mask).sum()/union) if union else None,
            raw_lab_de2000=float(delta_e(mx['lab'],my['lab'])) if mx['measurement_valid'] and my['measurement_valid'] else None))
    write(a.out/'paired-model-disagreement.json',dict(samples=paired,
        interpretation='Changes in model response, not generated-label accuracy or guaranteed color preservation.'))
    write(a.out/'comparison.json',dict(kind='frozen_model_response_to_transfer_appearance',summary=summary,
        frames_decoded_per_video=93,samples_per_video=len(wanted),model_pins=pins,
        interpreted_accuracy=False,training_eligible=False,hardware_actions=0,
        limits=['QA availability is not label accuracy or proof that generated liquid is physically correct.',
            'Both videos are resized/compressed inputs, so source results need not equal the native live study.',
            'One synthetic trajectory; generated onset/geometry require independent human review.']))
    with av.open(str(a.out/'generated-inspection-raw.mp4')) as c:
        if sum(1 for _ in c.decode(video=0))!=93:raise ValueError('Inspection export incomplete')
    write(a.out/'completed.json',dict(artifacts={p.name:sha(p) for p in a.out.iterdir() if p.is_file()}))
    print(json.dumps(summary))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['inputs','generated','study','out']:p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
