"""Decode a generated clip and publish source-aligned review evidence.

Edge agreement is only an image diagnostic. It cannot approve inherited labels
or establish that generated liquid dynamics/colors are physically correct.
"""
import argparse
import json
from pathlib import Path
import av
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont

from liquid_specialists_data import read,write,sha


def decode(path):
    with av.open(str(path)) as c:
        stream=c.streams.video[0];fps=float(stream.average_rate)
        frames=[f.to_ndarray(format='rgb24') for f in c.decode(video=0)]
    return frames,fps


def main(a):
    if a.out.exists():raise ValueError('Use a fresh review directory')
    preparation=read(a.inputs/'preparation.json')
    for rel,digest in preparation['artifacts'].items():
        if sha(a.inputs/rel)!=digest:raise ValueError('Prepared inputs changed: '+rel)
    source,fps=decode(a.inputs/'source-nominal.mp4');generated,generated_fps=decode(a.generated)
    if len(source)!=len(generated) or fps!=16 or generated_fps!=16:raise ValueError('Generated sequence cannot be associated frame-for-frame')
    if any(f.shape!=(720,960,3) for f in generated):raise ValueError('Unexpected generated resolution')
    mapping=read(a.inputs/'frame-map.json');stats=[];kernel=np.ones((5,5),np.uint8)
    for i,(x,y) in enumerate(zip(source,generated)):
        ex=cv2.Canny(cv2.cvtColor(x,cv2.COLOR_RGB2GRAY),100,200)>0
        ey=cv2.Canny(cv2.cvtColor(y,cv2.COLOR_RGB2GRAY),100,200)>0
        source_recall=float((ex&(cv2.dilate(ey.astype('uint8'),kernel)>0)).sum()/max(1,ex.sum()))
        generated_precision=float((ey&(cv2.dilate(ex.astype('uint8'),kernel)>0)).sum()/max(1,ey.sum()))
        f1=2*source_recall*generated_precision/max(1e-12,source_recall+generated_precision)
        stats.append(dict(frame=i,source_mapping=mapping[i],source_edge_recall=source_recall,
            generated_edge_precision=generated_precision,edge_f1_within_2px=f1,
            mean_absolute_rgb_difference=float(np.abs(x.astype(float)-y.astype(float)).mean())))
    a.out.mkdir(parents=True)
    # A contact sheet exposes whole-scene drift and the active vial separately.
    picks=[0,13,25,38,51,76,92];sheet=Image.new('RGB',(480*len(picks),2*360+2*430+90),(24,28,34));draw=ImageDraw.Draw(sheet);draw.font=ImageFont.load_default(size=20)
    for n,i in enumerate(picks):
        draw.text((n*480+8,8),f'Frame {i} / {i/16:.3f}s',fill='white')
        for row,arr in enumerate([source[i],generated[i]]):
            sheet.paste(Image.fromarray(arr).resize((480,360)),(n*480,30+row*380))
            # Same fixed crop in both rows; no generated-label assumption.
            crop=Image.fromarray(arr[60:690,360:585]).resize((154,430))
            sheet.paste(crop,(n*480+160,800+row*430))
    draw.text((8,395),'GENERATED (top row above is source)',fill='white')
    draw.text((8,785),'Active-vial crop: SOURCE above / GENERATED below',fill='white')
    sheet.save(a.out/'source-generated-contact-sheet.png')
    # Single-view generated-only video is the upstream output; no relabeling.
    report=dict(kind='generated_transfer_review_diagnostics',generated_video=str(a.generated),
        generated_video_sha256=sha(a.generated),input_preparation_sha256=sha(a.inputs/'preparation.json'),
        decoded_frames=len(generated),fps=generated_fps,size=[960,720],frames=stats,
        edge_f1_within_2px=dict(median=float(np.median([s['edge_f1_within_2px'] for s in stats])),
            minimum=min(s['edge_f1_within_2px'] for s in stats)),
        human_reviewed_geometry=False,training_eligible=False,inherited_labels_approved=False,
        real_detector_accuracy=None,limits=['Edge agreement does not score transparent liquid geometry or color preservation.',
            'Frame association is nominal sequence identity, not proof that generated events preserve source timing.',
            'Source is a repeated/resized 5 Hz synthetic development sequence, not new independent native frames.'])
    write(a.out/'review.json',report)
    write(a.out/'completed.json',dict(decoded_frames=len(generated),
        artifacts={p.name:sha(p) for p in a.out.iterdir() if p.is_file()}))
    print(json.dumps(dict(decoded_frames=len(generated),edge_f1=report['edge_f1_within_2px'],training_eligible=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['inputs','generated','out']:p.add_argument('--'+n,type=Path,required=True)
    main(p.parse_args())
