"""Prepare a small synthetic-only edge-control trial; does not generate or train."""
import argparse
from fractions import Fraction
import json
from pathlib import Path

import av
import cv2
import numpy as np
from PIL import Image

from liquid_specialists_data import read,write,sha
from render_liquid_pipeline import pixel_hash

SOURCE_REVISION='2ff49d0717af02057ae79bc75c00fbff9da1b4e7'
MODEL_REVISION='ce8440327c632d8313c3bde69db13b627ba5cae1'
COMMON=('A fixed industrial inspection camera observes a fluid mixing station. '
    'Transparent cylindrical vials remain stationary in their metal holders. '
    'The left vial under the nozzle accumulates colored liquid in exactly the '
    'same sequence as the input. Preserve camera position, perspective, vial and '
    'nozzle geometry, number of vials, liquid surface heights, liquid hue, narrow '
    'stream path, empty intervals and dispensing timing. ')
VARIANTS=[('neutral',19,'Photorealistic glass, brushed stainless steel and neutral diffuse factory lighting.'),
    ('cool-light',23,'Photorealistic glass and brushed stainless steel with modest cool overhead factory lighting.'),
    ('warm-light',29,'Photorealistic glass and brushed stainless steel with modest warm overhead factory lighting.')]


def video(path,images):
    with av.open(str(path),'w') as c:
        stream=c.add_stream('libx264',rate=16);stream.width=960;stream.height=720;stream.pix_fmt='yuv420p'
        stream.options={'crf':'10','preset':'medium'}
        for i,rgb in enumerate(images):
            frame=av.VideoFrame.from_ndarray(rgb,format='rgb24');frame.pts=i;frame.time_base=Fraction(1,16)
            for packet in stream.encode(frame):c.mux(packet)
        for packet in stream.encode():c.mux(packet)
    with av.open(str(path)) as c: decoded=sum(1 for _ in c.decode(video=0))
    if decoded!=len(images):raise ValueError('Incomplete transfer input video')
    return decoded


def main(a):
    if a.out.exists():raise ValueError('Use a fresh trial input directory')
    verified=read(a.verification/'verification.json')
    for rel,digest in verified['evidence_sha256'].items():
        if sha(a.source/rel)!=digest:raise ValueError('Verified source changed: '+rel)
    plan=read(a.source/'plan.json')
    if plan['episodes'][0]['id']!='nominal-reference':raise ValueError('Use the prior nominal synthetic reference')
    rows=[json.loads(s) for s in (a.source/'observer/frames.jsonl').read_text().splitlines()
        if json.loads(s)['public_source']['episode_id']=='episode-000']
    frames=[];edges=[];trace=[];cache={}
    # 93 frames at 16 Hz use zero-order hold of existing 5 Hz frames. No invented
    # optical flow, generated motion or new native simulator timestamps.
    for i in range(93):
        source_index=i*5//16;h=rows[source_index]['public_source']
        if source_index not in cache:
            path=a.source/'simulation/rgb'/f'episode-000-{source_index:05d}.png'
            rgb=np.array(Image.open(path).convert('RGB'))
            if pixel_hash(rgb)!=h['rgb_sha256']:raise ValueError('Source pixel identity differs')
            cache[source_index]=rgb
        resized=cv2.resize(cache[source_index],(960,720),interpolation=cv2.INTER_CUBIC)
        edge=cv2.Canny(cv2.cvtColor(resized,cv2.COLOR_RGB2GRAY),100,200)
        frames.append(resized);edges.append(np.repeat(edge[:,:,None],3,axis=2))
        trace.append(dict(input_frame=i,input_pts=i,time_base='1/16',source_frame=source_index,
            source_sim_time_s=h['sim_time_s'],source_rgb_sha256=h['rgb_sha256']))
    a.out.mkdir(parents=True)
    video(a.out/'source-nominal.mp4',frames);video(a.out/'edge-control.mp4',edges)
    specs=[]
    for name,seed,appearance in VARIANTS:
        spec=dict(name='vial-'+name,prompt=COMMON+appearance,video_path='source-nominal.mp4',
            seed=seed,guidance=3,num_steps=35,resolution='720',max_frames=93,
            num_video_frames_per_chunk=93,keep_input_resolution=True,show_input=False,
            show_control_condition=False,edge=dict(control_path='edge-control.mp4',control_weight=1.0))
        filename='transfer-'+name+'.json';write(a.out/filename,spec);specs.append(filename)
    # Reference images are source/control previews, not generated results.
    samples=[0,13,25,38,51,76,92];sheet=np.full((360*2,480*len(samples),3),245,np.uint8)
    for n,i in enumerate(samples):
        sheet[:360,n*480:(n+1)*480]=cv2.resize(frames[i],(480,360))
        sheet[360:,n*480:(n+1)*480]=cv2.resize(edges[i],(480,360))
    Image.fromarray(sheet).save(a.out/'source-control-contact-sheet.png')
    write(a.out/'frame-map.json',trace)
    manifest=dict(kind='cosmos_transfer_synthetic_input_preparation',generation_status='not_run',
        source_kind='actual Isaac RGB, prior nominal development reference; no real footage',
        source_verification_sha256=sha(a.verification/'verification.json'),
        upstream_repository='https://github.com/nvidia-cosmos/cosmos-transfer2.5',
        upstream_revision=SOURCE_REVISION,model='nvidia/Cosmos-Transfer2.5-2B',model_revision=MODEL_REVISION,
        variant='base edge control, one GPU, original safety checks enabled',
        source_native_size=[640,480],input_size=[960,720],input_fps=16,input_frames=93,
        input_duration_s=93/16,source_unique_frames=len(cache),
        temporal_mapping='zero-order hold of prior 5 Hz source; frame-map.json is authoritative',
        resize='bicubic; does not add native detail',training_eligible=False,inherited_labels_approved=False,
        run5_used=False,private_geometry_used_as_control=False,specifications=specs,
        review_required=['Vial/nozzle/holder geometry and camera stability',
            'First visible liquid, empty frames, surface height and stream timing',
            'Liquid hue preservation and temporal consistency; prompts are not a guarantee',
            'Per-output human geometry review before inheriting any source label',
            'Downstream value requires independently reviewed real Run 4 evaluation, not visual realism alone'],
        artifacts={p.name:sha(p) for p in sorted(a.out.iterdir()) if p.is_file()})
    write(a.out/'preparation.json',manifest)
    print(json.dumps(dict(status='inputs_prepared_not_generated',frames=93,source_frames=len(cache),specifications=specs)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['source','verification','out']:p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
