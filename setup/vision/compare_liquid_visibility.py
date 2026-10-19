"""Local, post-hoc visibility pilot; never changes replay/model/measurement QA."""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

import av
import cv2
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from replay.config import load_config
from replay.dataset import Catalog,DEFAULT_DATA,sha256
from replay.evidence import source_frame
from replay.geometry import OnnxDetector
from replay.proposals import CachedImageDetector


def clahe_rgb(rgb):
    lab=cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB)
    lab[:,:,0]=cv2.createCLAHE(clipLimit=2.,tileGridSize=(8,8)).apply(lab[:,:,0])
    return cv2.cvtColor(lab,cv2.COLOR_LAB2RGB)


def views(rgb,reference,frame_index,reference_index,box):
    if reference_index>=frame_index:raise ValueError('Reference must precede the current diagnostic frame')
    if rgb.shape!=reference.shape:raise ValueError('Reference shape mismatch')
    x,y,r,b=box
    if not 0<=x<r<=rgb.shape[1] or not 0<=y<b<=rgb.shape[0]:raise ValueError('Invalid diagnostic crop')
    enhanced=clahe_rgb(rgb)
    # Fixed gains, no per-frame auto stretching that could make tiny noise look
    # equally important as a large change. Neither map is a liquid mask.
    difference=np.clip(np.abs(rgb.astype(np.int16)-reference.astype(np.int16)).mean(axis=2)*4,0,255).astype(np.uint8)
    gradient=np.clip(np.abs(cv2.Sobel(cv2.cvtColor(enhanced,cv2.COLOR_RGB2GRAY),cv2.CV_32F,0,1,ksize=3))*2,0,255).astype(np.uint8)
    return dict(raw=rgb[y:b,x:r].copy(),clahe=enhanced[y:b,x:r],
                past_difference=cv2.cvtColor(difference[y:b,x:r],cv2.COLOR_GRAY2RGB),
                horizontal_edges=cv2.cvtColor(gradient[y:b,x:r],cv2.COLOR_GRAY2RGB))


def panel(image,title):
    h,w=image.shape[:2];scale=min(192/w,480/h)
    resized=cv2.resize(image,(round(w*scale),round(h*scale)),interpolation=cv2.INTER_NEAREST if title.endswith('edges') else cv2.INTER_LINEAR)
    tile=np.zeros((520,192,3),np.uint8);x=(192-resized.shape[1])//2
    tile[40:40+resized.shape[0],x:x+resized.shape[1]]=cv2.cvtColor(resized,cv2.COLOR_RGB2BGR)
    cv2.putText(tile,title,(5,25),0,.45,(255,255,255),1)
    return tile


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--cache-sha256',required=True)
    p.add_argument('--config',type=Path,default=Path('replay/examples/bootstrap-vision.json'))
    p.add_argument('--data',type=Path,default=DEFAULT_DATA);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    implementation_sha256=sha256(__file__)
    if args.out.exists():p.error('Use a new output directory')
    if sha256(args.cache)!=args.cache_sha256:raise ValueError('Cache hash mismatch')
    payload=json.loads(args.cache.read_text(encoding='utf-8'));run_id=payload['manifest']['run_id']
    camera=Catalog(args.data,include_annotations=False).runs[run_id]['cameras']['vial_fill']
    config=load_config(args.config)
    if camera['fps']!=30 or config.sample_stride!=6:
        raise ValueError('This bounded pilot requires 30 fps source and stride-6 proposal cache')
    cache=CachedImageDetector(args.cache,args.cache_sha256,config.vision_model_sha256,camera['sha256'],run_id)
    detector=OnnxDetector(config.vision_model_path,config.vision_model_sha256)
    # Chronological, first eight non-partial image-estimated starts. These
    # choose diagnostic clips only; they are not labels or true flow onsets.
    cases=[d for d in payload['detections'] if d['start_frame']>=60][:8]
    if len(cases)!=8:raise ValueError('Pilot requires eight non-partial candidate starts')
    args.out.mkdir(parents=True);(args.out/'frames').mkdir()
    references={};plans={};records=[];stills=[];offsets=(-6,0,6,18,36,72)
    for number,case in enumerate(cases):
        start=case['start_frame'];ref=start-30
        reference=source_frame(camera['path'],camera['sha256'],ref,camera['fps'],camera['start_pts_s'])
        cache.prepare_frame(ref);proposals=cache.detect(reference)
        vials=[b['box'] for b in proposals if b['class_id']==0 and b['box'][0]<306.5<b['box'][2]]
        if len(vials)!=1:raise ValueError('Pilot reference lacks a unique fill-position vial')
        vx,vy,vr,vb=vials[0]
        box=[max(0,vx-15),max(0,vy-10),min(640,vr+15),min(480,vb+10)]
        name=f'case{number+1}'
        reference_path=f'frames/{name}-reference-f{ref:06d}.png'
        cv2.imwrite(str(args.out/reference_path),cv2.cvtColor(reference,cv2.COLOR_RGB2BGR))
        references[number]=dict(rgb=reference,index=ref,box=box)
        records.append(dict(case=name,image_estimated_start=start,reference_frame=ref,
                            reference_source_rgb_sha256=hashlib.sha256(reference.tobytes()).hexdigest(),
                            reference_image=reference_path,reference_status='earlier image; same stationary empty vial NOT certified',
                            diagnostic_crop=box,samples=[]))
        for frame in range(start-15,start+106):plans[frame]=(number,start)
    output_index=0
    with av.open(camera['path']) as source,av.open(str(args.out/'visibility-comparison.mp4'),'w') as movie:
        source.streams.video[0].thread_type='AUTO'
        stream=movie.add_stream('libx264',rate=Fraction(str(camera['fps'])))
        stream.width=768;stream.height=564;stream.pix_fmt='yuv420p';stream.options={'crf':'20','preset':'veryfast'}
        for index,frame in enumerate(source.decode(video=0)):
            if index not in plans:
                if index>max(plans):break
                continue
            number,start=plans[index];reference=references[number];rgb=frame.to_ndarray(format='rgb24')
            images=views(rgb,reference['rgb'],index,reference['index'],reference['box'])
            strip=np.hstack([panel(value,name.replace('_',' ')) for name,value in images.items()])
            footer=np.zeros((44,768,3),np.uint8)
            text=f"case {number+1} | source f{index} | relative {(index-start)/camera['fps']:+.2f}s | ref f{reference['index']} (not certified empty)"
            cv2.putText(footer,text,(5,19),0,.45,(255,255,255),1)
            cv2.putText(footer,'VISIBILITY CUES ONLY: motion / glare / holder can also produce changes and edges',(5,38),0,.43,(200,200,200),1)
            rendered=np.vstack([strip,footer])
            encoded=av.VideoFrame.from_ndarray(rendered,format='bgr24');encoded.pts=output_index;encoded.time_base=Fraction(1,30)
            for packet in stream.encode(encoded):movie.mux(packet)
            output_index+=1
            if index-start in offsets:
                cache.prepare_frame(index);cached_boxes=cache.detect(rgb)
                raw_boxes=detector.detect(rgb)
                enhanced=clahe_rgb(rgb);enhanced_boxes=detector.detect(enhanced)
                stem=f'case{number+1}-f{index:06d}'
                cv2.imwrite(str(args.out/'frames'/(stem+'-comparison.png')),rendered)
                cv2.imwrite(str(args.out/'frames'/(stem+'-source.png')),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
                row=dict(frame_index=index,mp4_s=index/camera['fps'],relative_s=(index-start)/camera['fps'],
                         source_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                         source_image=f'frames/{stem}-source.png',comparison_image=f'frames/{stem}-comparison.png',
                         raw_proposals=raw_boxes,clahe_proposals=enhanced_boxes,
                         fresh_raw_matches_cached=(raw_boxes==cached_boxes))
                records[number]['samples'].append(row)
                if index-start==18:stills.append(rendered)
        for packet in stream.encode():movie.mux(packet)
    if output_index!=len(plans) or sum(len(c['samples']) for c in records)!=48:raise ValueError('Incomplete pilot export')
    cv2.imwrite(str(args.out/'contact-sheet.png'),np.vstack(stills))
    if sha256(__file__)!=implementation_sha256:raise ValueError('Pilot source changed while running')
    manifest=dict(schema_version=1,kind='posthoc_visibility_pilot',video_sha256=camera['sha256'],
                  model_sha256=config.vision_model_sha256,cache_sha256=args.cache_sha256,
                  implementation_sha256=implementation_sha256,cases=records,video_frames=output_index,video_fps=30,
                  source_color_preserved=True,measurement_pipeline_changed=False,
                  clahe=dict(color_space='OpenCV LAB',channel='L only',clip_limit=2.,tiles=[8,8]),
                  difference=dict(reference_offset_frames=-30,absolute_mean_rgb_gain=4),
                  edges=dict(input='CLAHE grayscale',derivative='Sobel dy',kernel=3,display_gain=2),
                  model_check='Frozen-model raw vs full-frame CLAHE inference on 48 diagnostic frames; predictions, not accuracy',
                  limits=['Image-estimated starts are not reviewed physical onset labels.',
                          'Past references are not certified empty/stationary same-vial images.',
                          'Fixed source crop is not motion compensation; reflections and motion cause false cues.',
                          'No training, checkpoint selection, reviewed accuracy score or QA change.',
                          'Enhanced/difference/edge pixels must not supply color measurements.'])
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(cases=len(records),stills=48,video_frames=output_index,out=str(args.out))),flush=True)


if __name__=='__main__':main()
