"""Local source-grounded fill walkthroughs; display only, no model/label changes."""
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import av
from PIL import Image, ImageDraw, ImageFont

from compare_station_tracking import CACHE, VIDEO, frames, CachedImageDetector, MODEL, sha256, write


TRACKING='eeda8d10126751166ef88cde8123960dca3af5c266067a7ca21db8c6d503916c'
BG='#142029';FG='#edf3f1';CYAN='#70cafa';GREEN='#8ee0aa';AMBER='#ffc575';MUTED='#adbdc8'
REASONS=dict(candidate_link='Candidate link', carrier_transition='Carrier moving / settling',
    no_fresh_station_vial='No fresh station vial', track_not_confirmed='Track not confirmed',
    outside_vial='Outside vial containment', multiple_liquid_candidates='Competing liquid boxes',
    ambiguous_active_vial='Ambiguous station vial')


def font(size):
    try:return ImageFont.truetype('C:/Windows/Fonts/arial.ttf',size)
    except OSError:return ImageFont.load_default()


def text(im,xy,value,size=20,color=FG):
    ImageDraw.Draw(im).text(xy,value,font=font(size),fill=color)


def linked(row):return any(l['association']=='candidate_link' for l in row['liquids'])


def painted(rgb,row):
    im=Image.fromarray(rgb);d=ImageDraw.Draw(im)
    for v in row['vials']:
        b=v['box'];d.rectangle(b,outline=CYAN,width=2)
        identity=v['track_id'].rsplit(':',1)[-1] if v['track_id'] else '?'
        d.text((max(2,b[0]-5),max(2,b[1]-19)),identity,font=font(15),fill=CYAN)
    for l in row['liquids']:
        d.rectangle(l['box'],outline=GREEN if l['association']=='candidate_link' else AMBER,width=2)
    return im


def frame_view(rgb,row,case,history,phase):
    im=Image.new('RGB',(1280,840),BG)
    text(im,(20,12),case['title']+' / Run 5',27)
    text(im,(20,48),f"Source frame {row['frame_index']}  |  MP4 {row['frame_index']/30:.2f}s  |  original detections sampled at 5 fps",18,MUTED)
    marked=painted(rgb,row);im.paste(marked,(20,85))
    # Display-only enlargement, fixed source crop, no input changes to detector.
    im.paste(marked.crop((260,170,355,435)).resize((190,530)),(690,85))
    text(im,(20,575),'Full camera frame',17,MUTED);text(im,(690,625),'2x source crop',16,MUTED)
    active=next((v for v in row['vials'] if v['active_station']),None)
    identity=active['track_id'].rsplit(':',1)[-1] if active and active['track_id'] else 'unassigned'
    text(im,(910,90),'VIAL TRACK',17,CYAN);text(im,(910,120),identity,32,CYAN)
    text(im,(910,163),row['state'].replace('_',' '),21)
    text(im,(910,208),'LIQUID DETECTION',17,GREEN)
    text(im,(910,240),f"{len(row['liquids'])} raw proposal(s)",23)
    for n,l in enumerate(row['liquids'][:3]):
        text(im,(910,280+n*62),REASONS.get(l['association'],l['association']),19,GREEN if l['association']=='candidate_link' else AMBER)
        text(im,(910,304+n*62),f"Detector confidence {l['confidence']:.2f}",16,MUTED)
    if not row['liquids']:text(im,(910,280),'No box above detector threshold',18,AMBER)
    text(im,(910,495),'IMAGE FILL CUE',17,MUTED)
    text(im,(910,523),phase.replace('_',' '),19)
    text(im,(910,565),'A link is not a QA pass.',18,AMBER)
    # Categorical history, not a fill-height/volume estimate. Future is blank.
    d=ImageDraw.Draw(im);x0,x1=195,1230;start,end=case['start'],case['end']
    lanes=[('Vial confirmed',675),('Liquid proposal / link',711)]
    for label,y in lanes:
        text(im,(20,y-3),label,16,MUTED);d.line((x0,y+12,x1,y+12),fill='#344653',width=1)
    for r in history:
        x=x0+(r['frame_index']-start)/(end-start)*(x1-x0)
        xr=x0+(r['frame_index']+6-start)/(end-start)*(x1-x0)
        c=CYAN if r['state']=='confirmed' else AMBER if r['state'] in ('tentative','carrier_moving','carrier_settling') else '#52616b'
        d.rectangle((x,675,min(xr,x1),697),fill=c)
        c=GREEN if linked(r) else AMBER if r['liquids'] else '#52616b'
        d.rectangle((x,711,min(xr,x1),733),fill=c)
    x=x0+(row['frame_index']-start)/(end-start)*(x1-x0);d.line((x,669,x,739),fill=FG,width=2)
    text(im,(195,745),f'{start/30:.1f}s',15,MUTED);text(im,(1160,745),f'{end/30:.1f}s',15,MUTED)
    text(im,(20,778),'Cyan: vial  |  Green: candidate liquid link  |  Amber: withheld  |  Gray: no proposal / no confirmed track',18)
    text(im,(20,810),'Current detector and QA unchanged. Liquid class scope is unreviewed; no separate stream class. No volume estimate.',16,MUTED)
    return im


def main(args):
    if args.out.exists():raise ValueError('Use a fresh output folder')
    if sha256(args.tracking)!=TRACKING or sha256(args.cache)!=CACHE or sha256(args.video)!=VIDEO:
        raise ValueError('Frozen source changed')
    raw=json.loads(args.cache.read_text());rows={r['frame_index']:r for r in map(json.loads,args.tracking.read_text().splitlines())}
    phases={f['frame_index']:f['fill_detection']['phase'] for f in raw['features']}
    cases=[]
    for n,title in ((2,'Blue fill'),(3,'Pale fill')):
        # Image-only event estimates select a replay interval, never tracking input.
        event=next(e for e in raw['detections'] if e['detection_id']==f'run5-fill{n}')
        cases.append(dict(case_id=f'run5-sequence-{n}',title=title,start=event['start_frame']-90,
            end=event['confirmed_frame']+120,
            navigation_source='Existing independent image-fill estimates; not reviewed timing labels',
            source_event=event,filename=f'run5-{title.lower().replace(" ","-")}-tracking.mp4'))
    args.out.mkdir(parents=True);outputs={};movies={};histories={c['case_id']:[] for c in cases};stills={}
    targets={}
    for c in cases:
        e=c['source_event'];targets[c['case_id']]=[c['start'],e['start_frame']+36,e['episodes'][0]['end_available_frame']+30,
            e['episodes'][1]['start_available_frame']+36,e['departure_frame']-12,e['confirmed_frame']]
        output=av.open(str(args.out/c['filename']),'w');stream=output.add_stream('libx264',rate=5)
        stream.width=1280;stream.height=840;stream.pix_fmt='yuv420p';stream.options={'crf':'21','preset':'veryfast'}
        movies[c['case_id']]=(output,stream);outputs[c['case_id']]=0
    detector=CachedImageDetector(args.cache,CACHE,MODEL,VIDEO,5)
    for f,rgb,proposals in frames(args.video,detector,6):
        for c in cases:
            if not c['start']<=f<=c['end']:continue
            row=rows[f]
            if row['detector_proposals']!=proposals or row['source_rgb_sha256']!=hashlib.sha256(rgb.tobytes()).hexdigest():
                raise ValueError('Tracking/source evidence mismatch')
            histories[c['case_id']].append(row)
            im=frame_view(rgb,row,c,histories[c['case_id']],phases[f]);output,stream=movies[c['case_id']]
            vf=av.VideoFrame.from_image(im);vf.pts=outputs[c['case_id']];vf.time_base=Fraction(1,5)
            for packet in stream.encode(vf):output.mux(packet)
            outputs[c['case_id']]+=1
            if f in targets[c['case_id']]:stills[(c['case_id'],f)]=(painted(rgb,row),row)
    manifests=[]
    for c in cases:
        output,stream=movies[c['case_id']]
        for packet in stream.encode():output.mux(packet)
        output.close()
        with av.open(str(args.out/c['filename'])) as video:count=sum(1 for _ in video.decode(video=0))
        if count!=outputs[c['case_id']]:raise ValueError('Incomplete MP4')
        # Six chronological crops; edges are the same unchanged source boxes.
        sheet=Image.new('RGB',(1320,700),BG)
        text(sheet,(16,10),c['title']+' / chronological source crops',25)
        for n,f in enumerate(targets[c['case_id']]):
            marked,r=stills[(c['case_id'],f)];sheet.paste(marked.crop((250,50,380,440)).resize((195,585)),(10+n*220,55))
            text(sheet,(10+n*220,645),f'Frame {f} / {f/30:.1f}s',16)
            text(sheet,(10+n*220,671),'linked' if linked(r) else 'withheld' if r['liquids'] else 'no liquid box',16,GREEN if linked(r) else AMBER)
        sheet.save(args.out/(c['case_id']+'-storyboard.png'))
        manifests.append(dict(c,exported_frames=count,duration_s=count/5,video_sha256=sha256(args.out/c['filename']),
            sample_states=dict(Counter('linked' if linked(r) else 'withheld' if r['liquids'] else 'no_proposal' for r in histories[c['case_id']])),
            storyboard_frames=targets[c['case_id']]))
    write(args.out/'manifest.json',dict(video_sha256=VIDEO,detector_sha256=MODEL,cache_sha256=CACHE,
        tracking_sha256=TRACKING,annotation_files_read=False,detector_changed=False,measurement_qa_changed=False,
        source_images_changed=False,rendering_only=True,fill_height_or_volume_inferred=False,cases=manifests))
    print(json.dumps(manifests),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video',type=Path,required=True);p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--tracking',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
