"""Local Run 5 MP4 diagnostics: identical frozen detections, causal association.

Only RGB and pinned raw proposals enter the tracker. Baseline IDs are read from
the previous frozen diagnostic, never from supplied vial/timing annotations.
"""
import argparse
from collections import Counter
from dataclasses import asdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

import av
import cv2
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from replay.box_tracking import StationTracker, TrackingSettings
from replay.config import load_config
from replay.dataset import sha256
from replay.proposals import CachedImageDetector


BASELINE_ROWS = 'e19049077814f077a8cb460fa7b1eb6a4f1f481a23744ad319b71462ceb3477c'
MODEL = '89ed11f7046171e5340dc45a9d528010de1cefbba21e163b23b7098151e0880e'
CACHE = 'aff504002184d856ac06451f1730677eace2e840b89110c2cb6c0af39a8850d8'
VIDEO = '6f87cec1aeec3b2bae6dac639a317c077a3f175f27abd5fc8430d6953d9d05b9'


def write(path, obj):
    Path(path).write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8')


def raw(rows):
    return [{k:r[k] for k in ('class_id','box','confidence')} for r in rows]


def frames(video, detector, stride):
    with av.open(str(video)) as container:
        stream=container.streams.video[0];stream.thread_type='AUTO'
        if (stream.width,stream.height,float(stream.average_rate),stream.frames)!=(640,480,30.,50942):
            raise ValueError('Unexpected source geometry/cadence/count')
        start=float((stream.start_time or 0)*stream.time_base)
        decoded=0
        for i, frame in enumerate(container.decode(video=0)):
            decoded+=1
            if i % stride:continue
            if frame.pts is None or abs(float(frame.pts*frame.time_base)-start-i/30)>1e-5:
                raise ValueError('Interrupted source PTS')
            rgb=frame.to_ndarray(format='rgb24');detector.prepare_frame(i)
            proposals=detector.detect(rgb)
            yield i,rgb,proposals
        if decoded!=50942:
            raise ValueError('Incomplete source decode')


def short_id(value):
    return value.rsplit(':',1)[-1] if value else '?'


def panel(rgb,row,new):
    im=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
    for v in row['vials']:
        b=v['box'];color=(255,210,70)
        cv2.rectangle(im,tuple(b[:2]),tuple(b[2:]),color,2)
        label=f"vial {short_id(str(v['track_id'])) if v['track_id'] else '?'}"
        if new:label+=' '+v['track_status']
        cv2.putText(im,label,(max(4,b[0]-15),max(16,b[1]-7)),0,.43,color,1,cv2.LINE_AA)
    for l in row['liquids']:
        b=l['box'];linked=l['association'] in ('unique_vial','candidate_link') and l.get('vial_track_id') is not None
        color=(90,230,90) if linked else (80,170,255)
        cv2.rectangle(im,tuple(b[:2]),tuple(b[2:]),color,2)
        label='liquid -> '+short_id(str(l['vial_track_id'])) if linked else 'liquid: '+l['association']
        # Put long status strings to the left, without changing source box.
        x=max(4,min(b[0]-15,640-int(len(label)*7)))
        cv2.putText(im,label,(x,min(474,b[3]+17)),0,.4,color,1,cv2.LINE_AA)
    return im


def overlay(rgb,old,new):
    canvas=np.zeros((600,1280,3),np.uint8)
    canvas[48:528,:640]=panel(rgb,old,False);canvas[48:528,640:]=panel(rgb,new,True)
    for x,title in ((8,'OLD: ByteTrack + per-frame containment'),(648,'NEW: causal station segments + guarded association')):
        cv2.putText(canvas,title,(x,23),0,.52,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(canvas,f"SOURCE frame {new['frame_index']} / {new['frame_index']/30:.2f}s   sampled 5 fps",(8,44),0,.44,(210,210,210),1)
    motion='unknown' if new['motion'] is None else f"{new['motion']:.2f}"
    cv2.putText(canvas,f"State: {new['state']}    image motion: {motion}",(648,550),0,.47,(255,255,255),1)
    cv2.putText(canvas,'Same raw detections, boxes and confidences. Cyan: vial. Green: candidate link. Amber: withheld.',(8,574),0,.47,(255,255,255),1)
    cv2.putText(canvas,'IDs are unverified station segments. Liquid scope unreviewed. No masks, color decisions or QA changes.',(8,596),0,.45,(180,210,255),1)
    return canvas


class Movie:
    def __init__(self,path):
        self.path=path;self.container=av.open(str(path),'w');self.count=0
        self.stream=self.container.add_stream('libx264',rate=5)
        self.stream.width=1280;self.stream.height=600;self.stream.pix_fmt='yuv420p'
        self.stream.options={'crf':'23','preset':'veryfast'}

    def add(self,image):
        frame=av.VideoFrame.from_ndarray(image,format='bgr24')
        frame.pts=self.count;frame.time_base=Fraction(1,5)
        for p in self.stream.encode(frame):self.container.mux(p)
        self.count+=1

    def close(self):
        for p in self.stream.encode():self.container.mux(p)
        self.container.close()


def main(args):
    if args.out.exists():raise ValueError('Use a fresh output directory')
    config=load_config(args.config)
    if config.liquid_segmentation_model_path or config.sample_stride!=6 or config.vision_model_sha256!=MODEL:
        raise ValueError('Requires unchanged box-only detector/config cadence')
    if sha256(config.vision_model_path)!=MODEL or sha256(args.video)!=VIDEO or sha256(args.baseline)!=BASELINE_ROWS:
        raise ValueError('Pinned video/model/previous diagnostic mismatch')
    settings=TrackingSettings(source_fps=30,stride=6,
        station_x=(config.stream_center_roi[0]+config.stream_center_roi[2])/2,
        motion_threshold=config.departure_motion,motion_confirm_s=config.departure_confirm_s,settle_s=config.carrier_settle_s)
    # Freeze settings and preservation hashes before looking at the comparison.
    args.out.mkdir(parents=True);(args.out/'stills').mkdir()
    preserved={str(p):sha256(p) for p in [Path('replay/config.py'),Path('replay/geometry.py'),
        Path('replay/measurement.py'),Path('replay/pipeline.py'),Path('replay/detection.py'),
        Path('replay/examples/bootstrap-vision.json'),Path(config.vision_model_path)]}
    plan=dict(settings=asdict(settings),model_sha256=MODEL,cache_sha256=CACHE,video_sha256=VIDEO,
        baseline_rows_sha256=BASELINE_ROWS,preserved_source_hashes=preserved,
        tracking_implementation_sha256=sha256(Path('replay/box_tracking.py')),
        exporter_sha256=sha256(Path(__file__)),
        annotations_used=False,calibration_annotations_used=False,measurement_pipeline_changed=False,
        containment_unchanged=.9,automatic_promotion=False)
    write(args.out/'plan.json',plan)
    detector=CachedImageDetector(args.cache,CACHE,MODEL,VIDEO,5)
    expected=list(range(0,50942,6))
    if sorted(detector.frames)!=expected:raise ValueError('Cache incomplete')
    old={r['frame_index']:r for r in map(json.loads,args.baseline.read_text(encoding='utf-8').splitlines())}
    if sorted(old)!=expected:raise ValueError('Previous diagnostic incomplete')
    tracker=StationTracker(settings,session_id='run5-tracking-review')
    rows={};counts=Counter();ids=set();old_ids=set();changed=[]
    with (args.out/'tracking.jsonl').open('w',encoding='utf-8') as output:
        for f,rgb,proposals in frames(args.video,detector,6):
            previous=old[f]
            if (previous['source_rgb_sha256']!=hashlib.sha256(rgb.tobytes()).hexdigest() or
                    raw(previous['vials']+previous['liquids'])!=proposals):
                raise ValueError('Baseline/cache raw evidence differs')
            row=tracker.update(rgb,proposals,f)
            if raw(row['vials']+row['liquids'])!=proposals or row['detector_proposals']!=proposals:
                raise ValueError('Tracker modified raw proposals')
            row.update(mp4_s=f/30,source_rgb_sha256=previous['source_rgb_sha256'])
            rows[f]=row;output.write(json.dumps(row)+'\n')
            counts['state_'+row['state']]+=1
            for l in row['liquids']:counts['liquid_'+l['association']]+=1
            for e in row['events']:counts['event_'+e['kind']]+=1
            ids.update(v['track_id'] for v in row['vials'] if v['track_id'])
            old_ids.update(v['track_id'] for v in previous['vials'] if v['track_id'])
            old_links=[l.get('vial_track_id') is not None for l in previous['liquids']]
            new_links=[l.get('vial_track_id') is not None for l in row['liquids']]
            if old_links!=new_links:changed.append(f)
    write(args.out/'first-pass.json',dict(counts=dict(counts),old_ids=len(old_ids),new_ids=len(ids),changed_link_frames=len(changed)))
    print('Tracking pass complete: '+json.dumps(dict(counts)),flush=True)
    # Prefix processing independently reproduces the first 120 outputs. A restart
    # uses a different session namespace and cannot inherit old track state.
    prefix=StationTracker(settings,session_id='run5-tracking-review')
    restarted=StationTracker(settings,session_id='run5-restarted-review')
    full=Movie(args.out/'run5-tracking-full.mp4')
    early=Movie(args.out/'run5-early-pours.mp4')
    transitions=Movie(args.out/'run5-carrier-transitions.mp4')
    onsets=[1374,2760,4152,5400,6660,8550,10152,11754]
    early_windows=[dict(anchor=f,start=((f-30)//6)*6,end=((f+150)//6)*6,
                        basis='Fixed prior image-estimated neighborhood; not reviewed physical onset') for f in onsets]
    transition_frames=[f for f,row in rows.items() if any(e['kind']=='carrier_transition_confirmed' for e in row['events'])]
    chosen=[transition_frames[int(i)] for i in np.linspace(0,len(transition_frames)-1,min(8,len(transition_frames)),dtype=int)]
    transition_windows=[dict(anchor=f,start=max(0,((f-60)//6)*6),end=min(expected[-1],((f+150)//6)*6),
                             basis='New tracker image-motion confirmation; diagnostic selection only') for f in chosen]
    target_stills={2796,1410,25470,expected[-1]}
    for f in chosen[:3]:target_stills.update([f-12,f,f+30])
    targets={f for f in target_stills if f in rows};still_files=[]
    prefix_checked=restart_checked=0
    for f,rgb,proposals in frames(args.video,detector,6):
        row=rows[f]
        if prefix_checked<120:
            fresh=prefix.update(rgb,proposals,f)
            if any(fresh[k]!=row[k] for k in fresh):raise ValueError('Prefix causality mismatch')
            prefix_checked+=1
        if 25002<=f<25122:
            r=restarted.update(rgb,proposals,f)
            if restart_checked==0 and (r['liquids'] and any(l['vial_track_id'] for l in r['liquids'])):
                raise ValueError('Restart inherited an association')
            if any(v['track_id'] and not v['track_id'].startswith('run5-restarted-review:') for v in r['vials']):
                raise ValueError('Restart namespace leaked')
            restart_checked+=1
        image=overlay(rgb,old[f],row);full.add(image)
        if any(w['start']<=f<=w['end'] for w in early_windows):early.add(image)
        if any(w['start']<=f<=w['end'] for w in transition_windows):transitions.add(image)
        if f in targets:
            name=f'stills/frame{f:06d}.png'
            if not cv2.imwrite(str(args.out/name),image):raise IOError(name)
            still_files.append(dict(frame_index=f,mp4_s=f/30,image=name,source_rgb_sha256=row['source_rgb_sha256']))
    for movie in (full,early,transitions):movie.close()
    video_checks=[]
    for movie in (full,early,transitions):
        with av.open(str(movie.path)) as container:
            decoded=sum(1 for _ in container.decode(video=0))
        if decoded!=movie.count:raise ValueError('Incomplete exported MP4')
        video_checks.append(dict(file=movie.path.name,frames=decoded,fps=5,sha256=sha256(movie.path)))
    if full.count!=8491 or prefix_checked!=120 or restart_checked!=20:
        raise ValueError('Incomplete comparison/causality/restart verification')
    if any(sha256(p)!=digest for p,digest in preserved.items()):
        raise ValueError('Detector/measurement/QA source changed')
    if (sha256(Path('replay/box_tracking.py'))!=plan['tracking_implementation_sha256'] or
            sha256(Path(__file__))!=plan['exporter_sha256']):
        raise ValueError('Tracking/export implementation changed during execution')
    result=dict(plan,run_id=5,source_frames=50942,sampled_frames=8491,
        old_unique_candidate_track_ids=len(old_ids),new_unique_candidate_station_ids=len(ids),counts=dict(counts),
        changed_link_frames=len(changed),raw_boxes_and_confidences_unchanged_on_every_sample=True,
        source_rgb_hashes_verified_on_every_sample=True,prefix_samples_verified=prefix_checked,
        restart_samples_verified=restart_checked,reviewed_tracking_accuracy=None,
        reviewed_association_accuracy=None,video_checks=video_checks,stills=still_files,
        early_windows=early_windows,transition_windows=transition_windows,
        limitations=['Station IDs are provisional; count is not number of verified physical vials.',
            'Same 5 fps cached detections; not native-30-fps tracker validation.',
            'Motion/settling settings are reused from existing image-only detector; lighting can also trigger motion.',
            'No boxes for missing detections; same-position undetected swaps can still escape this tracker.',
            'Legacy liquid may contain stream/body; link does not verify scope, pixels, color or release.',
            'Stricter association trades availability for explicit uncertainty; counts are not accuracy.',
            'Run 5 repeatedly inspected; no independent test claim.',
            'Opt-in tracking diagnostic; production measurement/QA tracking unchanged.'])
    write(args.out/'comparison.json',result)
    print(json.dumps(dict(out=str(args.out),old_ids=len(old_ids),new_ids=len(ids),videos=video_checks)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video',type=Path,required=True);p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path('replay/examples/bootstrap-vision.json'))
    main(p.parse_args())
