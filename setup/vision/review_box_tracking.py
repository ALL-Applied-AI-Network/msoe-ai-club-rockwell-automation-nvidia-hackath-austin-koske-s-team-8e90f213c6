"""Box-only video diagnostic: frozen two-class detections + ByteTrack vial IDs.

No segmentation, measurement QA, recipes or evaluation annotations are consumed.
This exports candidates for review, never reviewed labels or accuracy claims.
"""
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import av
import cv2
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from replay.config import load_config
from replay.dataset import Catalog,DEFAULT_DATA,sha256
from replay.proposals import CachedImageDetector


def associate_liquid(liquid,vials):
    """Associate by current raw boxes only; never invent a missing detection."""
    a=liquid['box'];area=(a[2]-a[0])*(a[3]-a[1]);candidates=[]
    for index,vial in enumerate(vials):
        b=vial['box']
        inside=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
        if inside/max(1,area)>=.9: candidates.append(index)
    status='unique_vial' if len(candidates)==1 else 'multiple_vials' if candidates else 'no_vial'
    index=candidates[0] if len(candidates)==1 else None
    return dict(association=status,vial_detection_index=index,
                vial_track_id=vials[index]['track_id'] if index is not None else None)


class VialTracker:
    def __init__(self,stride):
        from ultralytics.trackers.byte_tracker import BYTETracker
        # Defaults from pinned Ultralytics, with the original 30-frame lost
        # buffer expressed in sampled updates. No color/measurement QA changes.
        self.settings=dict(track_high_thresh=.25,track_low_thresh=.1,new_track_thresh=.25,
                           track_buffer=max(1,round(30/stride)),match_thresh=.8,fuse_score=True)
        self.tracker=BYTETracker(SimpleNamespace(**self.settings))

    def update(self,proposals,shape):
        from ultralytics.engine.results import Boxes
        vials=[dict(p,track_id=None) for p in proposals if p['class_id']==0]
        rows=np.asarray([[*p['box'],p['confidence'],0] for p in vials],np.float32).reshape(-1,6)
        tracks=self.tracker.update(Boxes(rows,orig_shape=shape))
        for track in tracks:
            index=int(track[-1])
            if not 0<=index<len(vials): raise ValueError('Tracker returned an invalid current detection index')
            # Paint the fresh detection, not a Kalman-predicted/smoothed box.
            vials[index]['track_id']=int(track[4])
        liquids=[dict(p,**associate_liquid(p,vials)) for p in proposals if p['class_id']==1]
        return vials,liquids


def overlay(rgb,vials,liquids,index,fps):
    image=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
    for v in vials:
        box=v['box'];color=(255,180,70)
        cv2.rectangle(image,tuple(box[:2]),tuple(box[2:]),color,2)
        label=f"vial {v['confidence']:.2f} ID {v['track_id'] if v['track_id'] is not None else '?'}"
        cv2.putText(image,label,(box[0],max(20,box[1]-7)),0,.43,color,1)
    for liquid in liquids:
        box=liquid['box'];color=(70,240,100) if liquid['association']=='unique_vial' else (60,80,255)
        cv2.rectangle(image,tuple(box[:2]),tuple(box[2:]),color,2)
        label=f"liquid {liquid['confidence']:.2f} -> ID {liquid['vial_track_id'] or '?'}"
        cv2.putText(image,label,(max(0,box[0]-25),min(475,box[3]+15)),0,.42,color,1)
    header=np.zeros((40,1280,3),np.uint8)
    cv2.putText(header,'SOURCE',(8,26),0,.65,(255,255,255),1)
    cv2.putText(header,f'RAW BOXES + CANDIDATE TRACKS   frame {index} / {index/fps:.1f}s',(648,26),0,.55,(255,255,255),1)
    return np.vstack([header,np.hstack([cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),image])]),image


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--cache-sha256',required=True)
    parser.add_argument('--config',type=Path,default=Path('replay/examples/bootstrap-vision.json'))
    parser.add_argument('--data',type=Path,default=DEFAULT_DATA)
    parser.add_argument('--run',type=int,default=5)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists(): parser.error('Use a new output directory to preserve previous evidence')
    config=load_config(args.config)
    if config.liquid_segmentation_model_path: parser.error('Box review requires the box-only detector config')
    if sha256(config.vision_model_path)!=config.vision_model_sha256: raise ValueError('Pinned detector hash mismatch')
    run=Catalog(args.data,include_annotations=False).runs[args.run];camera=run['cameras']['vial_fill']
    detector=CachedImageDetector(args.cache,args.cache_sha256,config.vision_model_sha256,camera['sha256'],args.run)
    expected=list(range(0,camera['frames'],config.sample_stride))
    if sorted(detector.frames)!=expected: raise ValueError('Cache must contain every sampled frame for this recording')
    tracker=VialTracker(config.sample_stride)
    args.out.mkdir(parents=True);(args.out/'stills').mkdir()
    reviews=set(expected[int(i)] for i in np.linspace(0,len(expected)-1,60))
    counters=Counter();track_ids=set();review_rows=[];sample_count=decoded=0
    output_fps=Fraction(str(camera['fps']))/config.sample_stride
    with av.open(camera['path']) as container,av.open(str(args.out/'boxes.mp4'),'w') as movie,(args.out/'boxes.jsonl').open('w',encoding='utf-8') as events:
        container.streams.video[0].thread_type='AUTO'
        stream=movie.add_stream('libx264',rate=output_fps)
        stream.width=1280;stream.height=520;stream.pix_fmt='yuv420p';stream.options={'crf':'24','preset':'veryfast'}
        for index,frame in enumerate(container.decode(video=0)):
            decoded+=1
            if index%config.sample_stride: continue
            if frame.pts is None or abs(float(frame.pts*frame.time_base)-camera['start_pts_s']-index/camera['fps'])>1/camera['fps']:
                raise ValueError('Unsupported or interrupted presentation timestamps')
            rgb=frame.to_ndarray(format='rgb24');detector.prepare_frame(index)
            vials,liquids=tracker.update(detector.detect(rgb),rgb.shape[:2])
            row=dict(frame_index=index,mp4_s=index/camera['fps'],source_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),vials=vials,liquids=liquids)
            events.write(json.dumps(row)+'\n')
            counters['samples_with_vial']+=bool(vials);counters['samples_with_liquid']+=bool(liquids)
            counters['samples_with_both']+=bool(vials and liquids)
            counters['samples_with_multiple_vials']+=len(vials)>1
            for liquid in liquids:counters['liquid_'+liquid['association']]+=1
            track_ids.update(v['track_id'] for v in vials if v['track_id'] is not None)
            image,annotated=overlay(rgb,vials,liquids,index,camera['fps'])
            video_frame=av.VideoFrame.from_ndarray(image,format='bgr24');video_frame.pts=sample_count
            video_frame.time_base=1/output_fps
            for packet in stream.encode(video_frame):movie.mux(packet)
            sample_count+=1
            if index in reviews:
                source=f'stills/f{index:06d}-source.png';view=f'stills/f{index:06d}-boxes.png'
                if not cv2.imwrite(str(args.out/source),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)):raise IOError(source)
                if not cv2.imwrite(str(args.out/view),annotated):raise IOError(view)
                review_rows.append(dict(row,source_image=source,overlay_image=view,review_status='unreviewed',
                                        corrected_boxes=None,physical_vial_identity=None))
        for packet in stream.encode():movie.mux(packet)
    if decoded!=camera['frames'] or sample_count!=len(expected):raise ValueError('Incomplete recording decode')
    import ultralytics
    manifest=dict(schema_version=1,kind='box_only_tracking_review',run_id=args.run,video_sha256=camera['sha256'],
                  source_video=str(camera['path']),model_sha256=config.vision_model_sha256,
                  detector_cache_sha256=args.cache_sha256,decoded_source_frames=decoded,samples=sample_count,
                  source_fps=camera['fps'],sample_stride=config.sample_stride,output_fps=float(output_fps),
                  tracker='ByteTrack vial-only; liquids associated to current vial detections',
                  tracker_settings=tracker.settings,ultralytics=ultralytics.__version__,
                  cached_detector_confidence_floor=.35,source_rgb_hashes_verified=True,
                  unique_candidate_track_ids=len(track_ids),counts=dict(counters),
                  reviewed_detection_metrics=None,reviewed_tracking_metrics=None,
                  limitations=['Presence counts are not precision or recall.','Candidate track IDs are not verified physical identities.',
                               'Raw boxes are unreviewed; no mask or color claim follows.',
                               'Cached detections omit confidence below 0.35; ByteTrack low-confidence matching is not evaluated.',
                               'Sampled 5 fps review is not native-frame-rate tracking validation.',
                               'Run 5 has already been inspected; it is not a new blind test.'])
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    (args.out/'review.json').write_text(json.dumps(dict(schema_version=1,kind='unreviewed_box_candidates',frames=review_rows),indent=2)+'\n',encoding='utf-8')
    cards=''.join(f'<figure><a href="{r["source_image"]}"><img loading="lazy" src="{r["overlay_image"]}"></a><figcaption>Frame {r["frame_index"]} / {r["mp4_s"]:.1f}s — unreviewed</figcaption></figure>' for r in review_rows)
    page='<!doctype html><meta charset="utf-8"><title>Vial / liquid box review</title><style>body{font:17px system-ui;margin:24px;background:#121820;color:#eee}video{width:min(100%,1280px)}section{display:flex;flex-wrap:wrap}figure{margin:8px}img{width:320px}a{color:#8cf}</style>'
    page+='<h1>Vial / liquid box review</h1><p>Frozen two-class detector + ByteTrack. Blue: vial. Green: liquid associated with one current vial. Red: unassociated/ambiguous liquid. Track IDs are provisional.</p><p>No segmentation or color QA. This video is for reviewing box placement and continuity; it does not establish accuracy.</p><video controls preload="metadata" src="boxes.mp4"></video><p><a href="manifest.json">Provenance and presence counts</a> · <a href="review.json">60-frame correction interface</a></p><section>'+cards+'</section>'
    (args.out/'index.html').write_text(page,encoding='utf-8')
    print(json.dumps(manifest),flush=True)


if __name__=='__main__':main()
