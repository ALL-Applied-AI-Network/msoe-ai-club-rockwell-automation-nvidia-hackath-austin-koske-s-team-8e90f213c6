"""Export every changed-link neighborhood as a local diagnostic review video."""
import argparse
from collections import Counter
import json
from pathlib import Path

from compare_station_tracking import (BASELINE_ROWS, CACHE, MODEL, VIDEO,
    CachedImageDetector, Movie, frames, overlay, sha256, write)


def main(args):
    if sha256(args.baseline)!=BASELINE_ROWS or sha256(args.video)!=VIDEO:
        raise ValueError('Original comparison input mismatch')
    if (args.trial/'association-review.json').exists():
        raise ValueError('Do not overwrite review export')
    new={r['frame_index']:r for r in map(json.loads,(args.trial/'tracking.jsonl').read_text(encoding='utf-8').splitlines())}
    old={r['frame_index']:r for r in map(json.loads,args.baseline.read_text(encoding='utf-8').splitlines())}
    if sorted(new)!=list(range(0,50942,6)) or sorted(old)!=sorted(new):
        raise ValueError('Incomplete comparison rows')
    cases=[];windows=[]
    for f,row in new.items():
        if len(row['liquids'])!=len(old[f]['liquids']):raise ValueError('Raw detection count changed')
        changed=[]
        for i,(a,b) in enumerate(zip(old[f]['liquids'],row['liquids'])):
            if (a.get('vial_track_id') is not None)!=(b.get('vial_track_id') is not None):
                changed.append(dict(liquid_index=i,raw_box=b['box'],old_candidate_id=a['vial_track_id'],
                    new_candidate_id=b['vial_track_id'],new_reason=b['association']))
        if changed:
            cases.append(dict(frame_index=f,mp4_s=f/30,source_rgb_sha256=row['source_rgb_sha256'],
                changed_links=changed,review_status='unreviewed',physical_identity=None))
            start,end=max(0,f-30),min(50940,f+30)
            if windows and start<=windows[-1]['end']+6:
                windows[-1]['end']=max(end,windows[-1]['end'])
            else:windows.append(dict(start=start,end=end))
    if not cases:raise ValueError('No changed links to export')
    movie=Movie(args.trial/'run5-association-review.mp4')
    detector=CachedImageDetector(args.cache,CACHE,MODEL,VIDEO,5)
    snapshots=[]
    first_reason={}
    for case in cases:
        for change in case['changed_links']:
            first_reason.setdefault(change['new_reason'],case['frame_index'])
    for f,rgb,proposals in frames(args.video,detector,6):
        if new[f]['detector_proposals']!=proposals:raise ValueError('Changed raw evidence')
        if any(w['start']<=f<=w['end'] for w in windows):
            im=overlay(rgb,old[f],new[f]);movie.add(im)
            if f in first_reason.values():
                import cv2
                name=f'stills/association-frame{f:06d}.png'
                if not cv2.imwrite(str(args.trial/name),im):raise IOError(name)
                snapshots.append(dict(frame_index=f,image=name))
    movie.close()
    import av
    with av.open(str(movie.path)) as container:decoded=sum(1 for _ in container.decode(video=0))
    if decoded!=movie.count:raise ValueError('Incomplete MP4')
    changes=[change for c in cases for change in c['changed_links']]
    result=dict(schema_version=1,kind='changed_candidate_link_review_not_ground_truth',
        tracking_jsonl_sha256=sha256(args.trial/'tracking.jsonl'),comparison_sha256=sha256(args.trial/'comparison.json'),
        source_video_sha256=VIDEO,cache_sha256=CACHE,changed_frames=len(cases),changed_proposals=len(changes),
        removed_links=sum(c['new_candidate_id'] is None for c in changes),
        added_links=sum(c['old_candidate_id'] is None for c in changes),
        new_reasons=dict(Counter(c['new_reason'] for c in changes)),cases=cases,windows=windows,
        snapshots=snapshots,video=dict(file=movie.path.name,frames=movie.count,fps=5,sha256=sha256(movie.path)),
        annotations_modified=False,reviewed_accuracy=None)
    write(args.trial/'association-review.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('cases','windows')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trial',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--video',type=Path,required=True);p.add_argument('--cache',type=Path,required=True)
    main(p.parse_args())
