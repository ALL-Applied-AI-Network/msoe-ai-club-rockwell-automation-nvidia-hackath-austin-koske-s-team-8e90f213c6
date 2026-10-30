"""Independently verify finished inference after manifest-only failure; fresh copy.

The failed original/logs/helpers remain intact. No GPU inference is repeated.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(a):
    import av
    import numpy as np
    if a.out.exists():raise FileExistsError(a.out)
    if (a.source/'completed.json').exists():raise ValueError('Original already completed; inspect rather than recover')
    plan=read(a.source/'plan.json');report=read(a.source/'comparison.json')
    for name,digest in plan['helper_sha256'].items():assert sha(a.source/'helpers'/name)==digest
    for model in plan['models'].values():assert sha(model['file'])==model['sha256']
    inventory=[json.loads(s) for s in (a.probe/'frames.jsonl').read_text().splitlines()];wanted={r['source_frame']:r for r in inventory}
    passes=[]
    for name in ['warmup.jsonl','pass-1.jsonl','pass-2.jsonl']:
        rows=[json.loads(s)['arms'] for s in (a.source/name).read_text().splitlines()]
        assert len(rows)==264
        for row,expected in zip(rows,inventory):
            for arm in plan['arms']:
                for k in ['case_id','source_frame','pts','time_base','source_rgb_sha256']:assert row[arm][k]==expected[k]
            assert row['legacy_sam']['vials']==row['synthetic_sam']['vials']==row['temporal']['vials']
        passes.append(rows)
    for arm in plan['arms']:
        first=[r[arm] for r in passes[1]];second=[r[arm] for r in passes[2]];meta=report['arms'][arm]
        assert sum(r['measurement']['measurement_valid'] for r in first)==meta['qa_pass']
        assert sum(r['proposal_available'] for r in first)==meta['proposal_frames']
        assert sum(r['tall_mask'] for r in first)==meta['tall_mask_frames']
        for stage in meta['latency_ms']:
            v=[r['timing_ms'][stage] for r in first+second]
            assert float(np.median(v))==meta['latency_ms'][stage]['median']
            assert float(np.percentile(v,95))==meta['latency_ms'][stage]['p95']
        for x,y in zip(first,second):
            assert x['raw_masks']==y['raw_masks']
            for k in ['lab','rgb','measurement_valid']:assert x['measurement'][k]==y['measurement'][k]
            assert x['measurement']['geometry']['mask']==y['measurement']['geometry']['mask']
    assert sha(a.video)==plan['source_video_sha256']
    n=0
    with av.open(str(a.video)) as container:
        for i,frame in enumerate(container.decode(video=0)):
            if i>max(wanted):break
            if i not in wanted:continue
            row=wanted[i];rgb=frame.to_ndarray(format='rgb24')
            assert hashlib.sha256(rgb.tobytes()).hexdigest()==row['source_rgb_sha256']
            assert frame.pts==row['pts'] and str(frame.time_base)==row['time_base'];n+=1
    assert n==264
    with av.open(str(a.source/'sam-vs-temporal.mp4')) as container:
        assert container.streams.video[0].average_rate==5
        decoded=0
        for frame in container.decode(video=0):
            assert (frame.width,frame.height)==(1920,560);decoded+=1
    assert decoded==240
    tasks=read(a.source/'review-drafts.json')['tasks'];assert len(tasks)==72
    for task in tasks:
        assert sha(a.probe/task['image'])==task['image_sha256']
        for draft in task['drafts']:assert sha(a.source/draft['mask'])==draft['mask_sha256']
    shutil.copytree(a.source,a.out,ignore=shutil.ignore_patterns('__pycache__'))
    (a.out/'recovery.json').write_text(json.dumps(dict(original=str(a.source),
        reason='Manifest-only IsADirectoryError on helpers/__pycache__ after all inference/video outputs finished.',
        inference_repeated=False,original_preserved=True,source_frames_redecoded=264,video_frames_redecoded=240,
        raw_and_final_masks_colors_QA_match_between_passes=True,timing_statistics_recomputed=True,
        finalizer_sha256=sha(__file__)),indent=2)+'\n')
    completed=dict(artifacts={p.name:sha(p) for p in a.out.iterdir() if p.is_file()},
        helpers={p.name:sha(p) for p in (a.out/'helpers').iterdir() if p.is_file()},
        draft_masks={p.name:sha(p) for p in (a.out/'draft-masks').iterdir() if p.is_file()},
        original_manifest_failure_recovered=True,local_gpu_used=False,run5_used=False,app_changed=False)
    (a.out/'completed.json').write_text(json.dumps(completed,indent=2)+'\n')
    print(json.dumps(dict(completed=str(a.out),source_frames=264,video_frames=240,inference_repeated=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['source','probe','video','out']:p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
