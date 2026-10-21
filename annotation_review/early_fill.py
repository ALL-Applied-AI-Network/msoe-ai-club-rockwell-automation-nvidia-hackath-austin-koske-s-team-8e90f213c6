"""Prepare unreviewed exact-frame onset/mask tasks from a public review request.

No label generation, timing adjudication, replay imports or workspace writes.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

from .core import sha

SCHEMA = 'first-visible-liquid-review-request/v1'
OFFSETS = [-60, -30, 0, 15, 30, 45, 60, 75, 90, 120, 150, 180]


def validate_request(doc):
    if doc.get('schema') != SCHEMA or doc.get('source', {}).get('run_id') != 4:
        raise ValueError('Expected an explicit Run 4 early-fill review request')
    if doc.get('review_status') != 'unreviewed' or doc.get('approved_labels') is not False:
        raise ValueError('This builder only accepts unreviewed navigation requests')
    source = doc['source']
    if source.get('camera') != 'vial_fill' or not source.get('sha256') or not source.get('frame_count'):
        raise ValueError('Missing source identity')
    relative = Path(source['relative_path'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Unsafe video locator')
    seen = set()
    for case in doc['cycles']:
        key, anchor = case['case_id'], case['provisional_anchor_frame']
        if key in seen or not isinstance(key, str) or not key.replace('-', '').isalnum():
            raise ValueError('Unsafe or duplicate case identity')
        seen.add(key)
        if type(anchor) is not int or not 60 <= anchor < source['frame_count']-180:
            raise ValueError('Anchor cannot supply the predeclared review neighborhood')
        if case.get('anchor_status') != 'image_detector_proposal_not_reviewed_onset':
            raise ValueError('An image anchor cannot approve visibility timing')
    if not seen:
        raise ValueError('Empty review request')
    return doc


def build(request, dataset, out):
    import av
    import cv2
    request, dataset, out = Path(request), Path(dataset).resolve(), Path(out)
    doc = validate_request(json.loads(request.read_text()))
    if out.exists():
        raise ValueError('Use a fresh review package; preserve prior evidence')
    source = (dataset/doc['source']['relative_path']).resolve()
    if not source.is_relative_to(dataset) or sha(source) != doc['source']['sha256']:
        raise ValueError('Source video identity differs')
    out.mkdir(parents=True)
    shutil.copy2(request, out/'request.json')
    shutil.copy2(__file__, out/'executed-builder.py')
    (out/'frames').mkdir()
    (out/'clips').mkdir()
    tasks, cycles, mask_tasks, decode_checks = [], [], [], []
    for cycle in doc['cycles']:
        key, anchor = cycle['case_id'], cycle['provisional_anchor_frame']
        # Two seconds before and six seconds after the image-derived anchor.
        # Boundary identities use decoded indices; actual times come from PTS.
        start, end = anchor-60, anchor+180
        selected = {anchor+offset for offset in OFFSETS}
        clip = out/'clips'/f'{key}.mp4'
        writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*'mp4v'), 30., (640, 480))
        if not writer.isOpened():
            raise RuntimeError('Cannot encode source review clip')
        trace = []
        try:
            with av.open(str(source)) as c:
                for index, frame in enumerate(c.decode(video=0)):
                    if index > end:
                        break
                    if index < start:
                        continue
                    if frame.pts is None:
                        raise ValueError('Missing native video PTS')
                    rgb = frame.to_ndarray(format='rgb24')
                    digest = hashlib.sha256(rgb.tobytes()).hexdigest()
                    frame_identity = dict(frame_index=index, video_pts=frame.pts, time_base=str(frame.time_base),
                                          video_pts_seconds=float(frame.pts*frame.time_base), source_rgb_sha256=digest)
                    # The PNG at the inclusive end is a mask task, while the MP4
                    # remains an eight-second [start,end) navigation clip.
                    if index < end:
                        writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                        trace.append(frame_identity)
                    if index in selected:
                        name = f'frames/{key}-frame{index:06d}.png'
                        image = out/name
                        if not cv2.imwrite(str(image), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                            raise RuntimeError('Cannot save exact source PNG')
                        decoded_rgb = cv2.cvtColor(cv2.imread(str(image)), cv2.COLOR_BGR2RGB)
                        if hashlib.sha256(decoded_rgb.tobytes()).hexdigest() != digest:
                            raise ValueError('PNG pixel round-trip changed')
                        sample = f'early_{key}_frame{index}'
                        provenance = dict(kind='video_frame', camera='vial_fill', run='4', filename=image.name,
                            image_relative_path=name, sha256=sha(image), width=frame.width, height=frame.height,
                            video_relative_path=doc['source']['relative_path'], source_video_sha256=doc['source']['sha256'],
                            capture_timestamp=None, **frame_identity,
                            clock_status='Native MP4 PTS; no still/wall-clock or process-clock association')
                        tasks.append(dict(sample_id=sample, cohort='run4_early_liquid', assignee='', priority=20,
                            source=provenance, reason='Unreviewed early-fill onset/mask review neighborhood',
                            context_references=dict(kind=SCHEMA, request_sha256=sha(request), cycle=key,
                                provisional_anchor_frame=anchor, anchor_status=cycle['anchor_status'],
                                purpose='Navigation only; no approved onset, presence or mask label'),
                            reviewed_label=None, review_status='unreviewed'))
                        mask_tasks.append(dict(sample_id=sample, frame_index=index, image=name,
                            image_sha256=provenance['sha256'], source_video_sha256=doc['source']['sha256'],
                            presence_status='unreviewed', reviewed_mask_path=None, reviewer=None,
                            notes='', mask_sha256=None, conventions_version=None))
        finally:
            writer.release()
        with av.open(str(clip)) as c:
            decoded = sum(1 for _ in c.decode(video=0))
        if decoded != end-start or len(trace) != decoded:
            raise ValueError('Incomplete review navigation clip')
        trace_path = out/'clips'/f'{key}-source-frames.json'
        trace_path.write_text(json.dumps(trace, indent=2)+'\n')
        cycles.append(dict(case_id=key, source_video_sha256=doc['source']['sha256'],
            clip_relative_path=clip.relative_to(out).as_posix(), source_start_frame=start,
            source_end_frame_exclusive=end, decoded_frames=decoded, provisional_anchor_frame=anchor,
            anchor_status=cycle['anchor_status'], reviewed_first_visible_frame=None,
            review_status='unreviewed', reviewer=None, notes=''))
        decode_checks.append(dict(path=clip.relative_to(out).as_posix(), decoded_frames=decoded, sha256=sha(clip)))
    if len(tasks) != len(doc['cycles'])*len(OFFSETS):
        raise ValueError('Missing exact-frame task')
    if sha(source) != doc['source']['sha256']:
        raise ValueError('Source video changed during preparation')
    def save(name, value):
        (out/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    save('cases.json', dict(schema='first-vial-review-cases/v1', cases=tasks,
        review_status='unreviewed', approved_labels=False, source_request_sha256=sha(request)))
    save('onset-review.json', dict(schema='first-visible-liquid-onset-review/v1', cycles=cycles,
        instructions='Choose exact zero-based decoded source frame. Use uncertain or left/right_censored when visibility cannot be established. Image anchors are not truth.'))
    save('mask-review.json', dict(schema='first-visible-liquid-mask-review/v1', tasks=mask_tasks,
        mask_format='Full source-sized 640x480 binary PNG: 0 background, 255 liquid; include only reviewer-defined visible pooled liquid. Null means unreviewed, never confirmed empty.'))
    with (out/'onset-review.csv').open('w', newline='', encoding='utf-8') as f:
        fields = ['case_id', 'source_video_sha256', 'review_status', 'first_visible_frame',
                  'last_definitely_absent_frame', 'reviewer', 'conventions_version', 'notes']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for cycle in cycles:
            writer.writerow(dict(case_id=cycle['case_id'], source_video_sha256=doc['source']['sha256'], review_status='unreviewed'))
    save('verification.json', dict(cycles=len(cycles), exact_source_pngs=len(tasks),
        source_video_sha256=doc['source']['sha256'], source_native_frame_count=doc['source']['frame_count'],
        decoded_review_videos=decode_checks, approved_onsets=0, approved_masks=0,
        existing_workspaces_modified=False, run5_used=False,
        artifacts={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()}))
    return {'cycles':len(cycles), 'exact_source_pngs':len(tasks), 'approved_labels':False}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('request', 'dataset', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(build(a.request, a.dataset, a.out)))
