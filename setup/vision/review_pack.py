"""Read-only selection of 210 real stills; separate human review, no label repair."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import shutil
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
CLASSES = ('vial', 'liquid_body', 'stream')
CONTRACT = 'active-station-visible-v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')


def decisions():
    return {c: dict(status='unreviewed', boxes=[], notes='') for c in CLASSES}


def blank(manifest, manifest_hash):
    return dict(schema_version=1, annotation_contract=CONTRACT,
        pack_id=manifest['pack_id'], manifest_sha256=manifest_hash,
        training_eligible=False, reviewer='', saved_at=None,
        records=[dict(sample_id=s['sample_id'], image_sha256=s['image_sha256'],
            source_run=s['source_run'], split_role=s['split_role'],
            review_status='unreviewed', reviewed_by=None, reviewed_at=None,
            decisions=decisions(), notes='') for s in manifest['samples']])


def validate(document, manifest, manifest_hash):
    """Fail closed on identity/geometry; submission does not approve training."""
    def require(condition, message):
        if not condition:
            raise ValueError(message)
    require(document['schema_version'] == 1 and document['annotation_contract'] == CONTRACT, 'Unknown contract')
    require(document['pack_id'] == manifest['pack_id'] and document['manifest_sha256'] == manifest_hash, 'Different pack/manifest')
    require(document['training_eligible'] is False, 'Review cannot approve training')
    expected = {s['sample_id']: s for s in manifest['samples']}
    records = document['records']
    require(len(records) == len(expected) and len({r['sample_id'] for r in records}) == len(expected), 'Missing/duplicate records')
    for r in records:
        require(r['sample_id'] in expected, 'Unknown sample')
        s = expected[r['sample_id']]
        require(all(r[k] == s[k] for k in ('image_sha256', 'source_run', 'split_role')), 'Changed source identity/split')
        require(r['review_status'] in ('unreviewed', 'draft', 'submitted_for_adjudication'), 'Unknown review status')
        submitted = r['review_status'] == 'submitted_for_adjudication'
        if submitted:
            require(bool(str(r.get('reviewed_by') or '').strip()) and bool(r.get('reviewed_at')), 'Submission needs reviewer/time')
            datetime.fromisoformat(r['reviewed_at'].replace('Z', '+00:00'))
        require(set(r['decisions']) == set(CLASSES), 'Missing/unknown class')
        for name, d in r['decisions'].items():
            require(d['status'] in ('unreviewed', 'present', 'absent', 'uncertain'), 'Unknown decision')
            if submitted:
                require(d['status'] != 'unreviewed', 'Incomplete submission')
                require(d['status'] != 'uncertain' or bool(d['notes'].strip()), 'Uncertainty needs explanation')
            if d['status'] == 'present':
                require(len(d['boxes']) >= 1, 'Present needs visible box')
            else:
                require(not d['boxes'], 'Only present decisions can have boxes')
            require(name == 'stream' or len(d['boxes']) <= 1, 'Only one active vial/body')
            for box in d['boxes']:
                b = box['xyxy']
                require(len(b) == 4 and all(type(x) in (int, float) and math.isfinite(x) for x in b), 'Nonfinite box')
                require(0 <= b[0] < b[2] <= s['width'] and 0 <= b[1] < b[3] <= s['height'], 'Invalid/outside box')
                require(set(box['flags']) == {'occluded', 'truncated', 'low_contrast'} and
                        all(type(v) is bool for v in box['flags'].values()), 'Invalid flags')
    return Counter(r['review_status'] for r in records)


def proposals(path):
    out = []
    for line in path.read_text(encoding='utf-8').splitlines():
        c, x, y, w, h = map(float, line.split())
        if c not in (0, 1):
            raise ValueError('Unexpected legacy class')
        out.append(dict(legacy_class_id=int(c), legacy_class_name='vial' if c == 0 else 'colored_liquid_scope_unreviewed',
                        xyxy=[(x-w/2)*640, (y-h/2)*480, (x+w/2)*640, (y+h/2)*480]))
    return out


def source_rows(root):
    inventory = read(root/'provenance.json')
    rows = []
    for s in inventory['samples']:
        if s.get('parent_id'):
            continue
        if s['source_run'] not in (1, 2, 3, 4):
            raise ValueError('Only original Runs 1-4 allowed')
        if s['split'] != ('val' if s['source_run'] == 4 else 'train'):
            raise ValueError('Changed run split')
        image, label = root/s['image'], root/s['labels']
        if sha(image) != s['image_sha256'] or sha(label) != s['actual_label_sha256']:
            raise ValueError('Frozen source changed')
        p = proposals(label)
        with Image.open(image) as im:
            if im.size != (640, 480):
                raise ValueError('Unexpected image dimensions')
            # Sampling cue only. Fixed lower station ROI is never a target/mask.
            rgb = np.asarray(im.convert('RGB'), dtype=float)[330:410, 285:335]/255
            brightness = float(np.mean(rgb))
            chroma = float(np.mean(rgb.max(axis=2)-rgb.min(axis=2)))
        body = next((o['xyxy'] for o in p if o['legacy_class_id'] == 1), None)
        rows.append(dict(s, proposals=p, brightness=brightness, chroma=chroma,
                         scope_flag=bool(body and body[1] < 200),
                         body_height=body[3]-body[1] if body else None))
    return rows


def select(rows):
    """Deterministic cue-enriched sample, never a semantic label decision."""
    selected, groups = {}, []
    known = {1: ['vial_fill_20260922_170544_697177'],
             2: ['vial_fill_20260922_192633_265636'],
             3: ['vial_fill_20260923_184720_822117'], 4: []}
    for run, count in ((1, 60), (2, 60), (3, 60), (4, 30)):
        candidates = sorted([s for s in rows if s['source_run'] == run], key=lambda s: s['source_id'])
        chosen = {}
        def add(s, reason, group=None):
            identity = s['source_id']
            if identity not in chosen:
                chosen[identity] = dict(s, selection_cues=[], sequence_groups=[])
            if reason not in chosen[identity]['selection_cues']:
                chosen[identity]['selection_cues'].append(reason)
            if group:
                chosen[identity]['sequence_groups'].append(group)
        # Include known pale reference neighborhoods, then ranked provisional
        # absent->present / apparent level-reset transitions from saved proposals.
        anchors = [i for i, s in enumerate(candidates) if s['source_id'] in known[run]]
        ranks = []
        for i in range(2, len(candidates)-2):
            current, prev = candidates[i], candidates[i-1]
            score = (100 if current['body_height'] is not None and prev['body_height'] is None else 0)
            if current['body_height'] and prev['body_height']:
                score += max(0, prev['body_height']-current['body_height'])
            ranks.append((-score, i))
        ranks.sort()
        for _, i in ranks:
            if len(anchors) >= (3 if run == 4 else 6):
                break
            if all(abs(i-j) >= 5 for j in anchors):
                anchors.append(i)
        for n, i in enumerate(anchors):
            neighborhood = candidates[max(0, i-2):i+3]
            group = f'run{run}-sequence{n+1:02d}'
            groups.append(dict(group_id=group, source_run=run,
                anchor_sample_id=candidates[i]['source_id'],
                reason='Known pale reference neighborhood' if candidates[i]['source_id'] in known[run]
                       else 'Provisional teacher presence/height transition; NOT verified onset',
                sample_ids=[s['source_id'] for s in neighborhood]))
            for s in neighborhood:
                add(s, 'early_or_transition_context_candidate', group)
        # All existing body/stream-scope flags when space permits; these are not
        # automatically declared errors. Then bright/low-chroma lower-station
        # candidates, short-pool proposals, no-liquid proposals and broad coverage.
        buckets = [
            ('body_stream_scope_review', [s for s in candidates if s['scope_flag']], 100),
            ('pale_low_contrast_candidate', sorted([s for s in candidates if s['body_height'] is not None], key=lambda s: -(s['brightness']-s['chroma'])), count//5),
            ('small_pool_candidate', sorted([s for s in candidates if s['body_height']], key=lambda s: s['body_height']), count//10),
            ('no_liquid_proposal_review_not_verified_negative', [s for s in candidates if s['body_height'] is None], count//10)]
        for reason, bucket, limit in buckets:
            added = 0
            for s in bucket:
                if len(chosen) >= count or added >= limit:
                    break
                if s['source_id'] not in chosen:
                    add(s, reason); added += 1
        # Evenly distributed across the remaining recording, no random leakage.
        remaining = [s for s in candidates if s['source_id'] not in chosen]
        need = count-len(chosen)
        for i in np.linspace(0, len(remaining)-1, need, dtype=int) if need else []:
            add(remaining[int(i)], 'recording_coverage')
        if len(chosen) != count:
            raise ValueError('Cannot satisfy per-run selection')
        selected.update(chosen)
    return sorted(selected.values(), key=lambda s: (s['source_run'], s['source_id'])), groups


def font(size):
    try:
        return ImageFont.truetype('C:/Windows/Fonts/arial.ttf', size)
    except OSError:
        return ImageFont.load_default()


def sheet(samples, out, title, prefix, cols=4):
    w, h = 400, 350
    canvas = Image.new('RGB', (cols*w, 70+math.ceil(len(samples)/cols)*h), '#19212a')
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 14), title, fill='white', font=font(24))
    for n, s in enumerate(samples):
        x, y = (n % cols)*w, 70+(n//cols)*h
        with Image.open(out/s['image']) as im:
            canvas.paste(im.resize((384, 288)), (x+8, y))
        draw.text((x+8, y+290), f"{s['queue_index']:03d} Run {s['source_run']} / {s['assigned_reviewer']}", fill='#7be3f0', font=font(17))
        draw.text((x+8, y+312), s['sample_id'].replace('vial_fill_', ''), fill='white', font=font(14))
    path = f'{prefix}.jpg'
    canvas.save(out/path, quality=88)
    return path


def build(args):
    root, out = args.source.resolve(), args.out.resolve()
    if out.exists():
        raise ValueError('Use a fresh output folder')
    rows = source_rows(root)
    selected, groups = select(rows)
    if len({s['image_sha256'] for s in selected}) != 210:
        raise ValueError('Duplicate source pixels')
    out.mkdir(parents=True)
    for d in ('images', 'proposals', 'reviews', 'contact-sheets', 'sequences'):
        (out/d).mkdir()
    samples = []
    for n, s in enumerate(selected):
        image = f"images/{s['source_id']}.jpg"
        label = f"proposals/{s['source_id']}.txt"
        shutil.copyfile(root/s['image'], out/image)
        shutil.copyfile(root/s['labels'], out/label)
        stamp = datetime.strptime(s['source_id'].removeprefix('vial_fill_'), '%Y%m%d_%H%M%S_%f').isoformat()
        samples.append(dict(sample_id=s['source_id'], queue_index=n+1,
            source_run=s['source_run'], split_role='development_only' if s['source_run'] == 4 else 'training_candidate',
            image=image, image_sha256=sha(out/image), width=640, height=480,
            source_image=str(root/s['image']), source_label=str(root/s['labels']),
            legacy_label_file=label, legacy_label_sha256=sha(out/label),
            proposal_status='unreviewed_unchanged_legacy_teacher_not_body_ground_truth',
            proposals=s['proposals'], selection_cues=s['selection_cues'], sequence_groups=s['sequence_groups'],
            selection_metrics=dict(lower_station_brightness=s['brightness'], lower_station_chroma=s['chroma'],
                                   legacy_body_height=s['body_height'], legacy_top_above_200=s['scope_flag']),
            supplied_image_timestamp=stamp, timestamp_status='filename_wall_time_unverified_not_mp4_time',
            mp4_frame_index=None, mp4_timestamp_s=None,
            assigned_reviewer=('Aleyna', 'Zoe', 'Reagan')[n % 3], calibration=False))
    # Three per run, chosen across scope/pale/transition queues for independent
    # calibration. They remain part of the same 210 unique images.
    for run in (1, 2, 3, 4):
        rs = [s for s in samples if s['source_run'] == run]
        known = {1: 'vial_fill_20260922_170544_697177',
                 2: 'vial_fill_20260922_192633_265636',
                 3: 'vial_fill_20260923_184720_822117'}
        picks = [s for s in rs if s['sample_id'] == known.get(run)]
        for cue in ('body_stream_scope_review', 'no_liquid_proposal_review_not_verified_negative', 'small_pool_candidate'):
            if len(picks) == 3:
                break
            picks.append(next((s for s in rs if cue in s['selection_cues'] and s not in picks),
                              next(s for s in rs if s not in picks)))
        for s in picks:
            s['calibration'] = True
    manifest = dict(schema_version=1, annotation_contract=CONTRACT,
        pack_id='aleyna-visible-boxes-v1-'+hashlib.sha256(
            ''.join(s['image_sha256'] for s in samples).encode()).hexdigest()[:12],
        source_inventory=str(root/'provenance.json'), source_inventory_sha256=sha(root/'provenance.json'),
        training_eligible=False, automatic_label_changes=False, run5_included=False,
        selection_policy='Deterministic cue-enriched review; cues and legacy proposals are not truth',
        source_still_cadence='Approximately 2 seconds; no sub-second onset claims or video clock mapping',
        classes=[dict(class_id=i, name=c) for i, c in enumerate(CLASSES)], samples=samples, sequence_groups=groups)
    write(out/'manifest.json', manifest)
    mh = sha(out/'manifest.json')
    template = blank(manifest, mh)
    validate(template, manifest, mh)
    write(out/'reviews/blank-review.json', template)
    shutil.copyfile(HERE/'LABEL-CONTRACT.md', out/'LABEL-CONTRACT.md')
    shutil.copyfile(HERE/'review-schema.json', out/'review-schema.json')
    payload = json.dumps(dict(manifest=manifest, review=template), ensure_ascii=True).replace('</', '<\\/')
    page = (HERE/'review_editor.html').read_text(encoding='utf-8').replace('/*PACK_DATA*/null', payload)
    (out/'review.html').write_text(page, encoding='utf-8')
    sheets = []
    for start in range(0, len(samples), 20):
        sheets.append(sheet(samples[start:start+20], out, 'Raw RGB review queue - proposals hidden',
                            f'contact-sheets/queue-{start+1:03d}-{min(start+20,210):03d}'))
    sheets.append(sheet([s for s in samples if s['calibration']], out,
                        '12 shared calibration frames - independently review before discussing', 'calibration'))
    by_id = {s['sample_id']: s for s in samples}
    for group in groups:
        group_samples = [by_id[i] for i in group['sample_ids']]
        sheet(group_samples, out, group['group_id']+' - chronological stills; NOT physical onset labels',
              'sequences/'+group['group_id'], cols=5)
    for name in ('Aleyna', 'Zoe', 'Reagan'):
        write(out/'reviews'/f'{name.lower()}-queue.json', dict(pack_id=manifest['pack_id'],
            assigned_sample_ids=[s['sample_id'] for s in samples if s['assigned_reviewer'] == name],
            independent_calibration_sample_ids=[s['sample_id'] for s in samples if s['calibration']]))
    verification = dict(pack_id=manifest['pack_id'], manifest_sha256=mh, frames=len(samples),
        per_run=dict(Counter(s['source_run'] for s in samples)),
        per_reviewer=dict(Counter(s['assigned_reviewer'] for s in samples)),
        cues=dict(Counter(c for s in samples for c in s['selection_cues'])),
        sequence_groups=len(groups), calibration_frames=sum(s['calibration'] for s in samples),
        original_sources_unchanged=True, all_review_decisions_unreviewed=True,
        proposal_boxes_copied_into_review=0, run5_frames=0, training_export_created=False)
    for s in samples:
        if sha(s['source_image']) != s['image_sha256'] or sha(s['source_label']) != s['legacy_label_sha256']:
            raise ValueError('Source mutated')
        if sha(out/s['image']) != s['image_sha256'] or sha(out/s['legacy_label_file']) != s['legacy_label_sha256']:
            raise ValueError('Copy mismatch')
    write(out/'verification.json', verification)
    (out/'START-HERE.txt').write_text(
        'EXTRACT THE WHOLE ZIP FIRST. Then double-click review.html. No server/install required.\n'
        'Read LABEL-CONTRACT.md. Review shared calibration first, then your assigned 70-frame queue.\n'
        'Raw RGB first; legacy proposals optional and hidden initially. Draw new boxes yourself.\n'
        'Mark all three classes present/absent/uncertain, then Submit for adjudication.\n'
        'Save named JSON progress frequently; give separate calibration and queue files to Aleyna.\n'
        'No source labels changed. No reviewed labels or training data have been created automatically.\n'
        '210 unique frames: Runs 1/2/3 each 60 training candidates; Run 4 30 development only.\n'
        '12 shared calibration frames are included in those 210. Assignment filters do not delete other records.\n'
        'Image filename wall times are not verified video positions; still cadence is about 2 seconds.\n'
        'Contact sheets and temporal context are in contact-sheets/ and sequences/.\n'
        'Editor downloads are JSON for adjudication, not YOLO label files.\n', encoding='utf-8')
    archive = out.with_suffix('.zip')
    if archive.exists():
        raise ValueError('Archive already exists')
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(out.rglob('*')):
            if path.is_file():
                z.write(path, Path(out.name)/path.relative_to(out))
    print(json.dumps(dict(verification, archive=str(archive), archive_sha256=sha(archive)), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('build'); p.add_argument('--source', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p = sub.add_parser('validate'); p.add_argument('--pack', type=Path, required=True); p.add_argument('--review', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'build':
        build(args)
    else:
        print(dict(validate(read(args.review), read(args.pack/'manifest.json'), sha(args.pack/'manifest.json'))))
