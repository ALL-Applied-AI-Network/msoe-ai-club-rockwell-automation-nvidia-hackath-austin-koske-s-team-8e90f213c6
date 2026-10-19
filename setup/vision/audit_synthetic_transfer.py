"""Read-only, local RGB/target audit of the frozen 2026-10-19 transfer pilot.

No fitting, label edits, QA changes, or renderer calibration. Run 5 is diagnostic.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont


COLORS = {0: '#54dcff', 1: '#ffc34a', 2: '#f18cff'}
NAMES = {0: 'vial', 1: 'body', 2: 'stream'}
BG = '#191d24'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def labels(path):
    result = []
    for line in path.read_text().splitlines():
        c, x, y, w, h = map(float, line.split())
        result.append(dict(class_id=int(c), box=[(x-w/2)*640, (y-h/2)*480,
                                                (x+w/2)*640, (y+h/2)*480]))
    return result


def font(size=20):
    return ImageFont.truetype('C:/Windows/Fonts/arial.ttf', size)


def text(im, xy, value, size=20, color='white'):
    ImageDraw.Draw(im).text(xy, value, font=font(size), fill=color)


def marked(im, boxes):
    im = im.copy()
    draw = ImageDraw.Draw(im)
    for obj in boxes:
        c, b = obj['class_id'], obj['box']
        draw.rectangle(b, outline=COLORS[c], width=2)
    return im


def crop(im):
    # Display-only enlargement; inference and targets remain full-frame.
    return im.crop((250, 40, 370, 435)).resize((240, 790))


def panel(title, subtitle, images, footer, path):
    """images = [(caption, image), ...], all at the same display size."""
    w, h = images[0][1].size
    canvas = Image.new('RGB', (max(1100, len(images)*(w+24)+24), h+190), BG)
    text(canvas, (24, 14), title, 26)
    text(canvas, (24, 51), subtitle, 18)
    for i, (caption, im) in enumerate(images):
        x = 24+i*(w+24)
        text(canvas, (x, 88), caption, 17)
        canvas.paste(im, (x, 118))
    for i, line in enumerate(footer):
        text(canvas, (24, h+130+i*23), line, 17)
    canvas.save(path)


def quantiles(values):
    return [float(x) for x in np.percentile(values, [0, 25, 50, 75, 100])] if values else []


def geometry(rows):
    by_class = {}
    for c in (0, 1):
        boxes = [b['box'] for row in rows for b in row['targets'] if b['class_id'] == c]
        by_class[NAMES[c]] = dict(count=len(boxes),
            top_y_quantiles=quantiles([b[1] for b in boxes]),
            width_quantiles=quantiles([b[2]-b[0] for b in boxes]),
            height_quantiles=quantiles([b[3]-b[1] for b in boxes]),
            unique_boxes=len({tuple(round(v, 3) for v in b) for b in boxes}),
            top_above_200px=sum(b[1] < 200 for b in boxes))
    return by_class


def main(args):
    root, out = args.experiment.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out/'sources').mkdir()
    (out/'cases').mkdir()
    real_root, renders = root/'data/real', root/'renders'
    real = read(real_root/'provenance.json')
    originals = [dict(s, targets=labels(real_root/s['labels']))
                 for s in real['samples'] if not s.get('parent_id')]
    for s in originals:
        if sha(real_root/s['image'])!=s['image_sha256'] or sha(real_root/s['labels'])!=s['actual_label_sha256']:
            raise ValueError(f'Frozen real source changed: {s["source_id"]}')
    by_id = {s['source_id']: s for s in originals}
    synthetic = [json.loads(line) for line in (renders/'manifest.jsonl').read_text().splitlines()]
    synth_by_id = {s['sample_id']: s for s in synthetic}
    synthetic_rows = [dict(source_id=s['sample_id'], targets=[dict(class_id=0 if o['instance_id']==101 else 1,
        box=o['bbox_xyxy']) for o in s['annotations'] if o['instance_id'] in (101, 201) and o['present']])
        for s in synthetic]
    stats = dict(real={f'run{run}': geometry([s for s in originals if s['source_run']==run])
                      for run in (1, 2, 3, 4)}, synthetic=geometry(synthetic_rows),
        synthetic_palette=dict(Counter(s['parameters'].get('palette', 'original-fixture') for s in synthetic)),
        synthetic_positive_level_quantiles=quantiles([s['parameters']['level'] for s in synthetic if s['parameters']['level']>0]),
        stream_present_body_absent=sum(s['parameters']['level']==0 and s['parameters']['stream']!='off' for s in synthetic),
        synthetic_training_scope='Only active vial 101 and accumulated body 201; stream 301 excluded',
        quantile_order=['min', 'q25', 'median', 'q75', 'max'],
        top_above_200px_policy='Geometric review flag only; not an adjudicated error or filter')
    cases = []

    def add_real(identity):
        s = by_id[identity]
        image_path, label_path = real_root/s['image'], real_root/s['labels']
        assert sha(image_path)==s['image_sha256'] and sha(label_path)==s['actual_label_sha256']
        case = dict(id=identity, origin='real', source_run=s['source_run'],
            source_image=str(image_path), source_sha256=sha(image_path),
            source_label=str(label_path), label_sha256=sha(label_path), targets=s['targets'],
            target_status='Unreviewed teacher proposals; unchanged', training_used=s['split']=='train')
        cases.append(case)
        return Image.open(image_path).convert('RGB')

    def add_synthetic(identity):
        s = synth_by_id[identity]
        path = renders/s['image_file']
        assert sha(path)==s['image_sha256']
        targets = [dict(class_id={101:0, 201:1, 301:2}[o['instance_id']], box=o['bbox_xyxy'])
                   for o in s['annotations'] if o['instance_id'] in (101, 201, 301) and o['present']]
        for o in s['annotations']:
            assert sha(renders/o['mask_file'])==o['mask_sha256']
        cases.append(dict(id=identity, origin='synthetic', source_image=str(path), source_sha256=sha(path),
            parameters=s['parameters'], targets=targets, native_annotations=s['annotations'],
            masks_root=str(renders), training_used=True,
            target_status='Geometric proxy; body/vial training targets; stream review only',
            physical_optical_calibration=False))
        return Image.open(path).convert('RGB')

    real_ids = ['vial_fill_20260922_170544_697177', 'vial_fill_20260922_170546_705450',
                'vial_fill_20260922_170548_711253', 'vial_fill_20260923_184720_822117',
                'vial_fill_20260923_184726_838430', 'vial_fill_20260923_184809_074191',
                'vial_fill_20260922_170600_743982']
    ims = {i: add_real(i) for i in real_ids}
    sim_ids = [s['sample_id'] for s in synthetic[:12]]
    ims.update({i: add_synthetic(i) for i in sim_ids})
    index = {c['id']: c for c in cases}

    panel('01  Same station, different image formation',
          'Original RGB, identical display scale. Illustrative stages/colors, not matched volumes or recipes.',
          [('Real Run 1 / blue body', ims[real_ids[-1]]), ('GB10 / mid blue proxy', ims['07-mid-blue'])],
          ['Real: textured/soft image, variable labels, curved surface and bubbles. Render: cleaner glass, flat proxy surface.',
           'The camera is approximately aligned; this is not a fresh physical camera/optical calibration.'],
          out/'01-scene-gap.png')

    show = real_ids[:3]+['03-tiny-pale', '01-first-stream']
    panel('02  One class currently has inconsistent target scope',
          'Saved targets unchanged: cyan=vial, yellow=liquid/body, pink=stream (raw render only; NOT a training class).',
          [(cap, crop(marked(ims[i], index[i]['targets']))) for cap, i in zip(
              ['Real 17:05:44.697', 'Real +2.008 seconds', 'Real +4.014 seconds', 'Render 2% body', 'Render stream-only'], show)],
          ['Real teacher liquid boxes can span stream + body, then switch to the body. Flag for review; do not copy this into new truth.',
           'All 96 stream-only render scenes have NO liquid training target. Earliest stream detection was never taught as a class.'],
          out/'02-target-contract.png')

    show = [real_ids[3], real_ids[4], '03-tiny-pale', '04-tiny-clear-proxy', '10-high-pale-stopped', '11-high-clear-proxy-stopped']
    panel('03  Pale liquid: inspect RGB separately from geometry',
          'Unmarked source crops. See sheet 04 for opaque geometry masks; these do not verify RGB liquid appearance.',
          [(cap, crop(ims[i])) for cap, i in zip(['Real pale / low', 'Real pale / higher', 'Render pale / 2%',
             'Render clear / 2%', 'Render pale / 90%', 'Render clear / 90%'], show)],
          ['Real early fills show bubbles, wall wetting and a curved moving surface. Proxies use uniform cylinders and one surface material.',
           'Pale/clear RGB can barely change while its geometric target stays equally positive. Review observability before admission.'],
          out/'03-pale-rgb.png')
    mask_images = []
    for i in show[2:]:
        s = synth_by_id[i]
        body = next(o for o in s['annotations'] if o['instance_id']==201)
        mask = Image.open(renders/body['mask_file']).convert('RGB')
        mask_images.append((i, crop(mask)))
    panel('04  These masks certify geometry, not visible RGB liquid',
          'Native body masks at the same crop/scale as sheet 03. Top/bottom bounds include the projected surface ellipse.',
          mask_images,
          ['Mask pass hides vial shells and transparent guard, makes fluid opaque, and retains opaque holder/label/nozzle.',
           'A correct native ID and mask-derived box can still be a poor target for transparent, refracted RGB.'],
          out/'04-native-masks.png')

    # Extract original Run 5 frames by sequential decode; no supplied event annotations.
    requested = {2730, 2760, 2796, 2820, 2865, 1410, 4188}
    video = args.data/'vial_fill/videos/vial_fill_video_5.mp4'
    video_hash = sha(video)
    frozen = {row['frame_index']: row for row in map(json.loads,
        (root/'evaluation/sim-only/video-predictions.jsonl').read_text().splitlines())}
    rgbs = {}
    with av.open(str(video)) as container:
        start = float((container.streams.video[0].start_time or 0)*container.streams.video[0].time_base)
        for n, frame in enumerate(container.decode(video=0)):
            if n > max(requested): break
            if n not in requested: continue
            assert abs(float(frame.pts*frame.time_base)-start-n/30)<1/30
            rgb = frame.to_ndarray(format='rgb24')
            digest = hashlib.sha256(rgb.tobytes()).hexdigest()
            if n in frozen: assert digest==frozen[n]['source_rgb_sha256']
            im = Image.fromarray(rgb); rgbs[n] = im
            filename = out/'sources'/f'run5-frame{n}.png'; im.save(filename)
            case = dict(id=f'run5-frame{n}', origin='real_video', source_run=5,
                source_video=str(video.resolve()), source_video_sha256=video_hash,
                frame_index=n, mp4_s=n/30, source_rgb_sha256=digest, exported_image=str(filename),
                training_used=False, targets=None,
                predictions={key:frozen[n][key] for key in ('baseline','candidate')} if n in frozen else None,
                target_status='No new ground truth; image-estimated onset 2760 only for display offsets')
            cases.append(case)
    strip_frames = [2730, 2760, 2796, 2820, 2865]
    panel('05  Real pale early pour: a sequence, not a static fill level',
          'Run 5, original pixels. Offsets use image-estimated start frame 2760; physical onset has NOT been adjudicated.',
          [(f'f{n} / {n/30:.2f}s ({(n-2760)/30:+.1f}s)', crop(rgbs[n])) for n in strip_frames],
          ['Notice the thin stream, turbulent bottom pool and changing surface; a full geometric cylinder is not the earliest signal.',
           'Synthetic positive fill starts at 2%; no wetting-film/droplet/turbulence sequence is simulated. Run 5 remains diagnostic only.'],
          out/'05-real-early-pale.png')
    panels = []
    for n in (1410, 2796, 4188):
        for key in ('baseline', 'candidate'):
            panels.append((f'f{n} {"current" if key=="baseline" else "sim-only"}',
                           crop(marked(rgbs[n], frozen[n][key]))))
    panel('06  Frozen model failures on original Run 5 frames',
          'Same original pixels, full-frame preprocessing, confidence 0.35 and NMS 0.45. No crop/contrast inference here.',
          panels,
          ['Current / sim-only alternate. Sim-only adds oversized or duplicate liquid boxes; both miss the pale low pool at f2796.',
           'These are unchanged saved predictions; boxes describe model outputs, not measurement masks or reviewed truth.'],
          out/'06-transfer-errors.png')

    # Provide every flagged ORIGINAL real target for Aleyna, without editing or excluding it.
    review = [dict(source_id=s['source_id'], source_run=s['source_run'], source_image=str(real_root/s['image']),
        source_label=str(real_root/s['labels']), image_sha256=s['image_sha256'],
        label_sha256=s['actual_label_sha256'], targets=s['targets'], reason='Liquid target top above y=200; review body/stream scope',
        status='needs_human_review_not_adjudicated') for s in originals
        if any(b['class_id']==1 and b['box'][1]<200 for b in s['targets'])]
    (out/'real-target-review.json').write_text(json.dumps(review, indent=2)+'\n')
    scored = read(root/'evaluation/sim-only/scoring-predictions.json')
    score_sources = {s['id']: s for s in read(root/'evaluation/sim-only/teacher-agreement.json')['sources']}
    eval_review = []
    for s in scored:
        if not any(b['class_id']==1 and b['box'][1]<200 for b in s['targets']): continue
        image_path = args.teacher/'review-pack/vial_fill/images'/f'{s["source_id"]}.jpg'
        label_path = args.teacher/'review-pack/vial_fill/labels'/f'{s["source_id"]}.txt'
        identity = score_sources[s['source_id']]
        assert sha(image_path)==identity['image_sha256'] and sha(label_path)==identity['label_sha256']
        eval_review.append(dict(source_id=s['source_id'], source_run=5, source_image=str(image_path.resolve()),
            source_label=str(label_path.resolve()), image_sha256=identity['image_sha256'],
            label_sha256=identity['label_sha256'], targets=s['targets'],
            reason='Liquid target top above y=200; review body/stream scope',
            status='needs_human_review_not_adjudicated', training_used=False))
    (out/'run5-target-review.json').write_text(json.dumps(eval_review, indent=2)+'\n')
    example = next(s for s in scored if s['source_id']=='vial_fill_20260923_194709_622176')
    source = next(s for s in eval_review if s['source_id']==example['source_id'])
    im = Image.open(source['source_image']).convert('RGB')
    panel('07  Evaluation targets also need review',
          'Run 5 still 2026-09-23 19:47:09.622176. All targets/predictions unchanged; current display confidence 0.35.',
          [('Original RGB', crop(im)), ('Teacher targets', crop(marked(im, example['targets']))),
           ('Current detector', crop(marked(im, [p for p in example['baseline'] if p['confidence']>=.35]))),
           ('Sim-only detector', crop(marked(im, [p for p in example['candidate'] if p['confidence']>=.35])))],
          ['The teacher spans most of the vial; both models localize a small bottom pool. Teacher agreement can penalize a plausible body box.',
           'This does not excuse the oversized predictions in sheet 06. Review labels before calling agreement scores physical accuracy.'],
          out/'07-evaluation-label-issue.png')
    cases.append(dict(source, id=source['source_id'], origin='real_evaluation_still',
        exported_image=source['source_image'], target_status='Unreviewed evaluation teacher proposal; unchanged',
        predictions={key:[p for p in example[key] if p['confidence']>=.35] for key in ('baseline','candidate')}))
    stats['run5_teacher_geometry'] = geometry([dict(targets=s['targets']) for s in scored])
    stats['run5_scope_review_records'] = len(eval_review)
    stats['real_scope_review_records'] = len(review)
    for case in cases:
        if case['id'] in ims:
            im = ims[case['id']]
            raw, overlay = out/'cases'/f'{case["id"]}-rgb.png', out/'cases'/f'{case["id"]}-targets.png'
            im.save(raw); marked(im, case['targets']).save(overlay)
            case['exported_image'], case['target_overlay'] = str(raw), str(overlay)
    result = dict(schema_version=1, kind='frozen_synthetic_transfer_diagnostic',
        no_training=True, no_label_edits=True, no_qa_changes=True,
        experiment=str(root), source_inventory_sha256=sha(real_root/'provenance.json'),
        render_manifest_sha256=sha(renders/'manifest.jsonl'), video_sha256=video_hash,
        statistics=stats, cases=cases,
        limitations=['Real targets are unreviewed teacher proposals, including evaluation targets.',
            'Examples are purposefully selected, not a random accuracy sample.',
            'Pale is a visual description, not an inferred ingredient or recipe.',
            'Mask pass is geometric visibility through removed transparent shells, not RGB semantic truth.',
            'Run 5 excluded from fitting/selection but repeatedly inspected; no blind-test claim.',
            'Causes are hypotheses unless directly established by target/source inspection; no causal ablation performed.'])
    (out/'audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(output=str(out), unique_real_originals=len(originals),
        native_renders=len(synthetic), real_scope_review=len(review), run5_scope_review=len(eval_review),
        sheets=7, cases=len(cases), no_training=True)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--teacher', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    main(parser.parse_args())
