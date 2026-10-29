"""Review-first Jev prelabels for the tested 640x480 vial camera.

Crop + compact prompts, one conditional full-frame fallback, YOLO candidates and
a review queue. No labels are auto-approved. Source images are never modified.
"""
import argparse
import csv
import getpass
import json
import os
from pathlib import Path
import shutil

from PIL import Image

from tune import ROI, crop_cut, definition, digest, mapped, needs_fallback, request
from run import to_yolo


def combine(row, primary, fallback=None):
    answers = primary['response']['answers']
    clipped = crop_cut(answers)
    records = []
    for class_id, target in enumerate(('vial', 'liquid')):
        box = mapped(answers[target + '_box']['box'], row)
        presence = answers[target + '_present']['noul']
        confidence = answers[target + '_box']['confidence']
        source = 'compact_crop'
        if target == 'liquid' and not (presence >= .8 and confidence >= .5) and fallback and fallback.get('ok'):
            other = fallback['response']['answers']
            p2, c2 = other['liquid_present']['noul'], other['liquid_box']['confidence']
            if p2 >= .8 and c2 >= .5:
                box, presence, confidence, source = other['liquid_box']['box'], p2, c2, 'compact_full_fallback'
        record = dict(sample_id=row['id'], class_id=class_id, target=target, box=box,
                      presence=presence, confidence=confidence, source=source,
                      strict=presence >= .8 and confidence >= .5,
                      draft=presence >= .5, manual_only=clipped,
                      review_priority='high' if clipped or presence < .8 or confidence < .5 else 'normal',
                      annotation_status='unreviewed_model_candidate')
        records.append(record)
    return records


def export_pack(out, rows, records):
    pack = out / 'review-pack'
    for directory in ('images', 'labels', 'manual_review_images'):
        (pack / directory).mkdir(parents=True, exist_ok=True)
    queue = []
    for row in rows:
        group = [r for r in records if r['sample_id'] == row['id']]
        manual = any(r['manual_only'] for r in group)
        target_dir = 'manual_review_images' if manual else 'images'
        dest = pack / target_dir / (row['id'] + '.jpg')
        if Path(row['file']).suffix.lower() in ('.jpg', '.jpeg'):
            shutil.copyfile(row['file'], dest)
        else:
            Image.open(row['file']).convert('RGB').save(dest, quality=95)
        lines = []
        for record in group:
            if record['draft'] and not manual:
                lines.append(str(record['class_id']) + ' ' + ' '.join(f'{v:.8f}' for v in to_yolo(record['box'])))
        if not manual:
            (pack / 'labels' / (row['id'] + '.txt')).write_text('\n'.join(lines) + ('\n' if lines else ''))
        queue.append(dict(sample_id=row['id'], status='manual_full_image' if manual else 'review_candidates',
                          candidate_boxes=len(lines), high_priority=manual or any(r['draft'] and r['review_priority']=='high' for r in group),
                          note='All labels require review. No model confidence constitutes approval.'))
    (pack / 'classes.txt').write_text('fill_position_vial\ncolored_liquid\n')
    (pack / 'annotations.json').write_text(json.dumps(records, indent=2) + '\n')
    with (pack / 'review-queue.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(queue[0]))
        writer.writeheader(); writer.writerows(queue)
    (pack / 'README.txt').write_text(
        'UNREVIEWED PRELABELS, NOT TRAINING GROUND TRUTH\n'
        'YOLO class 0=fill_position_vial; class 1=colored_liquid.\n'
        'Coordinates refer to the ORIGINAL full image, not an enlarged crop.\n'
        'Import images/ and labels/ as YOLO candidates into your annotation tool.\n'
        'Correct/reject boxes and check for missing objects on EVERY image.\n'
        'manual_review_images/ contains crop-boundary failures that must be drawn on full frames.\n'
        'These are partial labels for two task-specific targets; nozzles and neighboring vials are not labeled.\n'
        'Do not use the same recording in both training and evaluation without an explicit split plan.\n')
    return queue


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=50)
    args = parser.parse_args()
    if not 1 <= args.limit <= 200:
        parser.error('Choose a bounded batch of 1-200 images')
    if args.out.exists() and any(args.out.iterdir()):
        parser.error('Use a fresh output directory to preserve prior reviews')
    paths = sorted(p for p in args.images.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))[:args.limit]
    if not paths:
        parser.error('No PNG/JPEG images found')
    rows = []
    for i, path in enumerate(paths):
        with Image.open(path) as image:
            if image.size != (640, 480):
                parser.error(f'{path.name}: this tested camera profile requires 640x480; recalibrate its crop for other views')
        rows.append(dict(id=f'{i:04d}_{path.stem}', file=str(path.resolve()), split='new_batch', source_size=[640, 480]))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'manifest.json').write_text(json.dumps(rows, indent=2) + '\n')
    (args.out / 'prompt-definitions.json').write_text(json.dumps({p:definition(p) for p in ('compact_crop','compact_full')},indent=2)+'\n')
    for profile in ('compact_crop', 'compact_full'):
        for name in ('images', 'responses'):
            (args.out / profile / name).mkdir(parents=True, exist_ok=True)
    key = os.environ.get('TYPESAFE_API_KEY') or getpass.getpass('TypeSafe API key (hidden, memory only): ')
    records = []
    for row in rows:
        image = Image.open(row['file']).convert('RGB')
        crop_path = args.out / 'compact_crop/images' / (row['id'] + '.jpg')
        image.crop(ROI).resize((280, 800), Image.Resampling.LANCZOS).save(crop_path, quality=95)
        current = dict(row, crop_xyxy=ROI, input_file=str(crop_path), input_sha256=digest(crop_path.read_bytes()))
        primary = request(current, 'compact_crop', args.out / 'compact_crop', {}, key)
        if not primary.get('ok'):
            raise SystemExit('API request failed; inspect the saved error')
        fallback = None
        if needs_fallback(primary['response']['answers']):
            full_path = args.out / 'compact_full/images' / (row['id'] + '.jpg')
            if Path(row['file']).suffix.lower() in ('.jpg', '.jpeg'):
                shutil.copyfile(row['file'], full_path)
            else:
                image.save(full_path, quality=95)
            full = dict(row, crop_xyxy=None, input_file=str(full_path), input_sha256=digest(full_path.read_bytes()))
            fallback = request(full, 'compact_full', args.out / 'compact_full', {}, key)
        records.extend(combine(current, primary, fallback))
    del key
    queue = export_pack(args.out, rows, records)
    print(json.dumps(dict(images=len(rows), candidate_boxes=sum(r['candidate_boxes'] for r in queue),
                          manual_full_images=sum(r['status']=='manual_full_image' for r in queue),
                          output=str(args.out / 'review-pack')), indent=2))


if __name__ == '__main__':
    main()
