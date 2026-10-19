"""Prepare the small, fixed pilot without modifying raw data or labels."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--vial-colors', type=Path, help='Optional existing analysis/tables/vial_colors.csv for four settled-frame stress cases')
args = parser.parse_args()
if (args.out / 'manifest.json').exists():
    parser.error('Use a new output directory; an experiment manifest already exists')
(args.out / 'images').mkdir(parents=True, exist_ok=True)
rows = []
with (args.data / 'vial_fill/image_index.csv').open(newline='') as handle:
    index = list(csv.DictReader(handle))
for run in range(1, 6):
    group = [row for row in index if int(row['video_id']) == run]
    if not group:
        raise ValueError(f'No still images for recording {run}')
    for fraction in (.15, .65):
        source = group[int((len(group) - 1) * fraction)]
        path = args.data / 'vial_fill/images' / source['filename']
        row = dict(id=f'run{run}_{int(fraction * 100)}', source=str(path), video_id=run,
                   timestamp=source['timestamp'], kind='real', selection='15% and 65% of ordered still-image index')
        Image.open(path).convert('RGB').save(args.out / 'images' / (row['id'] + '.jpg'), quality=95)
        rows.append(row)
if args.vial_colors:
    with args.vial_colors.open(newline='') as handle:
        colors = list(csv.DictReader(handle))
    path = args.data / 'vial_fill/videos/vial_fill_video_5.mp4'
    cap = cv2.VideoCapture(str(path))
    try:
        for vial in (5, 8, 11, 23):
            source = next(row for row in colors if int(row['vial_id']) == vial and row['roi'] == 'lower')
            frame = int(source['post_first']) + 3
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, bgr = cap.read()
            if not ok:
                raise ValueError(f'Could not decode frame {frame}')
            row = dict(id=f'run5_vial{vial:02d}_settled', source=str(path), video_id=5,
                       frame_index=frame, decoder_next_frame=cap.get(cv2.CAP_PROP_POS_FRAMES),
                       kind='real', selection='purposive settled color/height coverage')
            Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).save(args.out / 'images' / (row['id'] + '.jpg'), quality=95)
            rows.append(row)
    finally:
        cap.release()
rows.append(dict(id='blank_control', kind='control', selection='synthetic blank negative control'))
Image.new('RGB', (640, 480), '#9b9b9b').save(args.out / 'images/blank_control.jpg', quality=95)
for row in rows:
    row['image'] = 'images/' + row['id'] + '.jpg'
    path = args.out / row['image']
    row['input_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    row['width'], row['height'] = Image.open(path).size
(args.out / 'manifest.json').write_text(json.dumps(rows, indent=2) + '\n', encoding='utf-8')
print(f'Prepared {len(rows)} inputs in {args.out}')
