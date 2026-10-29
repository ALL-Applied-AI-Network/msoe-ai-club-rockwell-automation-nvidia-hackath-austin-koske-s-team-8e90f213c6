"""Bounded Jev 1.14 bbox experiment. Reads images; never edits source labels.

Credentials come from TYPESAFE_API_KEY or a hidden prompt, never an argument/file.
Outputs are review candidates, including the YOLO-format text files.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import getpass
import hashlib
import html
import json
import math
import os
from pathlib import Path
import statistics
import time
import uuid

import requests
from PIL import Image, ImageDraw, ImageFont

ENDPOINT = 'https://api.typesafe.ai/v1/systemone'
MODEL = 'jev-1.14.0'
TARGETS = {
    'fill_position_vial': {
        'class_id': 0, 'color': '#20d9ff',
        'presence': 'Is the glass cylindrical vial directly beneath the central vertical metal dispensing nozzle visible in `photo`?',
        'box': 'The entire glass cylindrical vial directly beneath the central vertical metal dispensing nozzle in `photo`: tight box from the upper glass rim to the bottom of the glass base, including glass portions obscured by its barcode label or holder. Exclude the metal nozzle, black holder and metal base plate. Do not select the neighboring vial to the right.',
    },
    'colored_liquid': {
        'class_id': 1, 'color': '#ffb52e',
        'presence': 'Is visibly colored liquid (blue, green, yellow, orange or brown) inside the glass vial directly beneath the dispensing nozzle in `photo`? Do not count clear liquid, empty grey glass, reflections, barcode labels, or liquid in a neighboring vial.',
        'box': 'The visibly colored liquid body inside the glass vial directly beneath the dispensing nozzle in `photo`: tight box from its meniscus down to its bottom and across its full visible width. Exclude empty glass above the liquid, barcode labels, a falling stream above the liquid body and the neighboring vial to the right.',
    },
    'dispense_nozzle': {
        'class_id': 2, 'color': '#ff66cf',
        'presence': 'Is the vertical cylindrical metal dispensing nozzle above the primary vial visible in `photo`?',
        'box': 'The visible vertical cylindrical metal dispensing nozzle above the primary vial in `photo`. Box the metal tube from where it enters the top of the image to its lowest metal tip. Exclude glass, liquid streams, yellow supply hose, background machinery and horizontal supports.',
    },
}


def questions():
    result = {}
    for name, target in TARGETS.items():
        result[name + '_present'] = {'type': 'noul', 'instructions': target['presence']}
        result[name + '_box'] = {'type': 'box', 'instructions': target['box']}
    return result


def valid_box(box):
    try:
        left, top, right, bottom = [float(box[k]) for k in ('left', 'top', 'right', 'bottom')]
        return all(math.isfinite(v) and 0 <= v <= 1 for v in (left, top, right, bottom)) and left < right and top < bottom
    except (KeyError, TypeError, ValueError):
        return False


def to_yolo(box):
    assert valid_box(box), 'Invalid normalized box'
    left, top, right, bottom = [box[k] for k in ('left', 'top', 'right', 'bottom')]
    return ((left + right) / 2, (top + bottom) / 2, right - left, bottom - top)


def predict(row, directory, key):
    response_path = directory / 'responses' / (row['id'] + '.json')
    if response_path.exists():
        existing = json.loads(response_path.read_text())
        if existing.get('ok'):
            if existing['input_sha256'] != row['input_sha256'] or existing['questions'] != questions():
                raise ValueError('Cached request differs from current input/questions')
            return existing
    reference = str(uuid.uuid4())
    payload = {'model': MODEL, 'state': {'photo': reference, 'view': row.get('view_description', 'Full camera frame.')},
               'assets': {reference: {'type': 'image', 'part': 'photo-file'}}, 'questions': questions()}
    started = time.perf_counter()
    result = {'id': row['id'], 'input_sha256': row['input_sha256'],
              'requested_model': MODEL, 'questions': questions(),
              'images_header': True, 'utc': datetime.now(timezone.utc).isoformat()}
    for attempt in range(3):
        try:
            with open(directory / row['image'], 'rb') as handle:
                response = requests.post(ENDPOINT,
                    headers={'Authorization': 'Bearer ' + key, 'TypeSafe-Danger-Zone-Images': 'true'},
                    files=[('request', (None, json.dumps(payload), 'application/json')),
                           ('photo-file', (row['id'] + '.jpg', handle, 'image/jpeg'))],
                    timeout=(15, 90), allow_redirects=False)
            if response.status_code in (429, 529) and attempt < 2:
                try:
                    delay = min(30, max(2 ** (attempt + 1), float(response.headers.get('Retry-After', '0'))))
                except ValueError:
                    delay = 2 ** (attempt + 1)
                time.sleep(delay)
                continue
            result['http_status'] = response.status_code
            if response.status_code != 200:
                result.update(ok=False, error=response.text[:3000].replace(key, '[redacted]'))
            else:
                body = response.json()
                result.update(ok=True, response=body)
                for name in TARGETS:
                    a = body['answers']
                    assert a[name + '_present']['type'] == 'noul'
                    assert 0 <= a[name + '_present']['noul'] <= 1
                    assert a[name + '_box']['type'] == 'box'
                    assert valid_box(a[name + '_box']['box'])
                    assert 0 <= a[name + '_box']['confidence'] <= 1
        except Exception as exc:
            result.update(ok=False, error=(type(exc).__name__ + ': ' + str(exc)).replace(key, '[redacted]'))
        break
    result['seconds'] = round(time.perf_counter() - started, 3)
    response_path.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(f"{row['id']}: {'OK' if result.get('ok') else 'ERROR'} ({result['seconds']}s)", flush=True)
    return result


def render(directory, rows, results, presence_threshold=.8, confidence_threshold=.5):
    annotations, tiles, cards = [], [], []
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 14) if Path('C:/Windows/Fonts/arial.ttf').exists() else ImageFont.load_default()
    for row in rows:
        result = results.get(row['id'])
        if not result or not result.get('ok'):
            continue
        original = Image.open(directory / row['image']).convert('RGB')
        gated, raw = original.copy(), original.copy()
        display_rows, yolo = [], []
        for name, target in TARGETS.items():
            answers = result['response']['answers']
            presence = answers[name + '_present']['noul']
            answer = answers[name + '_box']
            box, confidence = answer['box'], answer['confidence']
            accepted = presence >= presence_threshold and confidence >= confidence_threshold
            record = dict(sample_id=row['id'], class_name=name, class_id=target['class_id'],
                          box=box, confidence=confidence, presence=presence, exported=accepted,
                          annotation_status='unreviewed_model_candidate', control=row['kind']=='control')
            annotations.append(record)
            xy = tuple(round(box[k] * (original.width if k in ('left', 'right') else original.height))
                       for k in ('left', 'top', 'right', 'bottom'))
            for image in ([raw, gated] if accepted else [raw]):
                draw = ImageDraw.Draw(image)
                draw.rectangle(xy, outline=target['color'], width=3)
                text = f'{name}: p={presence:.2f} c={confidence:.2f}'
                label_y = max(0, xy[1] - 18)
                rect = draw.textbbox((xy[0], label_y), text, font=font)
                label_x = min(xy[0], max(0, image.width - (rect[2] - rect[0]) - 5))
                rect = draw.textbbox((label_x, label_y), text, font=font)
                draw.rectangle(rect, fill='#101820')
                draw.text((label_x, label_y), text, fill=target['color'], font=font)
            if accepted:
                yolo.append(str(target['class_id']) + ' ' + ' '.join(f'{v:.8f}' for v in to_yolo(box)))
            display_rows.append(f'<tr><td>{name}</td><td>{presence:.3f}</td><td>{confidence:.3f}</td><td>{"candidate" if accepted else "suppressed"}</td></tr>')
        (directory / 'labels' / (row['id'] + '.txt')).write_text('\n'.join(yolo) + ('\n' if yolo else ''), encoding='utf-8')
        gated.save(directory / 'overlays' / (row['id'] + '.jpg'), quality=94)
        raw.save(directory / 'raw_overlays' / (row['id'] + '.jpg'), quality=94)
        tile = Image.new('RGB', (640, 518), '#101820')
        thumbnail = gated.copy()
        thumbnail.thumbnail((640, 480))
        tile.paste(thumbnail, ((640 - thumbnail.width) // 2, 28))
        ImageDraw.Draw(tile).text((8, 6), row['id'] + ' | ' + row['kind'] + ' | UNREVIEWED', fill='white', font=font)
        tiles.append(tile)
        cards.append(f'<article><h2>{html.escape(row["id"])}</h2><p>{html.escape(row.get("source", "synthetic blank negative control"))}</p><div class="pair"><a href="{row["image"]}"><img src="{row["image"]}" alt="Original"></a><a href="overlays/{row["id"]}.jpg"><img class="overlay" src="overlays/{row["id"]}.jpg" data-name="{row["id"]}" alt="Predicted boxes"></a></div><table><tr><th>Target</th><th>Presence</th><th>Box confidence</th><th>Export</th></tr>{"".join(display_rows)}</table></article>')
    sheet = Image.new('RGB', (1280, max(1, math.ceil(len(tiles) / 2)) * 518), '#101820')
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % 2) * 640, (i // 2) * 518))
    sheet.save(directory / 'contact-sheet.jpg', quality=94)
    (directory / 'annotations.json').write_text(json.dumps(annotations, indent=2) + '\n', encoding='utf-8')
    successful = [r for r in results.values() if r.get('ok')]
    tokens = sum(r['response'].get('usage', {}).get('input_tokens', 0) for r in successful)
    summary = dict(requests=len(results), successful=len(successful), model=sorted({r['response']['model'] for r in successful}),
                   returned_boxes=len(annotations), candidate_boxes=sum(r['exported'] for r in annotations),
                   input_tokens=tokens, output_tokens=sum(r['response'].get('usage', {}).get('output_tokens', 0) for r in successful),
                   estimated_api_cost_usd=tokens * .042 / 1_000_000,
                   median_request_seconds=statistics.median(r['seconds'] for r in successful) if successful else None,
                   thresholds=dict(presence=presence_threshold, box_confidence=confidence_threshold),
                   note='Thresholds are experimental, not calibrated. YOLO files are partial, unreviewed candidates, not complete ground truth. Blank control is not a training image.')
    (directory / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    page = '''<!doctype html><meta charset="utf-8"><title>Jev box labeling review</title>
<style>body{background:#101820;color:#e8f0f4;font:16px system-ui;max-width:1450px;margin:30px auto;padding:0 20px}h1{color:#20d9ff}article{background:#172735;padding:20px;margin:22px 0;border-radius:12px}p{overflow-wrap:anywhere;color:#c5d6e1}.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}img{width:100%}table{width:100%;border-collapse:collapse;margin-top:15px}td,th{text-align:left;padding:8px;border-bottom:1px solid #365065}a{color:#20d9ff}label{display:block;padding:12px;background:#254255;position:sticky;top:0}</style>
<h1>Jev 1.14: vial labeling experiment</h1><p>Original left, predicted boxes right. Cyan: fill-position vial. Orange: colored liquid. Pink: metal nozzle. These are unreviewed candidates. Existing data and labels were not changed.</p>'''
    page += '<pre>' + html.escape(json.dumps(summary, indent=2)) + '</pre>'
    page += '<label><input id="raw" type="checkbox"> Show all raw boxes, including targets rejected by the presence check</label>' + ''.join(cards)
    page += '<script>document.getElementById("raw").onchange=e=>document.querySelectorAll(".overlay").forEach(im=>{im.src=(e.target.checked?"raw_overlays/":"overlays/")+im.dataset.name+".jpg";im.parentElement.href=im.src})</script>'
    (directory / 'review.html').write_text(page, encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--review-only', action='store_true')
    args = parser.parse_args()
    for subdir in ('responses', 'overlays', 'raw_overlays', 'labels'):
        (args.out / subdir).mkdir(parents=True, exist_ok=True)
    rows = json.loads((args.out / 'manifest.json').read_text())
    results = {}
    if args.review_only:
        for row in rows:
            path = args.out / 'responses' / (row['id'] + '.json')
            if path.exists(): results[row['id']] = json.loads(path.read_text())
    else:
        key = os.environ.get('TYPESAFE_API_KEY') or getpass.getpass('TypeSafe API key (hidden, memory only): ')
        first = predict(rows[0], args.out, key)
        results[rows[0]['id']] = first
        if not first.get('ok'):
            raise SystemExit('First request failed; inspect response JSON before spending more requests.')
        with ThreadPoolExecutor(max_workers=3) as pool:
            pending = {pool.submit(predict, row, args.out, key): row['id'] for row in rows[1:]}
            for future in as_completed(pending):
                results[pending[future]] = future.result()
        del key
    render(args.out, rows, results)


if __name__ == '__main__':
    main()
