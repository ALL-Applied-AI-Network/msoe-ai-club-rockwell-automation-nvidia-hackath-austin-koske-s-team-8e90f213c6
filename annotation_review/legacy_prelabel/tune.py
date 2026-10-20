"""Compare bounded prompt variants, then lock one before fresh-frame validation.

No source labels change. The API key is held in process memory only. Results and
the exact prompts are saved outside Git. Reference boxes are assistant spot checks.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import getpass
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
import uuid

import requests
from PIL import Image

from run import valid_box, to_yolo

MODEL = 'jev-1.14.0'
ROI = [240, 40, 380, 440]
REFERENCE_IDS = {'BLUE': 'run5_vial08_settled', 'ORANGE': 'run5_vial05_settled',
                 'GREEN': 'run5_vial11_settled', 'CLEAR': 'run2_15'}
PROFILES = ('compact_full', 'compact_crop', 'reference_crop')
STRICT = {'presence': .8, 'box_confidence': .5}
DRAFT = {'presence': .5, 'box_confidence': None}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def definition(name):
    crop = name != 'compact_full'
    target = ('the single glass vial in this crop; a black holder crosses its middle, '
              'and the vial continues below the holder' if crop else
              'the glass vial nearest to and below the metal dispensing nozzle, left of image center; '
              'a black holder crosses its middle, and the vial continues below the holder')
    criteria = {'true': 'Colored liquid is visible, including blue, cyan, green, orange, red, yellow or brown. Dark or nearly black liquid counts.',
                'false': 'The vial looks empty, or only transparent colorless liquid is visible.'}
    if name == 'reference_crop':
        criteria = {'true': {'description': criteria['true'], 'examples': ['@REF_BLUE@', '@REF_ORANGE@', '@REF_GREEN@']},
                    'false': {'description': criteria['false'], 'examples': ['@REF_CLEAR@']}}
    return {'name': name, 'model': MODEL, 'crop_xyxy': ROI if crop else None,
            'state': {'target': target, 'note': 'Label only visible appearance. Do not infer recipe, volume or whether the equipment is running.'},
            'questions': {
                'vial_present': {'type': 'noul', 'instructions': 'Is `target` visible in `photo`?'},
                'vial_box': {'type': 'box', 'instructions': 'The complete glass body of `target` in `photo`, from the top rim to the bottom glass base. Include the section below the black holder.'},
                'liquid_present': {'type': 'noul', 'instructions': 'Does `target` in `photo` contain visibly colored liquid?', 'criteria': criteria},
                'liquid_box': {'type': 'box', 'instructions': 'The colored liquid body inside `target` in `photo`, from its top surface to the bottom of the fluid. Include dark fluid; exclude the falling stream above the fluid surface.'},
                'liquid_color': {'type': 'choice', 'instructions': 'What is the visible liquid color inside `target` in `photo`?',
                    'criteria': {'blue': 'Blue, cyan or teal liquid.', 'green': 'Green or very dark green liquid.',
                                 'orange': 'Orange, red or brown liquid.', 'yellow': 'Yellow liquid.',
                                 'empty_clear': 'Empty vial or transparent colorless liquid.',
                                 'uncertain': 'Cannot determine the visible liquid color.'}},
                'left_cut': {'type': 'noul', 'instructions': 'Does the glass body of `target` continue beyond the LEFT edge of `photo`?'},
                'right_cut': {'type': 'noul', 'instructions': 'Does the glass body of `target` continue beyond the RIGHT edge of `photo`?'},
            }}


def prepare(out, split, profile):
    folder = out / profile / split
    for subdir in ('images', 'responses', 'labels_strict', 'labels_draft'):
        (folder / subdir).mkdir(parents=True, exist_ok=True)
    rows = json.loads((out / f'{split}-manifest.json').read_text())
    for row in rows:
        source = Image.open(row['file']).convert('RGB')
        row['source_size'] = list(source.size)
        row['crop_xyxy'] = definition(profile)['crop_xyxy']
        dest = folder / 'images' / (row['id'] + '.jpg')
        if row['crop_xyxy']:
            source = source.crop(ROI).resize((280, 800), Image.Resampling.LANCZOS)
            source.save(dest, quality=95)
        else:
            dest.write_bytes(Path(row['file']).read_bytes())
        row['input_file'] = str(dest)
        row['input_sha256'] = digest(dest.read_bytes())
    (folder / 'manifest.json').write_text(json.dumps(rows, indent=2) + '\n')
    return folder, rows


def reference_assets(out):
    rows = json.loads((out / 'dev-manifest.json').read_text())
    refs = {}
    refdir = out / 'reference-images'
    refdir.mkdir(exist_ok=True)
    for key, sample_id in REFERENCE_IDS.items():
        row = next(r for r in rows if r['id'] == sample_id)
        image = Image.open(row['file']).convert('RGB').crop(ROI).resize((280, 800), Image.Resampling.LANCZOS)
        path = refdir / (key.lower() + '.jpg')
        image.save(path, quality=95)
        refs[key] = path
    return refs


def request(row, profile, folder, refs, key, spec=None):
    spec = spec or definition(profile)
    fingerprint = digest(canonical({'spec': spec, 'input': row['input_sha256'],
                                   'references': {k: digest(v.read_bytes()) for k, v in refs.items()}}))
    path = folder / 'responses' / (row['id'] + '.json')
    if path.exists():
        cached = json.loads(path.read_text())
        if cached.get('ok') and cached['fingerprint'] == fingerprint:
            return cached
        if cached.get('ok'):
            raise ValueError('Cached prompt/input changed; use a new experiment directory')
    photo_id = str(uuid.uuid4())
    assets = {photo_id: {'type': 'image', 'format': 'jpg', 'data_base64': base64.b64encode(Path(row['input_file']).read_bytes()).decode()}}
    qtext = json.dumps(spec['questions'])
    reference_manifest = {}
    if profile == 'reference_crop':
        for name, refpath in refs.items():
            ref_id = str(uuid.uuid4())
            qtext = qtext.replace('@REF_' + name + '@', ref_id)
            assets[ref_id] = {'type': 'image', 'format': 'jpg', 'data_base64': base64.b64encode(refpath.read_bytes()).decode()}
            reference_manifest[name] = dict(path=str(refpath), sha256=digest(refpath.read_bytes()))
    payload = {'model': MODEL, 'state': dict(photo=photo_id, **spec['state']),
               'assets': assets, 'questions': json.loads(qtext)}
    result = dict(id=row['id'], profile=profile, split=row['split'], fingerprint=fingerprint,
                  definition=spec, references=reference_manifest, input_sha256=row['input_sha256'],
                  utc=datetime.now(timezone.utc).isoformat())
    started = time.perf_counter()
    for attempt in range(3):
        try:
            response = requests.post('https://api.typesafe.ai/v1/systemone',
                headers={'Authorization': 'Bearer ' + key, 'TypeSafe-Danger-Zone-Images': 'true'},
                json=payload, timeout=(15, 90), allow_redirects=False)
            if response.status_code in (429, 529) and attempt < 2:
                time.sleep(2 ** (attempt + 1))
                continue
            result['http_status'] = response.status_code
            if response.status_code != 200:
                result.update(ok=False, error=response.text[:2000].replace(key, '[redacted]'))
            else:
                body = response.json()
                assert body['model'] == MODEL
                for name, question in spec['questions'].items():
                    answer = body['answers'][name]
                    assert answer['type'] == question['type']
                    if question['type'] == 'box':
                        assert valid_box(answer['box']) and 0 <= answer['confidence'] <= 1
                    elif question['type'] == 'noul':
                        assert 0 <= answer['noul'] <= 1
                    else:
                        assert answer['choice'] in question['criteria']
                result.update(ok=True, response=body)
        except Exception as exc:
            result.update(ok=False, error=str(exc).replace(key, '[redacted]'))
        break
    result['seconds'] = round(time.perf_counter() - started, 3)
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(f"{profile}/{row['id']}: {'OK' if result.get('ok') else 'ERROR'} {result['seconds']}s", flush=True)
    return result


def mapped(box, row):
    width, height = row['source_size']
    left, top, right, bottom = row['crop_xyxy'] or [0, 0, width, height]
    return {'left': (left + box['left'] * (right - left)) / width,
            'right': (left + box['right'] * (right - left)) / width,
            'top': (top + box['top'] * (bottom - top)) / height,
            'bottom': (top + box['bottom'] * (bottom - top)) / height}


def iou(box, reference, size):
    if reference is None:
        return None
    a = [box[k] * (size[0] if k in ('left', 'right') else size[1]) for k in ('left', 'top', 'right', 'bottom')]
    b = reference
    area = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return area / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - area)


def crop_cut(answers):
    box = answers['vial_box']['box']
    return (max(answers['left_cut']['noul'], answers['right_cut']['noul']) >= .5
            or box['left'] <= .02 or box['right'] >= .98)


def needs_fallback(answers):
    p, c = answers['liquid_present']['noul'], answers['liquid_box']['confidence']
    color = answers['liquid_color']['choice']
    colored = color not in ('empty_clear', 'uncertain')
    return crop_cut(answers) or ((p >= .5 or colored) and not (p >= .8 and c >= .5))


def evaluate(folder, rows, results):
    annotations = []
    for row in rows:
        result = results.get(row['id'])
        if not result or not result.get('ok'):
            continue
        answers = result['response']['answers']
        clipped = crop_cut(answers) if row['crop_xyxy'] else False
        labels = {'strict': [], 'draft': []}
        for class_id, name in enumerate(('vial', 'liquid')):
            box = answers[name + '_box']['box']
            source_box = mapped(box, row)
            p, c = answers[name + '_present']['noul'], answers[name + '_box']['confidence']
            positive = row['vial_visible'] if name == 'vial' else row['colored_liquid_visible']
            reference = row.get('vial_reference_xyxy') if name == 'vial' else row.get('liquid_reference_xyxy')
            overlap = iou(source_box, reference, row['source_size'])
            record = dict(sample_id=row['id'], target=name, presence=p, confidence=c,
                          input_box=box, original_image_box=source_box, visible_reference=positive,
                          approximate_reference_iou=overlap, strict=p >= .8 and c >= .5,
                          draft=p >= .5, crop_cut_flag=clipped,
                          review_priority='high' if clipped or c < .5 or p < .8 else 'normal',
                          annotation_status='unreviewed_model_candidate')
            annotations.append(record)
            for policy in labels:
                if record[policy]:
                    labels[policy].append(str(class_id) + ' ' + ' '.join(f'{v:.8f}' for v in to_yolo(source_box)))
        for policy, lines in labels.items():
            (folder / ('labels_' + policy) / (row['id'] + '.txt')).write_text('\n'.join(lines) + ('\n' if lines else ''))
    (folder / 'annotations.json').write_text(json.dumps(annotations, indent=2) + '\n')
    report = {'successful_requests': sum(r.get('ok', False) for r in results.values()),
              'input_tokens': sum(r['response']['usage']['input_tokens'] for r in results.values() if r.get('ok')),
              'median_seconds': statistics.median(r['seconds'] for r in results.values() if r.get('ok'))}
    for target in ('vial', 'liquid'):
        group = [r for r in annotations if r['target'] == target]
        report[target] = {'visible': sum(r['visible_reference'] for r in group)}
        for policy in ('strict', 'draft'):
            report[target][policy] = {
                'visible_retained': sum(r['visible_reference'] and r[policy] for r in group),
                'absent_retained': sum(not r['visible_reference'] and r[policy] for r in group),
                'useful_reference_boxes': sum(r['visible_reference'] and r[policy] and (r['approximate_reference_iou'] or 0) >= .5 for r in group)}
        values = [r['approximate_reference_iou'] for r in group if r['visible_reference'] and r['approximate_reference_iou'] is not None]
        report[target]['mean_approximate_iou_all_raw_boxes'] = statistics.mean(values) if values else None
    report['notes'] = 'Assistant visual references, not independent ground truth. Crop outputs mapped to source images. Strict vs draft separates prompting from filtering policy.'
    (folder / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--phase', choices=('dev', 'holdout'), required=True)
    parser.add_argument('--profile', choices=PROFILES)
    args = parser.parse_args()
    definitions = {name: definition(name) for name in PROFILES}
    spec_path = args.out / 'prompt-definitions.json'
    if spec_path.exists():
        assert json.loads(spec_path.read_text()) == definitions, 'Frozen definitions changed; create a new experiment'
    else:
        spec_path.write_text(json.dumps(definitions, indent=2) + '\n')
    if args.phase == 'holdout':
        locked = json.loads((args.out / 'locked-profile.json').read_text())
        assert args.profile in (locked['primary'], locked['fallback'])
        assert digest(canonical(definition(args.profile))) == locked['definitions_sha256'][args.profile]
        assert digest((args.out / 'holdout-manifest.json').read_bytes()) == locked['holdout_manifest_sha256']
    refs = reference_assets(args.out)
    key = os.environ.get('TYPESAFE_API_KEY') or getpass.getpass('TypeSafe API key (hidden, memory only): ')
    for profile in ([args.profile] if args.profile else PROFILES):
        folder, rows = prepare(args.out, args.phase, profile)
        if args.phase == 'holdout' and profile == locked['fallback']:
            selected = []
            for row in rows:
                result = json.loads((args.out / locked['primary'] / 'holdout/responses' / (row['id'] + '.json')).read_text())
                if result.get('ok') and needs_fallback(result['response']['answers']):
                    selected.append(row)
            rows = selected
            (folder / 'manifest.json').write_text(json.dumps(rows, indent=2) + '\n')
            if not rows:
                print('No fallback requests needed', flush=True)
                continue
        results = {}
        first = request(rows[0], profile, folder, refs, key)
        if not first.get('ok'):
            raise SystemExit('First request failed; inspect saved error before proceeding')
        results[rows[0]['id']] = first
        with ThreadPoolExecutor(max_workers=3) as pool:
            pending = {pool.submit(request, row, profile, folder, refs, key): row['id'] for row in rows[1:]}
            for future in as_completed(pending):
                results[pending[future]] = future.result()
        evaluate(folder, rows, results)
    del key


if __name__ == '__main__':
    main()
