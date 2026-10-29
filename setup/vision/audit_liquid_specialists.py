"""Run 4 development diagnostics; never changes labels, checkpoints or app settings."""
import argparse
from collections import Counter
import json
from pathlib import Path

import cv2
import numpy as np

from evaluate_crop_contrast import matched
from evaluate_liquid_specialists import SpecialistPair
from liquid_specialists_data import ARMS, WEIGHTS, read, sha, write


THRESHOLDS = (.1, .2, .35, .5, .65, .8, .9)


def development_rows(experiment):
    """Validate provenance before opening any image; Run 5 is never an input."""
    plan = read(experiment / 'experiment-plan.json')
    data = experiment / 'data/all'
    provenance = read(data / 'provenance.json')
    if (plan['excluded_from_fit_and_selection'] != [5]
            or sha(data / 'provenance.json') != plan['data']['all']['provenance_sha256']):
        raise ValueError('Development provenance changed')
    rows = [row for row in provenance['samples'] if row['split'] == 'val']
    held_out = set(provenance['held_out_source_hashes'])
    if len(rows) != 157 or len({r['source_id'] for r in rows}) != len(rows):
        raise ValueError('Development inventory changed')
    for row in rows:
        if (row['source_run'] != 4 or row.get('parent_id')
                or row['source_sha256'] in held_out or row['image_sha256'] in held_out):
            raise ValueError('Only original Run 4 development images are allowed')
        if (sha(data / row['image']) != row['image_sha256']
                or sha(data / row['labels']) != row['actual_label_sha256']):
            raise ValueError('Development image or label changed')
    return data, rows


def targets_for(data, row):
    targets = []
    for line in (data / row['labels']).read_text().splitlines():
        c, x, y, w, h = map(float, line.split())
        if c != 0 or not (0 < w <= 1 and 0 < h <= 1):
            raise ValueError('Invalid liquid-only target')
        targets.append(dict(class_id=1, box=[(x-w/2)*640, (y-h/2)*480,
                                            (x+w/2)*640, (y+h/2)*480]))
    return targets


def at_threshold(rows, threshold):
    """Rematch after filtering: removed low-score boxes cannot claim targets."""
    tp = fp = fn = empty_alarms = multiple = nested = 0
    details = []
    strata = {name: dict(targets=0, detected=0) for name in ('height_le_32', 'height_32_to_64', 'height_gt_64')}
    for row in rows:
        proposals = [p for p in row['predictions'] if p['confidence'] >= threshold]
        matches = matched(proposals, row['targets'])[:, 0]
        true = int(matches.sum())
        if len(row['targets']) > 1:
            raise ValueError('Height strata require the frozen single-liquid-target images')
        if row['targets']:
            box = row['targets'][0]['box']
            height = box[3] - box[1]
            name = 'height_le_32' if height <= 32 else 'height_32_to_64' if height <= 64 else 'height_gt_64'
            strata[name]['targets'] += 1
            strata[name]['detected'] += true
        tp += true
        fp += len(proposals) - true
        fn += len(row['targets']) - true
        empty_alarms += bool(proposals) and not row['targets']
        multiple += len(proposals) > 1
        for i, left in enumerate(proposals):
            a = np.asarray(left['box'])
            for right in proposals[i+1:]:
                b = np.asarray(right['box'])
                intersection = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])).prod()
                smaller = min((a[2:]-a[:2]).prod(), (b[2:]-b[:2]).prod())
                nested += smaller > 0 and intersection / smaller >= .9
        details.append(dict(source_id=row['source_id'], targets=len(row['targets']),
                            predictions=len(proposals), unmatched=len(proposals)-true,
                            missed=len(row['targets'])-true))
    return dict(confidence=threshold, tp=tp, unmatched_boxes=fp, missed_targets=fn,
                precision=tp/(tp+fp) if tp+fp else 0.,
                recall=tp/(tp+fn) if tp+fn else 0.,
                frames_with_multiple_boxes=int(multiple), nested_prediction_pairs=int(nested),
                provisional_negative_frames_with_predictions=int(empty_alarms),
                target_height_strata=strata), details


def summarize(rows):
    from ultralytics.utils.metrics import ap_per_class
    statistics = [(matched(r['predictions'], r['targets']),
                   np.array([p['confidence'] for p in r['predictions']]),
                   np.ones(len(r['predictions'])), np.ones(len(r['targets']))) for r in rows]
    tp, confidence, classes, targets = [np.concatenate([s[i] for s in statistics]) for i in range(4)]
    ap = ap_per_class(tp, confidence, classes, targets, plot=False, names={1: 'colored_liquid'})[5]
    operating_points = [at_threshold(rows, value)[0] for value in THRESHOLDS]
    return dict(frames=len(rows), targets=int(len(targets)),
                provisional_negative_frames=sum(not r['targets'] for r in rows),
                liquid_ap50=float(ap[0, 0]), liquid_ap50_95=float(ap[0].mean()),
                operating_points=operating_points)


def gallery(data, sources, predictions, output, short_only=False):
    """Review development disagreements without assigning physical semantics."""
    names = list(predictions)
    detail = {name: at_threshold(rows, .35)[1] for name, rows in predictions.items()}
    ranks = Counter()
    allowed = {r['source_id'] for r in predictions[names[0]] if r['targets'] and
               max(t['box'][3]-t['box'][1] for t in r['targets']) <= 64} if short_only else None
    for rows in detail.values():
        for row in rows:
            if allowed is not None and row['source_id'] not in allowed:
                continue
            ranks[row['source_id']] += row['unmatched'] + row['missed']
    selected = sorted(ranks, key=lambda key: (-ranks[key], key))[:12]
    lookup = {r['source_id']: r for r in sources}
    images = {}
    for key in selected:
        image = cv2.imread(str(data / lookup[key]['image']))
        if image is None or image.shape != (480, 640, 3):
            raise ValueError('Development image geometry changed')
        images[key] = image
    rows_by_name = {name: {r['source_id']: r for r in rows} for name, rows in predictions.items()}
    # One full-source sheet per case; display scaling never affects inference.
    (output / 'cases').mkdir()
    for key in selected:
        panels = []
        for name in ['Provisional targets', *names]:
            image = images[key].copy()
            first = rows_by_name[names[0]][key]
            boxes = first['targets'] if name == 'Provisional targets' else [
                p for p in rows_by_name[name][key]['predictions'] if p['confidence'] >= .35]
            for p in boxes:
                b = np.rint(p['box']).astype(int)
                cv2.rectangle(image, tuple(b[:2]), tuple(b[2:]), (80, 240, 100), 2)
            header = np.zeros((36, 640, 3), np.uint8)
            cv2.putText(header, name, (8, 24), 0, .65, (255, 255, 255), 1)
            panels.append(cv2.resize(np.vstack([header, image]), (384, 310)))
        while len(panels) % 3:
            panels.append(np.zeros_like(panels[0]))
        canvas = np.vstack([np.hstack(panels[i:i+3]) for i in range(0, len(panels), 3)])
        if not cv2.imwrite(str(output / 'cases' / (key + '.png')), canvas):
            raise IOError('Development gallery export failed')
    write(output / 'case-index.json', dict(
        selection='Top 12 Run 4 aggregate unmatched + missed at fixed .35; ties by source ID',
        geometry_scope='Supplied liquid target height <=64 source pixels' if short_only else 'All development images',
        human_adjudication=False, cases=[dict(source_id=key, disagreement_score=ranks[key],
            image_sha256=lookup[key]['image_sha256'], file='cases/' + key + '.png') for key in selected]))


def main(a):
    if a.out.exists():
        raise ValueError('Use a fresh development output')
    data, sources = development_rows(a.experiment)
    for name, expected in WEIGHTS.items():
        if sha(a.experiment / 'pretrained' / name) != expected:
            raise ValueError('Pretrained asset changed')
    if sha(a.baseline) != '10856c18a8a5c485e832adcc6b841b739e726a969dc043fe8db64c49163bb361':
        raise ValueError('Baseline changed')
    a.out.mkdir(parents=True)
    plan_hash = sha(a.experiment / 'experiment-plan.json')
    manifest = dict(schema_version=1, kind='liquid_run4_development_audit',
        development_run=4, run5_images_used=False, confidence_grid=list(THRESHOLDS),
        ap_confidence=.001, nms_iou=.45, nested_pair_intersection_over_smaller_area=.9,
        experiment_plan_sha256=plan_hash, script_sha256=sha(Path(__file__)),
        checkpoint_hashes={}, baseline_checkpoint_sha256=sha(a.baseline),
        sources=[{k: r[k] for k in ('source_id', 'source_run', 'image_sha256', 'actual_label_sha256')}
                 for r in sources],
        limits=['Unreviewed teacher agreement, including provisional negative absences.',
                'Run 4 already selected checkpoints: development evidence, not generalization.',
                'Threshold diagnostics are rematched at each cutoff; historical Run 5 reports retain their original matching.',
                'Geometric nesting does not adjudicate body/stream or correct labels.',
                'No operating threshold selected, detector promoted, or app/measurement QA changed.'])
    write(a.out / 'audit-plan.json', manifest)
    predictions = {}
    for arm in ARMS:
        model = read(a.experiment / 'models' / arm / 'model.json')
        digest = sha(a.experiment / 'models' / arm / 'best.pt')
        if (model['checkpoint_sha256'] != digest or model['experiment_plan_sha256'] != plan_hash
                or model['excluded_runs'] != [5] or model['development_runs'] != [4]):
            raise ValueError('Checkpoint ancestry changed')
        manifest['checkpoint_hashes'][arm] = digest
        pair = SpecialistPair(a.baseline, a.experiment, arm, a.device)
        rows = []
        baseline = []
        for offset in range(0, len(sources), 16):
            batch = sources[offset:offset+16]
            rgb = [cv2.cvtColor(cv2.imread(str(data/r['image'])), cv2.COLOR_BGR2RGB) for r in batch]
            results = pair.predict(rgb, .001)
            for i, source in enumerate(batch):
                common = dict(source_id=source['source_id'], targets=targets_for(data, source))
                rows.append(dict(**common, predictions=[p for p in results[1][i] if p['class_id'] == 1]))
                baseline.append(dict(**common, predictions=[p for p in results[0][i] if p['class_id'] == 1]))
        if 'Deployed' not in predictions:
            predictions['Deployed'] = baseline
        elif predictions['Deployed'] != baseline:
            raise ValueError('Baseline predictions changed between arms')
        predictions[arm] = rows
        write(a.out / (arm + '-predictions.json'), rows)
        print(json.dumps(dict(arm=arm, **summarize(rows))), flush=True)
        del pair
    write(a.out / 'Deployed-predictions.json', predictions['Deployed'])
    gallery(data, sources, predictions, a.out)
    import torch, ultralytics, transformers
    manifest['evaluation_runtime'] = dict(torch=torch.__version__, ultralytics=ultralytics.__version__,
        transformers=transformers.__version__, gpu=torch.cuda.get_device_name(0) if 'cuda' in a.device else None)
    write(a.out / 'comparison.json', dict(**manifest, models={name: summarize(rows) for name, rows in predictions.items()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('experiment', 'baseline', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    main(parser.parse_args())
