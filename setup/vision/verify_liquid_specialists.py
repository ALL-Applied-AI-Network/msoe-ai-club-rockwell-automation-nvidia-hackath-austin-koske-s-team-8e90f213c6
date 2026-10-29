"""Verify the assembled four-arm comparison, including full video decoding."""
import argparse
import csv
import json
from pathlib import Path

import av
import cv2

from evaluate_crop_contrast import matched
from liquid_specialists_data import ARMS, WEIGHTS, read, sha, write


def verify(experiment, evaluation, out):
    comparison = read(evaluation/'comparison.json')
    if (set(comparison['arms']) != set(ARMS) or comparison['detector_promoted']
            or comparison['measurement_qa_changed'] or comparison['run5_fit_or_selection']):
        raise ValueError('Comparison scope changed')
    plan_hash = sha(experiment/'experiment-plan.json')
    for name, expected in read(experiment/'preparation.json').items():
        if sha(experiment/name) != expected:
            raise ValueError('Frozen helper/plan changed: '+name)
    for name, expected in WEIGHTS.items():
        if sha(experiment/'pretrained'/name) != expected:
            raise ValueError('Initialization changed: '+name)
    for name, expected in comparison['image_sha256'].items():
        if sha(evaluation/name) != expected or cv2.imread(str(evaluation/name)) is None:
            raise ValueError('Summary image failed integrity check')
    result = dict(schema_version=1, artifact_kind='verified_liquid_specialist_comparison',
        experiment_plan_sha256=plan_hash, comparison_sha256=sha(evaluation/'comparison.json'),
        source_inventory_sha256=None, arms={}, total_videos=0, total_decoded_output_frames=0,
        detector_promoted=False, measurement_qa_changed=False, run5_fit_or_selection=False,
        verified_script_sha256=sha(Path(__file__)), image_sha256=comparison['image_sha256'],
        limits=['Agreement with unreviewed teacher labels, not human accuracy or measurement QA.',
                'Run 5 repeatedly inspected; not a blind final test.',
                'Early clips use image-detector starts, not reviewed first-visible-liquid/flow-onset timestamps.',
                'Blue vial proposals are identical across models; no improved tracking claim.',
                'Family scores are not calibrated. No automatic detector promotion.'])
    inventory = None
    baseline_rows = None
    for arm, report in comparison['arms'].items():
        folder = evaluation/arm
        model = report['model']
        actual_model = read(experiment/'models'/arm/'model.json')
        if (model != actual_model or model['excluded_runs'] != [5]
                or model['experiment_plan_sha256'] != plan_hash
                or sha(experiment/'models'/arm/'best.pt') != model['checkpoint_sha256']):
            raise ValueError('Model changed: '+arm)
        if read(folder/'completed-arm.json') != report:
            raise ValueError('Completed-arm report differs')
        metrics = read(folder/'teacher-agreement.json')
        if {k: v for k, v in metrics.items() if k != 'sources'} != report['teacher_agreement']:
            raise ValueError('Metrics changed')
        if inventory is None:
            inventory = metrics['sources']
            import hashlib
            result['source_inventory_sha256'] = hashlib.sha256(json.dumps(inventory, sort_keys=True).encode()).hexdigest()
        if metrics['sources'] != inventory or metrics['frames'] != 816:
            raise ValueError('Scoring inventory differs')
        if metrics['candidate']['per_class'][0] != metrics['baseline']['per_class'][0]:
            raise ValueError('Vial branch changed')
        rows = read(folder/'scoring-predictions.json')
        identity = [(r['source_id'], r['targets'], r['baseline']) for r in rows]
        if baseline_rows is None:
            baseline_rows = identity
        if identity != baseline_rows:
            raise ValueError('Targets/baseline predictions differ across arms')
        unmatched = multiple = unmatched_frames = 0
        for row in rows:
            # Preserve the original report's AP-floor matching for comparison.
            tp = matched(row['candidate'], row['targets'])[:, 0]
            indices = [i for i, p in enumerate(row['candidate']) if p['class_id'] == 1 and p['confidence'] >= .35]
            count = sum(not tp[i] for i in indices)
            unmatched += count; unmatched_frames += count > 0; multiple += len(indices) > 1
        diagnostic = dict(counts=dict(frames_multiple_liquid_boxes=multiple,
            unmatched_boxes=unmatched, unmatched_frames=unmatched_frames),
            matching='Historical AP-floor one-to-one assignment, then confidence .35',
            adjudicated=False)
        target = folder/'unmatched-box-review.json'
        if target.exists():
            if read(target)['counts'] != diagnostic['counts']:
                raise ValueError('Unmatched-box diagnostic differs')
        else:
            write(target, diagnostic)
        videos = []
        for video in report['decoded_videos']:
            path = folder/video['file']
            if sha(path) != video['sha256']:
                raise ValueError('Video bytes changed')
            with av.open(str(path)) as movie:
                frames = sum(1 for _ in movie.decode(video=0))
            if frames != video['frames']:
                raise ValueError('Incomplete video decode')
            videos.append(dict(file=str(path), frames=frames, sha256=video['sha256']))
            result['total_videos'] += 1; result['total_decoded_output_frames'] += frames
        for path in (experiment/'models'/arm).rglob('*.csv'):
            if b'\0' in path.read_bytes():
                raise ValueError('NUL byte in completed training CSV')
            with path.open(newline='') as f:
                curve = list(csv.DictReader(f))
            if len(curve) != model['epochs_completed']:
                raise ValueError('Training curve length differs')
        liquid = next(c for c in metrics['candidate']['per_class'] if c['class_id'] == 1)
        result['arms'][arm] = dict(model=model, liquid_metrics=liquid,
            unmatched=diagnostic['counts'], provisional_negatives=report['provisional_negatives'],
            source_video_sha256=report['video']['source_video_sha256'],
            decoded_source_frames=report['video']['source_frames_decoded'],
            decoded_videos=videos, paired_timing=report['timing'])
        print(arm, 'verified', flush=True)
    if result['total_videos'] != 12 or result['total_decoded_output_frames'] != 39292:
        raise ValueError('Unexpected delivered media inventory')
    if out.exists():
        raise ValueError('Do not overwrite verification evidence')
    write(out, result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('experiment', 'evaluation', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    verify(a.experiment, a.evaluation, a.out)
