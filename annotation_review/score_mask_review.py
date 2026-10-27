"""Score frozen pool predictions against explicit, non-demonstration human reviews.

Run with the CPU data environment (NumPy/Pillow). Does not fit, select, tune, or
approve anything; uncertain/unreviewed frames are excluded and coverage is shown.
"""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())


def validated_decisions(review, tasks):
    """Replay history rather than trusting editable latest_decisions snapshots."""
    import numpy as np
    if review.get('schema') != 'visible-pool-human-review/v1' or review.get('demonstration') is not False:
        raise ValueError('Explicit, non-demonstration human review required')
    if review.get('training_eligible') is not False or review.get('source_run') != 4:
        raise ValueError('Expected evaluation-only Run 4 review')
    states, intervals = {}, {}
    for seq, event in enumerate(review['history'], 1):
        if event.get('seq') != seq or not str(event.get('reviewer', '')).strip():
            raise ValueError('Missing ordered history/reviewer')
        if event['kind'] == 'onset':
            a, b = tasks.get(event['last_empty']), tasks.get(event['first_pool'])
            if not a or not b or a['case_id'] != event['case_id'] or b['case_id'] != event['case_id']:
                raise ValueError('Onset case/source mismatch')
            if a['frame_index'] >= b['frame_index'] or states.get(a['sample_id'], {}).get('status') != 'empty' or states.get(b['sample_id'], {}).get('status') != 'pool':
                raise ValueError('Onset endpoints must be ordered and human-reviewed')
            intervals[event['case_id']] = event
            continue
        if event['kind'] != 'frame':
            raise ValueError('Unknown history event')
        source = event['source']; task = tasks.get(source['sample_id'])
        keys = ['sample_id', 'image_sha256', 'source_rgb_sha256', 'source_video_sha256', 'frame_index', 'pts', 'time_base']
        if task is None or any(source.get(k) != task[k] for k in keys):
            raise ValueError('Reviewed source identity differs')
        rle = event['mask']
        if rle['width'] != 640 or rle['height'] != 480:
            raise ValueError('Reviewed mask dimensions differ')
        mask = np.zeros(640*480, dtype=np.uint8); pos = 0
        for i, count in enumerate(rle['counts']):
            if type(count) is not int or count < 0 or pos + count > len(mask):
                raise ValueError('Invalid reviewed mask RLE')
            if i % 2:
                mask[pos:pos+count] = 1
            pos += count
        if pos != len(mask) or event['status'] not in ['empty', 'pool', 'uncertain']:
            raise ValueError('Invalid reviewed mask or status')
        if event['status'] == 'empty' and mask.any() or event['status'] == 'pool' and not mask.any():
            raise ValueError('Review status and mask disagree')
        if event['status'] == 'uncertain' and not event['notes'].strip():
            raise ValueError('Uncertainty requires notes')
        states[source['sample_id']] = dict(event, decoded_mask=mask.reshape(480, 640).astype(bool))
    return states, intervals


def main(a):
    import numpy as np
    from PIL import Image
    draft = read(a.probe / 'review-drafts.json'); review = read(a.review)
    manifest = read(a.packet / 'manifest.json')
    if review.get('packet_id') != manifest['packet_id'] or review.get('source_video_sha256') != draft['source_video_sha256']:
        raise ValueError('Packet/source mismatch')
    tasks = {t['sample_id']: t for t in draft['tasks']}
    states, intervals = validated_decisions(review, tasks)
    scored = {k: v for k, v in states.items() if v['status'] != 'uncertain'}
    if not scored:
        raise ValueError('No definite human decisions to score')
    summary = {}
    for mode in ['single_rgb', 'paired_diff', 'paired_diff_gain']:
        tp = fp = fn = tn = 0; ious = []; dices = []; rows = []
        for sid, event in scored.items():
            task = tasks[sid]; pred = next(d for d in task['drafts'] if d['mode'] == mode)
            path = a.probe / pred['mask']
            if sha(path) != pred['mask_sha256'] or sha(a.probe / task['image']) != task['image_sha256']:
                raise ValueError('Frozen source/draft changed')
            mask = np.asarray(Image.open(path)) >= 128
            accepted = pred['presence_probability'] >= .5 and bool(mask.any())
            if not accepted:
                mask = np.zeros_like(mask)
            positive = event['status'] == 'pool'
            tp += positive and accepted; fp += not positive and accepted
            fn += positive and not accepted; tn += not positive and not accepted
            inter = int((mask & event['decoded_mask']).sum())
            union = int((mask | event['decoded_mask']).sum())
            iou = inter / union if union else 1.
            dice = 2 * inter / max(1, int(mask.sum() + event['decoded_mask'].sum()))
            if positive:
                ious.append(iou); dices.append(dice)
            rows.append(dict(sample_id=sid, status=event['status'], accepted=accepted, iou=iou,
                dice=dice, reviewer=event['reviewer'], draft_assisted=event['draft_model_sha256'] is not None))
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        summary[mode] = dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=precision, recall=recall,
            positive_mean_mask_iou=sum(ious)/len(ious) if ious else None,
            positive_mean_dice=sum(dices)/len(dices) if dices else None, frames=rows)
    frames = [json.loads(s) for s in (a.probe / 'frames.jsonl').read_text().splitlines()]
    onset = {}
    for case, event in intervals.items():
        if states[event['last_empty']]['status'] != 'empty' or states[event['first_pool']]['status'] != 'pool':
            onset[case] = dict(status='stale_endpoint_review_required'); continue
        absent, present = tasks[event['last_empty']], tasks[event['first_pool']]
        t_absent = float(absent['pts'] * Fraction(absent['time_base']))
        t_present = float(present['pts'] * Fraction(present['time_base']))
        onset[case] = dict(human_interval_frames=[absent['frame_index'], present['frame_index']], modes={})
        for mode in summary:
            predicted = next((r for r in frames if r['case_id'] == case and r['modes'][mode]['presence_gate_passed']), None)
            onset[case]['modes'][mode] = None if predicted is None else dict(
                first_sampled_activation_frame=predicted['source_frame'],
                delay_interval_s=[float(predicted['pts'] * Fraction(predicted['time_base'])) - t_present,
                                  float(predicted['pts'] * Fraction(predicted['time_base'])) - t_absent])
    a.out.mkdir(parents=True, exist_ok=False)
    report = dict(review_sha256=sha(a.review), packet_id=review['packet_id'], reviewed=len(states),
        scored=len(scored), uncertain=len(states)-len(scored), unreviewed=len(tasks)-len(states),
        models=summary, onset=onset, team_approved=review.get('team_approved', False),
        reference_status='Explicit human reviews; team approval is separate.',
        training=False, checkpoint_selection_changed=False, thresholds_changed=False,
        limitations=['Six correlated development clips; no production or independent-test claim.',
            'Draft-assisted reviews may be anchored to model suggestions. Include independent raw-first review.',
            'Onset delay is interval-valued and sampled; a negative value may be an early false activation.'])
    (a.out / 'comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(reviewed=len(states), scored=len(scored), out=str(a.out))))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['probe', 'packet', 'review', 'out']:
        p.add_argument('--' + name, type=Path, required=True)
    main(p.parse_args())
