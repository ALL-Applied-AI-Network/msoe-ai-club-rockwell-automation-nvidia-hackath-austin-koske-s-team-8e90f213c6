"""Offline one-to-one evaluation. Never imported by the image detector."""
import numpy as np


def interval_iou(left, right):
    intersection = max(0, min(left['end_frame'], right['end_frame']) - max(left['start_frame'], right['start_frame']) + 1)
    union = max(left['end_frame'], right['end_frame']) - min(left['start_frame'], right['start_frame']) + 1
    return intersection / union


def timing(values, fps):
    if not values:
        return None
    x = np.asarray(values)
    return dict(count=len(values), signed_median_frames=float(np.median(x)),
                absolute_median_frames=float(np.median(abs(x))), absolute_p95_frames=float(np.percentile(abs(x), 95)),
                absolute_max_frames=int(max(abs(x))),
                absolute_median_mp4_s=float(np.median(abs(x)) / fps),
                absolute_p95_mp4_s=float(np.percentile(abs(x), 95) / fps))


def evaluate(detections, annotations, manifest, min_iou=.5):
    if not 0 < min_iou <= 1:
        raise ValueError('Match IoU must lie in (0, 1]')
    truth = [item.to_dict() for item in annotations]
    n, m = len(detections), len(truth)
    # Chronological dynamic programming: maximize match count, then total IoU.
    # Each annotation and each detection can be used only once.
    score = np.zeros((n + 1, m + 1))
    action = np.zeros((n + 1, m + 1), dtype=int)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            options = [(score[i - 1, j], 1), (score[i, j - 1], 2)]
            overlap = interval_iou(detections[i - 1], truth[j - 1])
            if overlap >= min_iou:
                options.append((score[i - 1, j - 1] + 1 + overlap / (min(n, m) + 1), 3))
            score[i, j], action[i, j] = max(options)
    pairs = []
    i, j = n, m
    while i and j:
        if action[i, j] == 3:
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif action[i, j] == 1:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    matched = []
    for i, j in pairs:
        d, a = detections[i], truth[j]
        matched.append(dict(detection_id=d['detection_id'], annotation_vial_id=a['vial_id'],
                            iou=interval_iou(d, a), detected_start_frame=d['start_frame'], annotated_start_frame=a['start_frame'],
                            detected_end_frame=d['end_frame'], annotated_end_frame=a['end_frame'],
                            confirmed_frame=d['confirmed_frame'], departure_frame=d['departure_frame'],
                            start_error_frames=d['start_frame'] - a['start_frame'],
                            end_error_frames=d['end_frame'] - a['end_frame'],
                            confirmation_delay_frames=d['confirmed_frame'] - a['end_frame'],
                            partial_start_possible=d['partial_start_possible'], episodes=len(d['episodes'])))
    used_d, used_a = {i for i, _ in pairs}, {j for _, j in pairs}
    fps = manifest['fps']
    count = len(pairs)
    summary = dict(validation_available=bool(truth), labeled_fills=m, detected_fills=n,
                   matched_fills=count, missed_fills=m - count if truth else None,
                   extra_detections=n - count if truth else None,
                   precision=count / n if n and truth else None, recall=count / m if m else None,
                   start_error=timing([row['start_error_frames'] for row in matched if not row['partial_start_possible']], fps),
                   end_error=timing([row['end_error_frames'] for row in matched], fps),
                   confirmation_delay=timing([row['confirmation_delay_frames'] for row in matched], fps))
    return dict(schema_version=1, run_id=manifest['run_id'],
                evaluation_scope='development; Run 5 inspected while choosing pilot thresholds; not held out',
                annotation_status='supplied boundaries, not independently adjudicated',
                matching_rule=dict(method='chronological_one_to_one_maximum_cardinality_then_iou', minimum_iou=min_iou),
                time_units='decoded frames and MP4 seconds; no physical-time or warning-lead-time claim',
                boundary_definition='first confirmed visible stream to final visible stream stop; published after carrier departure',
                provenance={key: manifest.get(key) for key in ('video_sha256', 'annotation_sha256', 'config_id', 'implementation_id', 'fps', 'sample_count')},
                summary=summary, matches=matched,
                unmatched_detection_ids=[d['detection_id'] for i, d in enumerate(detections) if i not in used_d] if truth else None,
                unmatched_annotation_vial_ids=[a['vial_id'] for j, a in enumerate(truth) if j not in used_a],
                open_fill_at_eof=manifest.get('open_fill_at_eof'))
