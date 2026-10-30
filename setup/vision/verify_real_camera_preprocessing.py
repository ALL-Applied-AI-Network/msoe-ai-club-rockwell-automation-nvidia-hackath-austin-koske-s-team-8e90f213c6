"""Independently recount paired evidence and verify every raw/corrected RGB identity."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys

import av
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from liquid_specialists_data import read, write, sha
from render_liquid_pipeline import pixel_hash
from replay.geometry import decode_mask
from stage2_camera_reference import PATCHES


def panel(rgb, row, title):
    canvas = np.full((550, 640, 3), 24, np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if row is not None:
        mask = decode_mask(row['measurement']['geometry']['mask']).astype(bool)
        bgr[mask] = (.55*bgr[mask]+.45*np.array([50, 220, 220])).astype(np.uint8)
        zoom = cv2.resize(bgr[280:460, 260:360], (120, 216))
        bgr[255:471, 502:622] = zoom
    canvas[35:515] = bgr
    cv2.putText(canvas, title, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, .52, (235, 235, 235), 1, cv2.LINE_AA)
    state = 'unavailable' if row is None else str(bool(row['measurement']['measurement_valid']))
    cv2.putText(canvas, 'Sample QA: '+state+' | Yellow: sampled model geometry', (12, 537),
                cv2.FONT_HERSHEY_SIMPLEX, .48, (220, 220, 220), 1, cv2.LINE_AA)
    return canvas


def main(a):
    if a.out.exists():
        raise ValueError('Use a fresh verification output')
    complete = read(a.input/'completed.json'); plan = read(a.input/'plan.json')
    for name, digest in complete['artifacts'].items():
        if sha(a.input/name) != digest:
            raise ValueError('Candidate artifact changed: '+name)
    for name, digest in plan['current_helpers_sha256'].items():
        if sha(a.input/'helpers'/name) != digest:
            raise ValueError('Retained executed dependency changed: '+name)
    if sha(a.source) != plan['source_sha256']:
        raise ValueError('Source file differs')
    raw_path = a.baseline/'pytorch-pass0/frames.jsonl'
    if sha(raw_path) != plan['baseline_frames_sha256']:
        raise ValueError('Raw model evidence differs')
    raw = {r['source_frame']: r for r in map(json.loads, raw_path.read_text().splitlines())}
    rows = [json.loads(s) for s in (a.input/'frames.jsonl').read_text().splitlines()]
    if len(rows) != len(raw) or len({r['source_frame'] for r in rows}) != len(raw):
        raise ValueError('Missing or duplicate candidate samples')
    candidate = {r['source_frame']: r for r in rows}; summary = read(a.input/'comparison.json')
    calibration = read(a.input/'calibration.json')
    a.out.mkdir(parents=True); shutil.copy2(__file__, a.out/'executed-verifier.py')
    totals = Counter(); baseline = Counter(); proposed = Counter(); differences = []; ious = []
    rendered = []; frames = 0
    selected = {r['source_frame'] for r in summary['qa_disagreements']}
    with av.open(str(a.source)) as c:
        for index, f in enumerate(c.decode(video=0)):
            frames += 1
            if index not in candidate:
                continue
            row = candidate[index]; original = raw[index]; rgb = f.to_ndarray(format='rgb24')
            if not (pixel_hash(rgb) == row['raw_rgb_sha256'] == original['source_rgb_sha256']
                    and f.pts == row['pts'] == original['pts']
                    and str(f.time_base) == row['time_base'] == original['time_base']):
                raise ValueError('Raw pixel/clock mismatch')
            seconds = float(f.pts*f.time_base)
            if seconds != row['seconds']:
                raise ValueError('Candidate seconds differ from native PTS')
            totals['samples'] += 1
            if seconds < 5:
                expected_reason = 'startup_reference_not_yet_available'; corrected = None
            else:
                values = []
                for key in calibration['selected']:
                    x, y, r, b = PATCHES[key]
                    values.append(np.median(rgb[y:b, x:r].reshape(-1, 3), axis=0))
                values = np.asarray(values); reference = np.asarray(calibration['reference_rgb'])
                ratios = values/reference; gain = np.median(ratios, axis=0)
                spread = float(np.max(np.abs(ratios/gain-1)))
                healthy = bool(np.isfinite(values).all() and (reference >= 12).all() and (values < 245).all()
                    and (gain >= .65).all() and (gain <= 1.55).all() and spread <= calibration['maximum_disagreement'])
                expected_reason = None if healthy else 'unhealthy_background_reference'
                corrected = np.clip(np.rint(rgb.astype(float)/gain), 0, 255).astype(np.uint8) if healthy else None
                np.testing.assert_allclose(gain, row['camera_preprocessing']['gain'], rtol=0, atol=0)
                if spread != row['camera_preprocessing']['disagreement']:
                    raise ValueError('Camera-gate disagreement differs')
            if expected_reason != row['camera_preprocessing']['reason']:
                raise ValueError('Camera eligibility differs')
            model = row['model_output']
            if corrected is None:
                totals[expected_reason] += 1
                if model is not None or row['processed_rgb_sha256'] is not None:
                    raise ValueError('Unavailable input contains candidate model evidence')
                continue
            if model is None or not (pixel_hash(corrected) == row['processed_rgb_sha256'] == model['source_rgb_sha256']):
                raise ValueError('Corrected input/model identity differs')
            if model['source_frame'] != index or model['pts'] != f.pts or model['time_base'] != str(f.time_base):
                raise ValueError('Candidate model native identity differs')
            if row['hardware_action'] != 'none' or model['hardware_command'] != 'none':
                raise ValueError('Unexpected action')
            totals['paired_available'] += 1
            for previous, counter in [(original, baseline), (model, proposed)]:
                counter.update(dict(qa_pass=int(previous['measurement']['measurement_valid']),
                    liquid_proposed=int(any(p['class_id'] == 1 for p in previous['predictions'])),
                    mask_nonempty=int(bool(previous['measurement']['geometry']['mask_pixels']))))
            if original['measurement']['measurement_valid'] != model['measurement']['measurement_valid']:
                differences.append(dict(source_frame=index, seconds=seconds,
                    raw_qa=bool(original['measurement']['measurement_valid']), corrected_qa=bool(model['measurement']['measurement_valid'])))
            left = decode_mask(original['measurement']['geometry']['mask']).astype(bool)
            right = decode_mask(model['measurement']['geometry']['mask']).astype(bool)
            union = int((left | right).sum())
            if union:
                ious.append(float((left & right).sum()/union))
            if index in selected:
                name = f'qa-difference-frame{index:06d}.png'
                image = np.concatenate([panel(rgb, original, f'RAW BASELINE | {seconds:.1f}s'),
                    panel(corrected, model, f'CAMERA-CORRECTED INPUT | {seconds:.1f}s')], axis=1)
                if not cv2.imwrite(str(a.out/name), image):
                    raise ValueError('PNG export failed')
                rendered.append(dict(path=name, sha256=sha(a.out/name), source_frame=index))
    if dict(totals) != summary['counts'] or frames != summary['native_decoded_frames']:
        raise ValueError('Frame/coverage totals differ')
    if dict(baseline) != summary['raw_on_camera_eligible_frames'] or dict(proposed) != summary['corrected_on_camera_eligible_frames']:
        raise ValueError('QA/model counts differ')
    if differences != summary['qa_disagreements'] or len(rendered) != len(selected):
        raise ValueError('QA differences differ')
    if len(ious) != summary['paired_nonempty_union_mask_iou']['samples'] or min(ious) != summary['paired_nonempty_union_mask_iou']['minimum'] or float(np.median(ious)) != summary['paired_nonempty_union_mask_iou']['median']:
        raise ValueError('Paired mask agreement differs')
    if sha(a.source) != plan['source_sha256']:
        raise ValueError('Source changed')
    write(a.out/'verification.json', dict(status='passed', source_sha256=sha(a.source),
        candidate_completion_sha256=sha(a.input/'completed.json'), native_frames=frames,
        independently_verified_raw_rgb_clock_and_correction=len(rows), recomputed_counts=dict(totals),
        qa_gains=sum(r['corrected_qa'] for r in differences), qa_losses=sum(r['raw_qa'] for r in differences),
        difference_pngs=rendered, run5_used=False, hardware_actions=0,
        approved_onset_labels=0, approved_masks=0, helper_sha256=sha(__file__)))
    print(json.dumps(dict(status='passed', counts=dict(totals), difference_pngs=len(rendered))))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'baseline', 'input', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    main(p.parse_args())
