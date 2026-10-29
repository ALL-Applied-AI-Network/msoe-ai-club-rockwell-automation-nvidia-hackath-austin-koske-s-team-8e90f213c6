"""Recompute artifact integrity and descriptive metrics for the NVIDIA pilot."""
import argparse
import json
from pathlib import Path
import shutil

from liquid_specialists_data import read, write, sha


def check_artifacts(folder, completed):
    count = 0
    for name, expected in completed['artifacts'].items():
        if sha(folder/name) != expected:
            raise ValueError(f'Artifact changed: {folder/name}')
        count += 1
    return count


def main(a):
    import av
    import cv2
    import numpy as np
    from benchmark_vial_tensorrt import compare
    from cosmos_event_review import parse_review, load_clip
    helper_snapshot = a.out.with_name(a.out.stem+'-helper.py')
    if a.out.exists() or helper_snapshot.exists():
        raise ValueError('Use a fresh verification path')
    cases = a.root/'event-cases-v1'
    trt = a.root/'tensorrt-fp16-v1'
    reviews = a.root/a.reviews
    media = a.root/'cosmos-demo-v1'
    manifest = read(cases/'manifest.json')
    benchmark = read(trt/'completed.json')
    cosmos = read(reviews/'completed.json')
    captions = a.root/'cosmos-caption-v1'
    caption_done = read(captions/'completed.json')
    demo = read(media/'completed.json')
    if sha(cases/'manifest.json') != read(reviews/'plan.json')['manifest_sha256'] or sha(cases/'manifest.json') != read(captions/'plan.json')['cases_sha256']:
        raise ValueError('Case manifest identity differs')
    if sha(captions/'completed.json') != demo['reviews_sha256']:
        raise ValueError('Demo uses different review evidence')
    artifacts = sum(check_artifacts(folder, done) for folder, done in
                    ((trt, benchmark), (reviews, cosmos), (captions, caption_done), (media, demo)))
    clip_frames = 0
    checked_samples = 0
    for case in manifest['cases']:
        artifacts += check_artifacts(cases/case['id'], case)
        with av.open(str(cases/case['video'])) as c:
            decoded = sum(1 for _ in c.decode(video=0))
        if decoded != case['decoded_frames']:
            raise ValueError('Clip decode count differs')
        clip_frames += decoded
        checked_samples += case['source_checked_samples']
    records = read(trt/'measurements.json')
    calculated = {}
    for arm in ('pytorch', 'tensorrt'):
        rr = [r for r in records if r['arm'] == arm]
        calculated[arm] = {'samples': len(rr),
            'host_ms_p50': float(np.median([r['host_end_to_end_ms'] for r in rr])),
            'host_ms_p95': float(np.percentile([r['host_end_to_end_ms'] for r in rr], 95)),
            'reported_inference_ms_p50': float(np.median([r['reported_speed_ms']['inference'] for r in rr]))}
        # The first frozen runner reused 'tensorrt' for its version string,
        # overwriting only that summary field. Preserve it and recompute from
        # hash-verified original per-frame timings; never rewrite the evidence.
        expected = benchmark[arm]
        if isinstance(expected, dict) and calculated[arm] != expected:
            raise ValueError('Benchmark aggregate differs')
        if not isinstance(expected, dict) and not (arm == 'tensorrt' and expected == '10.14.1.48.post1'):
            raise ValueError('Unexpected benchmark summary format')
    keyed = {(r['arm'], r['repeat'], r['source_frame']): r for r in records}
    parity = read(trt/'parity.json')
    for row in parity:
        found = compare(keyed['pytorch', row['repeat'], row['source_frame']]['boxes'],
                        keyed['tensorrt', row['repeat'], row['source_frame']]['boxes'])
        if any(row[k] != v for k, v in found.items()):
            raise ValueError('Box parity differs')
    matches = [p for row in parity for p in row['pairs']]
    aggregate = {'frame_pairs': len(parity), 'matched_boxes_iou50': len(matches),
        'unmatched_torch': sum(row['unmatched_torch'] for row in parity),
        'unmatched_tensorrt': sum(row['unmatched_tensorrt'] for row in parity),
        'iou_min': min((p['iou'] for p in matches), default=None),
        'iou_median': float(np.median([p['iou'] for p in matches])) if matches else None,
        'max_confidence_abs_difference': max((p['confidence_abs_difference'] for p in matches), default=None)}
    if aggregate != benchmark['parity']:
        raise ValueError('Box agreement summary differs')
    responses = []
    for case in manifest['cases']:
        _, _, trace = load_clip(cases/case['video'], 8)
        for mode in ('video_only', 'with_telemetry'):
            record = read(reviews/f'{case["id"]}-{mode}.json')
            value, error = parse_review(record['response'])
            if value != record['parsed'] or error != record['contract_error']:
                raise ValueError('Explanation contract differs')
            if record['video_sha256'] != sha(cases/case['video']) or record['hardware_command'] != 'none':
                raise ValueError('Unexpected review source/action')
            if record['sampled_input_frames'] != trace:
                raise ValueError('VLM sampled pixels or timestamps differ')
            responses.append(record)
    if len(responses) != cosmos['responses'] or sum(r['contract_error'] is None for r in responses) != cosmos['contract_passes']:
        raise ValueError('Cosmos counts differ')
    with av.open(str(media/'cosmos-event-review.mp4')) as c:
        demo_frames = sum(1 for _ in c.decode(video=0))
    if demo_frames != demo['decoded_frames']:
        raise ValueError('Demo decode count differs')
    expected_trace = []
    for case in manifest['cases']:
        for frame in read(cases/case['id']/'source-frames.json'):
            if frame['source_frame'] % 6 == 0:
                expected_trace.append({'output_frame': len(expected_trace), 'case': case['id'],
                    'source_frame': frame['source_frame'], 'source_rgb_sha256': frame['source_rgb_sha256'],
                    'pts': frame['pts'], 'time_base': frame['time_base']})
    if read(media/'source-trace.json') != expected_trace:
        raise ValueError('Demo source trace differs')
    mode_metrics = {}
    for mode in ('video_only', 'with_telemetry'):
        rr = [r for r in responses if r['mode'] == mode]
        mode_metrics[mode] = {'responses': len(rr), 'contract_passes': sum(r['contract_error'] is None for r in rr),
            'seconds_p50': float(np.median([r['seconds'] for r in rr])),
            'seconds_p95': float(np.percentile([r['seconds'] for r in rr], 95)),
            'peak_allocated_gib': max(r['peak_allocated_bytes'] for r in rr)/(1024**3),
            'token_limit_hits': sum(r['generation_hit_token_limit'] for r in rr)}
    caption_metrics = {}
    for mode in ('full_frame', 'station_crop'):
        rr = []
        for case in manifest['cases']:
            record = read(captions/f'{case["id"]}-{mode}.json')
            pixels, _, trace = load_clip(cases/case['video'], 8)
            if mode == 'station_crop':
                pixels = np.stack([cv2.resize(im[50:460, 250:355], (192, 768), interpolation=cv2.INTER_CUBIC) for im in pixels])
            digest = __import__('hashlib').sha256(pixels.tobytes()).hexdigest()
            if record['sampled_input_frames'] != trace or record['vlm_pixel_sha256'] != digest:
                raise ValueError('Caption input pixels differ')
            if record['status_origin'] != 'deterministic_recorded_evidence' or record['parsed']['quality_assessment'] != 'unavailable' or record['hardware_command'] != 'none':
                raise ValueError('Unexpected caption status/action')
            context = read(cases/case['context'])
            counts = __import__('re').search(r'passes (\d+)/(\d+) provisional', record['parsed']['operator_message'])
            if counts is None or list(map(int, counts.groups())) != [sum(s['sample_qa_passes'] for s in context['samples']), len(context['samples'])]:
                raise ValueError('Deterministic status counts differ from recorded evidence')
            rr.append(record)
        caption_metrics[mode] = {'responses': len(rr), 'seconds_p50': float(np.median([r['seconds'] for r in rr])),
            'seconds_p95': float(np.percentile([r['seconds'] for r in rr], 95)),
            'peak_allocated_gib': max(r['peak_allocated_bytes'] for r in rr)/(1024**3),
            'nonempty_captions': sum(bool(r['caption']) for r in rr),
            'token_limit_hits': sum(r['generation_hit_token_limit'] for r in rr),
            'semantic_accuracy': None}
    report = {'schema_version': 1, 'kind': 'nvidia_offline_pilot', 'artifacts_hash_verified': artifacts,
        'source_run': 4, 'run5_used': False, 'hardware_actions': 0,
        'event_clips': len(manifest['cases']), 'clip_decoded_frames': clip_frames,
        'source_samples_compared_with_frozen_evidence': checked_samples,
        'tensorrt': calculated, 'box_parity': benchmark['parity'],
        'original_summary_field_collision': isinstance(benchmark['tensorrt'], str),
        'baseline_pt_sha256': read(trt/'plan.json')['model_sha256'],
        'tensorrt_engine_sha256': benchmark['engine_sha256'],
        'environment_sha256': sha(a.root/'environment.json'),
        'verification_helper_sha256': sha(Path(__file__)),
        'vial_host_latency_ratio': calculated['pytorch']['host_ms_p50']/calculated['tensorrt']['host_ms_p50'],
        'vial_reported_inference_latency_ratio': calculated['pytorch']['reported_inference_ms_p50']/calculated['tensorrt']['reported_inference_ms_p50'],
        'cosmos_model': cosmos['model'], 'cosmos_revision': cosmos['revision'],
        'cosmos': mode_metrics, 'caption_probe': caption_metrics, 'demo_decoded_frames': demo_frames,
        'real_quality_accuracy': None, 'explanation_semantic_accuracy': None,
        'promoted': False, 'main_changed': False,
        'limits': ['Run 4 inspected development footage; no independent defect truth.',
                   'Contract compliance does not establish factual explanation accuracy.',
                   'VLM explanations are retrospective and never drive the action coordinator.',
                   'TensorRT timings cover only the vial branch, not the complete liquid pipeline.',
                   'Event clips are lossy MP4 exports; numerical telemetry is from original frozen source pixels.'],
        'input_markers': {str(folder.name): sha(folder/'completed.json') for folder in (trt, reviews, captions, media)}}
    helper_snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(__file__, helper_snapshot)
    write(a.out, report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--reviews', default='cosmos-review-v1')
    p.add_argument('--out', type=Path, required=True)
    main(p.parse_args())
