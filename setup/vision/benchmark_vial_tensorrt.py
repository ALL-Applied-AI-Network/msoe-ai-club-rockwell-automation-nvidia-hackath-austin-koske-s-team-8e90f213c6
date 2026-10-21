"""Paired YOLO vial-only PyTorch/TensorRT benchmark on exact Run 4 source frames."""
import argparse
import json
from pathlib import Path
import shutil
import time

from liquid_specialists_data import read, write, sha

BASELINE_SHA = '10856c18a8a5c485e832adcc6b841b739e726a969dc043fe8db64c49163bb361'


def overlap(a, b):
    x, y = max(a[0], b[0]), max(a[1], b[1])
    r, t = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, r-x)*max(0, t-y)
    union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union > 0 else 0.


def compare(a, b):
    remaining = list(range(len(b)))
    pairs = []
    for x in a:
        if not remaining:
            break
        j = max(remaining, key=lambda i: overlap(x[:4], b[i][:4]))
        iou = overlap(x[:4], b[j][:4])
        if iou >= .5:
            remaining.remove(j)
            pairs.append({'iou': iou, 'confidence_abs_difference': abs(x[4]-b[j][4])})
    return {'torch_boxes': len(a), 'tensorrt_boxes': len(b), 'matches_iou50': len(pairs),
            'unmatched_torch': len(a)-len(pairs), 'unmatched_tensorrt': len(b)-len(pairs), 'pairs': pairs}


def main(a):
    import av
    import cv2
    import numpy as np
    import torch
    import tensorrt
    import ultralytics
    from ultralytics import YOLO
    if a.out.exists():
        raise ValueError('Use a fresh output directory')
    if sha(a.model) != BASELINE_SHA:
        raise ValueError('Wrong vial checkpoint')
    frozen = read(a.evidence/'plan.json')
    if a.source.name != 'vial_fill_video_4.mp4' or sha(a.source) != frozen['source_sha256']:
        raise ValueError('Wrong Run 4 source')
    done = read(a.evidence/'completed.json')
    if sha(a.evidence/'frames.jsonl') != done['artifacts']['frames.jsonl']:
        raise ValueError('Frozen evidence changed')
    rows = [json.loads(s) for s in (a.evidence/'frames.jsonl').read_text().splitlines()]
    chosen = [rows[i] for i in np.linspace(0, len(rows)-1, a.samples).astype(int)]
    wanted = {r['source_frame']: r for r in chosen}
    images, identities = [], []
    with av.open(str(a.source)) as c:
        for index, f in enumerate(c.decode(video=0)):
            if index not in wanted:
                continue
            rgb = f.to_ndarray(format='rgb24')
            digest = __import__('hashlib').sha256(rgb.tobytes()).hexdigest()
            row = wanted[index]
            if digest != row['source_rgb_sha256'] or f.pts != row['pts'] or str(f.time_base) != row['time_base']:
                raise ValueError('Source sample differs')
            images.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            identities.append({'source_frame': index, 'rgb_sha256': digest, 'pts': f.pts, 'time_base': str(f.time_base)})
    if len(images) != a.samples:
        raise ValueError('Incomplete sample set')
    a.out.mkdir(parents=True)
    shutil.copy2(__file__, a.out/Path(__file__).name)
    checkpoint = a.out/'vial-source.pt'
    shutil.copy2(a.model, checkpoint)
    options = dict(imgsz=[480, 640], rect=False, conf=.35, iou=.45, classes=[0],
                   max_det=100, device=0, verbose=False, batch=1)
    write(a.out/'plan.json', {'kind': 'vial_branch_only', 'source_sha256': sha(a.source),
        'model_sha256': BASELINE_SHA, 'precision': 'fp16' if a.half else 'fp32',
        'predict_options': options, 'warmup_per_arm': 20, 'passes': a.passes,
        'sample_identities': identities, 'run5_used': False, 'changes_deployed_defaults': False})
    model = YOLO(str(checkpoint))
    started = time.perf_counter()
    engine = Path(model.export(format='engine', imgsz=[480, 640], half=a.half,
        dynamic=False, batch=1, workspace=2, simplify=False, device=0, verbose=False))
    export_seconds = time.perf_counter()-started
    accelerated = YOLO(str(engine), task='detect')
    models = {'pytorch': model, 'tensorrt': accelerated}
    for m in models.values():
        for _ in range(20):
            m.predict(images[0], **options)
    records = []
    for repeat in range(a.passes):
        # Alternate order to reduce warm-up/thermal ordering effects.
        order = ('pytorch', 'tensorrt') if repeat % 2 == 0 else ('tensorrt', 'pytorch')
        for arm in order:
            for identity, image in zip(identities, images):
                torch.cuda.synchronize()
                start = time.perf_counter()
                result = models[arm].predict(image, **options)[0]
                boxes = result.boxes.data.cpu().numpy().tolist()
                torch.cuda.synchronize()
                elapsed = (time.perf_counter()-start)*1000
                records.append({'arm': arm, 'repeat': repeat, 'source_frame': identity['source_frame'],
                                'host_end_to_end_ms': elapsed, 'reported_speed_ms': result.speed, 'boxes': boxes})
    write(a.out/'measurements.json', records)
    summary = {}
    for arm in models:
        subset = [r for r in records if r['arm'] == arm]
        summary[arm] = {'samples': len(subset), 'host_ms_p50': float(np.median([r['host_end_to_end_ms'] for r in subset])),
            'host_ms_p95': float(np.percentile([r['host_end_to_end_ms'] for r in subset], 95)),
            'reported_inference_ms_p50': float(np.median([r['reported_speed_ms']['inference'] for r in subset]))}
    keyed = {(r['arm'], r['repeat'], r['source_frame']): r for r in records}
    parity = [dict(source_frame=i['source_frame'], repeat=repeat,
              **compare(keyed['pytorch', repeat, i['source_frame']]['boxes'],
                        keyed['tensorrt', repeat, i['source_frame']]['boxes']))
              for repeat in range(a.passes) for i in identities]
    write(a.out/'parity.json', parity)
    pairs = [p for r in parity for p in r['pairs']]
    summary['parity'] = {'frame_pairs': len(parity), 'matched_boxes_iou50': len(pairs),
        'unmatched_torch': sum(r['unmatched_torch'] for r in parity),
        'unmatched_tensorrt': sum(r['unmatched_tensorrt'] for r in parity),
        'iou_min': min((p['iou'] for p in pairs), default=None),
        'iou_median': float(np.median([p['iou'] for p in pairs])) if pairs else None,
        'max_confidence_abs_difference': max((p['confidence_abs_difference'] for p in pairs), default=None)}
    summary.update({'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
        'tensorrt_version': tensorrt.__version__, 'ultralytics': ultralytics.__version__,
        'export_seconds': export_seconds, 'engine_sha256': sha(engine),
        'scope': 'vial-only inference; excludes decode, liquid detector, SAM and whole-pipeline latency',
        'accuracy': None, 'promoted': False,
        'artifacts': {p.name: sha(p) for p in a.out.iterdir() if p.is_file()}})
    write(a.out/'completed.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'evidence', 'model', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--samples', type=int, default=96)
    p.add_argument('--passes', type=int, default=3)
    p.add_argument('--half', action='store_true')
    main(p.parse_args())
