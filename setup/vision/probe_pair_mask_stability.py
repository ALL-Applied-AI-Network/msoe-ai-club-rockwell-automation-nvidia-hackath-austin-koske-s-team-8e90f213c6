"""Synthetic-only stress tests of frozen GB10 pool-mask/presence models."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def write(p, v): Path(p).write_text(json.dumps(v, indent=2, allow_nan=False) + '\n')


def main(a):
    import fcntl
    import os
    a.out.mkdir(parents=True, exist_ok=False)
    write(a.out / 'queued.json', dict(pid=os.getpid(), queued=time.time()))
    conditions = ['raw', 'current_exposure085', 'current_exposure070', 'current_exposure130',
                  'current_green085', 'reference_shift2', 'reference_shift5', 'other_empty_reference']
    write(a.out / 'plan.json', dict(conditions=conditions, helper_sha256=sha(__file__),
        synthetic_only=True, real_inputs=False, fitting=False, selection=False,
        interpretation='Geometric body supervision. Sensitivity tests, not optical visibility or physical accuracy.'))
    lock = (Path.home() / 'ai-club-tools/spark-gpu.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX)
    import torch
    sys.path.insert(0, str(a.helper.parent))
    from train_pair_masks import MODES, build_model, data, batch, features
    torch.set_num_threads(4)
    assert (a.models / 'completed.json').exists()
    val = [r for r in data(a.bundle) if r['split'] == 'val']
    empty = [r for r in val if r['state'] == 'empty']
    summary = {}
    for mode in MODES:
        root = a.models / mode
        meta = read(root / 'frozen-model.json')
        assert sha(a.helper) == meta['helper_sha256']
        assert not meta['training_real_runs'] and not meta['selection_real_runs'] and not meta['run5_used']
        assert sha(root / 'best.pt') == meta['checkpoint_sha256']
        model = build_model(3 if mode == 'single_rgb' else 6).cuda().eval()
        model.load_state_dict(torch.load(root / 'best.pt', map_location='cpu', weights_only=True)['model_state'])
        summary[mode] = dict(metadata=meta, conditions={})
        for condition in conditions:
            records = []
            with torch.inference_mode():
                for start in range(0, len(val), 8):
                    rows = val[start:start+8]
                    v = batch(rows, 'cuda')
                    if condition.startswith('current_exposure'):
                        gain = int(condition[-3:]) / 100
                        v['current'] = (v['current'] * gain).clamp(0, 1)
                        v['current_anchor'] = (v['current_anchor'] * gain).clamp(0, 1)
                    if condition == 'current_green085':
                        v['current'][:, 1] *= .85
                        v['current_anchor'][:, 1] *= .85
                    if condition.startswith('reference_shift'):
                        shift = int(condition[-1])
                        v['reference'] = torch.roll(v['reference'], (0, shift), (-2, -1))
                    if condition == 'other_empty_reference':
                        # Deliberately mismatched pose/illumination family; disjoint from each current group.
                        refs = [next(e for e in empty if e['group'] != row['group']) for row in rows]
                        v['reference'] = torch.stack([r['current'] for r in refs]).cuda()
                        v['reference_anchor'] = torch.stack([r['current_anchor'] for r in refs]).cuda()
                    logits, p = model(features(v['current'], v['reference'], v['current_anchor'], v['reference_anchor'], mode))
                    masks, truth, presence = logits.sigmoid() >= .5, v['mask'] > 0, p.sigmoid()
                    for i, row in enumerate(rows):
                        positive = bool(truth[i].any())
                        inter = (masks[i] & truth[i]).sum().item()
                        union = (masks[i] | truth[i]).sum().item()
                        records.append(dict(id=row['id'], state=row['state'], positive=positive,
                            presence=float(presence[i]), mask_pixels=int(masks[i].sum()),
                            geometric_iou=inter / union if union else 1.))
            positives = [r for r in records if r['positive']]
            negatives = [r for r in records if not r['positive']]
            metrics = dict(images=len(records), positive_images=len(positives), negative_images=len(negatives),
                positive_presence_detected=sum(r['presence'] >= .5 for r in positives),
                negative_presence_alarms=sum(r['presence'] >= .5 for r in negatives),
                negative_nonempty_masks=sum(r['mask_pixels'] > 0 for r in negatives),
                positive_mean_geometric_iou=sum(r['geometric_iou'] for r in positives) / len(positives))
            write(a.out / (mode + '-' + condition + '.json'), dict(metrics=metrics, records=records))
            summary[mode]['conditions'][condition] = metrics
        del model
        torch.cuda.empty_cache()
    write(a.out / 'comparison.json', summary)
    write(a.out / 'completed.json', dict(local_gpu_used=False, real_inputs=False, run5_used=False,
        artifacts={p.name: sha(p) for p in a.out.iterdir() if p.is_file()}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['models', 'helper', 'bundle', 'out']:
        p.add_argument('--' + name, type=Path, required=True)
    main(p.parse_args())
