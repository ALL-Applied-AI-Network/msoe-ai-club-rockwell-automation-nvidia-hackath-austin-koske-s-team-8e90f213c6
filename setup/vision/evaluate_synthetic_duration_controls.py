"""Post-freeze GB10 diagnostics for predeclared epoch-3/final duration controls.

These real development scores never select a checkpoint. The primary checkpoint
is already fixed by synthetic AP. Queue behind the running suite's GPU lock.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def write(p, value):
    Path(p).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main(a):
    import fcntl
    import os
    a.out.mkdir(parents=True, exist_ok=False)
    write(a.out / 'queued.json', dict(pid=os.getpid(), queued=time.time()))
    write(a.out / 'plan.json', dict(arms=['original', 'paired'], controls=['epoch-03.pt', 'last.pt'],
        purpose='Fixed duration diagnostics; primary checkpoint remains selected on synthetic AP only.',
        helper_sha256=sha(__file__), training=False, selection_real_runs=[], run5_used=False))
    lock = (Path.home() / 'ai-club-tools/spark-gpu.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX)
    if not (a.suite / 'completed-long.json').exists():
        raise ValueError('Both fits must finish successfully before controls are scored')
    import torch
    torch.set_num_threads(4)
    summary = {}
    for arm in ['original', 'paired']:
        root = a.suite / ('long-' + arm + '-v1')
        sys.path.insert(0, str(root / 'helpers'))
        from train_synthetic_liquid_pilot import evaluate, validate
        from train_early_liquid import build_resnet
        plan = validate(root)
        completed = read(root / 'completed.json')
        for name, expected in completed['artifacts'].items():
            if sha(root / name) != expected:
                raise ValueError('Completed fit changed: ' + name)
        real_root = a.bundle / 'tools/results/real-synthetic-detector-20261019/data/real'
        real = []
        for row in read(real_root / 'provenance.json')['samples']:
            assert row['source_run'] == 4 and row['split'] == 'val'
            assert sha(real_root / row['image']) == row['image_sha256']
            real.append(dict(source_id=row['source_id'], image=str(real_root / row['image']),
                label='real-evaluation-labels/' + row['source_id'] + '.txt'))
        model = build_resnet(root).cuda()
        summary[arm] = dict(primary=read(root / 'comparison.json'), metadata=read(root / 'frozen-model.json'), controls={})
        for name in ['epoch-03.pt', 'last.pt']:
            checkpoint = root / name
            state = torch.load(checkpoint, map_location='cpu', weights_only=True)
            model.load_state_dict(state['model_state'])
            synthetic, _ = evaluate(model, [r for r in plan['samples'] if r['split'] == 'val'], root, 'cuda')
            agreement, predictions = evaluate(model, real, root, 'cuda')
            label = arm + '-' + name.removesuffix('.pt')
            write(a.out / (label + '-predictions.json'), predictions)
            report = dict(epoch=state['epoch'], checkpoint_sha256=sha(checkpoint), synthetic_development=synthetic,
                real_run4_machine_agreement=agreement, training=False, used_for_checkpoint_selection=False)
            write(a.out / (label + '.json'), report)
            summary[arm]['controls'][name] = report
            print(json.dumps(dict(arm=arm, checkpoint=name, **report)), flush=True)
        del model
        torch.cuda.empty_cache()
    write(a.out / 'comparison.json', summary)
    write(a.out / 'completed.json', dict(artifacts={p.name: sha(p) for p in a.out.iterdir() if p.is_file()},
        local_gpu_used=False, run5_used=False, approved_labels=False, checkpoint_selection_changed=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['suite', 'bundle', 'out']:
        parser.add_argument('--' + name, type=Path, required=True)
    main(parser.parse_args())
