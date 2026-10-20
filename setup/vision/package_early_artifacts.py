"""Read completed artifacts on the training node; reject incomplete CSVs/weights."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import socket
import tarfile


ARMS = ('yolo11n-960', 'yolo11s-640', 'fasterrcnn-resnet50')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def package(root, out):
    if out.exists():
        raise ValueError('Archive exists; verify it rather than overwrite')
    contents = {}
    prepared = json.loads((root/'preparation.json').read_text())
    for name, expected in prepared.items():
        raw = (root/name).read_bytes()
        if digest(raw) != expected:
            raise ValueError('Frozen source/plan changed: '+name)
        contents[name] = raw
    contents['preparation.json'] = (root/'preparation.json').read_bytes()
    for arm in ARMS:
        prefix = 'models/'+arm+'/'
        model_raw = (root/prefix/'model.json').read_bytes()
        model = json.loads(model_raw)
        weights = (root/prefix/'best.pt').read_bytes()
        if model['checkpoint_sha256'] != digest(weights) or model['excluded_runs'] != [5]:
            raise ValueError('Incomplete/mismatched checkpoint: '+arm)
        curve_name = 'training.csv' if arm == 'fasterrcnn-resnet50' else 'train/results.csv'
        curve = (root/prefix/curve_name).read_bytes()
        if b'\0' in curve:
            raise ValueError('NUL in completed training CSV: '+arm)
        rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(curve.decode().splitlines())]
        key = 'run4_liquid_ap50_95' if arm == 'fasterrcnn-resnet50' else 'metrics/mAP50-95(B)'
        best = max(rows, key=lambda r: float(r[key]))
        if (len(rows) != model['epochs_completed'] or int(best['epoch']) != model['best_run4_epoch']
                or abs(float(best[key])-model['best_run4_liquid_ap50_95']) > 1e-8):
            raise ValueError('Training curve disagrees with model metadata: '+arm)
        for name, raw in [('best.pt', weights), ('model.json', model_raw), (curve_name, curve)]:
            contents[prefix+name] = raw
        extra = 'model-check.json' if arm == 'fasterrcnn-resnet50' else 'train/args.yaml'
        contents[prefix+extra] = (root/prefix/extra).read_bytes()
        print(arm, 'verified', len(rows), 'epochs; selected', model['best_run4_epoch'], flush=True)
    for name in ('job-id.txt', 'scheduler-status.txt'):
        if (root/name).exists():
            contents[name] = (root/name).read_bytes()
    for path in root.glob('slurm-286291_*.log'):
        raw = path.read_bytes()
        if b'\0' in raw:
            raise ValueError('NUL in completed log: '+path.name)
        contents[path.name] = raw
    evidence = dict(schema_version=1, hostname=socket.gethostname(),
        method='Read original completed files; validate raw bytes before packaging; no sanitization',
        source_sha256={name: digest(raw) for name, raw in contents.items()})
    contents['collection-read.json'] = (json.dumps(evidence, indent=2)+'\n').encode()
    partial = out.with_suffix(out.suffix+'.partial')
    with tarfile.open(partial, 'w:gz') as archive:
        for name, raw in contents.items():
            info = tarfile.TarInfo(name); info.size = len(raw)
            archive.addfile(info, io.BytesIO(raw))
    partial.rename(out)
    Path(str(out)+'.sha256').write_text(digest(out.read_bytes())+'  '+out.name+'\n')
    print(out.name, digest(out.read_bytes()), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); package(a.experiment, a.out)
