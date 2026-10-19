"""Bounded bootstrap experiment from an existing Jev box export; no renderer.

Run 5 is excluded from training, development and augmentation. These are weak
labels, not reviewed masks. The replay consumes only the exported ONNX model.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil


def prepare(source, output):
    import cv2
    import numpy as np
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest_bytes = (source / 'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    gallery = {r['id']: r for r in json.loads((source / 'gallery.json').read_text()) if r['camera'] == 'vial_fill'}
    vial = [r for r in manifest if r['camera'] == 'vial_fill']
    held = {r['source_sha256'] for r in vial if r['video_id'] == '5'}
    dev = {r['source_sha256'] for r in vial if r['video_id'] == '4'}
    rows, seen = [], set()
    for r in vial:
        run = int(r['video_id'])
        if run == 5 or r['source_sha256'] in held:
            continue
        split = 'val' if run == 4 else 'train'
        if split == 'train' and r['source_sha256'] in dev:
            continue
        g = gallery[r['id']]
        if g['status'] != 'review_candidates' or not g['boxes'] or r['source_sha256'] in seen:
            continue  # Manual/API/empty cases are unknown, not negative truth.
        seen.add(r['source_sha256'])
        image = source / 'review-pack/vial_fill/images' / r['source']
        label = source / 'review-pack/vial_fill/labels' / (r['id'] + '.txt')
        lines = label.read_text().strip().splitlines()
        for line in lines:
            values = list(map(float, line.split()))
            if len(values) != 5 or values[0] not in (0, 1) or not all(0 <= v <= 1 for v in values[1:]):
                raise ValueError(f'Invalid source box: {label}')
        if hashlib.sha256(image.read_bytes()).hexdigest() != r['source_sha256']:
            raise ValueError(f'Source changed: {image}')
        for folder in ('images', 'labels'):
            (output / folder / split).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(image, output / 'images' / split / image.name)
        shutil.copyfile(label, output / 'labels' / split / label.name)
        row = dict(source_id=r['id'], source_run=run, source_sha256=r['source_sha256'], split=split,
                   label_sha256=hashlib.sha256(label.read_bytes()).hexdigest(), label_status='unreviewed_jev_box',
                   image=f'images/{split}/{image.name}', labels=f'labels/{split}/{label.name}')
        rows.append(row)
        if split != 'train':
            continue
        # One exactly label-preserving translation per training image. This is
        # an augmentation, not a physical fluid simulation or new ground truth.
        seed = int(r['source_sha256'][:8], 16)
        dx = (-18, 18)[seed % 2]
        dy = (-5, 5)[seed // 2 % 2]
        rgb = cv2.imread(str(image))
        h, w = rgb.shape[:2]
        translated = cv2.warpAffine(rgb, np.float32([[1, 0, dx], [0, 1, dy]]), (w, h), borderMode=cv2.BORDER_REFLECT_101)
        transformed = []
        for line in lines:
            c, x, y, bw, bh = map(float, line.split())
            # Retain only transforms whose complete original boxes stay inside.
            if not (0 <= x + dx / w - bw / 2 < x + dx / w + bw / 2 <= 1 and
                    0 <= y + dy / h - bh / 2 < y + dy / h + bh / 2 <= 1):
                transformed = []
                break
            transformed.append(f'{int(c)} {x+dx/w:.8f} {y+dy/h:.8f} {bw:.8f} {bh:.8f}')
        if transformed:
            name = r['id'] + '-shift'
            cv2.imwrite(str(output / 'images/train' / (name + '.jpg')), translated)
            (output / 'labels/train' / (name + '.txt')).write_text('\n'.join(transformed) + '\n')
            rows.append(dict(row, source_id=name, parent_id=r['id'], augmentation=dict(kind='translation', dx=dx, dy=dy),
                             image=f'images/train/{name}.jpg', labels=f'labels/train/{name}.txt',
                             label_status='transformed_unreviewed_jev_box'))
    (output / 'data.yaml').write_text(f'path: {output.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: vial\n  1: colored_liquid\n')
    document = dict(schema_version=1, source_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
                    status='experimental_weak_supervision', reviewed_masks=False, held_out_runs=[5],
                    split_policy='Runs 1-3 train; Run 4 development; Run 5 excluded, including exact duplicate images',
                    counts=dict(Counter(r['split'] for r in rows)), samples=rows)
    (output / 'provenance.json').write_text(json.dumps(document, indent=2))
    print(json.dumps({k: v for k, v in document.items() if k != 'samples'}), flush=True)


def train(dataset, output, epochs, device):
    import torch
    import ultralytics
    from ultralytics import YOLO
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    # Absolute path avoids an account-wide Ultralytics weights_dir belonging
    # to another project. AMP checks download another model, so disable them.
    from urllib.request import urlretrieve
    pretrained = output / 'yolo11n.pt'
    urlretrieve('https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt',pretrained)
    model = YOLO(str(pretrained))
    model.train(data=str(Path(dataset) / 'data.yaml'), epochs=epochs, imgsz=640, batch=16,
                device=device, workers=0, seed=19, deterministic=True, project=str(output), name='train',
                exist_ok=False, patience=epochs, pretrained=True, plots=True, amp=False,
                hsv_h=0, hsv_s=0, hsv_v=.1, degrees=2, translate=.04, scale=.1,
                fliplr=0, flipud=0, mosaic=0, mixup=0, copy_paste=0)
    best = YOLO(str(output / 'train/weights/best.pt'))
    exported = Path(best.export(format='onnx', imgsz=640, opset=17, simplify=False, dynamic=False, nms=False))
    shutil.copyfile(exported, output / 'detector.onnx')
    info = dict(schema_version=1, status='experimental_weak_supervision', architecture='yolo11n_detect',
                classes=['vial', 'colored_liquid'], model_sha256=hashlib.sha256((output / 'detector.onnx').read_bytes()).hexdigest(),
                source_provenance_sha256=hashlib.sha256((Path(dataset)/'provenance.json').read_bytes()).hexdigest(),
                train_runs=[1,2,3], development_runs=[4], held_out_runs=[5], epochs=epochs,
                torch=torch.__version__, ultralytics=ultralytics.__version__, trained_on_reviewed_masks=False,
                pretrained_sha256=hashlib.sha256(pretrained.read_bytes()).hexdigest(),
                note='Box model only; mask extraction is separate. Development agreement with Jev is not segmentation accuracy.')
    (output / 'model.json').write_text(json.dumps(info, indent=2))
    print(json.dumps(info), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['prepare', 'train'])
    p.add_argument('--source', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--device', default='0')
    a = p.parse_args()
    if a.source is None:
        p.error('--source is required (existing Jev output or prepared dataset)')
    if a.command == 'prepare':
        prepare(a.source, a.out)
    else:
        train(a.source, a.out, a.epochs, a.device)


if __name__ == '__main__':
    main()
