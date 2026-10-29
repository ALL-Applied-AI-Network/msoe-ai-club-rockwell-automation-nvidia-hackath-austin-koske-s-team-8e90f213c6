"""Architecture/resolution pilots for visible liquid; only Runs 1-4 are consumed."""
import argparse
import csv
import json
from pathlib import Path
import platform
import random
import shutil
import time

from liquid_specialists_data import check, read, sha, write, yaml


ARMS = {
    'yolo11n-960': dict(family='yolo11n', initialization='yolo11n.pt', imgsz=960, batch=16),
    'yolo11s-640': dict(family='yolo11s', initialization='yolo11s.pt', imgsz=640, batch=16),
    'fasterrcnn-resnet50': dict(family='fasterrcnn-resnet50-fpn', initialization='fasterrcnn-coco.pth',
                              min_size=480, max_size=640, batch=8),
}
HELPERS = ['train_early_liquid.py', 'train-early-liquid-rosie.sbatch', 'liquid_specialists_data.py',
           'train_liquid_specialists.py', 'train_epoch_augmentation.py', 'train_mixed_detector.py',
           'detector_photometric.py', 'dino_liquid.py']


def prepare(source, assets, out):
    if out.exists():
        raise ValueError('Use a fresh experiment root')
    original = read(source/'experiment-plan.json')
    provenance = check(source/'data/all')
    out.mkdir(parents=True)
    shutil.copytree(source/'data/all', out/'data')
    yaml(out/'data')
    (out/'pretrained').mkdir()
    weights = {}
    for name in ('yolo11n.pt', 'yolo11s.pt', 'fasterrcnn-coco.pth'):
        path = assets/name if name == 'yolo11s.pt' else source/'pretrained'/name
        if name in original['pretrained'] and sha(path) != original['pretrained'][name]:
            raise ValueError('Initialization changed')
        shutil.copyfile(path, out/'pretrained'/name)
        weights[name] = sha(path)
    heights = {split: [float(line.split()[4])*480 for r in provenance['samples'] if r['split'] == split
                      for line in (out/'data'/r['labels']).read_text().splitlines()]
               for split in ('train', 'val')}
    write(out/'experiment-plan.json', dict(schema_version=1, kind='early_liquid_architecture_pilots',
        arms=ARMS, seed=19, train_runs=[1, 2, 3], development_runs=[4], excluded_from_fit_and_selection=[5],
        source_experiment_plan_sha256=sha(source/'experiment-plan.json'),
        data_provenance_sha256=sha(out/'data/provenance.json'), pretrained=weights,
        max_epochs=50, patience=10, checkpoint_selection='Run 4 overall liquid AP50-95',
        evaluation_confidence=.35, nms_iou=.45, ap_confidence=.001,
        yolo_base=original['yolo_rtdetr'],
        fasterrcnn=dict(trainable_resnet_stages=['layer2', 'layer3', 'layer4'],
            trainable_fpn=True, predictor='new background/liquid', optimizer='AdamW', lr=.0001,
            weight_decay=.0001, scheduler='cosine50', online_augmentation=False,
            normalization='ImageNet; FrozenBatchNorm2d', coco_anchor_sizes=[32, 64, 128, 256, 512]),
        height_inventory={split: dict(targets=len(v), height_le_32=sum(h <= 32 for h in v),
            height_le_64=sum(h <= 64 for h in v)) for split, v in heights.items()},
        limits=['Existing colored_liquid targets are provisional and can combine stream and body.',
            'Short supplied targets are a geometry proxy, not reviewed first-visible-liquid or physical onset labels.',
            'No development target <=32px; tiny-pool performance cannot be established with this inventory.',
            'Architecture/resolution family pilots, not a matched optimization benchmark.',
            'No new labels, synthesis, Run 5 fitting/tuning, detector promotion or measurement QA changes.']))
    here = Path(__file__).resolve().parent
    for name in HELPERS:
        (out/name).write_bytes((here/name).read_bytes().replace(b'\r\n', b'\n'))
    write(out/'preparation.json', {name: sha(out/name) for name in [*HELPERS, 'experiment-plan.json']})
    print(json.dumps(read(out/'experiment-plan.json')), flush=True)


def validate(root):
    plan = read(root/'experiment-plan.json')
    if plan['arms'] != ARMS or plan['excluded_from_fit_and_selection'] != [5]:
        raise ValueError('Frozen plan changed')
    for name, expected in read(root/'preparation.json').items():
        if sha(root/name) != expected:
            raise ValueError('Executed helper/plan changed: '+name)
    for name, expected in plan['pretrained'].items():
        if sha(root/'pretrained'/name) != expected:
            raise ValueError('Initialization changed')
    check(root/'data')
    if sha(root/'data/provenance.json') != plan['data_provenance_sha256']:
        raise ValueError('Inventory changed')
    return plan


def build_resnet(root):
    import torch
    from torchvision.models.detection import FasterRCNN
    from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.ops.misc import FrozenBatchNorm2d
    backbone = resnet_fpn_backbone(backbone_name='resnet50', weights=None, trainable_layers=3,
                                  norm_layer=FrozenBatchNorm2d)
    model = FasterRCNN(backbone, num_classes=91, min_size=480, max_size=640,
                      box_score_thresh=.001, box_nms_thresh=.45, box_detections_per_img=300)
    model.load_state_dict(torch.load(root/'pretrained/fasterrcnn-coco.pth', map_location='cpu', weights_only=True))
    model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, 2)
    return model


def resnet_fit(root, out, smoke=False):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from train_liquid_specialists import Frames, development_score
    torch.manual_seed(19); np.random.seed(19); random.seed(19)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = build_resnet(root).cuda()
    frozen_before = model.backbone.body.conv1.weight.detach().clone()
    train = DataLoader(Frames(root/'data', 'train'), batch_size=8, shuffle=True, num_workers=0,
                       collate_fn=lambda b: tuple(zip(*b)))
    val = DataLoader(Frames(root/'data', 'val'), batch_size=8, num_workers=0,
                     collate_fn=lambda b: tuple(zip(*b)))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.0001, weight_decay=.0001)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    curve = []; best = -1.; selected = 0; started = time.monotonic()
    if smoke:
        model.train()
        image = Frames(root/'data', 'train')[0][0].cuda()
        empty = dict(boxes=torch.empty((0, 4), device='cuda'), labels=torch.empty(0, dtype=torch.int64, device='cuda'))
        assert all(torch.isfinite(v) for v in model([image], [empty]).values())
    for epoch in range(1, 2 if smoke else 51):
        model.train(); losses = []
        for step, (images, targets) in enumerate(train):
            images = [x.cuda() for x in images]
            targets = [{k: v.cuda() for k, v in t.items()} for t in targets]
            optimizer.zero_grad(set_to_none=True)
            loss = sum(model(images, targets).values())
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 10.)
            optimizer.step(); losses.append(float(loss.detach()))
            if smoke and step == 1:
                break
        ap = development_score(model, list(val)[:1] if smoke else val, 'cuda')
        schedule.step()
        curve.append(dict(epoch=epoch, train_loss=float(np.mean(losses)), run4_liquid_ap50_95=ap,
                          elapsed_s=time.monotonic()-started))
        with (out/'training.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(curve[0])); writer.writeheader(); writer.writerows(curve)
        if ap > best:
            best = ap; selected = epoch
            torch.save(dict(model_state={k: v.detach().cpu() for k, v in model.state_dict().items()}, epoch=epoch), out/'best.pt')
        print(json.dumps(curve[-1]), flush=True)
        if epoch-selected >= 10:
            break
    if not torch.equal(frozen_before, model.backbone.body.conv1.weight):
        raise ValueError('Frozen stem changed')
    restored = build_resnet(root).cuda().eval()
    restored.load_state_dict(torch.load(out/'best.pt', map_location='cpu', weights_only=True)['model_state'])
    if smoke:
        model.eval(); image = Frames(root/'data', 'val')[0][0].cuda()
        with torch.no_grad():
            left = model([image])[0]; right = restored([image])[0]
        if not all(torch.equal(left[k], right[k]) for k in ('boxes', 'scores', 'labels')):
            raise ValueError('Checkpoint round trip changed predictions')
    write(out/'model-check.json', dict(frozen_stem_unchanged=True, checkpoint_reloaded=True,
        exact_prediction_roundtrip_checked=smoke, smoke=smoke))
    return len(curve), selected, best


def train(root, arm, smoke=False):
    import torch, ultralytics
    if ultralytics.__version__ != '8.4.182':
        raise ValueError('Pinned Ultralytics version changed')
    plan = validate(root)
    out = root/('resnet-smoke' if smoke else 'models/'+arm)
    out.mkdir(parents=True, exist_ok=False)
    if arm == 'fasterrcnn-resnet50':
        epochs, selected, ap = resnet_fit(root, out, smoke)
    else:
        from ultralytics import YOLO
        cfg = ARMS[arm]
        data = out/'data'; shutil.copytree(root/'data', data); yaml(data)
        options = {**plan['yolo_base'], 'imgsz': cfg['imgsz'], 'batch': cfg['batch']}
        YOLO(root/'pretrained'/cfg['initialization']).train(data=str(data/'data.yaml'), epochs=50,
            patience=10, seed=19, device=0, project=str(out), name='train', exist_ok=False, plots=True, **options)
        shutil.copyfile(out/'train/weights/best.pt', out/'best.pt')
        rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader((out/'train/results.csv').open())]
        best = max(rows, key=lambda r: float(r['metrics/mAP50-95(B)']))
        epochs, selected, ap = len(rows), int(best['epoch']), float(best['metrics/mAP50-95(B)'])
    if not smoke:
        write(out/'model.json', dict(schema_version=1, arm=arm, **ARMS[arm],
            checkpoint_sha256=sha(out/'best.pt'), experiment_plan_sha256=sha(root/'experiment-plan.json'),
            preparation_sha256=sha(root/'preparation.json'), data_provenance_sha256=plan['data_provenance_sha256'],
            train_runs=[1, 2, 3], development_runs=[4], excluded_runs=[5], epochs_completed=epochs,
            best_run4_epoch=selected, best_run4_liquid_ap50_95=ap,
            torch=torch.__version__, ultralytics=ultralytics.__version__, python=platform.python_version(),
            gpu=torch.cuda.get_device_name(0), provisional_targets=True, detector_promoted=False))
    validate(root)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['prepare', 'train', 'smoke-resnet'])
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--source', type=Path); p.add_argument('--assets', type=Path)
    p.add_argument('--arm', choices=list(ARMS))
    a = p.parse_args()
    if a.command == 'prepare':
        prepare(a.source, a.assets, a.out)
    else:
        train(a.out, 'fasterrcnn-resnet50' if a.command == 'smoke-resnet' else a.arm, a.command == 'smoke-resnet')
