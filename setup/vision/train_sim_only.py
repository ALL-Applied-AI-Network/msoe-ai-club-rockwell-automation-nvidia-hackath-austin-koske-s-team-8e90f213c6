"""Supplemental synthetic-only fine-tuning; fixed final checkpoint, no real selection."""
import argparse
import csv
from pathlib import Path
import random
import shutil

from train_mixed_detector import read, write, sha, validate_split


def create_plan(root):
    target = root/'sim-only-plan.json'
    if target.exists(): raise ValueError('Sim-only plan already frozen')
    base = read(root/'experiment-plan.json')
    write(target, dict(schema_version=1, arm='sim-only', base_plan_sha256=sha(root/'experiment-plan.json'),
        architecture=base['architecture'], seed=base['seed'], epochs=base['epochs'],
        pretrained_sha256=base['pretrained_sha256'], expected_unique_images=base['expected_synthetic_images'],
        training_images=2210, checkpoint_selection=f'Fixed final epoch {base["epochs"]}; no real checkpoint selection',
        training_real_runs=[], checkpoint_selection_real_runs=[], real_data_used_for_fit_or_selection=False,
        validation='Training-set synthetic resubstitution only; val=False during fitting; not held-out accuracy',
        source_scope=base['target_scope'], preprocessing=base['preprocessing'],
        generic_real_pretraining='Same COCO initialization as other arms; sim-only refers to CSI domain fine-tuning',
        authorization='Austin requested an additional exclusively simulated transfer trial on 2026-10-19',
        limits=['Added after real-only control diagnostic was inspected; not a new blind test.',
                'No real CSI pixels/labels used in fine-tuning or checkpoint selection.',
                'Training images repeated to match the other arms optimizer-step budget.',
                'Geometric proxy targets; no optical calibration or detector promotion.']))


def validate_rows(rows, unique, budget):
    if len(rows)!=budget or any(r['origin']!='synthetic' or r['source_run'] is not None or r['split']!='train' for r in rows):
        raise ValueError('Sim-only data must contain only synthetic training rows')
    validate_split(rows,set())
    if sum(not r.get('parent_id') for r in rows)!=unique:
        raise ValueError('Sim-only unique source count changed')


def prepare(root):
    plan=read(root/'sim-only-plan.json')
    if sha(root/'experiment-plan.json')!=plan['base_plan_sha256']: raise ValueError('Base plan changed')
    source=root/'data/mixed'; p=read(source/'provenance.json')
    original=[dict(r) for r in p['samples'] if r['origin']=='synthetic']
    if len(original)!=plan['expected_unique_images'] or any(r.get('parent_id') for r in original):
        raise ValueError('Expected the verified original synthetic batch')
    rows=[dict(r,donor_image=r['image'],donor_labels=r['labels']) for r in original]
    rng=random.Random(plan['seed'])
    for i in range(plan['training_images']-len(original)):
        donor=rng.choice(original); name=f'sim-repeat-{i:06d}'
        rows.append(dict(donor,source_id=name,parent_id=donor['source_id'],
            image=f'images/train/{name}.png',labels=f'labels/train/{name}.txt',
            donor_image=donor['image'],donor_labels=donor['labels']))
    validate_rows(rows,plan['expected_unique_images'],plan['training_images'])
    data=root/'data/sim-only';data.mkdir(parents=True,exist_ok=False)
    write(data/'provenance.json',dict(schema_version=1,arm='sim-only',samples=rows,
        plan_sha256=sha(root/'sim-only-plan.json'),source_provenance_sha256=sha(source/'provenance.json'),
        render_manifest_sha256=p['render_manifest_sha256'],render_verification_sha256=p['render_verification_sha256'],
        unique_synthetic_images=len(original),training_images=len(rows),real_pixels_or_labels_copied=False,
        source_metadata_note='Reads mixed inventory metadata, then selects synthetic rows only; no real image/label access.'))


def verify_files(data, rows):
    for row in rows:
        if sha(data/row['image'])!=row['image_sha256'] or sha(data/row['labels'])!=row['actual_label_sha256']:
            raise ValueError('Sim-only file changed')
    for folder,key in [('images','image'),('labels','labels')]:
        actual={p.relative_to(data).as_posix() for p in (data/folder).rglob('*') if p.is_file()}
        if actual!={row[key] for row in rows}:raise ValueError('Unlisted file in sim-only dataset')


def verify_epochs(path, epochs):
    with path.open(newline='',encoding='utf-8') as stream:
        completed=[int(row['epoch']) for row in csv.DictReader(stream)]
    if completed!=list(range(1,epochs+1)):
        raise ValueError('Sim-only fitting did not complete the fixed epoch budget')


def materialize(root):
    data=root/'data/sim-only'; p=read(data/'provenance.json');plan=read(root/'sim-only-plan.json')
    if p['plan_sha256']!=sha(root/'sim-only-plan.json'):raise ValueError('Sim-only plan changed')
    source=root/'data/mixed'
    if sha(source/'provenance.json')!=p['source_provenance_sha256']:raise ValueError('Synthetic source inventory changed')
    validate_rows(p['samples'],plan['expected_unique_images'],plan['training_images'])
    for row in p['samples']:
        for key,donor,hash_key in [('image','donor_image','image_sha256'),('labels','donor_labels','actual_label_sha256')]:
            original=source/row[donor];target=data/row[key]
            if sha(original)!=row[hash_key]:raise ValueError('Synthetic source changed')
            target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():shutil.copyfile(original,target)
    verify_files(data,p['samples'])


def train(root, pretrained):
    import torch
    import ultralytics
    from ultralytics import YOLO
    plan=read(root/'sim-only-plan.json');data=root/'data/sim-only';p=read(data/'provenance.json')
    if sha(root/'experiment-plan.json')!=plan['base_plan_sha256'] or p['plan_sha256']!=sha(root/'sim-only-plan.json'):
        raise ValueError('Frozen plans changed')
    if sha(pretrained)!=plan['pretrained_sha256']:raise ValueError('Initialization changed')
    materialize(root)
    # Ultralytics still performs a final validation. Its val source is synthetic
    # training data and the resulting best.pt is deliberately ignored.
    (data/'data.yaml').write_text(f'path: {data.resolve().as_posix()}\ntrain: images/train\nval: images/train\nnames:\n  0: vial\n  1: colored_liquid\n',encoding='utf-8')
    output=root/'models/sim-only';output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);model=YOLO(str(pretrained))
    model.train(data=str(data/'data.yaml'),epochs=plan['epochs'],imgsz=640,batch=16,device=0,workers=0,
        seed=plan['seed'],deterministic=True,project=str(output),name='train',exist_ok=False,patience=0,
        val=False,pretrained=True,plots=True,amp=False,hsv_h=0,hsv_s=0,hsv_v=.1,degrees=2,translate=.04,
        scale=.1,fliplr=0,flipud=0,mosaic=0,mixup=0,copy_paste=0)
    verify_epochs(output/'train/results.csv',plan['epochs'])
    checkpoint=output/'train/weights/last.pt'
    chosen=YOLO(str(checkpoint));exported=Path(chosen.export(format='onnx',imgsz=640,opset=17,simplify=False,dynamic=False,nms=False))
    shutil.copyfile(checkpoint,output/'checkpoint.pt');shutil.copyfile(exported,output/'detector.onnx')
    write(output/'model.json',dict(schema_version=1,arm='sim-only',architecture=plan['architecture'],
        classes=['vial','colored_liquid'],preprocessing=plan['preprocessing'],target_scope=plan['source_scope'],
        checkpoint_sha256=sha(output/'checkpoint.pt'),model_sha256=sha(output/'detector.onnx'),
        sim_only_plan_sha256=sha(root/'sim-only-plan.json'),base_plan_sha256=plan['base_plan_sha256'],
        data_provenance_sha256=sha(data/'provenance.json'),trainer_sha256=sha(__file__),
        shared_preparation_helper_sha256=sha(Path(__file__).with_name('train_mixed_detector.py')),
        pretrained_sha256=sha(pretrained),checkpoint_selection=plan['checkpoint_selection'],
        train_runs=[],development_runs=[],excluded_real_runs=[1,2,3,4,5],
        unique_synthetic_images=p['unique_synthetic_images'],training_images=p['training_images'],
        epochs=plan['epochs'],seed=plan['seed'],torch=torch.__version__,ultralytics=ultralytics.__version__,
        generic_real_pretraining=plan['generic_real_pretraining'],automatic_promotion=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['plan','prepare','materialize','train'])
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--pretrained',type=Path)
    args=parser.parse_args()
    if args.command=='plan':create_plan(args.out)
    elif args.command=='prepare':prepare(args.out)
    elif args.command=='materialize':materialize(args.out)
    else:train(args.out,args.pretrained)
