"""Derived one-class datasets; original annotations are never edited or approved."""
from collections import Counter
from pathlib import Path
import random
import shutil

from train_epoch_augmentation import check as check_source, INITIAL, REAL, BASELINE, read, write, sha
from train_mixed_detector import validate_split

ARMS={'yolo-all':dict(family='yolo11n',dataset='all'),
      'yolo-focused':dict(family='yolo11n',dataset='focused'),
      'rtdetr-all':dict(family='rtdetr-l',dataset='all'),
      'dinov3-all':dict(family='dinov3-convnext-small-fasterrcnn',dataset='all')}
WEIGHTS={'yolo11n.pt':INITIAL,
    'rtdetr-l.pt':'6de60b10d4bc566f00cda0f5b4d64afe4b66d48dc9695d2171effb7859d8e73f',
    'fasterrcnn-coco.pth':'258fb6c638b15964ddcdd1ae0748c5eef1be9e732750120cc857feed3faac384',
    'dinov3-convnext-small/model.safetensors':'8182492b73345ceb449ed6be0de486d778a73afb273ba2684de785a71ef1da9d',
    'dinov3-convnext-small/config.json':'71388aa45b2e6faa8be975655a16d87fd2e0f829bf2071ef1aacd0686cee3cdd',
    'dinov3-convnext-small/preprocessor_config.json':'960c41d1f3a7778b936365769a2d90550b318a6c0a53a0296957adacfe5e0dd7'}


def liquid_text(text):
    lines=[]
    for line in text.splitlines():
        fields=line.split()
        if len(fields)!=5 or fields[0] not in ('0','1'):raise ValueError('Unexpected source target')
        if fields[0]=='1':lines.append('0 '+' '.join(fields[1:]))
    return '\n'.join(lines)+ ('\n' if lines else '')


def select_units(original,positive):
    """721 paired units in both arms; ~80% positive in focused arm."""
    train=[r for r in original if r['split']=='train' and not r.get('parent_id')]
    positives=sorted([r for r in train if positive[r['source_id']]],key=lambda r:r['source_id'])
    negatives=sorted([r for r in train if not positive[r['source_id']]],key=lambda r:r['source_id'])
    if (len(train),len(positives),len(negatives))!=(721,354,367):raise ValueError('Frozen source balance changed')
    rng=random.Random(19);rng.shuffle(negatives)
    focused=positives+rng.choices(positives,k=222)+negatives[:145]
    rng.shuffle(focused)
    return train,focused


def yaml(data):
    (data/'data.yaml').write_text(f'path: {data.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: colored_liquid\n',encoding='utf-8')


def check(data):
    p=read(data/'provenance.json');validate_split(p['samples'],set(p['held_out_source_hashes']))
    if p['source_provenance_sha256']!=REAL or p['counts']!={'train':1442,'val':157}:raise ValueError('Derived ancestry/count changed')
    for r in p['samples']:
        if sha(data/r['image'])!=r['image_sha256'] or sha(data/r['labels'])!=r['actual_label_sha256']:
            raise ValueError('Derived image/label changed')
        if any(line.split()[0]!='0' for line in (data/r['labels']).read_text().splitlines()):raise ValueError('Unexpected derived class')
    return p


def prepare(source,assets,out):
    original=check_source(source);rows=original['samples']
    for name,expected in WEIGHTS.items():
        if sha(assets/name)!=expected:raise ValueError('Pretrained asset mismatch: '+name)
    out.mkdir(parents=True,exist_ok=False);shutil.copytree(assets,out/'pretrained')
    positive={r['source_id']:bool(liquid_text((source/r['labels']).read_text())) for r in rows}
    units,focused=select_units(rows,positive);children={r['parent_id']:r for r in rows if r.get('parent_id')}
    summary={}
    for arm,selected in [('all',units),('focused',focused)]:
        data=out/'data'/arm;new=[]
        donors=[]
        for i,donor in enumerate(selected):
            parent=f'{arm}-{i:04d}'
            donors.extend([(donor,parent,None),(children[donor['source_id']],parent+'-shift',parent)])
        donors.extend((r,r['source_id'],None) for r in rows if r['split']=='val')
        for donor,identity,parent in donors:
            split=donor['split'];image=f'images/{split}/{identity}.jpg';label=f'labels/{split}/{identity}.txt'
            for rel in (image,label):(data/rel).parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source/donor['image'],data/image)
            (data/label).write_text(liquid_text((source/donor['labels']).read_text()),encoding='utf-8')
            record=dict(source_id=identity,donor_id=donor['source_id'],source_run=donor['source_run'],
                source_sha256=donor['source_sha256'],image_sha256=sha(data/image),
                source_label_sha256=donor['actual_label_sha256'],actual_label_sha256=sha(data/label),
                split=split,origin='real',image=image,labels=label,positive=positive[donor['source_id']],
                label_status='derived_unreviewed_liquid_only',negative_status='provisional_label_absence_not_confirmed_empty')
            if parent:record['parent_id']=parent
            new.append(record)
        write(data/'provenance.json',dict(schema_version=1,source_provenance_sha256=REAL,
            samples=new,counts=dict(Counter(r['split'] for r in new)),held_out_source_hashes=original['held_out_source_hashes']))
        yaml(data);check(data)
        summary[arm]=dict(train_positive=sum(r['split']=='train' and r['positive'] for r in new),
            train_provisional_negative=sum(r['split']=='train' and not r['positive'] for r in new),
            val_positive=sum(r['split']=='val' and r['positive'] for r in new),
            val_provisional_negative=sum(r['split']=='val' and not r['positive'] for r in new),
            provenance_sha256=sha(data/'provenance.json'),unique_train_sources=len({r['source_sha256'] for r in new if r['split']=='train'}))
    # Validation stays identical after class remapping, independent of sampling arm.
    a=read(out/'data/all/provenance.json');b=read(out/'data/focused/provenance.json')
    if [r for r in a['samples'] if r['split']=='val']!=[r for r in b['samples'] if r['split']=='val']:raise ValueError('Validation differs')
    write(out/'experiment-plan.json',dict(schema_version=1,kind='liquid_specialists_and_non_yolo',arms=ARMS,
        seed=19,train_runs=[1,2,3],development_runs=[4],excluded_from_fit_and_selection=[5],
        data=summary,source_provenance_sha256=REAL,pretrained=WEIGHTS,baseline_onnx_sha256=BASELINE,
        max_epochs=50,patience=10,checkpoint_selection='Run 4 liquid AP50-95 only',
        additional_photometric=False,source_labels_changed=False,negative_review_status='provisional; Austin authorized existing vial-only labels',
        yolo_rtdetr=dict(imgsz=640,batch=16,amp=False,workers=0,deterministic=True,hsv_h=0,hsv_s=0,hsv_v=.1,
            degrees=2,translate=.04,scale=.1,fliplr=0,flipud=0,mosaic=0,mixup=0,copy_paste=0,close_mosaic=0),
        dino=dict(backbone='DINOv3 ConvNeXt Small',revision='421020015aa4266363db803b6049a529999afb1e',frozen=True,
            detector='TorchVision Faster R-CNN + trainable FPN256',head_initialization='COCO RPN and ROI box head; new liquid predictor',
            batch=8,lr=.0001,weight_decay=.0001,optimizer='AdamW',scheduler='cosine50',amp=False,
            min_size=480,max_size=640,online_augmentation=False),
        evaluation=dict(confidence=.35,nms_iou=.45,ap_confidence=.001,fixed_vial_detector=True),
        limits=['Teacher boxes/negative absences are unreviewed; agreement is not human accuracy.',
            'Run 5 is repeatedly inspected, not a blind final test.',
            'Non-YOLO comparison changes architecture, pretraining and native preprocessing; not an isolated architecture ablation.',
            'Focused arm resamples sources, retaining identical 1442-image epoch length; 222 positive units repeat.',
            'No label adjudication, stream class, app promotion, measurement QA or industrial connection.']))
    print(summary,flush=True)
