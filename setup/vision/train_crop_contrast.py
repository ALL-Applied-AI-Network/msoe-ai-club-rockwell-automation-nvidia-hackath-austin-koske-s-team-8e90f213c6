"""Immutable crop+CLAHE two-class detector experiment; Run 5 never fits/selects."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shutil

from crop_contrast import crop_labels,enhance_crop,read_labels


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def validate_split(rows,held_hashes):
    if not rows:raise ValueError('Empty experiment')
    by_id={r['source_id']:r for r in rows}
    if len(by_id)!=len(rows):raise ValueError('Duplicate source IDs')
    split_hashes={'train':set(),'val':set()}
    for row in rows:
        split=row['split'];run=row['source_run']
        if split not in split_hashes or run not in ([1,2,3] if split=='train' else [4]):
            raise ValueError('Source recording crosses fitting/development boundary')
        if row['source_sha256'] in held_hashes:raise ValueError('Run 5 source duplicate in fitting/development')
        split_hashes[split].add(row['source_sha256'])
        if row.get('parent_id'):
            parent=by_id.get(row['parent_id'])
            if not parent or parent['split']!=split or parent['source_run']!=run or parent['source_sha256']!=row['source_sha256']:
                raise ValueError('Augmentation crosses recording/split boundary')
    if split_hashes['train'] & split_hashes['val']:raise ValueError('Source duplicate across splits')


def prepare(source,teacher,out):
    import cv2
    import numpy as np
    provenance=read(source/'provenance.json');rows=provenance['samples']
    teacher_rows=read(teacher/'manifest.json')
    held={r['source_sha256'] for r in teacher_rows if r['camera']=='vial_fill' and r['video_id']=='5'}
    if not held:raise ValueError('Missing Run 5 exclusion hashes')
    validate_split(rows,held)
    parsed={r['source_id']:read_labels((source/r['labels']).read_text(encoding='utf-8')) for r in rows}
    coords=np.asarray([b['box'] for row in rows if row['split']=='train' for b in parsed[row['source_id']]])
    # Context chosen from train bounds ONLY, never supplied Run 5 labels/frames.
    roi=[max(0,math.floor(coords[:,0].min())-24),max(0,math.floor(coords[:,1].min())-16),
         min(640,math.ceil(coords[:,2].max())+24),min(480,math.ceil(coords[:,3].max())+16)]
    transformed={r['source_id']:crop_labels(parsed[r['source_id']],roi) for r in rows}
    out.mkdir(parents=True,exist_ok=False);data=out/'data';samples=[];pixel_hashes={'train':set(),'val':set()}
    for row in rows:
        image=source/row['image'];label=source/row['labels'];name=row['source_id'];split=row['split']
        if not row.get('parent_id') and (sha(image)!=row['source_sha256'] or sha(label)!=row['label_sha256']):
            raise ValueError('Original source image/label changed')
        rgb=cv2.cvtColor(cv2.imread(str(image)),cv2.COLOR_BGR2RGB)
        if rgb.shape!=(480,640,3):raise ValueError('Unexpected camera profile')
        processed=enhance_crop(rgb,roi)
        for folder in ['images','labels']:(data/folder/split).mkdir(parents=True,exist_ok=True)
        output_image=data/f'images/{split}/{name}.png';output_label=data/f'labels/{split}/{name}.txt'
        if not cv2.imwrite(str(output_image),cv2.cvtColor(processed,cv2.COLOR_RGB2BGR)):raise IOError(output_image)
        output_label.write_text(transformed[name],encoding='utf-8')
        rgb_sha=hashlib.sha256(processed.tobytes()).hexdigest();pixel_hashes[split].add(rgb_sha)
        samples.append(dict(row,original_input_image_sha256=sha(image),original_input_label_sha256=sha(label),
                            image=output_image.relative_to(data).as_posix(),labels=output_label.relative_to(data).as_posix(),
                            processed_image_sha256=sha(output_image),processed_label_sha256=sha(output_label),processed_rgb_sha256=rgb_sha))
    if pixel_hashes['train'] & pixel_hashes['val']:raise ValueError('Identical processed RGB across train/dev')
    spec=dict(schema_version=1,source_shape=[480,640],roi=roi,roi_derivation='All fitting-box bounds plus 24px horizontal/16px vertical context; no development/test data',
              crop_before_clahe=True,clahe=dict(color_space='OpenCV LAB',channel='L',clip_limit=2.,tile_grid=[8,8]),
              opencv=cv2.__version__,crop_coordinates='source xyxy with exclusive upper bounds')
    write(out/'preprocessing.json',spec)
    record=dict(schema_version=1,kind='cropped_contrast_box_training',train_runs=[1,2,3],development_runs=[4],excluded_runs=[5],
                source_provenance_sha256=sha(source/'provenance.json'),held_out_source_hashes=sorted(held),
                source_teacher_manifest_sha256=sha(teacher/'manifest.json'),preprocessing=spec,
                counts=dict(Counter(r['split'] for r in samples)),unique_original_train_frames=sum(r['split']=='train' and not r.get('parent_id') for r in samples),
                no_target_clipping_or_removal=True,source_label_status='unreviewed teacher boxes accepted as provisional targets',samples=samples)
    write(data/'provenance.json',record)
    (data/'data.yaml').write_text(f'path: {data.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: vial\n  1: colored_liquid\n',encoding='utf-8')
    plan=dict(schema_version=1,kind='frozen_crop_contrast_trial',fit_runs=[1,2,3],development_runs=[4],excluded_from_fit_and_selection=[5],
              epochs=12,seed=19,architecture='yolo11n_detect',pretrained='COCO yolo11n, same starting model as original detector',
              checkpoint_selection='Run 4 box mAP50-95 fitness only',confidence_for_videos=.35,nms_iou=.45,
              ap_scoring_confidence=.001,preprocessing=spec,data_provenance_sha256=sha(data/'provenance.json'),
              baseline_onnx_sha256='89ed11f7046171e5340dc45a9d528010de1cefbba21e163b23b7098151e0880e',
              limits=['Provisional teacher targets; not independent human truth.','Run 5 previously inspected; diagnostic comparison, not new blind test.',
                      'Original historical run used AMP; new reproducible trainer disables AMP checks/implicit downloads.',
                      'No measured color, measurement-QA change, threshold tuning or baseline replacement.'])
    write(out/'experiment-plan.json',plan)
    print(json.dumps(dict(roi=roi,counts=record['counts'],parents=record['unique_original_train_frames'],out=str(out))),flush=True)


def train(root,pretrained):
    import torch
    import ultralytics
    from ultralytics import YOLO
    plan=read(root/'experiment-plan.json');data=root/'data';p=read(data/'provenance.json')
    validate_split(p['samples'],set(p['held_out_source_hashes']))
    if sha(data/'provenance.json')!=plan['data_provenance_sha256']:raise ValueError('Prepared provenance changed')
    for row in p['samples']:
        if sha(data/row['image'])!=row['processed_image_sha256'] or sha(data/row['labels'])!=row['processed_label_sha256']:
            raise ValueError('Prepared data changed')
    # Host path is supplied at job execution; YAML itself carries no split logic.
    (data/'data.yaml').write_text(f'path: {data.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: vial\n  1: colored_liquid\n',encoding='utf-8')
    output=root/'model';output.mkdir(exist_ok=False);torch.set_num_threads(4)
    model=YOLO(str(pretrained))
    model.train(data=str(data/'data.yaml'),epochs=plan['epochs'],imgsz=640,batch=16,device=0,workers=0,
                seed=plan['seed'],deterministic=True,project=str(output),name='train',exist_ok=False,patience=plan['epochs'],
                pretrained=True,plots=True,amp=False,hsv_h=0,hsv_s=0,hsv_v=.1,degrees=2,translate=.04,scale=.1,
                fliplr=0,flipud=0,mosaic=0,mixup=0,copy_paste=0)
    best=YOLO(str(output/'train/weights/best.pt'))
    exported=Path(best.export(format='onnx',imgsz=640,opset=17,simplify=False,dynamic=False,nms=False))
    shutil.copyfile(exported,output/'detector.onnx')
    write(output/'model.json',dict(schema_version=1,architecture='yolo11n_detect',model_sha256=sha(output/'detector.onnx'),
          checkpoint_sha256=sha(output/'train/weights/best.pt'),pretrained_sha256=sha(pretrained),
          data_provenance_sha256=sha(data/'provenance.json'),experiment_plan_sha256=sha(root/'experiment-plan.json'),
          preprocessing=read(root/'preprocessing.json'),epochs=plan['epochs'],seed=plan['seed'],train_runs=[1,2,3],development_runs=[4],excluded_runs=[5],
          torch=torch.__version__,ultralytics=ultralytics.__version__,trainer_sha256=sha(__file__),transform_sha256=sha(Path(__file__).with_name('crop_contrast.py'))))
    print(json.dumps(read(output/'model.json')),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['prepare','train'])
    parser.add_argument('--source',type=Path);parser.add_argument('--teacher',type=Path);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--pretrained',type=Path);a=parser.parse_args()
    if a.command=='prepare':prepare(a.source,a.teacher,a.out)
    else:train(a.out,a.pretrained)
