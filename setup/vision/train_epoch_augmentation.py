"""Frozen real-only epoch/photometric comparison. No Run 5 fit or selection."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import shutil

from detector_photometric import Photometric, SPEC
from train_mixed_detector import validate_split

REAL='81851cba53c05a1811e81c32bc8328c4f4d5236e4f26bf6510505eae029b260c'
INITIAL='0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1'
BASELINE='89ed11f7046171e5340dc45a9d528010de1cefbba21e163b23b7098151e0880e'
ARMS={'plain12':dict(epochs=12,patience=12,extra_augmentation=False),
      'plain50':dict(epochs=50,patience=10,extra_augmentation=False),
      'augmented50':dict(epochs=50,patience=10,extra_augmentation=True)}


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')


def check(data):
    if sha(data/'provenance.json')!=REAL:raise ValueError('Frozen real dataset inventory changed')
    p=read(data/'provenance.json');validate_split(p['samples'],set(p['held_out_source_hashes']))
    if p['counts']!={'train':1442,'val':157} or any(r['origin']!='real' for r in p['samples']):
        raise ValueError('Unexpected data/augmentation ancestry')
    for r in p['samples']:
        if sha(data/r['image'])!=r['image_sha256'] or sha(data/r['labels'])!=r['actual_label_sha256']:
            raise ValueError('Frozen image/target bytes changed')
    return p


def plan():
    return dict(schema_version=1,kind='controlled_real_epoch_photometric_trial',arms=ARMS,
        seed=19,architecture='yolo11n_detect',pretrained_sha256=INITIAL,real_provenance_sha256=REAL,
        baseline_onnx_sha256=BASELINE,train_runs=[1,2,3],development_runs=[4],excluded_from_fit_and_selection=[5],
        train_samples=1442,unique_train_originals=721,development_samples=157,
        labels='Unchanged provisional teacher boxes; human review drafts excluded',
        checkpoint_selection='Run 4 combined class AP50-95 only; never Run 5',
        common=dict(imgsz=640,batch=16,amp=False,workers=0,deterministic=True,
            hsv_h=0,hsv_s=0,hsv_v=.1,degrees=2,translate=.04,scale=.1,
            fliplr=0,flipud=0,mosaic=0,mixup=0,copy_paste=0,close_mosaic=0),
        extra_photometric=SPEC,photo_rng='Independent NumPy Generator seed190021; does not consume geometry RNG',
        preprocessing='Original full-frame RGB; ordinary YOLO letterbox; no inference enhancement/crop',
        epochs_comparison_scope='Training budget incl. standard learning-rate horizon; not a fixed-LR update-count ablation',
        automatic_promotion=False,measurement_qa_changed=False,
        limits=['Run 5 repeatedly inspected; diagnostic, not blind final test.',
            'Development/scoring targets are unreviewed teacher proposals, not human accuracy.',
            'Historical deployed detector is an anchor; fresh12 is the matched runtime/FP32 control.',
            'Fifty epochs is a maximum budget; patience10 can stop earlier.',
            'No real labels, synthetic renderers, reviewed-label assignments or operator workflows changed.'])


def prepare(source,out):
    check(source);out.mkdir(parents=True,exist_ok=False)
    shutil.copytree(source,out/'data')
    write(out/'experiment-plan.json',plan())
    yaml(out/'data');check(out/'data')
    print(json.dumps(dict(out=str(out),plan_sha256=sha(out/'experiment-plan.json'))),flush=True)


def yaml(data):
    (data/'data.yaml').write_text(f'path: {data.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: vial\n  1: colored_liquid\n',encoding='utf-8')


def train(root,arm,pretrained):
    import platform
    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionTrainer
    if ultralytics.__version__!='8.4.182':raise ValueError('Pinned Ultralytics runtime differs')
    p=read(root/'experiment-plan.json')
    if p!=plan() or sha(pretrained)!=INITIAL:raise ValueError('Frozen plan/initialization changed')
    check(root/'data')
    output=root/'models'/arm;output.mkdir(parents=True,exist_ok=False)
    # Private copies prevent simultaneous Slurm arms racing over YOLO label
    # caches or machine-specific YAML. Image/target bytes remain identical.
    data=output/'data';shutil.copytree(root/'data',data);yaml(data);check(data)
    photo=Photometric(preview=output/'augmentation-preview')
    class ControlledTrainer(DetectionTrainer):
        def build_dataset(self,img_path,mode='train',batch=None):
            dataset=super().build_dataset(img_path,mode,batch)
            if mode=='train' and ARMS[arm]['extra_augmentation']:
                dataset.transforms.insert(0,photo)
            return dataset
    torch.set_num_threads(4)
    config=p['arms'][arm]
    model=YOLO(str(pretrained))
    model.train(trainer=ControlledTrainer,data=str(data/'data.yaml'),
        epochs=config['epochs'],patience=config['patience'],device=0,seed=p['seed'],
        project=str(output),name='train',exist_ok=False,pretrained=True,plots=True,**p['common'])
    if config['extra_augmentation'] and (not photo.counts['calls'] or not photo.counts['changed_images']):
        raise ValueError('Declared augmentation did not execute')
    if not config['extra_augmentation'] and photo.counts['calls']:
        raise ValueError('Extra augmentation entered plain arm')
    photo.save(output/'augmentation-execution.json')
    best=output/'train/weights/best.pt';checkpoint=YOLO(str(best))
    exported=Path(checkpoint.export(format='onnx',imgsz=640,opset=17,simplify=False,dynamic=False,nms=False))
    shutil.copyfile(exported,output/'detector.onnx');shutil.copyfile(best,output/'best.pt')
    epochs=list(csv.DictReader((output/'train/results.csv').read_text().splitlines()))
    write(output/'model.json',dict(schema_version=1,arm=arm,checkpoint_sha256=sha(output/'best.pt'),
        model_sha256=sha(output/'detector.onnx'),pretrained_sha256=INITIAL,
        experiment_plan_sha256=sha(root/'experiment-plan.json'),data_provenance_sha256=REAL,
        trainer_sha256=sha(__file__),photo_sha256=sha(Path(__file__).with_name('detector_photometric.py')),
        torch=torch.__version__,ultralytics=ultralytics.__version__,python=platform.python_version(),
        cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),epochs_completed=len(epochs),
        epoch_budget=config['epochs'],patience=config['patience'],seed=19,train_runs=[1,2,3],
        development_runs=[4],excluded_runs=[5],extra_augmentation=config['extra_augmentation'],
        automatic_promotion=False,measurement_qa_changed=False))
    check(root/'data');check(data)
    print(json.dumps(read(output/'model.json')),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','train'])
    p.add_argument('--source',type=Path);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--arm',choices=ARMS);p.add_argument('--pretrained',type=Path)
    a=p.parse_args()
    if a.command=='prepare':prepare(a.source,a.out)
    else:train(a.out,a.arm,a.pretrained)
