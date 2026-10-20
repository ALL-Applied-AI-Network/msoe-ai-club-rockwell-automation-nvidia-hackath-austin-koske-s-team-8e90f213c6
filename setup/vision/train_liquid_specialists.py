"""Frozen single-class sampling comparison and two non-YOLO detection pilots."""
import argparse
import csv
import json
from pathlib import Path
import platform
import random
import shutil
import time
from types import SimpleNamespace

import cv2
import numpy as np

from liquid_specialists_data import ARMS,WEIGHTS,prepare,check,yaml,read,write,sha


def validate_inputs(root,arm):
    plan=read(root/'experiment-plan.json');frozen=read(root/'preparation.json')
    if plan['excluded_from_fit_and_selection']!=[5] or plan['arms']!=ARMS:raise ValueError('Frozen plan changed')
    if frozen['experiment-plan.json']!=sha(root/'experiment-plan.json'):raise ValueError('Plan bytes changed')
    for name,expected in frozen.items():
        if name!='experiment-plan.json' and sha(root/name)!=expected:raise ValueError('Executed helper changed: '+name)
    for name,expected in WEIGHTS.items():
        if sha(root/'pretrained'/name)!=expected:raise ValueError('Pretrained bytes changed')
    data=root/'data'/ARMS[arm]['dataset'];check(data)
    if sha(data/'provenance.json')!=plan['data'][ARMS[arm]['dataset']]['provenance_sha256']:raise ValueError('Inventory changed')
    return plan,data


def metadata(root,arm,out,plan,data,checkpoint,epochs,best_epoch,best_ap):
    import torch,ultralytics
    write(out/'model.json',dict(schema_version=1,arm=arm,**ARMS[arm],
        checkpoint_sha256=sha(checkpoint),experiment_plan_sha256=sha(root/'experiment-plan.json'),
        data_provenance_sha256=sha(data/'provenance.json'),preparation_sha256=sha(root/'preparation.json'),
        train_runs=[1,2,3],development_runs=[4],excluded_runs=[5],
        epochs_completed=epochs,best_run4_epoch=best_epoch,best_run4_liquid_ap50_95=best_ap,
        seed=19,torch=torch.__version__,ultralytics=ultralytics.__version__,python=platform.python_version(),
        cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),
        checkpoint_file=checkpoint.name,provisional_negatives=True,measurement_qa_changed=False,automatic_promotion=False))


def train_ultralytics(root,arm):
    import torch,ultralytics
    from ultralytics import YOLO,RTDETR
    if ultralytics.__version__!='8.4.182':raise ValueError('Pinned runtime differs')
    plan,source=validate_inputs(root,arm);out=root/'models'/arm;out.mkdir(parents=True,exist_ok=False)
    data=out/'data';shutil.copytree(source,data);yaml(data)
    torch.set_num_threads(4)
    model=(RTDETR(root/'pretrained/rtdetr-l.pt') if arm=='rtdetr-all' else YOLO(root/'pretrained/yolo11n.pt'))
    model.train(data=str(data/'data.yaml'),epochs=50,patience=10,device=0,seed=19,
        project=str(out),name='train',exist_ok=False,pretrained=True,plots=True,**plan['yolo_rtdetr'])
    best=out/'train/weights/best.pt';shutil.copyfile(best,out/'best.pt')
    rows=[{k.strip():v for k,v in row.items()} for row in csv.DictReader((out/'train/results.csv').read_text().splitlines())]
    best_row=max(rows,key=lambda r:float(r['metrics/mAP50-95(B)']))
    metadata(root,arm,out,plan,data,out/'best.pt',len(rows),int(best_row['epoch']),float(best_row['metrics/mAP50-95(B)']))
    check(data);check(source)
    print(json.dumps(read(out/'model.json')),flush=True)


class Frames:
    def __init__(self,data,split):
        self.data=data;self.rows=[r for r in read(data/'provenance.json')['samples'] if r['split']==split]
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        import torch
        r=self.rows[i];bgr=cv2.imread(str(self.data/r['image']))
        if bgr is None or bgr.shape!=(480,640,3):raise ValueError('Source image geometry changed')
        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);boxes=[]
        for line in (self.data/r['labels']).read_text().splitlines():
            c,x,y,w,h=map(float,line.split())
            if c!=0:raise ValueError('Non-liquid target')
            boxes.append([(x-w/2)*640,(y-h/2)*480,(x+w/2)*640,(y+h/2)*480])
        image=torch.from_numpy(rgb.copy()).permute(2,0,1).float()/255
        return image,dict(boxes=torch.tensor(boxes,dtype=torch.float32).reshape(-1,4),labels=torch.ones(len(boxes),dtype=torch.int64))


def development_score(model,loader,device):
    import torch
    from ultralytics.engine.validator import BaseValidator
    from ultralytics.utils.metrics import ap_per_class,box_iou
    stats=[]
    model.eval()
    with torch.no_grad():
        for images,targets in loader:
            result=model([x.to(device) for x in images])
            for p,t in zip(result,targets):
                proposals=[dict(class_id=0,box=b.tolist(),confidence=float(s)) for b,s in zip(p['boxes'].cpu(),p['scores'].cpu())]
                expected=[dict(class_id=0,box=b.tolist()) for b in t['boxes']]
                tp=BaseValidator.match_predictions(SimpleNamespace(iouv=torch.linspace(.5,.95,10)),
                    torch.zeros(len(proposals),dtype=torch.int64),torch.zeros(len(expected),dtype=torch.int64),
                    box_iou(t['boxes'],p['boxes'].cpu())).numpy()
                stats.append((tp,np.array([p['confidence'] for p in proposals]),
                    np.zeros(len(proposals)),np.zeros(len(expected))))
    tp,conf,classes,target=[np.concatenate([r[i] for r in stats]) for i in range(4)]
    result=ap_per_class(tp,conf,classes,target,plot=False,names={0:'colored_liquid'})
    return float(result[5].mean())


def train_dino(root,smoke=False):
    import torch,transformers
    from torch.utils.data import DataLoader
    from dino_liquid import build,head_state,load
    if transformers.__version__!='4.57.1':raise ValueError('Pinned Transformers runtime differs')
    plan,data=validate_inputs(root,'dinov3-all');out=root/('dino-smoke' if smoke else 'models/dinov3-all')
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(19);np.random.seed(19);random.seed(19)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=build(root/'pretrained/dinov3-convnext-small',root/'pretrained/fasterrcnn-coco.pth').cuda()
    frozen_before=next(model.backbone.encoder.parameters()).detach().clone()
    if smoke:
        model.train();image=Frames(data,'train')[0][0].cuda()
        empty=dict(boxes=torch.empty((0,4),device='cuda'),labels=torch.empty(0,dtype=torch.int64,device='cuda'))
        if not all(torch.isfinite(v) for v in model([image],[empty]).values()):raise ValueError('Empty-target loss failed')
    loader=lambda split,shuffle:DataLoader(Frames(data,split),batch_size=8,shuffle=shuffle,num_workers=0,
        collate_fn=lambda batch:tuple(zip(*batch)))
    train=loader('train',True);val=loader('val',False)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0001,weight_decay=.0001)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=50)
    best=-1.;best_epoch=0;curve=[];started=time.monotonic()
    for epoch in range(1,2 if smoke else 51):
        model.train();losses=[]
        for step,(images,targets) in enumerate(train):
            images=[x.cuda() for x in images];targets=[{k:v.cuda() for k,v in t.items()} for t in targets]
            optimizer.zero_grad(set_to_none=True);components=model(images,targets);loss=sum(components.values())
            if not torch.isfinite(loss):raise ValueError('Nonfinite detection loss')
            loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],10.)
            optimizer.step();losses.append(float(loss.detach()))
            if smoke and step==1:break
        ap=development_score(model,val if not smoke else list(val)[:1],'cuda');schedule.step()
        curve.append(dict(epoch=epoch,train_loss=float(np.mean(losses)),run4_liquid_ap50_95=ap,elapsed_s=time.monotonic()-started))
        with (out/'training.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(curve[0]));writer.writeheader();writer.writerows(curve)
        if ap>best:
            best=ap;best_epoch=epoch;torch.save(dict(head_state=head_state(model),epoch=epoch),out/'best.pt')
        print(json.dumps(curve[-1]),flush=True)
        if epoch-best_epoch>=10:break
    if not torch.equal(frozen_before,next(model.backbone.encoder.parameters())):raise ValueError('Frozen backbone changed')
    restored=load(root/'pretrained/dinov3-convnext-small',out/'best.pt','cuda')
    # A checkpoint must reproduce its own predictions after save/reload.
    model.eval();image=Frames(data,'val')[0][0].cuda()
    with torch.no_grad():left=restored([image])[0];right=model([image])[0]
    if smoke and (not torch.allclose(left['boxes'],right['boxes'],atol=1e-4) or not torch.allclose(left['scores'],right['scores'],atol=1e-5)):
        raise ValueError('Head roundtrip changed predictions')
    write(out/'frozen-backbone-check.json',dict(first_parameter_unchanged=True,all_parameters_require_grad_false=all(not p.requires_grad for p in model.backbone.encoder.parameters()),
        smoke=smoke,transformers=transformers.__version__,torchvision=__import__('torchvision').__version__))
    if not smoke:metadata(root,'dinov3-all',out,plan,data,out/'best.pt',len(curve),best_epoch,best)
    check(data)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','train','smoke-dino'])
    p.add_argument('--out',type=Path,required=True);p.add_argument('--source',type=Path);p.add_argument('--assets',type=Path)
    p.add_argument('--arm',choices=list(ARMS));a=p.parse_args()
    if a.command=='prepare':prepare(a.source,a.assets,a.out)
    elif a.command=='smoke-dino':train_dino(a.out,True)
    elif a.arm=='dinov3-all':train_dino(a.out)
    else:train_ultralytics(a.out,a.arm)
