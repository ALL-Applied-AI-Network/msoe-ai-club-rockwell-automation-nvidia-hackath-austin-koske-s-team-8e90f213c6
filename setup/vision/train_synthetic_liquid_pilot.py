"""Bounded Faster R-CNN pilot with synthetic-only CSI fine-tuning/selection.

Uses existing immutable rendered images; does not modify scene assets. The final
fixed epoch is frozen before any real development images are opened for scoring.
Synthetic targets are geometric body boxes, not approved optical masks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'setup/vision'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def prepare(a):
    if a.out.exists():
        raise ValueError('Use a fresh experiment root')
    inventory=a.tools/'results/real-synthetic-detector-20261019/data/sim-only/provenance.json'
    # Local sim-only inventory is retained without a duplicate image tree. Its
    # hash-pinned synthetic files are materialized in the shared mixed package;
    # only the explicitly synthetic inventory rows below may be accessed.
    data=a.tools/'results/real-synthetic-detector-20261019/data/mixed'
    source=read(inventory)
    if source['real_pixels_or_labels_copied'] is not False:
        raise ValueError('Synthetic-only source boundary changed')
    unique={}
    for row in source['samples']:
        if row['origin']!='synthetic' or row['source_run'] is not None or row['split']!='train':
            raise ValueError('A source is not an exclusively synthetic training image')
        if row['native_sample_id'] in unique:
            continue
        if sha(data/row['image'])!=row['image_sha256'] or sha(data/row['labels'])!=row['actual_label_sha256']:
            raise ValueError('Synthetic source bytes differ')
        unique[row['native_sample_id']]=row
    if len(unique)!=768:
        raise ValueError('Expected original 768-scene synthetic fixture')
    # Identical image bytes remain in one split even if scene IDs differ.
    hashes=sorted({r['image_sha256'] for r in unique.values()})
    random.Random(19).shuffle(hashes);val=set(hashes[:max(1,round(len(hashes)*.2))])
    a.out.mkdir(parents=True);(a.out/'pretrained').mkdir();(a.out/'helpers').mkdir()
    init=a.tools/'results/early-liquid-20261020/pretrained/fasterrcnn-coco.pth'
    original=read(a.tools/'results/early-liquid-20261020/experiment-plan.json')
    if sha(init)!=original['pretrained']['fasterrcnn-coco.pth']:
        raise ValueError('Generic COCO initialization differs')
    shutil.copy2(init,a.out/'pretrained/fasterrcnn-coco.pth')
    samples=[]
    for sid,row in sorted(unique.items()):
        split='val' if row['image_sha256'] in val else 'train'
        image_rel=f'data/images/{split}/{sid}.png';label_rel=f'data/labels/{split}/{sid}.txt'
        image=a.out/image_rel;label=a.out/label_rel
        image.parent.mkdir(parents=True,exist_ok=True);label.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(data/row['image'],image)
        lines=[]
        for line in (data/row['labels']).read_text().splitlines():
            fields=line.split()
            if int(fields[0])==1:
                lines.append('0 '+' '.join(fields[1:]))
            elif int(fields[0])!=0:
                raise ValueError('Unexpected class in two-class synthetic source')
        label.write_text('\n'.join(lines)+('\n' if lines else ''),encoding='utf-8')
        samples.append(dict(source_id=sid,origin='synthetic',source_run=None,split=split,
            source_image_sha256=row['image_sha256'],source_label_sha256=row['actual_label_sha256'],
            image=image_rel,image_sha256=sha(image),label=label_rel,label_sha256=sha(label),liquid_targets=len(lines)))
    train_hashes={s['image_sha256'] for s in samples if s['split']=='train'}
    if train_hashes & {s['image_sha256'] for s in samples if s['split']=='val'}:
        raise ValueError('Image-byte leakage across synthetic split')
    helpers=[Path(__file__),ROOT/'setup/vision/train_early_liquid.py',ROOT/'setup/vision/experiment_sim_real_cpu.py',ROOT/'setup/vision/experiment_sim_real_robustness.py']
    helper_pins={p.name:sha(p) for p in helpers}
    for p in helpers:
        shutil.copy2(p,a.out/'helpers'/p.name)
    plan=dict(schema_version=1,kind='synthetic_only_fasterrcnn_liquid_pilot',seed=19,epochs=3,batch_size=4,
        optimizer='AdamW',lr=.0001,weight_decay=.0001,architecture='Faster R-CNN ResNet50-FPN',
        initialization='Generic COCO weights; no CSI real-trained checkpoint',pretrained_sha256=sha(init),
        trainable_backbone_stages=['layer2','layer3','layer4'],new_predictor='background/liquid body',
        training_real_runs=[],selection_real_runs=[],run5_used=False,checkpoint_selection='Fixed final epoch 3; no early stopping or real-data selection',
        synthetic_split='20% image-hash groups using seed19; remaining 80% train; duplicated source images never cross splits.',
        counts={split:dict(images=sum(s['split']==split for s in samples),positive=sum(s['split']==split and s['liquid_targets']>0 for s in samples),targets=sum(s['liquid_targets'] for s in samples if s['split']==split)) for split in ['train','val']},
        source_provenance_sha256=sha(inventory),helpers=helper_pins,samples=samples,
        augmentation='Uniform raw/exposure085/exposure115/green085/blur_sigma12/jpeg_q40 on synthetic training RGB only, Python RNG seed19; labels unchanged.' if a.augment else 'none',
        target_scope='Accumulated liquid body; drop synthetic vial class0 and remap body class1 to training class0 without coordinate changes. Geometric boxes, not visible-pool mask truth.',
        real_evaluation='After final checkpoint freeze, score the existing 157 original Run 4 machine-reference stills. Report as development agreement; no fitting/checkpoint selection.',
        limitations=['Small fixed-budget family pilot, not an optimized or matched architecture benchmark.',
            'Underlying 768-scene renderer has restricted camera/geometry and imperfect early/pale/clear liquid optics.',
            'Synthetic geometric development AP is not physical or optical accuracy.',
            'No additional synthetic rendering or teammate asset changes occur in this pilot.'])
    write(a.out/'experiment-plan.json',plan)
    write(a.out/'preparation.json',dict(plan_sha256=sha(a.out/'experiment-plan.json'),helpers=helper_pins))
    print(json.dumps(dict(prepared=str(a.out),counts=plan['counts'])),flush=True)


def validate(root):
    plan=read(root/'experiment-plan.json');prep=read(root/'preparation.json')
    if sha(root/'experiment-plan.json')!=prep['plan_sha256'] or plan['training_real_runs'] or plan['selection_real_runs'] or plan['run5_used']:
        raise ValueError('Frozen plan or training boundary changed')
    for name,expected in prep['helpers'].items():
        if sha(root/'helpers'/name)!=expected:
            raise ValueError('Frozen helper snapshot differs')
    if sha(root/'pretrained/fasterrcnn-coco.pth')!=plan['pretrained_sha256']:
        raise ValueError('Pretrained bytes differ')
    for s in plan['samples']:
        if s['origin']!='synthetic' or s['source_run'] is not None or sha(root/s['image'])!=s['image_sha256'] or sha(root/s['label'])!=s['label_sha256']:
            raise ValueError('A derived synthetic sample differs')
    return plan


def evaluate(model,rows,root,device):
    import torch
    from liquid_agreement_metrics import score
    output=[];model.eval()
    with torch.inference_mode():
        for row in rows:
            rgb=np.asarray(Image.open(root/row['image']).convert('RGB')).copy()
            targets=[]
            for line in (root/row['label']).read_text().splitlines():
                _,x,y,w,h=map(float,line.split());targets.append(dict(class_id=1,box=[(x-w/2)*640,(y-h/2)*480,(x+w/2)*640,(y+h/2)*480]))
            affine=None
            if 'inference_view' in row:
                from vial_input_view import make_view
                rgb,affine=make_view(rgb,row['view_vials'],row['inference_view'])
            tensor=torch.from_numpy(rgb).permute(2,0,1).float().to(device)/255.
            native=model([tensor])[0]
            predictions=[dict(class_id=1,box=b.tolist(),confidence=float(s)) for b,s in zip(native['boxes'].cpu(),native['scores'].cpu())]
            if 'inference_view' in row:
                from vial_input_view import map_box
                predictions=[dict(p,box=box) for p in predictions if (box:=map_box(p['box'],affine,inverse=True)) is not None]
            output.append(dict(source_id=row['source_id'],targets=targets,predictions=predictions))
    return score(output,1),output


def fit(a):
    import torch
    from torch.utils.data import Dataset,DataLoader
    # Execute the retained architecture helper rather than a mutable checkout copy.
    sys.path.insert(0,str(a.out/'helpers'))
    from train_early_liquid import build_resnet
    plan=validate(a.out)
    if (a.out/'training-started.json').exists():
        raise ValueError('Fit already started; inspect artifacts instead of duplicating it')
    write(a.out/'training-started.json',dict(plan_sha256=sha(a.out/'experiment-plan.json'),device=torch.cuda.get_device_name()))
    torch.manual_seed(19);np.random.seed(19);random.seed(19);torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False

    class Frames(Dataset):
        def __init__(self):
            self.rows=[s for s in plan['samples'] if s['split']=='train']
        def __len__(self):
            return len(self.rows)
        def __getitem__(self,index):
            row=self.rows[index];rgb=np.asarray(Image.open(a.out/row['image']).convert('RGB')).copy();boxes=[]
            if plan['augmentation']!='none':
                from experiment_sim_real_robustness import transform,CONDITIONS
                rgb=transform(rgb,random.choice(CONDITIONS))
            for line in (a.out/row['label']).read_text().splitlines():
                _,x,y,w,h=map(float,line.split());boxes.append([(x-w/2)*640,(y-h/2)*480,(x+w/2)*640,(y+h/2)*480])
            return torch.from_numpy(rgb).permute(2,0,1).float()/255.,dict(boxes=torch.tensor(boxes,dtype=torch.float32).reshape(-1,4),labels=torch.ones(len(boxes),dtype=torch.int64))

    model=build_resnet(a.out).cuda()
    train=DataLoader(Frames(),batch_size=4,shuffle=True,num_workers=0,collate_fn=lambda b:tuple(zip(*b)))
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0001,weight_decay=.0001)
    started=time.perf_counter();curve=[]
    epochs=int(plan.get('epochs',3));selection=plan.get('selection_policy','fixed_final')
    best_ap=-1.;selected_epoch=epochs;selection_curve=[]
    for epoch in range(1,epochs+1):
        model.train();losses=[]
        for step,(images,targets) in enumerate(train):
            optimizer.zero_grad(set_to_none=True)
            loss=sum(model([im.cuda() for im in images],[{k:v.cuda() for k,v in t.items()} for t in targets]).values())
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite synthetic training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.);optimizer.step();losses.append(float(loss.detach()))
            if (step+1)%40==0:
                print(json.dumps(dict(epoch=epoch,step=step+1,mean_loss=float(np.mean(losses)),elapsed_s=time.perf_counter()-started)),flush=True)
        entry=dict(epoch=epoch,mean_loss=float(np.mean(losses)),steps=len(losses),elapsed_s=time.perf_counter()-started)
        curve.append(entry);write(a.out/'training-curve.json',curve);print(json.dumps(entry),flush=True)
        if epoch==3 and plan.get('retain_epoch3_control'):
            torch.save(dict(model_state={k:v.detach().cpu() for k,v in model.state_dict().items()},epoch=3),a.out/'epoch-03.pt')
        if selection=='synthetic_ap50_95' and (epoch%3==0 or epoch==epochs):
            metrics,_=evaluate(model,[s for s in plan['samples'] if s['split']=='val'],a.out,'cuda')
            ap=metrics['ap50_95_101point'];selection_curve.append(dict(epoch=epoch,synthetic_ap50_95=ap))
            write(a.out/'synthetic-selection-curve.json',selection_curve)
            if ap>best_ap:
                best_ap=ap;selected_epoch=epoch
                torch.save(dict(model_state={k:v.detach().cpu() for k,v in model.state_dict().items()},epoch=epoch),a.out/'synthetic-best.pt')
            print(json.dumps(dict(epoch=epoch,synthetic_selection_ap=ap,selected_epoch=selected_epoch)),flush=True)
    checkpoint=a.out/'final.pt'
    if selection=='synthetic_ap50_95':
        torch.save(dict(model_state={k:v.detach().cpu() for k,v in model.state_dict().items()},epoch=epochs),a.out/'last.pt')
        shutil.copy2(a.out/'synthetic-best.pt',checkpoint)
        model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['model_state'])
    else:
        torch.save(dict(model_state={k:v.detach().cpu() for k,v in model.state_dict().items()},epoch=epochs),checkpoint)
    write(a.out/'frozen-model.json',dict(architecture=plan['architecture'],checkpoint_sha256=sha(checkpoint),epoch=selected_epoch,fit_epochs=epochs,
        checkpoint_selection=plan['checkpoint_selection'],training_real_runs=[],selection_real_runs=[],training_images=plan['counts']['train']['images'],
        source_plan_sha256=sha(a.out/'experiment-plan.json'),augmentation=plan['augmentation'],input_view=plan.get('input_view'),generic_real_pretraining='COCO initialization; CSI fine-tuning synthetic only',torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name()))
    # Real evaluation is deliberately after the frozen checkpoint marker.
    synthetic,syn_rows=evaluate(model,[s for s in plan['samples'] if s['split']=='val'],a.out,'cuda')
    write(a.out/'synthetic-development-predictions.json',syn_rows)
    data=a.tools/'results/real-synthetic-detector-20261019/data/real';prov=read(data/'provenance.json');real=[]
    view_model=None;view_evidence=[]
    if plan.get('input_view'):
        from ultralytics import YOLO
        from vial_input_view import VIAL_OPTIONS
        view=plan['input_view']
        if sha(view['vial_checkpoint'])!=view['vial_checkpoint_sha256']:raise ValueError('Input-view vial model differs')
        view_model=YOLO(view['vial_checkpoint'])
    for row in prov['samples']:
        if row['split']!='val':
            continue
        if row['source_run']!=4 or sha(data/row['image'])!=row['image_sha256'] or sha(data/row['labels'])!=row['actual_label_sha256']:
            raise ValueError('Real development source differs')
        # Existing two-class source becomes a temporary liquid-only evaluation view.
        rel='real-evaluation-labels/'+row['source_id']+'.txt';path=a.out/rel;path.parent.mkdir(exist_ok=True)
        lines=['0 '+' '.join(line.split()[1:]) for line in (data/row['labels']).read_text().splitlines() if line.split()[0]=='1']
        path.write_text('\n'.join(lines)+('\n' if lines else ''),encoding='utf-8')
        evaluation_row=dict(source_id=row['source_id'],image=str(data/row['image']),label=rel)
        if view_model is not None:
            result=view_model.predict([str(data/row['image'])],**VIAL_OPTIONS)[0]
            vials=[dict(class_id=int(p[5]),box=p[:4].tolist(),confidence=float(p[4])) for p in result.boxes.data.cpu().numpy() if int(p[5])==0]
            evaluation_row.update(inference_view=view,view_vials=vials)
            view_evidence.append(dict(source_id=row['source_id'],image_sha256=row['image_sha256'],vial_predictions=vials))
        real.append(evaluation_row)
    if view_model is not None:write(a.out/'real-input-view-evidence.json',view_evidence)
    if len(real)!=157:
        raise ValueError('Expected 157 Run 4 evaluation images')
    real_metrics,real_rows=evaluate(model,real,a.out,'cuda');write(a.out/'real-development-predictions.json',real_rows)
    report=dict(schema_version=1,kind=plan['kind'],checkpoint_sha256=sha(checkpoint),synthetic_development=synthetic,
        real_run4_machine_agreement=real_metrics,training_images=plan['counts']['train']['images'],synthetic_development_images=plan['counts']['val']['images'],
        real_evaluation_images=157,training_real_runs=[],selection_real_runs=[],run5_used=False,promoted=False,
        observed_proxy_ap_gap=synthetic['ap50_95_101point']-real_metrics['ap50_95_101point'],
        accuracy_gap_interpretation='Different geometric/provisional target sources and scene distributions; proxy generalization gap, not a calibrated physical accuracy gap.',
        real_mask_or_defect_accuracy=None,limits=plan['limitations'])
    write(a.out/'comparison.json',report)
    write(a.out/'completed.json',dict(fit_epochs=epochs,selected_epoch=selected_epoch,model_sha256=sha(checkpoint),plan_sha256=sha(a.out/'experiment-plan.json'),
        artifacts={p.name:sha(p) for p in a.out.iterdir() if p.is_file()}))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','fit'])
    p.add_argument('--tools',type=Path,default=Path.home()/'ai-club-tools')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--augment',action='store_true',help='Prepare synthetic RGB nuisance augmentation arm')
    a=p.parse_args()
    (prepare if a.action=='prepare' else fit)(a)
