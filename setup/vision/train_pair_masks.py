"""GB10-only synthetic RGB/difference segmentation and presence ablations.

Geometric supervision is not reviewed optical mask truth. Real images never
enter fitting or checkpoint selection. References at deployment require review.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

ROI=[256,240,368,432]
ANCHOR=[32,160,80,208]
MODES=['single_rgb','paired_diff','paired_diff_gain']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def build_model(channels):
    import torch
    from torch import nn
    import torch.nn.functional as F
    def block(a,b):return nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.GroupNorm(4,b),nn.SiLU(),
        nn.Conv2d(b,b,3,padding=1),nn.GroupNorm(4,b),nn.SiLU())
    class Net(nn.Module):
        def __init__(self):
            super().__init__();self.a=block(channels,16);self.b=block(16,32);self.c=block(32,64);self.d=block(64,128)
            self.u=block(192,64);self.v=block(96,32);self.w=block(48,16)
            self.mask=nn.Conv2d(16,1,1);self.presence=nn.Linear(256,1)
        def forward(self,x):
            a=self.a(x);b=self.b(F.avg_pool2d(a,2));c=self.c(F.avg_pool2d(b,2));d=self.d(F.avg_pool2d(c,2))
            p=self.presence(torch.cat([d.mean((2,3)),d.amax((2,3))],1)).squeeze(1)
            u=self.u(torch.cat([F.interpolate(d,size=c.shape[-2:],mode='bilinear',align_corners=False),c],1))
            v=self.v(torch.cat([F.interpolate(u,size=b.shape[-2:],mode='bilinear',align_corners=False),b],1))
            w=self.w(torch.cat([F.interpolate(v,size=a.shape[-2:],mode='bilinear',align_corners=False),a],1))
            return self.mask(w).squeeze(1),p
    return Net()


def features(current,reference,current_anchor,reference_anchor,mode):
    import torch
    if mode=='single_rgb':return current
    if mode=='paired_diff_gain':
        ratio=((reference_anchor+.005)/(current_anchor+.005)).clamp(.65,1.55)
        current=(current*ratio[:,:,None,None]).clamp(0,1)
    difference=current-reference
    return torch.cat([difference,difference.abs()],1)


def data(bundle):
    import numpy as np
    import torch
    from PIL import Image
    rows=read(bundle/'specs/paired-native.json');byid={r['id']:r for r in rows}
    manifest=read(bundle/'manifest.json');items=[]
    x,y,r,b=ROI;a,c,d,e=ANCHOR
    for row in rows:
        ref=byid[f'pair-{row["group"]:03d}-empty'];rel='data/paired/images/'+row['id']+'.png'
        refrel='data/paired/images/'+ref['id']+'.png';maskrel='data/paired/masks/'+row['id']+'.png'
        for name in [rel,refrel,maskrel]:
            if sha(bundle/name)!=manifest[name]:raise ValueError('Synthetic pixels or masks changed')
        rgb=np.asarray(Image.open(bundle/rel).convert('RGB'),dtype=np.float32)/255
        reference=np.asarray(Image.open(bundle/refrel).convert('RGB'),dtype=np.float32)/255
        mask=np.asarray(Image.open(bundle/maskrel))>0
        items.append(dict(id=row['id'],group=row['group'],state=row['state'],split=row['split'],
            current=torch.from_numpy(rgb[y:b,x:r].transpose(2,0,1).copy()),
            reference=torch.from_numpy(reference[y:b,x:r].transpose(2,0,1).copy()),
            current_anchor=torch.tensor(np.median(rgb[c:e,a:d],axis=(0,1))),
            reference_anchor=torch.tensor(np.median(reference[c:e,a:d],axis=(0,1))),
            mask=torch.from_numpy(mask[y:b,x:r].astype(np.float32))))
    train={r['group'] for r in items if r['split']=='train'};val={r['group'] for r in items if r['split']=='val'}
    assert len(train)==36 and len(val)==12 and not train&val
    return items


def batch(items,device):
    import torch
    return {key:torch.stack([r[key] for r in items]).to(device) for key in ['current','reference','current_anchor','reference_anchor','mask']}


def evaluate(model,items,mode,condition='raw'):
    import torch
    model.eval();records=[]
    with torch.inference_mode():
        for start in range(0,len(items),8):
            group=items[start:start+8];v=batch(group,'cuda')
            if condition=='current_exposure085':v['current']*=.85;v['current_anchor']*=.85
            if condition=='reference_shift2':v['reference']=torch.roll(v['reference'],(0,2),(-2,-1))
            logits,presence=model(features(v['current'],v['reference'],v['current_anchor'],v['reference_anchor'],mode))
            mask=logits.sigmoid()>=.5;truth=v['mask']>0;prob=presence.sigmoid()
            for i,row in enumerate(group):
                inter=(mask[i]&truth[i]).sum().item();union=(mask[i]|truth[i]).sum().item();positive=bool(truth[i].any())
                records.append(dict(id=row['id'],state=row['state'],positive=positive,presence=float(prob[i]),
                    presence_correct=bool((prob[i]>=.5).item())==positive,mask_pixels=int(mask[i].sum()),
                    geometric_iou=inter/union if union else 1.))
    positives=[r for r in records if r['positive']];negatives=[r for r in records if not r['positive']]
    accuracy=sum(r['presence_correct'] for r in records)/len(records)
    iou=sum(r['geometric_iou'] for r in positives)/len(positives)
    return dict(condition=condition,images=len(records),presence_accuracy=accuracy,positive_mean_geometric_iou=iou,
        negative_presence_alarms=sum(r['presence']>=.5 for r in negatives),negative_frames=len(negatives),
        negative_nonempty_masks=sum(r['mask_pixels']>0 for r in negatives),selection_score=accuracy+iou),records


def main(a):
    import fcntl
    import os
    a.out.mkdir(parents=True,exist_ok=False)
    write(a.out/'queued.json',dict(pid=os.getpid(),queued=time.time(),gpu='GB10',local_gpu_used=False))
    write(a.out/'suite-plan.json',dict(modes=MODES,epochs=60,batch_size=8,seed=19,roi=ROI,background_anchor=ANCHOR,
        initialization='Random; no generic pretraining or real CSI fitting',optimizer='AdamW lr.001 weight_decay.0001',
        mask_threshold=.5,presence_threshold=.5,checkpoint_selection='Highest synthetic presence accuracy plus mean positive geometric IoU, every5 epochs; earliest tie.',
        supervision='Native geometric body mask; stream-only/empty are body negatives, not optical pixel truth.',
        augmentation='Shared RGB gains/bias, independent small gains/bias/noise, reference shifts up to1px. Encoded RGB only.',
        training_real_runs=[],selection_real_runs=[],run5_used=False,helper_sha256=sha(__file__)))
    lock=(Path.home()/'ai-club-tools/spark-gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX)
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    write(a.out/'running.json',dict(pid=os.getpid(),started=time.time(),device=torch.cuda.get_device_name(),torch=torch.__version__))
    items=data(a.bundle);train=[r for r in items if r['split']=='train'];val=[r for r in items if r['split']=='val']
    for mode in MODES:
        out=a.out/mode;out.mkdir();torch.manual_seed(19)
        model=build_model(3 if mode=='single_rgb' else 6).cuda()
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        torch.manual_seed(2719)  # Match data order/augmentation across input modes.
        best=-1.;curve=[];selected_epoch=None;started=time.perf_counter()
        for epoch in range(1,61):
            model.train();losses=[];order=torch.randperm(len(train)).tolist()
            for start in range(0,len(train),8):
                v=batch([train[i] for i in order[start:start+8]],'cuda');n=v['current'].shape[0]
                gain=.8+.4*torch.rand(n,3,1,1,device='cuda');bias=.04*(torch.rand(n,3,1,1,device='cuda')-.5)
                for key,anchor in [('current','current_anchor'),('reference','reference_anchor')]:
                    independent=.95+.1*torch.rand(n,3,1,1,device='cuda');offset=.02*(torch.rand(n,3,1,1,device='cuda')-.5)
                    v[key]=((v[key]*gain+bias)*independent+offset+torch.randn_like(v[key])/255).clamp(0,1)
                    v[anchor]=((v[anchor]*gain.squeeze(-1).squeeze(-1)+bias.squeeze(-1).squeeze(-1))*independent.squeeze(-1).squeeze(-1)+offset.squeeze(-1).squeeze(-1)).clamp(0,1)
                shift=tuple(torch.randint(-1,2,(2,)).tolist());v['reference']=torch.roll(v['reference'],shift,(-2,-1))
                logits,presence=model(features(v['current'],v['reference'],v['current_anchor'],v['reference_anchor'],mode))
                positive=v['mask'].flatten(1).sum(1)>0;prob=logits.sigmoid()
                dice=(2*(prob*v['mask']).sum((1,2))+1)/(prob.sum((1,2))+v['mask'].sum((1,2))+1)
                loss=F.binary_cross_entropy_with_logits(logits,v['mask'],pos_weight=torch.tensor(3.,device='cuda'))
                if positive.any():loss=loss+(1-dice[positive].mean())
                loss=loss+.5*F.binary_cross_entropy_with_logits(presence,positive.float())
                if not torch.isfinite(loss):raise ValueError('Nonfinite mask/presence loss')
                optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();losses.append(float(loss.detach()))
            entry=dict(epoch=epoch,mean_loss=sum(losses)/len(losses),elapsed_s=time.perf_counter()-started)
            if epoch%5==0:
                metrics,_=evaluate(model,val,mode);entry['synthetic_development']=metrics
                if metrics['selection_score']>best:
                    best=metrics['selection_score'];selected_epoch=epoch
                    torch.save(dict(model_state={k:v.detach().cpu() for k,v in model.state_dict().items()},mode=mode,epoch=epoch),out/'best.pt')
                print(json.dumps(dict(mode=mode,**entry)),flush=True)
            curve.append(entry);write(out/'curve.json',curve)
        state=torch.load(out/'best.pt',map_location='cpu',weights_only=True);model.load_state_dict(state['model_state'])
        write(out/'frozen-model.json',dict(mode=mode,epoch=selected_epoch,fit_epochs=60,checkpoint_sha256=sha(out/'best.pt'),
            roi=ROI,anchor=ANCHOR,architecture='Small GroupNorm U-Net16/32/64/128 + presence head',
            initialization='Random',training_images=len(train),development_images=len(val),
            training_real_runs=[],selection_real_runs=[],run5_used=False,helper_sha256=sha(__file__)))
        report={}
        for condition in ['raw','current_exposure085','reference_shift2']:
            metrics,records=evaluate(model,val,mode,condition);report[condition]=metrics;write(out/(condition+'-predictions.json'),records)
        write(out/'synthetic-development.json',report)
        write(out/'completed.json',dict(checkpoint_sha256=sha(out/'best.pt'),artifacts={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
        del model;torch.cuda.empty_cache()
    write(a.out/'completed.json',dict(modes=MODES,finished=time.time(),real_training=False,run5_used=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
