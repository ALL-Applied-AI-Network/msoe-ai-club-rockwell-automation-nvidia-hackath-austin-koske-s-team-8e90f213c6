"""GB10-only DINOv3 direct-mask pilot, synthetic training/development only."""
import argparse
import fcntl
from pathlib import Path
import time

from train_pair_masks import data,batch,read,write,sha,ROI,ANCHOR


def evaluate(model,items):
    import torch
    records=[];model.eval()
    with torch.inference_mode():
        for start in range(0,len(items),8):
            group=items[start:start+8];v=batch(group,'cuda')
            logits,p=model(v['current'],v['reference'],v['current_anchor'],v['reference_anchor']);masks=logits.sigmoid()>=.5;presence=p.sigmoid()
            for i,row in enumerate(group):
                truth=v['mask'][i]>0;positive=bool(truth.any());prediction=masks[i];intersection=int((truth&prediction).sum());union=int((truth|prediction).sum())
                records.append(dict(id=row['id'],group=row['group'],state=row['state'],positive=positive,presence=float(presence[i]),
                    presence_correct=bool(presence[i]>=.5)==positive,geometric_iou=intersection/union if union else 1.,mask_pixels=int(prediction.sum())))
    positive=[r for r in records if r['positive']];negative=[r for r in records if not r['positive']]
    accuracy=sum(r['presence_correct'] for r in records)/len(records);iou=sum(r['geometric_iou'] for r in positive)/len(positive)
    result=dict(images=len(records),presence_accuracy=accuracy,positive_mean_geometric_iou=iou,
        negative_presence_alarms=sum(r['presence']>=.5 for r in negative),negative_frames=len(negative),selection_score=accuracy+iou,
        per_state={state:dict(images=sum(r['state']==state for r in records),presence_correct=sum(r['state']==state and r['presence_correct'] for r in records)) for state in sorted({r['state'] for r in records})})
    return result,records


def run(root):
    import torch
    import torch.nn.functional as F
    from dinov3_liquid_mask import DinoLiquidMask
    if any((root/n).exists() for n in ('running.json','fitted-v1','completed.json')):raise ValueError('Inspect existing job; duplicate refused')
    plan=read(root/'experiment-plan.json')
    for name,digest in plan['helpers'].items():
        if sha(root/name)!=digest:raise ValueError('Frozen helper differs: '+name)
    for name,digest in plan['backbone_files'].items():
        if sha(root/'pretrained'/name)!=digest:raise ValueError('Generic encoder differs')
    lock=(Path.home()/'ai-club-tools/spark-gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    write(root/'running.json',dict(started=time.time(),gpu=torch.cuda.get_device_name(0),plan_sha256=sha(root/'experiment-plan.json'),local_gpu_used=False))
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    items=data(root/'training');train=[r for r in items if r['split']=='train'];dev=[r for r in items if r['split']=='val']
    if len(train)!=144 or len(dev)!=48:raise ValueError('Unexpected synthetic family split')
    out=root/'fitted-v1';out.mkdir();frozen={}
    try:
        for mode in plan['modes']:
            arm=out/mode;arm.mkdir();torch.manual_seed(19);model=DinoLiquidMask(root/'pretrained',mode).cuda()
            optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001,weight_decay=.0001)
            torch.manual_seed(2719);curve=[];best=-1;selected=None
            for epoch in range(1,plan['epochs']+1):
                model.train();losses=[];order=torch.randperm(len(train)).tolist()
                for start in range(0,len(train),8):
                    v=batch([train[i] for i in order[start:start+8]],'cuda');n=len(v['current'])
                    gain=.8+.4*torch.rand(n,3,1,1,device='cuda');bias=.04*(torch.rand(n,3,1,1,device='cuda')-.5)
                    for key,anchor in [('current','current_anchor'),('reference','reference_anchor')]:
                        independent=.95+.1*torch.rand(n,3,1,1,device='cuda');offset=.02*(torch.rand(n,3,1,1,device='cuda')-.5)
                        v[key]=((v[key]*gain+bias)*independent+offset+torch.randn_like(v[key])/255).clamp(0,1)
                        v[anchor]=((v[anchor]*gain.squeeze(-1).squeeze(-1)+bias.squeeze(-1).squeeze(-1))*independent.squeeze(-1).squeeze(-1)+offset.squeeze(-1).squeeze(-1)).clamp(0,1)
                    v['reference']=torch.roll(v['reference'],tuple(torch.randint(-1,2,(2,)).tolist()),(-2,-1))
                    logits,presence=model(v['current'],v['reference'],v['current_anchor'],v['reference_anchor']);positive=v['mask'].flatten(1).sum(1)>0;prob=logits.sigmoid()
                    dice=(2*(prob*v['mask']).sum((1,2))+1)/(prob.sum((1,2))+v['mask'].sum((1,2))+1)
                    loss=F.binary_cross_entropy_with_logits(logits,v['mask'],pos_weight=torch.tensor(3.,device='cuda'))+.5*F.binary_cross_entropy_with_logits(presence,positive.float())
                    if positive.any():loss=loss+1-dice[positive].mean()
                    if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                    optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5.);optimizer.step();losses.append(float(loss.detach()))
                entry=dict(epoch=epoch,mean_loss=sum(losses)/len(losses))
                if epoch%5==0:
                    metrics,_=evaluate(model,dev);entry['synthetic_development']=metrics
                    if metrics['selection_score']>best:
                        best=metrics['selection_score'];selected=epoch;torch.save(dict(decoder_state=model.decoder_state(),mode=mode,epoch=epoch),arm/'best.pt')
                    print(mode,epoch,metrics,flush=True)
                curve.append(entry);write(arm/'curve.json',curve)
            checkpoint=torch.load(arm/'best.pt',map_location='cpu',weights_only=True);model.load_decoder(checkpoint['decoder_state']);metrics,records=evaluate(model,dev)
            meta=dict(mode=mode,selected_epoch=selected,checkpoint_sha256=sha(arm/'best.pt'),synthetic_development=metrics,
                encoder_parameters=sum(p.numel() for p in model.encoder.parameters()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                generic_pretraining='DINOv3 ConvNeXt Small LVD-1689M',CSI_training='144 synthetic images /36 families',real_training_runs=[],run5_used=False,
                input_roi=ROI,reference_anchor=ANCHOR,SAM_used=False,backbone_frozen=True)
            write(arm/'development-predictions.json',records);write(arm/'frozen.json',meta);frozen[mode]=meta
        write(out/'all-models-frozen.json',dict(utc=time.time(),models=frozen,plan_sha256=sha(root/'experiment-plan.json')))
        write(root/'completed.json',dict(finished=time.time(),gpu='GB10',fits=len(frozen),model_metadata_sha256=sha(out/'all-models-frozen.json'),real_accuracy=None))
    except BaseException as error:write(root/'failed.json',dict(type=type(error).__name__,error=str(error)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root.resolve())
