"""Train a small crop segmentation model, choosing its checkpoint on Run 4 only."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import time


def validate_split(provenance):
    if provenance['train_runs']!=[1,2,3] or provenance['development_runs']!=[4] or provenance['excluded_runs']!=[5]:
        raise ValueError('Training/development split is not the frozen Run 1-3 / Run 4 split')
    rows=provenance['samples']
    for row in rows:
        if row['split'] not in ('train','val') or row['source_run'] not in ((1,2,3) if row['split']=='train' else (4,)):
            raise ValueError('A source sample crosses the frozen recording split')
    training={row['crop_rgb_sha256'] for row in rows if row['split']=='train'}
    development={row['crop_rgb_sha256'] for row in rows if row['split']=='val'}
    if not training or not development or training & development:
        raise ValueError('Missing split or duplicated crop across train/development')


def train(data,output,epochs):
    import cv2
    import numpy as np
    import torch
    from torch.utils.data import DataLoader,Dataset
    from tiny_unet import TinyUNet
    provenance=json.loads((data/'provenance.json').read_text())
    validate_split(provenance)
    for split in ('train','val'):
        if hashlib.sha256((data/f'{split}.npz').read_bytes()).hexdigest()!=provenance[f'{split}_sha256']:
            raise ValueError('Prepared crop archive hash mismatch')
    output.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(19); torch.cuda.manual_seed_all(19); np.random.seed(19); random.seed(19)
    torch.set_num_threads(4); cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    if not torch.cuda.is_available():
        raise RuntimeError('This experiment requires an allocated GPU')
    class Crops(Dataset):
        def __init__(self,split):
            values=np.load(data/f'{split}.npz',allow_pickle=False)
            self.images,self.masks=values['images'],values['masks']; self.augment=split=='train'
        def __len__(self): return len(self.images)
        def __getitem__(self,index):
            image,mask=self.images[index].copy(),self.masks[index].copy()
            if self.augment:
                h,w=mask.shape
                matrix=cv2.getRotationMatrix2D((w/2,h/2),0,np.random.uniform(.95,1.05))
                matrix[:,2]+=np.random.uniform([-5,-8],[5,8])
                image=cv2.warpAffine(image,matrix,(w,h),flags=cv2.INTER_LINEAR,borderValue=(114,114,114))
                mask=cv2.warpAffine(mask,matrix,(w,h),flags=cv2.INTER_NEAREST,borderValue=0)
                image=np.clip(image.astype(np.float32)*np.random.uniform(.9,1.1),0,255)
            return torch.from_numpy(image.transpose(2,0,1).astype(np.float32)/255),torch.from_numpy(mask[None].astype(np.float32))
    training,validation=Crops('train'),Crops('val')
    model=TinyUNet().cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs)
    pos=float(training.masks.mean()); weight=min(10.,(1-pos)/max(pos,1e-6))
    bce=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight,device='cuda'))
    def loss(logits,target):
        pred=logits.sigmoid(); dice=1-(2*(pred*target).sum((1,2,3))+1)/(pred.sum((1,2,3))+target.sum((1,2,3))+1)
        return bce(logits,target)+dice.mean()
    train_loader=DataLoader(training,batch_size=16,shuffle=True,num_workers=0)
    val_loader=DataLoader(validation,batch_size=16,shuffle=False,num_workers=0)
    best=-1; best_epoch=None; history=[]; started=time.perf_counter()
    for epoch in range(1,epochs+1):
        model.train(); losses=[]
        for image,target in train_loader:
            image,target=image.cuda(),target.cuda()
            optimizer.zero_grad(set_to_none=True); value=loss(model(image),target)
            value.backward(); optimizer.step(); losses.append(float(value.detach()))
        scheduler.step(); model.eval(); intersections=union=tp=fp=fn=0; val_losses=[]
        with torch.no_grad():
            for image,target in val_loader:
                image,target=image.cuda(),target.cuda(); logits=model(image); val_losses.append(float(loss(logits,target)))
                pred=logits>0; truth=target>0
                tp+=int((pred&truth).sum()); fp+=int((pred&~truth).sum()); fn+=int((~pred&truth).sum())
        iou=tp/max(1,tp+fp+fn)
        row=dict(epoch=epoch,train_loss=sum(losses)/len(losses),val_loss=sum(val_losses)/len(val_losses),
                 provisional_mask_iou=iou,provisional_mask_precision=tp/max(1,tp+fp),provisional_mask_recall=tp/max(1,tp+fn))
        history.append(row); print(json.dumps(row),flush=True)
        if iou>best:
            best,best_epoch=iou,epoch
            torch.save({key:value.detach().cpu() for key,value in model.state_dict().items()},output/'best.pth')
    model.load_state_dict(torch.load(output/'best.pth',map_location='cpu',weights_only=True)); model.cpu().eval()
    torch.onnx.export(model,torch.zeros(1,3,384,128),str(output/'sampling-segmenter.onnx'),
                      input_names=['rgb_crop'],output_names=['liquid_logits'],opset_version=17,dynamo=False)
    with (output/'epochs.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(history[0])); writer.writeheader(); writer.writerows(history)
    report=dict(schema_version=1,architecture='tiny_unet_16_32_64_binary_sampling_interior',parameters=sum(p.numel() for p in model.parameters()),
                model_sha256=hashlib.sha256((output/'sampling-segmenter.onnx').read_bytes()).hexdigest(),
                source_provenance_sha256=hashlib.sha256((data/'provenance.json').read_bytes()).hexdigest(),
                train_runs=[1,2,3],development_runs=[4],excluded_runs=[5],train_images=len(training),development_images=len(validation),
                epochs=epochs,best_epoch=best_epoch,best_provisional_mask_iou=best,seed=19,
                mask_probability_threshold=.5,input_size=[128,384],torch=torch.__version__,
                elapsed_s=time.perf_counter()-started,gpu=torch.cuda.get_device_name(0),
                training_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                architecture_script_sha256=hashlib.sha256((Path(__file__).parent/'tiny_unet.py').read_bytes()).hexdigest(),
                supervision='unreviewed conservative sampling masks; not full liquid outlines or human truth',
                label_positive_fraction=pos,positive_loss_weight=weight)
    (output/'model.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--data',type=Path,required=True); p.add_argument('--out',type=Path,required=True); p.add_argument('--epochs',type=int,default=40); a=p.parse_args(); train(a.data,a.out,a.epochs)
