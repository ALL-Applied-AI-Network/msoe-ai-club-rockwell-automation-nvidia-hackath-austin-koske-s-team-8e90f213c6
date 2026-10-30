"""SAM-free liquid mask/presence head over a frozen generic DINOv3 encoder.

This is a custom fluid decoder, not Meta's published segmentation checkpoint.
The earlier CSI-trained DINO/Faster R-CNN head is never loaded.
"""
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel


class DinoLiquidMask(nn.Module):
    def __init__(self,backbone,mode='rgb',width=32):
        super().__init__()
        if mode not in ('rgb','paired'):raise ValueError('Unknown mask input mode')
        self.mode=mode;self.encoder=AutoModel.from_pretrained(str(backbone),local_files_only=True)
        self.encoder.requires_grad_(False);self.encoder.eval()
        sizes=self.encoder.config.hidden_sizes
        if len(sizes)!=4:raise ValueError('Expected ConvNeXt four-stage features')
        factor=1 if mode=='rgb' else 3
        self.projections=nn.ModuleList(nn.Conv2d(c*factor,width,1) for c in sizes)
        self.fuse=nn.Sequential(nn.Conv2d(width,width,3,padding=1),nn.GroupNorm(8,width),nn.SiLU())
        self.rgb_detail=nn.Sequential(nn.Conv2d(3*factor,16,3,padding=1),nn.GroupNorm(4,16),nn.SiLU())
        self.mask_head=nn.Sequential(nn.Conv2d(width+16,width,3,padding=1),nn.GroupNorm(8,width),nn.SiLU(),nn.Conv2d(width,1,1))
        self.presence_head=nn.Linear(width*2,1)
        self.register_buffer('mean',torch.tensor([.485,.456,.406])[None,:,None,None])
        self.register_buffer('std',torch.tensor([.229,.224,.225])[None,:,None,None])

    def train(self,mode=True):
        super().train(mode);self.encoder.eval();return self

    def forward(self,current,reference,current_anchor,reference_anchor):
        if current.ndim!=4 or current.shape[1]!=3 or current.shape!=reference.shape:raise ValueError('Matched NCHW RGB required')
        if self.mode=='paired':
            gain=((reference_anchor+.005)/(current_anchor+.005)).clamp(.65,1.55)
            current=(current*gain[:,:,None,None]).clamp(0,1)
        n=len(current)
        images=current if self.mode=='rgb' else torch.cat([current,reference],0)
        with torch.no_grad():
            states=self.encoder((images-self.mean)/self.std,output_hidden_states=True).hidden_states[1:]
        if len(states)!=4:raise ValueError('Unexpected encoder feature contract')
        projected=[]
        for feature,layer in zip(states,self.projections):
            if self.mode=='paired':
                cur,ref=feature[:n],feature[n:];difference=cur-ref
                feature=torch.cat([cur,difference,difference.abs()],1)
            projected.append(layer(feature))
        size=projected[0].shape[-2:]
        pyramid=self.fuse(sum(F.interpolate(f,size=size,mode='bilinear',align_corners=False) for f in projected))
        presence=self.presence_head(torch.cat([pyramid.mean((2,3)),pyramid.amax((2,3))],1)).squeeze(1)
        detail=current
        if self.mode=='paired':
            difference=current-reference;detail=torch.cat([current,difference,difference.abs()],1)
        upsampled=F.interpolate(pyramid,size=current.shape[-2:],mode='bilinear',align_corners=False)
        logits=self.mask_head(torch.cat([upsampled,self.rgb_detail(detail)],1)).squeeze(1)
        return logits,presence

    def decoder_state(self):
        return {k:v.detach().cpu() for k,v in self.state_dict().items() if not k.startswith('encoder.')}

    def load_decoder(self,state):
        missing,unexpected=self.load_state_dict(state,strict=False)
        if unexpected or any(not k.startswith('encoder.') for k in missing):raise ValueError('Incomplete/incompatible segmentation head')
