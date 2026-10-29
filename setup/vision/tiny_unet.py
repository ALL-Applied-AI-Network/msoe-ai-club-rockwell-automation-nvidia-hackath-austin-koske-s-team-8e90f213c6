"""Small binary U-Net; explicit architecture for the provisional-mask experiment."""
import torch
from torch import nn


def block(left,right):
    return nn.Sequential(nn.Conv2d(left,right,3,padding=1),nn.ReLU(),
                         nn.Conv2d(right,right,3,padding=1),nn.ReLU())


class TinyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.a=block(3,16); self.b=block(16,32); self.c=block(32,64)
        self.pool=nn.MaxPool2d(2)
        self.up_b=block(96,32); self.up_a=block(48,16)
        self.out=nn.Conv2d(16,1,1)

    def forward(self,x):
        a=self.a(x); b=self.b(self.pool(a)); c=self.c(self.pool(b))
        c=nn.functional.interpolate(c,scale_factor=2,mode='bilinear',align_corners=False)
        b=self.up_b(torch.cat([b,c],dim=1))
        b=nn.functional.interpolate(b,scale_factor=2,mode='bilinear',align_corners=False)
        return self.out(self.up_a(torch.cat([a,b],dim=1)))
