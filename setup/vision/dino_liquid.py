"""Custom liquid detector: frozen DINOv3 ConvNeXt + TorchVision Faster R-CNN."""
from collections import OrderedDict
import torch
from torch import nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import FeaturePyramidNetwork,MultiScaleRoIAlign
from transformers import AutoModel


class DinoPyramid(nn.Module):
    def __init__(self,backbone):
        super().__init__()
        self.encoder=AutoModel.from_pretrained(str(backbone),local_files_only=True)
        self.encoder.requires_grad_(False);self.encoder.eval()
        self.fpn=FeaturePyramidNetwork(self.encoder.config.hidden_sizes,256)
        self.out_channels=256

    def train(self,mode=True):
        super().train(mode);self.encoder.eval();return self

    def forward(self,x):
        with torch.no_grad():features=self.encoder(x,output_hidden_states=True).hidden_states[1:]
        if len(features)!=4:raise ValueError('Unexpected DINOv3 feature contract')
        return self.fpn(OrderedDict((str(i),f) for i,f in enumerate(features)))


def build(backbone,coco_head=None):
    model=FasterRCNN(DinoPyramid(backbone),num_classes=2,min_size=480,max_size=640,
        image_mean=[.485,.456,.406],image_std=[.229,.224,.225],
        rpn_anchor_generator=AnchorGenerator(((32,),(64,),(128,),(256,)),((.5,1.,2.),)*4),
        box_roi_pool=MultiScaleRoIAlign(['0','1','2','3'],output_size=7,sampling_ratio=2),
        box_score_thresh=.001,box_nms_thresh=.45,box_detections_per_img=300)
    if coco_head:
        state=torch.load(coco_head,map_location='cpu',weights_only=True)
        for prefix,module in [('rpn.',model.rpn),('roi_heads.box_head.',model.roi_heads.box_head)]:
            module.load_state_dict({k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)})
    return model


def head_state(model):
    return {k:v.detach().cpu() for k,v in model.state_dict().items() if not k.startswith('backbone.encoder.')}


def load(backbone,checkpoint,device):
    model=build(backbone)
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    missing,unexpected=model.load_state_dict(state['head_state'],strict=False)
    if unexpected or not missing or any(not k.startswith('backbone.encoder.') for k in missing):
        raise ValueError('Detector head checkpoint is incomplete/incompatible')
    return model.to(device).eval()
