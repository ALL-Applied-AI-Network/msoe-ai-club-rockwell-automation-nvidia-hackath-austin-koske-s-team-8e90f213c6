"""Pinned binary sampling-interior model on a current detected vial crop."""
import hashlib
from pathlib import Path

import cv2
import numpy as np

from .geometry import encode_mask

INPUT_SIZE = (128, 384)  # width, height; shared with preparation/training


def crop_box(vial, shape):
    x0,y0,x1,y1=vial
    w,h=x1-x0,y1-y0
    return [max(0,int(x0-.10*w)),max(0,int(y0-.03*h)),
            min(shape[1],int(x1+.10*w)),min(shape[0],int(y1+.03*h))]


def crop_input(rgb,vial):
    box=crop_box(vial,rgb.shape)
    x0,y0,x1,y1=box
    return cv2.resize(rgb[y0:y1,x0:x1],INPUT_SIZE,interpolation=cv2.INTER_LINEAR),box


class SamplingSegmenter:
    def __init__(self,path,sha256):
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=sha256:
            raise ValueError('Sampling segmenter does not match the pinned model hash')
        cv2.setNumThreads(min(4,cv2.getNumThreads()))
        self.net=cv2.dnn.readNetFromONNX(str(path))

    def propose(self,rgb,vial):
        image,box=crop_input(rgb,vial)
        self.net.setInput(cv2.dnn.blobFromImage(image,1/255.,INPUT_SIZE,swapRB=False))
        logits=self.net.forward()[0,0]
        x0,y0,x1,y1=box
        logits=cv2.resize(logits,(x1-x0,y1-y0),interpolation=cv2.INTER_LINEAR)
        # Threshold fixed before Run 5 evaluation. This is a learned proposal;
        # all existing geometry, motion, color and window QA still follows.
        candidate=(logits>0).astype(np.uint8)  # sigmoid probability > .5
        count,labels,stats,_=cv2.connectedComponentsWithStats(candidate)
        proposals=[]
        for k in range(1,count):
            if stats[k,cv2.CC_STAT_AREA]<60:
                continue
            x,y,w,h,area=map(int,stats[k])
            mask=np.zeros(rgb.shape[:2],np.uint8)
            mask[y0:y1,x0:x1]=(labels==k)
            probability=1/(1+np.exp(-np.clip(logits[labels==k],-30,30)))
            proposals.append(dict(class_id=1,box=[x0+x,y0+y,x0+x+w,y0+y+h],
                                  confidence=float(probability.mean()),candidate_mask=encode_mask(mask)))
        return proposals
