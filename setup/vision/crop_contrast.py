"""Shared source-coordinate crop and lightness CLAHE for detector experiments."""
import cv2
import numpy as np


def enhance_crop(rgb,roi):
    x,y,r,b=roi
    if not 0<=x<r<=rgb.shape[1] or not 0<=y<b<=rgb.shape[0]:raise ValueError('Invalid crop')
    crop=rgb[y:b,x:r].copy()
    lab=cv2.cvtColor(crop,cv2.COLOR_RGB2LAB)
    lab[:,:,0]=cv2.createCLAHE(clipLimit=2.,tileGridSize=(8,8)).apply(lab[:,:,0])
    return cv2.cvtColor(lab,cv2.COLOR_LAB2RGB)


def read_labels(text,shape=(480,640)):
    h,w=shape;boxes=[]
    for line in text.splitlines():
        if not line.strip():continue
        values=list(map(float,line.split()))
        if len(values)!=5 or not np.isfinite(values).all():raise ValueError('Invalid box label')
        c,x,y,bw,bh=values
        if c not in (0,1) or not 0<bw<=1 or not 0<bh<=1:raise ValueError('Invalid class/size')
        box=[(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h]
        if box[0]<-1e-4 or box[1]<-1e-4 or box[2]>w+1e-4 or box[3]>h+1e-4:raise ValueError('Box outside source')
        boxes.append(dict(class_id=int(c),box=box))
    return boxes


def crop_labels(boxes,roi):
    x,y,r,b=roi;w,h=r-x,b-y;lines=[]
    for item in boxes:
        x0,y0,x1,y1=item['box']
        if x0<x-1e-4 or y0<y-1e-4 or x1>r+1e-4 or y1>b+1e-4:
            raise ValueError('Crop truncates a target; do not silently remove or clip it')
        lines.append(f"{item['class_id']} {((x0+x1)/2-x)/w:.9f} {((y0+y1)/2-y)/h:.9f} {(x1-x0)/w:.9f} {(y1-y0)/h:.9f}")
    return '\n'.join(lines)+'\n'


def project_boxes(predictions,roi):
    x,y,r,b=roi;result=[]
    for item in predictions:
        box=item['box']
        if not 0<=box[0]<box[2]<=r-x or not 0<=box[1]<box[3]<=b-y:
            raise ValueError('Prediction outside crop coordinates')
        result.append(dict(item,box=[float(box[0]+x),float(box[1]+y),float(box[2]+x),float(box[3]+y)]))
    return result
