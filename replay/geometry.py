"""Causal box detections, short-lived vial tracks and conservative liquid masks.

The learned boxes are proposals, not proof of segmentation accuracy. No mask is
propagated into a frame lacking fresh detections. No recipe/annotation is used.
"""
import hashlib
from pathlib import Path

import cv2
import numpy as np


def encode_mask(mask):
    ys, xs = np.where(mask)
    if not len(xs):
        return None
    x, y, w, h = int(xs.min()), int(ys.min()), int(xs.max()-xs.min()+1), int(ys.max()-ys.min()+1)
    flat = mask[y:y+h, x:x+w].astype(np.uint8).reshape(-1)
    changes = np.diff(np.r_[0, flat, 0].astype(np.int16))
    starts, ends = np.where(changes == 1)[0], np.where(changes == -1)[0]
    return dict(origin=[x,y], size=[w,h], runs=[[int(a),int(b-a)] for a,b in zip(starts,ends)])


def decode_mask(encoded, shape=(480,640)):
    mask = np.zeros(shape, dtype=np.uint8)
    if encoded is None:
        return mask
    x,y = encoded['origin']; w,h = encoded['size']
    if not 0 <= x < x+w <= shape[1] or not 0 <= y < y+h <= shape[0]:
        raise ValueError('Mask lies outside source frame')
    flat = np.zeros(w*h, dtype=np.uint8)
    for start,length in encoded['runs']:
        if start < 0 or length <= 0 or start+length > len(flat):
            raise ValueError('Invalid mask run')
        flat[start:start+length] = 1
    mask[y:y+h,x:x+w] = flat.reshape(h,w)
    return mask


def iou(a,b):
    inter = max(0,min(a[2],b[2])-max(a[0],b[0])) * max(0,min(a[3],b[3])-max(a[1],b[1]))
    return inter / max(1,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)


class OnnxDetector:
    def __init__(self,path,sha256):
        if not sha256 or hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha256:
            raise ValueError('Detector file does not match the explicitly pinned model hash')
        cv2.setNumThreads(min(4,cv2.getNumThreads()))
        self.net = cv2.dnn.readNetFromONNX(str(path))

    def detect(self,rgb):
        h,w = rgb.shape[:2]
        scale = 640/max(h,w)
        rw,rh = round(w*scale),round(h*scale)
        left,top = (640-rw)//2,(640-rh)//2
        canvas = np.full((640,640,3),114,np.uint8)
        canvas[top:top+rh,left:left+rw] = cv2.resize(rgb,(rw,rh))
        self.net.setInput(cv2.dnn.blobFromImage(canvas,1/255.,(640,640),swapRB=False))
        prediction = self.net.forward()[0].T
        result=[]
        for cls in (0,1):
            selected=prediction[prediction[:,4+cls]>=.35]
            if not len(selected):
                continue
            xywh=selected[:,:4].copy()
            xywh[:,:2] -= xywh[:,2:4]/2
            kept=cv2.dnn.NMSBoxes(xywh.tolist(),selected[:,4+cls].tolist(),.35,.45)
            for k in np.asarray(kept).reshape(-1):
                x,y,bw,bh=xywh[k]
                box=[max(0,int((x-left)/scale)),max(0,int((y-top)/scale)),
                     min(w,int((x+bw-left)/scale)),min(h,int((y+bh-top)/scale))]
                if box[2]>box[0] and box[3]>box[1]:
                    result.append(dict(class_id=cls,box=box,confidence=float(selected[k,4+cls])))
        return result


class VialGeometry:
    def __init__(self,config,detector=None):
        self.config=config
        self.detector=detector or (OnnxDetector(config.vision_model_path,config.vision_model_sha256) if config.vision_model_path else None)
        self.previous=None
        self.serial=0
        self.age=0

    def detect(self,rgb):
        return self.detector.detect(rgb) if self.detector else []

    def update(self,rgb,moving=False):
        boxes=self.detect(rgb)
        evidence=dict(model_sha256=self.config.vision_model_sha256, proposals=boxes, vial_box=None, liquid_box=None,
                      track_id=None, track_age=0, mask=None, mask_pixels=0,
                      verification_status='experimental_weak_supervision', quality_flags=[])
        flags=evidence['quality_flags']
        vials=[p for p in boxes if p['class_id']==0]
        if not vials:
            self.previous=None; self.age=0
            flags.append('vial_not_detected' if self.detector else 'vial_detector_not_configured')
            return evidence
        # Current fill position is a camera-profile constraint, not a color ROI.
        # Other visible vials cannot supply the color of the vial under the nozzle.
        center=(self.config.stream_center_roi[0]+self.config.stream_center_roi[2])/2
        candidates=[p for p in vials if p['box'][0] < center < p['box'][2] and p['box'][3]-p['box'][1]>180]
        if len(candidates)!=1:
            self.previous=None; self.age=0
            flags.append('fill_position_vial_missing_or_ambiguous')
            return evidence
        vial=candidates[0]['box']
        if moving or self.previous is None or iou(vial,self.previous)<.65:
            self.serial+=1; self.age=1
        else:
            self.age+=1
        self.previous=vial
        evidence.update(vial_box=vial,track_id=self.serial,track_age=self.age)
        if self.age<3:
            flags.append('vial_track_not_stable')
        liquids=[]
        for p in boxes:
            if p['class_id']!=1:
                continue
            a=p['box']; area=(a[2]-a[0])*(a[3]-a[1])
            inside=max(0,min(a[2],vial[2])-max(a[0],vial[0]))*max(0,min(a[3],vial[3])-max(a[1],vial[1]))
            if inside/max(1,area)>=.9:
                liquids.append(p)
        if len(liquids)!=1:
            flags.append('liquid_missing_or_ambiguous_inside_vial')
            return evidence
        liquid=liquids[0]['box']; evidence['liquid_box']=liquid
        vx,vy,vr,vb=vial; vw,vh=vr-vx,vb-vy
        # Exclude glass walls/base and the upper label/holder. Sampling the lower
        # interior intentionally sacrifices coverage for dark/clear small fills.
        x0=max(liquid[0]+3,int(vx+.2*vw)); x1=min(liquid[2]-3,int(vr-.2*vw))
        y0=max(liquid[1]+4,int(vy+.58*vh)); y1=min(liquid[3]-5,int(vb-.07*vh))
        mask=np.zeros(rgb.shape[:2],np.uint8)
        if x1<=x0 or y1<=y0:
            flags.append('liquid_interior_too_small')
            return evidence
        hsv=cv2.cvtColor(rgb[y0:y1,x0:x1],cv2.COLOR_RGB2HSV)
        candidate=((hsv[:,:,1]>=45)&(hsv[:,:,2]>=12)&(hsv[:,:,2]<240)).astype(np.uint8)
        # Keep a coherent colored interior, never white glare, clear glass,
        # neutral background, or a disconnected edge reflection.
        candidate=cv2.morphologyEx(candidate,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
        n,labels,stats,_=cv2.connectedComponentsWithStats(candidate)
        if n>1:
            largest=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))
            region=(labels==largest).astype(np.uint8)
            region=cv2.erode(region,np.ones((3,3),np.uint8))
            mask[y0:y1,x0:x1]=region
        pixels=int(mask.sum())
        evidence.update(mask=encode_mask(mask),mask_pixels=pixels)
        if pixels<60 or pixels/max(1,(x1-x0)*(y1-y0))<.35:
            flags.append('insufficient_coherent_colored_liquid')
        return evidence
