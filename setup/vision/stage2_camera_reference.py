"""Experimental background-referenced camera features; no product calibration.

Correction anchors are fixed background patches, never liquid pixels. A patch is
not assumed to be a certified neutral standard. All outputs remain diagnostic.
"""
from dataclasses import dataclass
import numpy as np
from replay.measurement import rgb_to_lab


PATCHES={f'x{x}-y{y}':[x,y,x+48,y+48] for y in (160,224,288) for x in (32,112,192)}
METHODS=('raw','srgb_gain','linear_gain')


def linearize(rgb):
    x=np.asarray(rgb,float)/255
    return np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)


def encode(linear):
    x=np.maximum(linear,0)
    return np.clip(np.where(x<=.0031308,12.92*x,1.055*x**(1/2.4)-.055)*255,0,255)


def correction(pixels,current_anchors,reference_anchors,method):
    values=np.asarray(pixels,float);anchors=np.asarray(current_anchors,float);reference=np.asarray(reference_anchors,float)
    if method=='raw':return values.copy(),np.ones(3),0.
    if anchors.shape!=reference.shape or anchors.ndim!=2 or anchors.shape[1]!=3:
        raise ValueError('Anchor RGB arrays must match')
    if not np.isfinite([anchors,reference]).all() or np.any(reference<12) or np.any(anchors>=245):
        return None,None,None
    if method=='linear_gain':
        ratios=linearize(anchors)/np.maximum(linearize(reference),1e-6)
        gain=np.median(ratios,axis=0)
        corrected=encode(linearize(values)/gain)
    elif method=='srgb_gain':
        ratios=anchors/np.maximum(reference,1e-6);gain=np.median(ratios,axis=0)
        corrected=np.clip(values/gain,0,255)
    else:raise ValueError('Unknown correction')
    disagreement=float(np.max(np.abs(ratios/gain-1)))
    return corrected,gain,disagreement


def feature(pixels):
    values=np.asarray(pixels,float)
    usable=(values.max(axis=1)<245)&(values.max(axis=1)>3)
    if not usable.any():return None,0.
    return rgb_to_lab(np.median(values[usable],axis=0)),float(usable.mean())


def perturb(rgb,gain=(1.,1.,1.),offset=0.,domain='linear'):
    if domain=='linear':return encode(np.maximum(0,linearize(rgb)*np.asarray(gain)+offset))
    if domain=='srgb':return np.clip(np.asarray(rgb,float)*np.asarray(gain)+offset,0,255)
    raise ValueError('Unknown nuisance domain')


@dataclass(frozen=True)
class ReferenceHealth:
    minimum_gain: float=.65
    maximum_gain: float=1.55
    maximum_patch_disagreement: float=.03

    def accepts(self,gain,disagreement):
        return bool(gain is not None and disagreement is not None and
                    np.isfinite(gain).all() and np.isfinite(disagreement) and
                    np.all(gain>=self.minimum_gain) and np.all(gain<=self.maximum_gain) and
                    disagreement<=self.maximum_patch_disagreement)


def anchor_health(current,reference,maximum_disagreement):
    """One shared sRGB-domain gate, independent of correction method."""
    current=np.asarray(current,float);reference=np.asarray(reference,float)
    if current.shape!=reference.shape or not np.isfinite([current,reference]).all() or np.any(reference<12) or np.any(current>=245):
        return False,None,None
    ratio=current/reference;gain=np.median(ratio,axis=0)
    spread=float(np.max(np.abs(ratio/gain-1)))
    healthy=ReferenceHealth(maximum_patch_disagreement=maximum_disagreement).accepts(gain,spread)
    return healthy,gain,spread
