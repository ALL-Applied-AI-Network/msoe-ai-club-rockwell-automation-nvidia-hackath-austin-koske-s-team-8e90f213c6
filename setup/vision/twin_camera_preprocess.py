"""Experimental full-frame correction from prior background references only."""
import numpy as np
from stage2_camera_reference import anchor_health
from twin_diagnostics import PATCHES,SETTINGS


def anchors(rgb):
    return np.asarray([np.median(rgb[y:y+48,x:x+48].reshape(-1,3),axis=0) for x,y in PATCHES])


def normalize(rgb,reference):
    current=anchors(rgb)
    healthy,gain,spread=anchor_health(current,reference,SETTINGS['maximum_anchor_disagreement'])
    metadata=dict(healthy=bool(healthy),gain=gain.tolist() if gain is not None else None,spread=spread,
        anchors=current.tolist(),method='rounded uint8 sRGB division before all inference/QA',
        clipped_output_fraction=None)
    if not healthy:return None,metadata
    corrected=rgb.astype(float)/gain
    metadata['clipped_output_fraction']=float(((corrected<0)|(corrected>255)).any(axis=2).mean())
    return np.clip(np.rint(corrected),0,255).astype(np.uint8),metadata
