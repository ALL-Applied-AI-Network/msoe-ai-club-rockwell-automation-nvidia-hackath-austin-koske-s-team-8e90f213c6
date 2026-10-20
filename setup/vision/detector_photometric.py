"""Training-only image changes; never touches box targets or global RNG."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


SPEC = dict(contrast=dict(probability=.5, factor=[.85,1.15], center=127.5),
    blur=dict(probability=.15, kernel=3, sigma=[.2,.7]),
    noise=dict(probability=.25, sigma_8bit=[1.,4.]),
    order=['contrast','blur','noise'], hue_shift=False, geometric_change=False)


class Photometric:
    def __init__(self, seed=190021, preview=None):
        self.rng=np.random.default_rng(seed);self.preview=Path(preview) if preview else None
        self.counts=Counter();self.examples=[]

    def __call__(self, labels):
        im=labels['img']
        if im.dtype!=np.uint8 or im.ndim!=3 or im.shape[2]!=3:
            raise ValueError('Photometric augmentation expects uint8 BGR')
        self.counts['calls']+=1;params={};result=im.copy()
        if self.rng.random()<.5:
            factor=float(self.rng.uniform(.85,1.15));params['contrast_factor']=factor
            result=np.clip((result.astype(np.float32)-127.5)*factor+127.5,0,255).astype(np.uint8)
        if self.rng.random()<.15:
            sigma=float(self.rng.uniform(.2,.7));params['blur_sigma']=sigma
            result=cv2.GaussianBlur(result,(3,3),sigmaX=sigma)
        if self.rng.random()<.25:
            sigma=float(self.rng.uniform(1.,4.));params['noise_sigma_8bit']=sigma
            result=np.clip(result.astype(np.float32)+self.rng.normal(0,sigma,result.shape),0,255).astype(np.uint8)
        for key in params:self.counts[key]+=1
        changed=not np.array_equal(im,result)
        self.counts['changed_images']+=changed
        if changed and self.preview and len(self.examples)<12:
            self.preview.mkdir(parents=True,exist_ok=True);n=len(self.examples)
            for name,value in [('original',im),('augmented',result)]:
                if not cv2.imwrite(str(self.preview/f'{n:02d}-{name}.png'),value):raise IOError('Preview save failed')
            self.examples.append(dict(index=n,source_image=labels.get('im_file'),parameters=params,
                original_pixels_sha256=hashlib.sha256(im.tobytes()).hexdigest(),
                augmented_pixels_sha256=hashlib.sha256(result.tobytes()).hexdigest()))
        labels['img']=result
        return labels

    def save(self, path):
        Path(path).write_text(json.dumps(dict(spec=SPEC,counts=dict(self.counts),examples=self.examples,
            label_geometry_changed=False,validation_augmented=False),indent=2)+'\n',encoding='utf-8')
