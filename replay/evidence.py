"""Exact decoded source-frame and mask previews; never re-run the tracker."""
from functools import lru_cache
import hashlib

import av
import cv2
import numpy as np

from .geometry import decode_mask


@lru_cache(maxsize=8)
def source_frame(path, video_sha256, index, fps, start_pts_s):
    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO'
        target = start_pts_s + index / fps
        container.seek(int(target / float(stream.time_base)), stream=stream, backward=True)
        for frame in container.decode(video=0):
            if frame.pts is None:
                continue
            at = round((float(frame.pts * frame.time_base) - start_pts_s) * fps)
            if at == index:
                return frame.to_ndarray(format='rgb24')
            if at > index:
                break
    raise ValueError('Exact measurement frame could not be recovered')


def images(rgb, geometry):
    mask=decode_mask(geometry.get('mask') if geometry else None,rgb.shape[:2])
    overlay=rgb.copy()
    selected=mask.astype(bool)
    overlay[selected]=(rgb[selected]*.45+np.array([20,240,180])*.55).astype(np.uint8)
    if geometry:
        for key,color in (('vial_box',(90,180,255)),('liquid_box',(255,190,70))):
            box=geometry.get(key)
            if box:
                cv2.rectangle(overlay,tuple(box[:2]),tuple(box[2:]),color,1)
    def png(array):
        ok,encoded=cv2.imencode('.png',array)
        if not ok:
            raise OSError('Could not encode measurement evidence')
        return encoded.tobytes()
    return dict(source=png(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)),mask=png(mask*255),
                overlay=png(cv2.cvtColor(overlay,cv2.COLOR_RGB2BGR)))


def preview(camera, feature, kind):
    if kind not in ('source','mask','overlay'):
        raise ValueError('Unknown measurement evidence kind')
    rgb=source_frame(camera['path'],camera['sha256'],feature['frame_index'],camera['fps'],camera['start_pts_s'])
    if feature.get('source_rgb_sha256') and hashlib.sha256(rgb.tobytes()).hexdigest()!=feature['source_rgb_sha256']:
        raise ValueError('Recovered frame pixels differ from original measurement input')
    return images(rgb,feature.get('geometry'))[kind]
