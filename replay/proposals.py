"""Reuse pinned per-image detector outputs for controlled replay comparisons.

Only model proposals keyed by source RGB/frame are consumed. Cached colors,
timing decisions, track states, recipe context and annotations are ignored.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path


class CachedImageDetector:
    def __init__(self,path,sha256,model_sha256,video_sha256,run_id):
        raw=Path(path).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=sha256:
            raise ValueError('Image proposal cache hash mismatch')
        payload=json.loads(raw); manifest=payload['manifest']
        if (manifest['video_sha256']!=video_sha256 or manifest['run_id']!=run_id or
                manifest.get('geometry_model_sha256')!=model_sha256 or
                manifest.get('config',{}).get('liquid_segmentation_model_path')):
            raise ValueError('Image proposal cache recording or original detector does not match')
        self.frames={}
        for f in payload['features']:
            frame=f['frame_index']; geometry=f['geometry']
            if type(frame)!=int or frame<0 or frame in self.frames or geometry['model_sha256']!=model_sha256:
                raise ValueError('Invalid cached detector frame/model')
            boxes=deepcopy(geometry['proposals'])
            for b in boxes:
                box=b['box']
                if (b['class_id'] not in (0,1) or len(box)!=4 or not all(type(v)==int for v in box) or
                        not 0<=box[0]<box[2]<=640 or not 0<=box[1]<box[3]<=480 or
                        not math.isfinite(b['confidence']) or not 0<=b['confidence']<=1 or 'candidate_mask' in b):
                    raise ValueError('Invalid cached image proposal')
            self.frames[frame]=dict(rgb_sha256=f['source_rgb_sha256'],boxes=boxes)
        self.current=None

    def prepare_frame(self,index):
        self.current=self.frames.get(index)
        if self.current is None:
            raise ValueError('Requested frame is absent from pinned image proposals')

    def detect(self,rgb):
        current=self.current; self.current=None
        if current is None or hashlib.sha256(rgb.tobytes()).hexdigest()!=current['rgb_sha256']:
            raise ValueError('Source pixels do not match pinned image proposals')
        return deepcopy(current['boxes'])
