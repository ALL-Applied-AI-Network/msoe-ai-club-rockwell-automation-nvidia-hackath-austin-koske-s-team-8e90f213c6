"""Separate, lazy tracking evidence. Never modifies monitor features or QA."""
from bisect import bisect_right
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import av
import cv2
import numpy as np

from .box_tracking import StationTracker, TrackingSettings
from .dataset import sha256
from .pipeline import write_json


class TrackingUnavailable(ValueError):
    pass


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def settings_for(camera, config):
    if not config.vision_model_sha256:
        raise TrackingUnavailable('Choose the box-detector replay configuration to use station tracking.')
    if config.liquid_segmentation_model_path:
        raise TrackingUnavailable('Station tracking needs the box-only configuration; learned-mask proposals are not raw detector boxes.')
    if ((camera['width'], camera['height'], camera['fps'], config.sample_stride) != (640,480,30.,6)
            or tuple(config.image_size) != (640,480)):
        raise TrackingUnavailable('Station tracking is available for the 640×480, 30 fps, stride-six camera profile.')
    return TrackingSettings(source_fps=camera['fps'], stride=config.sample_stride,
        station_x=(config.stream_center_roi[0]+config.stream_center_roi[2])/2,
        motion_threshold=config.departure_motion, motion_confirm_s=config.departure_confirm_s,
        settle_s=config.carrier_settle_s)


class TrackingOverlay:
    def __init__(self, payload):
        self.manifest = payload['manifest']
        self.rows = payload['rows']
        self.frames = [r['frame_index'] for r in self.rows]

    def snapshot(self, frame):
        if type(frame) is not int or not 0 <= frame < self.manifest['frame_count']:
            raise ValueError('Frame lies outside the recording')
        i = bisect_right(self.frames, frame)-1
        row = self.rows[i] if i >= 0 else None
        return dict(status='ready', purpose='optional_tracking_display_only', frame_index=frame,
            sample_frame_index=row['frame_index'] if row else None,
            sample_matches_playhead=bool(row and row['frame_index']==frame),
            sample_age_frames=frame-row['frame_index'] if row else None,
            row=deepcopy(row), measurement_authorized=False,
            provenance=deepcopy(self.manifest['provenance']))


def prepare_overlay(camera, data, config, cache_root, progress=None):
    settings = settings_for(camera, config)
    inventory = []
    for f in data.features:
        g = f.get('geometry') or {}
        if (g.get('model_sha256') != config.vision_model_sha256 or
                not f.get('source_rgb_sha256') or 'proposals' not in g or g.get('liquid_proposal_method')):
            raise TrackingUnavailable('Original hash-bound detector proposals are unavailable for this analysis.')
        inventory.append(dict(frame_index=f['frame_index'], rgb_sha256=f['source_rgb_sha256'], proposals=g['proposals']))
    if not inventory or [r['frame_index'] for r in inventory] != sorted({r['frame_index'] for r in inventory}):
        raise ValueError('Tracking input must be nonempty and strictly ordered')
    if inventory[-1]['frame_index'] >= camera['frames']:
        raise ValueError('Tracking input exceeds source recording')
    provenance = dict(schema_version=1, video_sha256=camera['sha256'], model_sha256=config.vision_model_sha256,
        raw_input_sha256=digest(inventory), settings=asdict(settings),
        implementation_sha256=hashlib.sha256((Path(__file__).read_bytes()+
            Path(__file__).with_name('box_tracking.py').read_bytes())).hexdigest(),
        runtime=dict(av=av.__version__, opencv=cv2.__version__, numpy=np.__version__),
        annotations_consumed=False, measurement_pipeline_changed=False)
    key = digest(provenance)
    cache = Path(cache_root)/f'run{data.payload["manifest"]["run_id"]}-{key[:20]}.json'
    if sha256(camera['path']) != camera['sha256']:
        raise ValueError('Recording changed after cataloging')
    if cache.is_file():
        payload = json.loads(cache.read_text(encoding='utf-8'))
        if (payload['manifest']['provenance'] != provenance or
                payload['manifest']['frame_count'] != camera['frames'] or
                payload['manifest']['rows_sha256'] != digest(payload['rows']) or
                len(payload['rows']) != len(inventory)):
            raise ValueError('Tracking cache identity/content mismatch')
        for row, source in zip(payload['rows'], inventory):
            if (row['frame_index'] != source['frame_index'] or row['source_rgb_sha256'] != source['rgb_sha256'] or
                    row['detector_proposals'] != source['proposals']):
                raise ValueError('Cached tracking differs from original image evidence')
        if progress: progress(1.)
        return TrackingOverlay(payload)
    wanted = {r['frame_index']:r for r in inventory}
    tracker = StationTracker(settings, session_id='overlay-'+key[:16])
    rows=[];decoded=0
    with av.open(camera['path']) as container:
        container.streams.video[0].thread_type='AUTO'
        for frame_index, frame in enumerate(container.decode(video=0)):
            decoded += 1
            source = wanted.get(frame_index)
            if source is None: continue
            if frame.pts is None or abs(float(frame.pts*frame.time_base)-camera['start_pts_s']-frame_index/camera['fps'])>1e-5:
                raise ValueError('Source presentation timestamps do not match tracking input')
            rgb=frame.to_ndarray(format='rgb24')
            if hashlib.sha256(rgb.tobytes()).hexdigest() != source['rgb_sha256']:
                raise ValueError('Tracking source pixels differ from the measurement input')
            row=tracker.update(rgb, source['proposals'], frame_index)
            row.update(source_rgb_sha256=source['rgb_sha256'], mp4_s=frame_index/camera['fps'])
            rows.append(row)
            if progress and len(rows)%100 == 0: progress(frame_index/max(1,camera['frames']-1))
    if decoded != camera['frames'] or len(rows) != len(inventory):
        raise ValueError('Incomplete tracking source decode')
    payload=dict(manifest=dict(provenance=provenance, frame_count=camera['frames'], rows_sha256=digest(rows)),rows=rows)
    write_json(cache,payload)
    if progress: progress(1.)
    return TrackingOverlay(payload)
