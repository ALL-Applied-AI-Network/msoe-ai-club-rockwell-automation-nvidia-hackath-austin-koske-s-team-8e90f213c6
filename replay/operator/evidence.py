"""Read verified frozen inference; never rerun a model or read evaluation labels."""
from bisect import bisect_right
from copy import deepcopy
from functools import lru_cache
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from replay.measurement import delta_e


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


class Evidence:
    def __init__(self, root, video, arm='pytorch-pass0'):
        self.root, self.video = Path(root).resolve(), Path(video).resolve()
        plan = json.loads((self.root/'plan.json').read_text(encoding='utf-8-sig'))
        complete = json.loads((self.root/'completed.json').read_text(encoding='utf-8-sig'))
        if sha(self.root/'plan.json') != complete['plan_sha256']:
            raise ValueError('Frozen measurement/model plan changed')
        if arm not in ('pytorch-pass0', 'tensorrt-pass0'):
            raise ValueError('Choose an explicitly completed inference arm')
        folder = self.root/arm
        if sha(folder/'completed.json') != complete['runs'][arm]:
            raise ValueError('Inference completion marker changed')
        marker = json.loads((folder/'completed.json').read_text(encoding='utf-8-sig'))
        for name in ('frames.jsonl', 'events.json'):
            if sha(folder/name) != marker['artifacts'][name]:
                raise ValueError('Frozen evidence changed: '+name)
        if sha(self.video) != plan['source_sha256']:
            raise ValueError('Video differs from the frozen inference source')
        self.rows = [json.loads(s) for s in (folder/'frames.jsonl').read_text(encoding='utf-8').splitlines()]
        if not self.rows:
            raise ValueError('Empty inference evidence')
        self.frames = [r['source_frame'] for r in self.rows]
        self.seconds = [r['seconds'] for r in self.rows]
        if (len(set(self.frames)) != len(self.frames) or self.frames != sorted(self.frames)
                or not all(math.isfinite(s) for s in self.seconds)
                or any(b <= a for a, b in zip(self.seconds, self.seconds[1:]))):
            raise ValueError('Source evidence must have unique increasing frame/clock identities')
        for row in self.rows:
            if type(row['source_frame']) is not int or type(row['pts']) is not int or Fraction(row['time_base']) <= 0 or float(row['pts']*Fraction(row['time_base'])) != row['seconds']:
                raise ValueError('Sample clocks must preserve native PTS and time base')
        self.by_frame = {r['source_frame']: r for r in self.rows}
        self.width, self.height = plan['qa_config']['image_size']
        self.threshold = float(plan['qa_config']['review_threshold_de2000'])
        self.spread_limit = float(plan['qa_config']['max_temporal_de2000'])
        self.profile = digest(dict(kind='operator_endpoint_last_three_samples/v1',
                                   qa_config=plan['qa_config'], qa_code=plan['qa_code_sha256'],
                                   models=plan['frozen_models'], arm=arm))
        self.identity = dict(source_video_sha256=plan['source_sha256'],
                             plan_sha256=complete['plan_sha256'],
                             inference_frames_sha256=sha(folder/'frames.jsonl'),
                             measurement_profile=self.profile, arm=arm, models=plan['frozen_models'])
        events = json.loads((folder/'events.json').read_text(encoding='utf-8-sig'))
        self.navigation = [dict(label=f'Sample fill {i+1}', seconds=self.by_frame[e['frame_index']]['seconds'], source_frame=e['frame_index'])
                           for i, e in enumerate(e for e in events if e['kind'] == 'fill_started')]
        self.public = dict(source_name=self.video.name, run_id=plan.get('source_run'),
                           source_mode='recorded_demonstration', width=self.width, height=self.height,
                           duration_s=self.seconds[-1], sample_count=len(self.rows),
                           navigation=self.navigation, identity=self.identity,
                           threshold_de2000=self.threshold,
                           geometry_status='unreviewed_model_proposals',
                           hardware_connected=False, metered_progress_available=False)

    def at(self, seconds):
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or not 0 <= seconds <= self.seconds[-1]+1:
            raise ValueError('Playhead is outside this recording')
        return self.rows[max(0, bisect_right(self.seconds, seconds)-1)]

    def frame(self, index):
        if type(index) is not int or index not in self.by_frame:
            raise ValueError('Expected an exact sampled source frame')
        return self.by_frame[index]

    def sample(self, seconds):
        row = self.at(seconds); m = row['measurement']
        return dict(source_frame=row['source_frame'], seconds=row['seconds'], pts=row['pts'],
                    time_base=row['time_base'], rgb=m['rgb'], lab=m['lab'],
                    measurement_valid=m['measurement_valid'], quality_flags=m['quality_flags'],
                    geometry_status='unreviewed_model_proposals',
                    mask_pixels=m['geometry']['mask_pixels'], visual_track_id=m['geometry']['track_id'],
                    image_phase=row['fill_detection']['phase'],
                    source_rgb_sha256=row['source_rgb_sha256'])

    def endpoint(self, start, end):
        self.frame(start); row = self.frame(end)
        selected = self.rows[max(0, bisect_right(self.frames, end)-3):bisect_right(self.frames, end)]
        flags = []
        if len(selected) != 3 or selected[0]['source_frame'] < start:
            flags.append('not_enough_current_vial_samples')
        if any(not r['measurement']['measurement_valid'] for r in selected):
            flags.append('sample_quality_unavailable')
        tracks = {r['measurement']['geometry']['track_id'] for r in selected}
        if len(tracks) != 1 or None in tracks:
            flags.append('visual_track_changed')
        if any(b['seconds']-a['seconds'] > .400001 for a, b in zip(selected, selected[1:])):
            flags.append('sample_gap')
        candidates = {r['fill_detection'].get('detection_id') for r in self.rows
                      if start <= r['source_frame'] <= end and r['fill_detection'].get('detection_id')}
        if len(candidates) > 1:
            flags.append('record_spans_image_fill_candidates')
        lab = rgb = spread = None
        if not flags:
            lab = np.median([r['measurement']['lab'] for r in selected], axis=0).tolist()
            rgb = np.median([r['measurement']['rgb'] for r in selected], axis=0).tolist()
            spread = max(delta_e(r['measurement']['lab'], lab) for r in selected)
            if spread > self.spread_limit:
                flags.append('unstable_endpoint_color')
        return dict(source_frame=end, seconds=row['seconds'], sample_frames=[r['source_frame'] for r in selected],
                    measurement_valid=not flags, quality_flags=flags, lab=lab, rgb=rgb,
                    spread_de2000=spread, geometry_status='unreviewed_model_proposals',
                    measurement_profile=self.profile, policy='operator_endpoint_last_three_samples/v1')

    @lru_cache(maxsize=24)
    def preview(self, frame, kind='overlay'):
        import av
        import cv2
        from replay.geometry import decode_mask
        row = self.frame(frame)
        if kind not in ('source', 'overlay', 'mask'):
            raise ValueError('Unknown evidence image kind')
        rgb = None
        with av.open(str(self.video)) as c:
            stream = c.streams.video[0]
            c.seek(row['pts'], stream=stream, backward=True)
            for f in c.decode(stream):
                if f.pts == row['pts'] and str(f.time_base) == row['time_base']:
                    rgb = f.to_ndarray(format='rgb24'); break
                if f.pts is not None and f.pts > row['pts']:
                    break
        if rgb is None or hashlib.sha256(rgb.tobytes()).hexdigest() != row['source_rgb_sha256']:
            raise ValueError('Decoded source pixels/clock do not match frozen inference')
        mask = decode_mask(row['measurement']['geometry']['mask'], (self.height, self.width)).astype(bool)
        if kind == 'mask':
            image = mask.astype(np.uint8)*255
        else:
            image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if kind == 'overlay':
                image[mask] = (.55*image[mask]+.45*np.array([65, 225, 220])).astype(np.uint8)
                box = row['measurement']['geometry'].get('vial_box')
                if box:
                    x, y, r, b = [int(v) for v in box]
                    cv2.rectangle(image, (x, y), (r, b), (235, 160, 55), 1)
        ok, buffer = cv2.imencode('.png', image)
        if not ok:
            raise ValueError('Could not encode evidence preview')
        return buffer.tobytes()
