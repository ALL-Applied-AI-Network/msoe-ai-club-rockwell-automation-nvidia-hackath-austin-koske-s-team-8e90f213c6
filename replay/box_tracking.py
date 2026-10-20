"""Opt-in causal station tracking. No detector, segmentation, recipe or QA edits.

IDs identify provisional station segments, not verified physical vials. Only
fresh boxes can own a liquid proposal. Missing boxes are never interpolated.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import math
import uuid

import cv2
import numpy as np


def intersection_fraction(a, b):
    inside = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    return inside/max(1, (a[2]-a[0])*(a[3]-a[1]))


def iou(a, b):
    inside = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    return inside/max(1, (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inside)


@dataclass(frozen=True)
class TrackingSettings:
    source_fps: float = 30.
    stride: int = 6
    station_x: float = 306.5
    minimum_vial_height: int = 180
    motion_threshold: float = 8.
    motion_confirm_s: float = .2
    settle_s: float = .6
    maximum_detection_gap_s: float = .6
    match_iou: float = .65
    confirm_samples: int = 3
    containment_minimum: float = .9

    def __post_init__(self):
        if not all(math.isfinite(v) for v in asdict(self).values()):
            raise ValueError('Nonfinite tracking settings')
        if self.source_fps <= 0 or type(self.stride) is not int or self.stride < 1:
            raise ValueError('Invalid input cadence')
        if not 0 < self.match_iou <= 1 or not .9 <= self.containment_minimum <= 1:
            raise ValueError('Containment may not be weakened below existing 90% gate')
        if (type(self.confirm_samples) is not int or self.confirm_samples < 1 or
                min(self.motion_threshold, self.motion_confirm_s, self.settle_s,
                    self.maximum_detection_gap_s, self.minimum_vial_height) <= 0):
            raise ValueError('Invalid tracking duration/confirmation')


class StationTracker:
    def __init__(self, settings=None, session_id=None):
        self.settings = settings or TrackingSettings()
        if session_id is None:
            session_id = str(uuid.uuid4())
        if not session_id:
            raise ValueError('A session identity is required')
        self.session_id = session_id
        self.serial = 0
        self.last_frame = None
        self.previous_thumb = None
        self.track = None
        self.motion_start = None
        self.last_motion = None
        self.transition_confirmed = False

    def elapsed(self, start, end, seconds):
        return end-start >= math.ceil(seconds*self.settings.source_fps-1e-9)

    def update(self, rgb, proposals, frame_index):
        c = self.settings
        if type(frame_index) is not int or frame_index < 0 or (self.last_frame is not None and frame_index <= self.last_frame):
            raise ValueError('Frame indices must strictly increase')
        if rgb.shape != (480, 640, 3) or rgb.dtype != np.uint8:
            raise ValueError('Station profile requires original 640x480 RGB uint8')
        for p in proposals:
            b = p['box']
            if (p['class_id'] not in (0, 1) or len(b) != 4 or
                    not all(type(v) in (int, float) and math.isfinite(v) for v in b) or
                    not 0 <= b[0] < b[2] <= 640 or not 0 <= b[1] < b[3] <= 480 or
                    not math.isfinite(p['confidence']) or not 0 <= p['confidence'] <= 1):
                raise ValueError('Invalid raw detector proposal')
        events = []
        def emit(kind, **values):
            events.append(dict(kind=kind, available_frame=frame_index, **values))
        def retire(reason):
            if self.track:
                emit('station_track_retired', track_id=self.track['id'], reason=reason)
                self.track = None
        gap = self.last_frame is not None and frame_index-self.last_frame != c.stride
        if gap:
            retire('interrupted_input')
            self.previous_thumb = None
            self.motion_start = self.last_motion = None
            self.transition_confirmed = False
            emit('input_discontinuity', previous_frame=self.last_frame)
        self.last_frame = frame_index
        # Exactly the existing image-derived thumbnail feature; recalculated from
        # current/past RGB. No cached fill events, colors or timing labels enter.
        thumb = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (32, 24)).astype(float)[10:23, 11:21]
        motion = float(np.abs(thumb-self.previous_thumb).mean()) if self.previous_thumb is not None else None
        self.previous_thumb = thumb
        moving = motion is not None and motion > c.motion_threshold
        if moving:
            self.last_motion = frame_index
            if self.motion_start is None:
                self.motion_start = frame_index
            if not self.transition_confirmed and self.elapsed(self.motion_start, frame_index, c.motion_confirm_s):
                self.transition_confirmed = True
                retire('confirmed_carrier_motion')
                emit('carrier_transition_confirmed', estimated_start_frame=self.motion_start)
        else:
            self.motion_start = None
        settling = self.last_motion is not None and not self.elapsed(self.last_motion, frame_index, c.settle_s)
        blocked = moving or settling
        if not blocked:
            self.transition_confirmed = False
        vials = [dict(deepcopy(p), track_id=None, track_status='off_station', active_station=False)
                 for p in proposals if p['class_id'] == 0]
        candidates = [i for i, v in enumerate(vials) if v['box'][0] < c.station_x < v['box'][2]
                      and v['box'][3]-v['box'][1] > c.minimum_vial_height]
        active_index = candidates[0] if len(candidates) == 1 else None
        if self.track and frame_index-self.track['last_seen'] > math.floor(c.maximum_detection_gap_s*c.source_fps+1e-9):
            retire('detection_gap_timeout')
        if len(candidates) > 1:
            retire('ambiguous_active_vial')
        if active_index is not None:
            v = vials[active_index]; v['active_station'] = True
            if blocked:
                v['track_status'] = 'carrier_moving' if moving else 'carrier_settling'
                if self.track:
                    self.track['hits'] = 0
                    # Fresh detections during a single unconfirmed motion spike
                    # are not missing detections. Keep only the internal candidate
                    # identity; never publish a link while movement is blocked.
                    self.track['last_seen'] = frame_index
            else:
                if self.track and iou(v['box'], self.track['box']) < c.match_iou:
                    retire('geometry_discontinuity')
                if self.track is None:
                    self.serial += 1
                    self.track = dict(id=f'{self.session_id}:s{self.serial:03d}', box=list(v['box']),
                                      last_seen=frame_index, hits=0, first_seen=frame_index)
                    emit('station_track_started', track_id=self.track['id'])
                elif frame_index-self.track['last_seen'] != c.stride:
                    self.track['hits'] = 0
                    emit('short_detection_gap_reacquired', track_id=self.track['id'])
                self.track.update(box=list(v['box']), last_seen=frame_index, hits=self.track['hits']+1)
                v.update(track_id=self.track['id'], track_status='confirmed' if self.track['hits'] >= c.confirm_samples else 'tentative')
        else:
            if self.track:
                self.track['hits'] = 0
            for i in candidates:
                vials[i]['track_status'] = 'ambiguous_station'
        liquids = [dict(deepcopy(p), association=None, vial_detection_index=None, vial_track_id=None,
                        containment_fraction=None, measurement_authorized=False, liquid_scope='unreviewed_legacy_class')
                   for p in proposals if p['class_id'] == 1]
        eligible = []
        for i, liquid in enumerate(liquids):
            if len(candidates) > 1:
                liquid['association'] = 'ambiguous_active_vial'
            elif active_index is None:
                liquid['association'] = 'no_fresh_station_vial'
            else:
                vial = vials[active_index]
                fraction = intersection_fraction(liquid['box'], vial['box'])
                liquid['containment_fraction'] = fraction
                if blocked:
                    liquid['association'] = 'carrier_transition'
                elif vial['track_status'] != 'confirmed':
                    liquid['association'] = 'track_not_confirmed'
                elif fraction < c.containment_minimum:
                    liquid['association'] = 'outside_vial'
                else:
                    eligible.append(i)
        # No confidence-based tie-breaking and no squeezing/clipping liquid boxes
        # into a vial. Association is a candidate link, never a verified body mask.
        for i in eligible:
            if len(eligible) != 1:
                liquids[i]['association'] = 'multiple_liquid_candidates'
            else:
                liquids[i].update(association='candidate_link', vial_detection_index=active_index,
                                  vial_track_id=vials[active_index]['track_id'])
        state = ('carrier_moving' if moving else 'carrier_settling' if settling else
                 'ambiguous_station' if len(candidates) > 1 else 'station_vial_missing' if active_index is None else
                 vials[active_index]['track_status'])
        return dict(frame_index=frame_index, session_id=self.session_id, state=state, motion=motion,
                    input_gap=gap, detector_proposals=deepcopy(proposals), vials=vials, liquids=liquids, events=events)
