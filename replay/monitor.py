"""Source-independent monitoring session. No camera/network connection code."""
from collections import deque
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
import time
import uuid

from .detection import FillDetector
from .measurement import Extractor
from .observations import build_detected_observations
from .recipes import RecipePlan
from .references import assess, validate_references


class InputInterrupted(ValueError):
    pass


def measurement_profile_id(config):
    code = b''.join((Path(__file__).parent / name).read_text(encoding='utf-8').encode() for name in ('measurement.py', 'observations.py'))
    return hashlib.sha256(code + config.fingerprint.encode()).hexdigest()[:20]


class MonitorSession:
    def __init__(self, config, fps, run_id, source_video_sha256, recipe_plan=None, references=None, session_id=None, retain_features=False):
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError('Source cadence must be finite and positive')
        self.config, self.fps, self.run_id, self.video_sha256 = config, fps, run_id, source_video_sha256
        self.session_id = session_id or uuid.uuid4().hex
        self.recipe_plan, self.references = recipe_plan or RecipePlan(), validate_references(references or [])
        self.profile_id = measurement_profile_id(config)
        self.extractor, self.detector = Extractor(config), FillDetector(config, fps, run_id)
        self.features, self.observations, self.session_events = [], [], []
        self.retain_features = retain_features
        # Only a bounded stationary tail is needed for finished measurements.
        self.tail = deque(maxlen=max(16, math.ceil(4 * fps / config.sample_stride)))
        self.last_frame = self.last_video_s = None
        self.first_frame = None
        self.seen_idle = False
        self.running = False

    def start(self):
        if self.running or self.session_events:
            raise ValueError('A session cannot be resumed; create a new MonitorSession and confirm recipe context')
        self.running = True
        self.session_events.append(dict(kind='session_started', session_id=self.session_id, frame_index=0))

    def stop(self, reason='operator_stop', terminal_frame=None):
        if not self.running:
            return
        self.running = False
        at = terminal_frame if terminal_frame is not None else self.last_frame if self.last_frame is not None else 0
        if self.detector.active:
            self.session_events.append(dict(kind='incomplete_fill_abandoned', session_id=self.session_id,
                frame_index=at, detection_id=self.detector.active['detection_id'], reason=reason))
        self.session_events.append(dict(kind='session_stopped', session_id=self.session_id, frame_index=at, reason=reason))

    def select_recipe(self, recipe_id, volumes_ml, dispense_order, recorded_by, description, active_from_frame=None):
        """Future approved adapters stamp operator choices from the owned cursor."""
        if not self.running:
            raise ValueError('Recipe input requires a started session')
        at = self.last_frame if self.last_frame is not None else 0
        selection = dict(selection_id='operator-' + uuid.uuid4().hex[:12], run_id=self.run_id,
                         source_video_sha256=self.video_sha256, available_frame=at,
                         active_from_frame=at if active_from_frame is None else active_from_frame,
                         recipe_id=recipe_id, volumes_ml=volumes_ml, dispense_order=dispense_order,
                         recorded_by=recorded_by, recorded_at_utc=datetime.now(timezone.utc).isoformat(),
                         provenance=dict(kind='manual_operator', description=description))
        document = self.recipe_plan.to_dict()
        document['selections'].append(selection)
        self.recipe_plan = RecipePlan(document)
        self.session_events.append(dict(kind='recipe_selection_recorded', frame_index=at,
                                       session_id=self.session_id, selection=selection))
        return selection

    def ingest(self, packet):
        if not self.running:
            raise ValueError('Start a new monitoring session before ingesting frames')
        if (type(packet.frame_index) is not int or packet.frame_index < 0 or
                (self.last_frame is not None and packet.frame_index - self.last_frame != self.config.sample_stride)):
            self.stop('interrupted_or_nonmonotonic_frame_sequence')
            raise InputInterrupted('Interrupted frame sequence: start a new session; do not bridge fills across missing input')
        if (self.last_video_s is not None and packet.video_s is not None and
                (not math.isfinite(packet.video_s) or packet.video_s <= self.last_video_s or
                 abs((packet.video_s - self.last_video_s) - self.config.sample_stride / self.fps) > .5 / self.fps)):
            self.stop('interrupted_timestamp_sequence')
            raise InputInterrupted('Interrupted timestamp sequence: start a new session')
        before = time.perf_counter()
        if self.first_frame is None:
            self.first_frame = packet.frame_index
            self.session_events[0]['frame_index'] = packet.frame_index
        try:
            measurement = self.extractor.measure(packet.rgb)
        except Exception:
            self.stop('invalid_frame_geometry_or_format')
            raise
        video_s = packet.video_s if packet.video_s is not None else packet.frame_index / self.fps
        if packet.video_s is None or not math.isfinite(video_s):
            video_s = packet.frame_index / self.fps
            measurement['quality_flags'].append('missing_presentation_timestamp')
            measurement['measurement_valid'] = False
        elif abs(video_s - packet.frame_index / self.fps) > .5 / self.fps:
            measurement['quality_flags'].append('non_cfr_timestamp_mapping')
            measurement['measurement_valid'] = False
        row = dict(frame_index=packet.frame_index, video_s=video_s, pts=packet.pts, time_base=packet.time_base,
                   estimated_wall_elapsed_s=packet.estimated_wall_elapsed_s, **measurement)
        previous_count = len(self.detector.completed)
        row['fill_detection'] = self.detector.update(row)
        if (row['fill_detection']['phase'] == 'waiting_for_fill' and row['stream_contrast'] < self.config.stream_off_contrast and
                row['holder_dark_fraction'] >= self.config.holder_dark_fraction_min and
                not any(f in row['quality_flags'] for f in ('missing_presentation_timestamp', 'non_cfr_timestamp_mapping'))):
            self.seen_idle = True
        if self.detector.active and not self.seen_idle:
            self.detector.active['partial_start_possible'] = True
            for event in self.detector.events[-2:]:
                if event['kind'] == 'fill_started':
                    event['partial_start_possible'] = True
        row['processing_ms'] = (time.perf_counter() - before) * 1000
        self.last_frame, self.last_video_s = packet.frame_index, video_s
        self.tail.append(row)
        if self.retain_features:
            self.features.append(row)
        if len(self.detector.completed) != previous_count:
            item = self.detector.completed[-1]
            observation = build_detected_observations(list(self.tail), [item], self.config, self.fps, self.run_id)[0]
            observation.update(source_video_sha256=self.video_sha256, source_annotation_sha256=None,
                               measurement_profile_id=self.profile_id, session_id=self.session_id)
            self.observations.append(observation)
        return row

    def decisions(self):
        result = []
        for raw in self.observations:
            observation = self.recipe_plan.bind(raw)
            result.append(dict(observation, assessment=assess(observation, self.references, self.config)))
        return result
