"""Image-only, incremental fill cycles. No annotation or recipe input.

Visible stream episodes are grouped until confirmed carrier departure. A pause
never closes a vial. Estimated boundaries and decision availability are distinct.
EOF never invents a departure. All durations use MP4 time, not physical seconds.
"""
import math


class FillDetector:
    def __init__(self, config, fps, run_id):
        self.config, self.fps, self.run_id = config, fps, run_id
        self.active = None
        self.completed, self.events = [], []
        self.on_candidate = self.off_candidate = self.move_candidate = None
        self.stream_on = False
        self.last_motion = None
        self.last_frame = None
        self.next_id = 1

    def elapsed(self, start, frame, seconds):
        return frame - start >= math.ceil(seconds * self.fps - 1e-9)

    def emit(self, kind, frame, **fields):
        self.events.append(dict(kind=kind, frame_index=frame, video_s=frame / self.fps,
                                source='image_fill_detector', **fields))

    def reset_cycle(self):
        self.active = None
        self.stream_on = False
        self.on_candidate = self.off_candidate = None

    def update(self, feature):
        c = self.config
        frame = feature['frame_index']
        if self.last_frame is not None and frame <= self.last_frame:
            raise ValueError('Detector samples must be strictly increasing')
        gap = self.last_frame is not None and frame - self.last_frame != c.sample_stride
        self.last_frame = frame
        if gap:
            self.on_candidate = self.off_candidate = self.move_candidate = None
            if self.active and 'detection_sample_gap' not in self.active['quality_flags']:
                self.active['quality_flags'].append('detection_sample_gap')
        moving = feature.get('motion') is not None and feature['motion'] > c.departure_motion
        if moving:
            self.last_motion = frame
            if self.move_candidate is None:
                self.move_candidate = frame
            if self.active and self.elapsed(self.move_candidate, frame, c.departure_confirm_s):
                item = self.active
                item.update(departure_frame=self.move_candidate, confirmed_frame=frame,
                            end_frame=item['last_stream_frame'] if self.stream_on else item.get('last_stream_off_frame', item['last_stream_frame']))
                if self.stream_on:
                    item['quality_flags'].append('visible_stream_at_departure')
                if item['stream_samples'] * c.sample_stride / self.fps < c.minimum_stream_s or item['peak_stream_contrast'] < c.stream_peak_min:
                    self.emit('fill_candidate_rejected', frame, detection_id=item['detection_id'], reason='insufficient_visible_stream_evidence')
                else:
                    self.completed.append(dict(item))
                    self.emit('fill_completed', frame, **item)
                self.reset_cycle()
            self.on_candidate = self.off_candidate = None
            return self.state(frame, 'carrier_moving')
        self.move_candidate = None
        if self.last_motion is not None and not self.elapsed(self.last_motion, frame, c.carrier_settle_s):
            return self.state(frame, 'carrier_settling')
        signal = feature.get('stream_contrast')
        valid = (signal is not None and math.isfinite(signal) and
                 feature.get('holder_dark_fraction', 0) >= c.holder_dark_fraction_min and
                 not any(flag in feature.get('quality_flags', []) for flag in
                         ('missing_presentation_timestamp', 'non_cfr_timestamp_mapping')))
        if not valid:
            self.on_candidate = self.off_candidate = None
            if self.active and 'stream_evidence_unavailable' not in self.active['quality_flags']:
                self.active['quality_flags'].append('stream_evidence_unavailable')
            return self.state(frame, 'detection_unavailable')
        if not self.stream_on:
            if signal >= c.stream_on_contrast:
                if self.on_candidate is None:
                    self.on_candidate = frame
                if self.elapsed(self.on_candidate, frame, c.stream_confirm_s):
                    self.stream_on = True
                    if self.active is None:
                        detection_id = f'run{self.run_id}-fill{self.next_id}'
                        self.active = dict(detection_id=detection_id, vial_id=self.next_id,
                                           start_frame=self.on_candidate, start_available_frame=frame,
                                           partial_start_possible=self.on_candidate == 0,
                                           stream_samples=0, peak_stream_contrast=signal,
                                           episodes=[], quality_flags=[])
                        self.next_id += 1
                        self.emit('fill_started', frame, detection_id=detection_id,
                                  vial_id=self.active['vial_id'], start_frame=self.on_candidate,
                                  partial_start_possible=self.on_candidate == 0)
                    self.active['episodes'].append(dict(start_frame=self.on_candidate, start_available_frame=frame))
                    self.emit('stream_started', frame, detection_id=self.active['detection_id'],
                              episode=len(self.active['episodes']), start_frame=self.on_candidate)
            else:
                self.on_candidate = None
        else:
            if signal < c.stream_off_contrast:
                if self.off_candidate is None:
                    self.off_candidate = frame
                if self.elapsed(self.off_candidate, frame, c.stream_confirm_s):
                    self.stream_on = False
                    self.active['last_stream_off_frame'] = self.off_candidate
                    self.active['episodes'][-1].update(end_frame=self.off_candidate, end_available_frame=frame)
                    self.emit('stream_stopped', frame, detection_id=self.active['detection_id'],
                              episode=len(self.active['episodes']), end_frame=self.off_candidate)
                    self.off_candidate = self.on_candidate = None
            else:
                self.off_candidate = None
        if self.active and self.stream_on and signal >= c.stream_off_contrast:
            self.active['last_stream_frame'] = frame
            self.active['stream_samples'] += 1
            self.active['peak_stream_contrast'] = max(signal, self.active['peak_stream_contrast'])
        return self.state(frame, 'visible_stream' if self.stream_on else 'ingredient_pause' if self.active else 'waiting_for_fill')

    def state(self, frame, phase):
        return dict(phase=phase, detection_id=self.active['detection_id'] if self.active else None,
                    vial_id=self.active['vial_id'] if self.active else None,
                    start_frame=self.active['start_frame'] if self.active else None,
                    stream_on=self.stream_on, episode=len(self.active['episodes']) if self.active else 0)


def detect(features, config, fps, run_id):
    detector = FillDetector(config, fps, run_id)
    for feature in features:
        feature['fill_detection'] = detector.update(feature)
    return detector.completed, detector.events
