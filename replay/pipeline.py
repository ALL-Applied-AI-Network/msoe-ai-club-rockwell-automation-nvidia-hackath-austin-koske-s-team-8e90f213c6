"""Causal image detection and completed measurements; optional annotation baseline."""
from bisect import bisect_left, bisect_right
import csv
import hashlib
import io
import json
from pathlib import Path
import platform
import time
import uuid

import av
import cv2
import numpy as np
import skimage

from .measurement import Extractor, delta_e
from .detection import detect


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, allow_nan=False, separators=(',', ':')), encoding='utf-8')
    temporary.replace(path)


def build_observations(features, annotations, config, fps, frame_count, run_id):
    frames = [item['frame_index'] for item in features]
    result = []
    for annotation in annotations:
        window_start = annotation.end_frame + int(round(config.window_start_after_end_s * fps))
        window_end = annotation.end_frame + int(round(config.window_end_after_end_s * fps))
        selected = features[bisect_left(frames, window_start):bisect_right(frames, window_end)]
        index = bisect_left(frames, window_end)
        available_frame = frames[index] if index < len(frames) else frame_count - 1
        accepted = [item for item in selected if item['measurement_valid']]
        flags = []
        if window_end >= frame_count:
            flags.append('recording_ends_before_measurement_window')
        if len(accepted) < config.minimum_window_samples:
            flags.append('insufficient_valid_post_fill_samples')
        if any('carrier_motion' in item['quality_flags'] for item in selected):
            flags.append('motion_during_measurement_window')
        lab = np.median([item['lab'] for item in accepted], axis=0).tolist() if accepted else None
        rgb = np.median([item['rgb'] for item in accepted], axis=0).tolist() if accepted else None
        spread = max((delta_e(item['lab'], lab) for item in accepted), default=None)
        if spread is not None and spread > config.max_temporal_de2000:
            flags.append('unstable_post_fill_color')
        result.append(dict(event_id=f'run{run_id}-vial{annotation.vial_id}', run_id=run_id,
                           **annotation.to_dict(), available_frame=available_frame,
                           available_video_s=available_frame / fps,
                           sample_frames=[item['frame_index'] for item in selected],
                           accepted_sample_frames=[item['frame_index'] for item in accepted],
                           lab=lab, rgb=rgb, temporal_spread_de2000=spread,
                           quality_flags=flags, measurement_valid=not flags and lab is not None,
                           measurement_config_id=config.fingerprint,
                           event_source='provided_fill_end_annotation',
                           partial_start_possible=annotation.start_frame == 0,
                           baseline_hold_first=annotation.position_in_batch == 1,
                           baseline_hold_first_two=annotation.position_in_batch <= 2))
    return result


def build_detected_observations(features, detections, config, fps, run_id):
    """Measure the stationary tail, publish only after departure confirmation.

    An ingredient pause is not a completed fill. Samples may precede the
    confirmation but can never include departure or the next vial. Recipe
    identity is unknown: matching labels to detections belongs to evaluation.
    """
    frames = [item['frame_index'] for item in features]
    observations = []
    for item in detections:
        tail_end = item['departure_frame'] - config.sample_stride
        span = int(round((config.window_end_after_end_s - config.window_start_after_end_s) * fps))
        start = max(item['end_frame'] + int(round(config.window_start_after_end_s * fps)), tail_end - span)
        selected = features[bisect_left(frames, start):bisect_right(frames, tail_end)]
        accepted = [row for row in selected if row['measurement_valid']]
        flags = list(dict.fromkeys(item['quality_flags']))
        if len(accepted) < config.minimum_window_samples:
            flags.append('insufficient_valid_post_fill_samples')
        if any('carrier_motion' in row['quality_flags'] for row in selected):
            flags.append('motion_during_measurement_window')
        lab = np.median([row['lab'] for row in accepted], axis=0).tolist() if accepted else None
        rgb = np.median([row['rgb'] for row in accepted], axis=0).tolist() if accepted else None
        spread = max((delta_e(row['lab'], lab) for row in accepted), default=None)
        if spread is not None and spread > config.max_temporal_de2000:
            flags.append('unstable_post_fill_color')
        observations.append(dict(item, event_id=item['detection_id'], run_id=run_id,
                                 recipe_id=None, volumes_ml={}, dispense_order=None, position_in_batch=None,
                                 previous_recipe_id=None, baseline_hold_first=None, baseline_hold_first_two=None,
                                 available_frame=item['confirmed_frame'], available_video_s=item['confirmed_frame'] / fps,
                                 sample_frames=[row['frame_index'] for row in selected],
                                 accepted_sample_frames=[row['frame_index'] for row in accepted],
                                 lab=lab, rgb=rgb, temporal_spread_de2000=spread,
                                 quality_flags=flags, measurement_valid=not flags and lab is not None,
                                 measurement_config_id=config.fingerprint, event_source='image_fill_detector'))
    return observations


def reference_for(observation, references):
    if observation['recipe_id'] is None:
        return None
    eligible = []
    for reference in references:
        if (str(reference['recipe_id']) != observation['recipe_id'] or
                reference['measurement_config_id'] != observation['measurement_config_id']):
            continue
        if reference['scope'] == 'same_run_development':
            if (reference['source_run_id'] != observation['run_id'] or
                    reference.get('source_video_sha256') != observation.get('source_video_sha256') or
                    reference.get('source_annotation_sha256') != observation.get('source_annotation_sha256') or
                    reference['available_frame'] >= observation['start_frame'] or
                    reference.get('source_event_id') == observation['event_id']):
                continue
        elif reference['scope'] != 'external_commissioning':
            continue
        eligible.append(reference)
    return eligible[-1] if eligible else None


def assess(observation, references, config):
    result = dict(status='unavailable', delta_e2000=None, reference_id=None,
                  expected_lab=None, threshold_de2000=config.review_threshold_de2000,
                  threshold_status='engineering_default_unvalidated',
                  explanation='', review_is_advisory=True)
    if not observation['measurement_valid']:
        result['explanation'] = 'Unable to assess: ' + ', '.join(observation['quality_flags'])
        return result
    if observation['recipe_id'] is None:
        result['explanation'] = 'Fill detected from images. Recipe context is unavailable; no recipe is inferred from color.'
        return result
    reference = reference_for(observation, references)
    if reference is None:
        result['explanation'] = 'No matching reference was available before this fill started.'
        return result
    score = delta_e(observation['lab'], reference['lab'])
    mismatch = score > config.review_threshold_de2000
    result.update(status='review' if mismatch else 'within_reference_band',
                  delta_e2000=score, reference_id=reference['reference_id'], expected_lab=reference['lab'],
                  explanation=('Appearance differs from the selected recipe reference; operator review requested.'
                               if mismatch else 'Appearance is within the experimental reference band; this is not a release decision.'))
    return result


def validate_references(references):
    seen = set()
    for item in references:
        required = ('reference_id', 'recipe_id', 'lab', 'scope', 'measurement_config_id')
        if any(key not in item for key in required):
            raise ValueError('Reference is missing required provenance fields')
        lab = np.asarray(item['lab'], dtype=float)
        if lab.shape != (3,) or not np.isfinite(lab).all() or item['reference_id'] in seen:
            raise ValueError('Invalid or duplicate reference')
        if item['scope'] == 'same_run_development':
            if not isinstance(item.get('available_frame'), int) or item['available_frame'] < 0 or not item.get('source_run_id'):
                raise ValueError('Same-run reference needs its source run and availability frame')
            if not item.get('source_video_sha256'):
                raise ValueError('Same-run reference needs a source video hash')
        elif item['scope'] == 'external_commissioning':
            if not item.get('source_description'):
                raise ValueError('External reference needs a commissioning source description')
        else:
            raise ValueError('Unknown reference scope')
        seen.add(item['reference_id'])
    return references


class RunData:
    def __init__(self, payload, config):
        self.payload, self.config = payload, config
        self.features = payload['features']
        self.frames = [item['frame_index'] for item in self.features]
        self.observations = payload['observations']
        self.annotations = payload['annotations']
        self.context_frames = [item['start_frame'] for item in self.annotations]
        self.image_mode = payload['manifest'].get('fill_source') == 'image'

    def completed(self, frame, references):
        return [dict(item, assessment=assess(item, references, self.config))
                for item in self.observations if item['available_frame'] <= frame]

    def snapshot(self, frame, references):
        frame = max(0, min(int(frame), self.payload['manifest']['frame_count'] - 1))
        index = bisect_right(self.frames, frame) - 1
        feature = self.features[index] if index >= 0 else None
        if self.image_mode:
            completed = self.completed(frame, references)
            detection = feature.get('fill_detection', {}) if feature else {}
            active = detection.get('vial_id')
            current = None if active else (completed[-1] if completed else None)
            context = dict(vial_id=active, recipe_id=None, volumes_ml={}, dispense_order=None,
                           position_in_batch=None) if active else current
            return dict(run_id=self.payload['manifest']['run_id'], frame_index=frame,
                        video_s=frame / self.payload['manifest']['fps'], phase=detection.get('phase', 'waiting_for_fill'),
                        feature=feature, context=context, completed=completed, current_completed=current,
                        visible_references=[], fill_source='image',
                        coverage=dict(available=len(completed), valid=sum(row['measurement_valid'] for row in completed), assessed=0))
        context_index = bisect_right(self.context_frames, frame) - 1
        annotation = self.annotations[context_index] if context_index >= 0 else None
        completed = self.completed(frame, references)
        current = next((item for item in reversed(completed) if annotation and item['vial_id'] == annotation['vial_id']), None)
        if annotation is None:
            phase = 'unlabeled' if not self.annotations else 'before_first_annotation'
        elif frame <= annotation['end_frame']:
            phase = 'annotated_fill'
        elif current is None:
            phase = 'collecting_completed_vial_window'
        else:
            phase = 'completed_vial_available'
        # End frame is display-only replay context. No early-warning rule uses it.
        context = dict(annotation) if annotation else None
        return dict(run_id=self.payload['manifest']['run_id'], frame_index=frame,
                    video_s=frame / self.payload['manifest']['fps'], phase=phase,
                    feature=feature, context=context, completed=completed, current_completed=current,
                    visible_references=[item for item in references if item['scope'] == 'external_commissioning' or
                                        (item['source_run_id'] == self.payload['manifest']['run_id'] and item['available_frame'] <= frame)],
                    coverage=dict(available=len(completed), valid=sum(item['measurement_valid'] for item in completed),
                                  assessed=sum(item['assessment']['status'] != 'unavailable' for item in completed)))

    def events(self, frame, references):
        fps = self.payload['manifest']['fps']
        events = []
        if self.image_mode:
            events = [event for event in self.payload['detection_events'] if event['frame_index'] <= frame]
            for item in self.completed(frame, references):
                events.append(dict(kind='completed_vial_assessment', frame_index=item['available_frame'],
                                   video_s=item['available_video_s'], observation=item))
            return sorted(events, key=lambda event: event['frame_index'])
        for annotation in self.annotations:
            if annotation['start_frame'] <= frame:
                events.append(dict(kind='recipe_context_available', frame_index=annotation['start_frame'],
                                   video_s=annotation['start_frame'] / fps, vial_id=annotation['vial_id'],
                                   recipe_id=annotation['recipe_id'], source='provided_recipe_annotation',
                                   volumes_ml=annotation['volumes_ml'], dispense_order=annotation['dispense_order'],
                                   previous_recipe_id=annotation['previous_recipe_id'],
                                   baseline_hold_first=annotation['position_in_batch'] == 1,
                                   baseline_hold_first_two=annotation['position_in_batch'] <= 2))
            if annotation['end_frame'] <= frame:
                events.append(dict(kind='annotated_fill_end', frame_index=annotation['end_frame'],
                                   video_s=annotation['end_frame'] / fps, vial_id=annotation['vial_id'],
                                   source='provided_fill_end_annotation'))
        for item in self.completed(frame, references):
            events.append(dict(kind='completed_vial_assessment', frame_index=item['available_frame'],
                               video_s=item['available_video_s'], observation=item))
        return sorted(events, key=lambda event: event['frame_index'])


def analyze(run, config, cache_root, progress=None, fill_source='image'):
    if fill_source not in ('image', 'annotations'):
        raise ValueError('Fill source must be image or annotations')
    camera = run['cameras']['vial_fill']
    implementation = hashlib.sha256(b''.join((Path(__file__).parent / name).read_bytes()
                                             for name in ('config.py', 'dataset.py', 'measurement.py', 'detection.py', 'pipeline.py'))).hexdigest()[:16]
    runtime = dict(python=platform.python_version(), architecture=platform.machine(),
                   numpy=np.__version__, av=av.__version__, opencv=cv2.__version__, skimage=skimage.__version__)
    provenance = dict(schema_version=config.schema_version, implementation_id=implementation, runtime=runtime, video_sha256=camera['sha256'],
                      annotation_sha256=run['annotation_sha256'], clock_sha256=run['clock_sha256'],
                      config_id=config.fingerprint, fill_source=fill_source)
    key = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()[:20]
    cache = Path(cache_root) / f'run{run["run_id"]}-{key}.json'
    if cache.is_file():
        return RunData(json.loads(cache.read_text(encoding='utf-8')), config)
    clock = None
    if run['clock_path']:
        clock = np.load(run['clock_path'], allow_pickle=False)
        if (clock.ndim != 1 or len(clock) != camera['frames'] or not np.isfinite(clock).all() or
                np.any(np.diff(clock) < 0)):
            raise ValueError('Clock map must be finite, monotonic and match the frame count')
        clock = clock - clock[0]
    extractor, features = Extractor(config), []
    started = time.perf_counter()
    missing_pts = 0
    with av.open(camera['path']) as container:
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO'
        last_frame = -1
        for frame_index, frame in enumerate(container.decode(stream)):
            last_frame = frame_index
            if frame_index % config.sample_stride:
                continue
            before = time.perf_counter()
            rgb = frame.to_ndarray(format='rgb24')
            measurement = extractor.measure(rgb)
            timestamp_valid = frame.pts is not None
            if not timestamp_valid:
                missing_pts += 1
                measurement['quality_flags'].append('missing_presentation_timestamp')
                measurement['measurement_valid'] = False
            video_s = float(frame.pts * frame.time_base) - camera['start_pts_s'] if timestamp_valid else frame_index / camera['fps']
            if abs(video_s - frame_index / camera['fps']) > .5 / camera['fps']:
                measurement['quality_flags'].append('non_cfr_timestamp_mapping')
                measurement['measurement_valid'] = False
            features.append(dict(frame_index=frame_index, video_s=video_s,
                                 pts=int(frame.pts) if timestamp_valid else None,
                                 time_base=str(frame.time_base),
                                 estimated_wall_elapsed_s=float(clock[frame_index]) if clock is not None else None,
                                 processing_ms=(time.perf_counter() - before) * 1000, **measurement))
            if progress and len(features) % 100 == 0:
                progress(frame_index / max(1, camera['frames'] - 1))
    actual_count = last_frame + 1
    if not features or actual_count != camera['frames']:
        raise ValueError(f'Decode incomplete: expected {camera["frames"]} frames, decoded {actual_count}')
    detections, detection_events = detect(features, config, camera['fps'], run['run_id']) if fill_source == 'image' else ([], [])
    observations = (build_detected_observations(features, detections, config, camera['fps'], run['run_id']) if fill_source == 'image'
                    else build_observations(features, run['annotations'], config, camera['fps'], actual_count, run['run_id']))
    for observation in observations:
        observation.update(source_video_sha256=camera['sha256'], source_annotation_sha256=run['annotation_sha256'] if fill_source == 'annotations' else None)
    manifest = dict(run_id=run['run_id'], **provenance, config=config.to_dict(),
                    source_cameras=run['cameras'], annotation_path=run['annotation_path'],
                    fps=camera['fps'], frame_count=actual_count, sample_count=len(features),
                    missing_pts_sample_count=missing_pts, annotation_count=len(run['annotations']),
                    analysis_seconds=time.perf_counter() - started,
                    processing_p95_ms=float(np.percentile([item['processing_ms'] for item in features], 95)),
                    measurement_kind='fixed_roi_camera_srgb_d65_lab',
                    clock_status='unreviewed OCR estimate' if clock is not None else 'wall clock unavailable',
                    mode='independent_image_fill_replay' if fill_source == 'image' else 'annotation_guided_completed_vial_replay',
                    detection_count=len(detections), recipe_context_source='unavailable' if fill_source == 'image' else 'provided_recipe_annotation',
                    open_fill_at_eof=features[-1].get('fill_detection', {}).get('detection_id'),
                    limitations=['no stage-aware early alert', 'image mode requires separate recipe context',
                                 'completion waits for carrier departure', 'visible stream boundary is not a pump signal',
                                 'no defect ground truth', 'fixed pilot ROI', 'unvalidated thresholds'])
    payload = dict(manifest=manifest, annotations=[item.to_dict() for item in run['annotations']],
                   features=features, observations=observations, detections=detections, detection_events=detection_events)
    write_json(cache, payload)
    if progress:
        progress(1.)
    return RunData(payload, config)


def export_csv(observations):
    fields = ['run_id', 'vial_id', 'recipe_id', 'start_frame', 'end_frame', 'available_frame', 'available_video_s',
              'measurement_valid', 'quality_flags', 'lab', 'rgb', 'measurement_config_id',
              'sample_frames', 'accepted_sample_frames', 'source_video_sha256', 'source_annotation_sha256',
              'event_source', 'partial_start_possible', 'baseline_hold_first', 'baseline_hold_first_two',
              'detection_id', 'start_available_frame', 'departure_frame', 'confirmed_frame', 'episodes',
              'status', 'delta_e2000', 'reference_id', 'threshold_de2000', 'threshold_status']
    output = io.StringIO(newline='')
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for item in observations:
        row = {key: item.get(key) for key in fields}
        row.update({key: item['assessment'][key] for key in ('status', 'delta_e2000', 'reference_id', 'threshold_de2000', 'threshold_status')})
        for key in ('quality_flags', 'lab', 'rgb', 'sample_frames', 'accepted_sample_frames', 'episodes'):
            row[key] = json.dumps(row[key], allow_nan=False)
        writer.writerow(row)
    return output.getvalue()
