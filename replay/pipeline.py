"""Causal image detection and completed measurements; optional annotation baseline."""
from bisect import bisect_right
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

from .observations import build_observations, build_detected_observations
from .references import reference_for, assess, validate_references
from .recipes import RecipePlan


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, allow_nan=False, separators=(',', ':')), encoding='utf-8')
    temporary.replace(path)


class RunData:
    def __init__(self, payload, config, recipe_plan=None):
        self.payload, self.config = payload, config
        self.features = payload['features']
        self.frames = [item['frame_index'] for item in self.features]
        self.observations = payload['observations']
        self.annotations = payload['annotations']
        self.context_frames = [item['start_frame'] for item in self.annotations]
        self.image_mode = payload['manifest'].get('fill_source') == 'image'
        self.recipe_plan = recipe_plan or RecipePlan()

    def completed(self, frame, references):
        items = [self.recipe_plan.bind(item) if self.image_mode else dict(item)
                 for item in self.observations if item['available_frame'] <= frame]
        return [dict(item, assessment=assess(item, references, self.config)) for item in items]

    def snapshot(self, frame, references):
        frame = max(0, min(int(frame), self.payload['manifest']['frame_count'] - 1))
        index = bisect_right(self.frames, frame) - 1
        feature = self.features[index] if index >= 0 else None
        if self.image_mode:
            completed = self.completed(frame, references)
            detection = feature.get('fill_detection', {}) if feature else {}
            active = detection.get('vial_id')
            current = None if active else (completed[-1] if completed else None)
            if active:
                context = self.recipe_plan.bind(dict(run_id=self.payload['manifest']['run_id'],
                    source_video_sha256=self.payload['manifest'].get('video_sha256'), vial_id=active,
                    start_frame=detection['start_frame'], available_frame=frame), until=frame)
            else:
                selection = self.recipe_plan.at(self.payload['manifest']['run_id'], self.payload['manifest'].get('video_sha256'), frame)
                context = dict(selection, vial_id=None, recipe_context=selection) if selection else current
            return dict(run_id=self.payload['manifest']['run_id'], frame_index=frame,
                        video_s=frame / self.payload['manifest']['fps'], phase=detection.get('phase', 'waiting_for_fill'),
                        feature=feature, context=context, completed=completed, current_completed=current,
                        visible_references=[r for r in references if r['source_run_id'] == self.payload['manifest']['run_id'] and
                                            r['available_frame'] <= frame], fill_source='image',
                        recipe_plan_sha256=self.recipe_plan.fingerprint,
                        coverage=dict(available=len(completed), valid=sum(row['measurement_valid'] for row in completed),
                                      assessed=sum(row['assessment']['status'] != 'unavailable' for row in completed)))
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
            events.extend(dict(event, video_s=event['frame_index'] / fps, source='monitor_session')
                          for event in self.payload.get('session_events', []) if event['frame_index'] <= frame)
            for s in self.recipe_plan.selections(self.payload['manifest']['run_id'], self.payload['manifest'].get('video_sha256'), frame):
                events.append(dict(kind='intended_recipe_active', frame_index=s['active_from_frame'],
                                   video_s=s['active_from_frame'] / fps, selection=s))
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
    implementation = hashlib.sha256(b''.join((Path(__file__).parent / name).read_text(encoding='utf-8').encode()
                                             for name in sorted(p.name for p in Path(__file__).parent.glob('*.py')))).hexdigest()[:16]
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
    from .inputs import RecordingSource
    from .monitor import MonitorSession
    source = RecordingSource(camera, config.sample_stride, clock)
    session = MonitorSession(config, camera['fps'], run['run_id'], camera['sha256'],
                             session_id='replay-' + camera['sha256'][:16], retain_features=True)
    started = time.perf_counter()
    session.start()
    try:
        for packet in source:
            session.ingest(packet)
            if progress and len(session.features) % 100 == 0:
                progress(packet.frame_index / max(1, camera['frames'] - 1))
        if source.decoded_frames != camera['frames']:
            session.stop('incomplete_recording_decode')
            raise ValueError(f'Decode incomplete: expected {camera["frames"]} frames, decoded {source.decoded_frames}')
    except Exception:
        session.stop('input_error')
        raise
    session.stop('end_of_recording', source.decoded_frames - 1)
    actual_count, features = source.decoded_frames, session.features
    if not features:
        raise ValueError('Recording contains no sampled frames')
    missing_pts = sum('missing_presentation_timestamp' in f['quality_flags'] for f in features)
    detections, detection_events = (session.detector.completed, session.detector.events) if fill_source == 'image' else ([], [])
    observations = (session.observations if fill_source == 'image' else
                    build_observations(features, run['annotations'], config, camera['fps'], actual_count, run['run_id']))
    for observation in observations:
        observation.update(source_video_sha256=camera['sha256'], source_annotation_sha256=run['annotation_sha256'] if fill_source == 'annotations' else None)
    manifest = dict(run_id=run['run_id'], **provenance, config=config.to_dict(),
                    source_cameras=run['cameras'], annotation_path=run['annotation_path'],
                    fps=camera['fps'], frame_count=actual_count, sample_count=len(features),
                    missing_pts_sample_count=missing_pts, annotation_count=len(run['annotations']),
                    analysis_seconds=time.perf_counter() - started,
                    processing_p95_ms=float(np.percentile([item['processing_ms'] for item in features], 95)),
                    measurement_kind='tracked_colored_liquid_mask_camera_srgb_d65_lab',
                    geometry_model_sha256=config.vision_model_sha256,
                    liquid_segmentation_model_sha256=config.liquid_segmentation_model_sha256,
                    image_proposal_cache_sha256=config.image_proposal_cache_sha256,
                    proposal_source='pinned_per_image_detector_cache' if config.image_proposal_cache_path else 'fresh_image_model_inference',
                    geometry_review_status='unreviewed_model_proposals',
                    measurement_profile_id=session.profile_id, input_kind='existing_recording', session_id=session.session_id,
                    clock_status='unreviewed OCR estimate' if clock is not None else 'wall clock unavailable',
                    mode='independent_image_fill_replay' if fill_source == 'image' else 'annotation_guided_completed_vial_replay',
                    detection_count=len(detections), recipe_context_source='separate_explicit_recipe_plan' if fill_source == 'image' else 'provided_recipe_annotation',
                    open_fill_at_eof=features[-1].get('fill_detection', {}).get('detection_id'),
                    limitations=['no stage-aware early alert', 'image mode requires separate recipe context',
                                 'completion waits for carrier departure', 'visible stream boundary is not a pump signal',
                                 'no defect ground truth', 'weak-label detector and heuristic liquid mask; no reviewed mask accuracy',
                                 'clear/neutral liquid unavailable', 'unvalidated thresholds'])
    payload = dict(manifest=manifest, annotations=[item.to_dict() for item in run['annotations']],
                   features=features, observations=observations, detections=detections, detection_events=detection_events,
                   session_events=session.session_events)
    write_json(cache, payload)
    if progress:
        progress(1.)
    return RunData(payload, config)


def export_csv(observations):
    fields = ['run_id', 'vial_id', 'recipe_id', 'start_frame', 'end_frame', 'available_frame', 'available_video_s',
              'measurement_valid', 'quality_flags', 'lab', 'rgb', 'measurement_config_id',
              'measurement_evidence', 'geometry_review_status', 'measurement_kind',
              'sample_frames', 'accepted_sample_frames', 'source_video_sha256', 'source_annotation_sha256',
              'event_source', 'partial_start_possible', 'baseline_hold_first', 'baseline_hold_first_two',
              'detection_id', 'start_available_frame', 'departure_frame', 'confirmed_frame', 'episodes',
              'recipe_definition_id', 'recipe_context', 'context_quality_flags', 'reference_rejections', 'measurement_profile_id',
              'expected_lab', 'reference_available_frame', 'reference_source_description',
              'status', 'delta_e2000', 'reference_id', 'threshold_de2000', 'threshold_status']
    output = io.StringIO(newline='')
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for item in observations:
        row = {key: item.get(key) for key in fields}
        row.update({key: item['assessment'][key] for key in ('status', 'delta_e2000', 'reference_id', 'threshold_de2000', 'threshold_status')})
        row.update({key: item['assessment'].get(key) for key in ('expected_lab', 'reference_rejections', 'reference_available_frame', 'reference_source_description')})
        for key in ('quality_flags', 'lab', 'rgb', 'sample_frames', 'accepted_sample_frames', 'episodes', 'measurement_evidence',
                    'recipe_context', 'context_quality_flags', 'reference_rejections', 'expected_lab'):
            row[key] = json.dumps(row[key], allow_nan=False)
        writer.writerow(row)
    return output.getvalue()
