"""Unchanged measurement-window and QA policy shared by replay and monitor."""
from bisect import bisect_left, bisect_right
import numpy as np
from .measurement import delta_e


def geometry_evidence(selected, accepted, flags):
    """Retain exact mask/frame pairs behind the aggregate, including failures."""
    if not any('geometry' in row for row in selected):
        return {}  # Legacy synthetic feature fixtures, not production frames.
    tracks = {row['geometry']['track_id'] for row in accepted if row.get('geometry')}
    if len(tracks) > 1:
        flags.append('vial_track_changed_during_measurement')
    if len(accepted) < len(selected):
        for row in selected:
            if not row['measurement_valid']:
                flags.extend(row.get('geometry',{}).get('quality_flags',[]))
    flags[:] = list(dict.fromkeys(flags))
    return dict(measurement_evidence=[dict(frame_index=row['frame_index'], video_s=row['video_s'],
                accepted=row['measurement_valid'], rgb=row['rgb'], lab=row['lab'], quality_flags=row['quality_flags'],
                geometry=row.get('geometry')) for row in selected],
                geometry_review_status='unreviewed_model_proposals',
                measurement_kind='tracked_colored_liquid_mask_camera_srgb_d65_lab')


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
        evidence = geometry_evidence(selected, accepted, flags)
        result.append(dict(event_id=f'run{run_id}-vial{annotation.vial_id}', run_id=run_id,
                           **annotation.to_dict(), **evidence, available_frame=available_frame,
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
        evidence = geometry_evidence(selected, accepted, flags)
        observations.append(dict(item, **evidence, event_id=item['detection_id'], run_id=run_id,
                                 recipe_id=None, volumes_ml={}, dispense_order=None, position_in_batch=None,
                                 previous_recipe_id=None, baseline_hold_first=None, baseline_hold_first_two=None,
                                 available_frame=item['confirmed_frame'], available_video_s=item['confirmed_frame'] / fps,
                                 sample_frames=[row['frame_index'] for row in selected],
                                 accepted_sample_frames=[row['frame_index'] for row in accepted],
                                 lab=lab, rgb=rgb, temporal_spread_de2000=spread,
                                 quality_flags=flags, measurement_valid=not flags and lab is not None,
                                 measurement_config_id=config.fingerprint, event_source='image_fill_detector'))
    return observations
