"""Diagnostic artifacts for label owners. No label adjudication or QA changes."""
import hashlib
from pathlib import Path

import av
import cv2

from .pipeline import write_json


def export_review(data, run, output, base_url='http://127.0.0.1:8765'):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = data.payload['manifest']
    fps, run_id = manifest['fps'], run['run_id']
    features = {row['frame_index']: row for row in data.features}
    cases, requested = [], {}
    for observation in data.observations:
        samples = []
        for index in observation['sample_frames']:
            row = features[index]
            samples.append(dict(frame_index=index, mp4_s=row['video_s'], qa_valid=row['measurement_valid'],
                                qa_flags=list(row['quality_flags']), motion=row['motion'],
                                roi_sensitivity_de2000=row.get('roi_sensitivity_de2000'),
                                replay_url=f'{base_url}/?run={run_id}&frame={index}'))
        boundary_frames = {key: observation.get(key) for key in
                           ('start_frame', 'end_frame', 'departure_frame', 'confirmed_frame', 'available_frame')}
        case = dict(detection_id=observation['event_id'], candidate_number=observation['vial_id'],
                    measurement_available=observation['measurement_valid'],
                    rejection_reasons=list(observation['quality_flags']),
                    boundaries={key: dict(frame_index=value, mp4_s=value / fps) for key, value in boundary_frames.items() if value is not None},
                    sample_frames=list(observation['sample_frames']), accepted_sample_frames=list(observation['accepted_sample_frames']),
                    samples=samples, measured_lab=observation['lab'], measured_rgb=observation['rgb'],
                    temporal_spread_de2000=observation['temporal_spread_de2000'],
                    replay_url=f'{base_url}/?run={run_id}&frame={observation["available_frame"]}',
                    representative_frames=[], review_status='unreviewed', reviewed_label=None)
        if not observation['measurement_valid']:
            # Every actual sampled window frame plus departure: show evidence,
            # including accepted samples in an otherwise rejected window.
            indices = set(observation['sample_frames'])
            indices.add(observation.get('departure_frame', observation['available_frame']))
            for index in sorted(indices):
                path = f'frames/{observation["event_id"]}-f{index:06d}.jpg'
                representative = dict(frame_index=index, mp4_s=index / fps, path=path,
                                      role='window_sample' if index in observation['sample_frames'] else 'departure',
                                      replay_url=f'{base_url}/?run={run_id}&frame={index}')
                case['representative_frames'].append(representative)
                requested.setdefault(index, []).append((output / path, representative))
        cases.append(case)
    if requested:
        (output / 'frames').mkdir(exist_ok=True)
        with av.open(run['cameras']['vial_fill']['path']) as container:
            container.streams.video[0].thread_type = 'AUTO'
            for index, frame in enumerate(container.decode(video=0)):
                for path, representative in requested.pop(index, []):
                    if not cv2.imwrite(str(path), frame.to_ndarray(format='bgr24')):
                        raise OSError(f'Could not save diagnostic frame: {path}')
                    representative['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
                if not requested:
                    break
        if requested:
            raise ValueError('Diagnostic frames lie outside the decoded recording')
    document = dict(schema_version=1, artifact_kind='measurement_qa_review',
                    purpose='Diagnostic handoff; not reviewed timing, recipe or defect ground truth',
                    source=dict(run_id=run_id, video_sha256=manifest['video_sha256'], camera='vial_fill',
                                fps=fps, frame_count=manifest['frame_count'],
                                time_convention='zero-based decoded frames and MP4 seconds; physical timing unverified'),
                    config=manifest['config'], config_id=manifest['config_id'], implementation_id=manifest['implementation_id'],
                    measurement_profile_id=manifest.get('measurement_profile_id'),
                    counts=dict(detected=len(cases), measurement_available=sum(c['measurement_available'] for c in cases),
                                rejected=sum(not c['measurement_available'] for c in cases)), cases=cases)
    path = output / 'manifest.json'
    write_json(path, document)
    return path
