"""Source-verified Run 4 event clips for an offline VLM experiment."""
import argparse
import json
from pathlib import Path
import shutil

from liquid_specialists_data import read, write, sha

CASES = [
    ('early-pool', 8, 16),
    ('first-pause', 20, 28),
    ('checkpoint', 64, 72),
    ('intermediate-pause', 145, 153),
    ('restart', 164, 172),
    ('later-pool', 272, 280),
]


def main(a):
    import av
    import cv2
    import numpy as np
    if a.out.exists():
        raise ValueError('Use a fresh output directory')
    plan = read(a.evidence / 'plan.json')
    done = read(a.evidence / 'completed.json')
    if a.source.name != 'vial_fill_video_4.mp4' or sha(a.source) != plan['source_sha256']:
        raise ValueError('Wrong Run 4 source')
    if sha(a.evidence / 'frames.jsonl') != done['artifacts']['frames.jsonl']:
        raise ValueError('Frozen evidence changed')
    rows = {r['source_frame']: r for r in map(json.loads, (a.evidence / 'frames.jsonl').read_text().splitlines())}
    a.out.mkdir(parents=True)
    shutil.copy2(__file__, a.out / Path(__file__).name)
    manifest = {'schema_version': 1, 'kind': 'offline_cosmos_event_review',
                'source_sha256': plan['source_sha256'], 'evidence_sha256': sha(a.evidence / 'frames.jsonl'),
                'run': 4, 'run5_used': False, 'human_labels_created': False,
                'quality_truth_available': False, 'hardware_actions': 0, 'cases': []}
    for name, start, end in CASES:
        folder = a.out / name
        folder.mkdir()
        writer = cv2.VideoWriter(str(folder / 'source.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 30., (640, 480))
        if not writer.isOpened():
            raise RuntimeError('Cannot encode event clip')
        frames = []
        telemetry = []
        try:
            with av.open(str(a.source)) as c:
                for index, frame in enumerate(c.decode(video=0)):
                    seconds = float(frame.pts * frame.time_base)
                    if seconds >= end:
                        break
                    if seconds < start:
                        continue
                    rgb = frame.to_ndarray(format='rgb24')
                    digest = __import__('hashlib').sha256(rgb.tobytes()).hexdigest()
                    frames.append({'source_frame': index, 'pts': frame.pts,
                                   'time_base': str(frame.time_base), 'seconds': seconds,
                                   'source_rgb_sha256': digest})
                    if index in rows:
                        row = rows[index]
                        if digest != row['source_rgb_sha256'] or frame.pts != row['pts'] or str(frame.time_base) != row['time_base']:
                            raise ValueError('Source pixels or native timing differ')
                        m = row['measurement']
                        telemetry.append({'source_frame': index, 'clip_seconds': seconds-start,
                                          'image_phase': row['fill_detection']['phase'],
                                          'detection_id': row['fill_detection']['detection_id'],
                                          'sample_qa_passes': m['measurement_valid'],
                                          'qa_flags': m['quality_flags'],
                                          'sampling_pixels': m['geometry']['mask_pixels']})
                    if len(frames) == 121:
                        cv2.imwrite(str(folder / 'poster.png'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                    writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        finally:
            writer.release()
        with av.open(str(folder / 'source.mp4')) as c:
            decoded = sum(1 for _ in c.decode(video=0))
        if decoded != len(frames) or decoded != (end-start)*30:
            raise ValueError('Incomplete event clip')
        context = {'provenance': 'frozen image-based experimental telemetry, not independent truth',
                   'source_run': 4, 'source_start_s': start, 'source_end_exclusive_s': end,
                   'recipe_id': None, 'acceptable_reference': None, 'metered_volume': None,
                   'authoritative_dispense_complete': None,
                   'geometry_review_status': 'unreviewed', 'physical_vial_identity': 'unconfirmed',
                   'policy_assessment': 'unavailable', 'policy_reason': 'Missing recipe/reference and reviewed geometry',
                   'hardware_command': 'none', 'samples': telemetry}
        write(folder / 'context.json', context)
        write(folder / 'source-frames.json', frames)
        manifest['cases'].append({'id': name, 'start_s': start, 'end_s': end,
                                  'decoded_frames': decoded, 'source_checked_samples': len(telemetry),
                                  'video': f'{name}/source.mp4', 'context': f'{name}/context.json',
                                  'artifacts': {f: sha(folder/f) for f in ('source.mp4', 'poster.png', 'context.json', 'source-frames.json')}})
    write(a.out / 'manifest.json', manifest)
    print(json.dumps({'cases': len(manifest['cases']), 'decoded_frames': sum(c['decoded_frames'] for c in manifest['cases'])}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--evidence', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    main(p.parse_args())
