"""Single-view local video of recorded events and draft Cosmos explanations."""
import argparse
from pathlib import Path
import shutil
import textwrap

from liquid_specialists_data import read, write, sha


def label(cv2, canvas, value, x, y, color=(225, 233, 240), size=.57):
    cv2.putText(canvas, value, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)


def main(a):
    import av
    import cv2
    import numpy as np
    if a.out.exists():
        raise ValueError('Use a fresh output directory')
    manifest = read(a.cases/'manifest.json')
    completed = read(a.reviews/'completed.json')
    if sha(a.source) != manifest['source_sha256']:
        raise ValueError('Wrong source video')
    a.out.mkdir(parents=True)
    shutil.copy2(__file__, a.out/Path(__file__).name)
    path = a.out/'cosmos-event-review.mp4'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 5., (1280, 720))
    if not writer.isOpened():
        raise RuntimeError('Cannot encode')
    count = 0
    trace = []
    try:
        for case in manifest['cases']:
            response_path = a.reviews/f'{case["id"]}-{a.response_mode}.json'
            if sha(response_path) != completed['artifacts'][response_path.name]:
                raise ValueError('Response changed')
            record = read(response_path)
            parsed = record['parsed']
            identities = {f['source_frame']: f for f in read(a.cases/case['id']/'source-frames.json')}
            with av.open(str(a.source)) as c:
                for index, frame in enumerate(c.decode(video=0)):
                    seconds = float(frame.pts*frame.time_base)
                    if seconds >= case['end_s']:
                        break
                    if seconds < case['start_s'] or index % 6:
                        continue
                    rgb = frame.to_ndarray(format='rgb24')
                    digest = __import__('hashlib').sha256(rgb.tobytes()).hexdigest()
                    expected = identities[index]
                    if digest != expected['source_rgb_sha256'] or frame.pts != expected['pts'] or str(frame.time_base) != expected['time_base']:
                        raise ValueError('Source pixels or timing changed')
                    canvas = np.full((720, 1280, 3), (23, 28, 38), np.uint8)
                    label(cv2, canvas, 'COSMOS EVENT REVIEW / OFFLINE EXPERIMENT', 24, 42, size=.83)
                    label(cv2, canvas, f'Run 4 | {case["id"]} | native video time {seconds:.1f}s', 24, 77)
                    canvas[115:595, 20:660] = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    caption_mode = record.get('status_origin') == 'deterministic_recorded_evidence'
                    lines = ['Cosmos3-Edge / recorded event caption' if caption_mode else 'Cosmos3-Edge / video + recorded context',
                             'Draft caption may be incorrect; review required', '',
                             'VLM CAPTION / UNVERIFIED', parsed['visible_summary'] if parsed else 'Response rejected by the display contract.', '',
                             'RECORDED STATUS / DETERMINISTIC' if caption_mode else 'OPERATOR MESSAGE',
                             parsed['operator_message'] if parsed else record['contract_error'], '',
                             'POLICY: UNAVAILABLE', 'No intended recipe or approved reference.', 'Hardware command: NONE']
                    y = 130
                    for line in lines:
                        for part in textwrap.wrap(line, 55) or ['']:
                            label(cv2, canvas, part, 690, y, size=.53)
                            y += 23
                    label(cv2, canvas, 'Retrospective clip explanation: available after the whole clip, not a live early-warning measurement.', 24, 643, size=.55)
                    label(cv2, canvas, 'The VLM cannot approve a vial or control equipment. This video does not demonstrate a real defect.', 24, 680, color=(130, 200, 245), size=.55)
                    writer.write(canvas)
                    trace.append({'output_frame': count, 'case': case['id'], 'source_frame': index,
                                  'source_rgb_sha256': digest, 'pts': frame.pts, 'time_base': str(frame.time_base)})
                    if count % 40 == 20:
                        cv2.imwrite(str(a.out/f'{case["id"]}-poster.png'), canvas)
                    count += 1
    finally:
        writer.release()
    with av.open(str(path)) as c:
        decoded = sum(1 for f in c.decode(video=0) if f.width == 1280 and f.height == 720)
    if decoded != count or count != 240:
        raise ValueError('Incomplete output video')
    write(a.out/'source-trace.json', trace)
    write(a.out/'completed.json', {'decoded_frames': decoded, 'duration_s': count/5,
        'reviews_sha256': sha(a.reviews/'completed.json'), 'hardware_actions': 0,
        'artifacts': {p.name: sha(p) for p in a.out.iterdir() if p.is_file()}})
    print(f'Fully decoded {decoded} frames')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'cases', 'reviews', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--response-mode', default='with_telemetry')
    main(p.parse_args())
