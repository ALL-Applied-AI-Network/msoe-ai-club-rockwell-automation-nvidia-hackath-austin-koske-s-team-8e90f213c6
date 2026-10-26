"""CPU-only source-exact export of an already completed mask inference probe."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(a):
    import av
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw
    a.out.mkdir(parents=True, exist_ok=False)
    plan = json.loads((a.probe / 'plan.json').read_text())
    if sha(a.video) != plan['source_video_sha256']:
        raise ValueError('Source video differs')
    rows = [json.loads(s) for s in (a.probe / 'frames.jsonl').read_text().splitlines()]
    wanted = {r['source_frame']: r for r in rows if any(
        c['source_start_frame'] <= r['source_frame'] < c['source_end_frame_exclusive']
        and (r['source_frame'] - c['source_start_frame']) % 6 == 0 for c in plan['cycles'])}
    assert len(wanted) == 240
    writer = cv2.VideoWriter(str(a.out / 'temporal-pool-candidate.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 5, (640, 550))
    if not writer.isOpened():
        raise RuntimeError('Video writer unavailable')
    samples = []
    count = 0
    try:
        with av.open(str(a.video)) as container:
            for index, frame in enumerate(container.decode(video=0)):
                if index > max(wanted):
                    break
                if index not in wanted:
                    continue
                row = wanted[index]
                rgb = frame.to_ndarray(format='rgb24')
                assert hashlib.sha256(rgb.tobytes()).hexdigest() == row['source_rgb_sha256']
                assert frame.pts == row['pts'] and str(frame.time_base) == row['time_base']
                pred = row['modes']['paired_diff_gain']
                mask = np.zeros((480, 640), dtype=np.uint8)
                encoded = pred['component_mask'] if pred['presence_gate_passed'] else None
                if encoded:
                    x, y = encoded['origin']; w, h = encoded['size']; flat = np.zeros(w*h, dtype=np.uint8)
                    for start, length in encoded['runs']:
                        flat[start:start+length] = 1
                    mask[y:y+h, x:x+w] = flat.reshape(h, w)
                display = rgb.copy()
                display[mask > 0] = (.60 * display[mask > 0] + .40 * np.array([45, 255, 150])).astype(np.uint8)
                image = Image.new('RGB', (640, 550), '#101925'); image.paste(Image.fromarray(display), (0, 0))
                d = ImageDraw.Draw(image)
                d.text((10, 488), 'Synthetic-only temporal pool candidate | Run 4 | unreviewed', fill='white')
                d.text((10, 508), f'{row["case_id"]} / native frame {index} / presence {pred["presence"]:.3f}', fill='white')
                qa = 'available' if pred['measurement']['measurement_valid'] else 'unavailable'
                d.text((10, 528), f'Color measurement: {qa} | no human mask/onset accuracy yet', fill='white')
                writer.write(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)); count += 1
                if (index - next(c['source_start_frame'] for c in plan['cycles'] if c['case_id'] == row['case_id'])) in [0, 60, 90, 150, 210]:
                    samples.append((row['case_id'], index, image.copy()))
    finally:
        writer.release()
    cap = cv2.VideoCapture(str(a.out / 'temporal-pool-candidate.mp4')); decoded = 0
    while cap.read()[0]:
        decoded += 1
    cap.release()
    assert count == decoded == 240
    board = Image.new('RGB', (5*320, 6*295), '#101925')
    for i, (_, _, im) in enumerate(samples):
        im.thumbnail((320, 275)); board.paste(im, ((i % 5)*320, (i // 5)*295))
    board.save(a.out / 'candidate-contact-sheet.png')
    report = dict(frames=decoded, fps=5, duration_s=48, source_rgb_and_pts_verified=True,
        source_video_sha256=sha(a.video), checkpoint_sha256=plan['models']['paired_diff_gain']['checkpoint_sha256'],
        helper_sha256=sha(__file__), inference_rerun=False, gpu_used=False, approved_labels=False,
        artifacts={p.name: sha(p) for p in a.out.iterdir() if p.is_file()})
    (a.out / 'completed.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['probe', 'video', 'out']:
        p.add_argument('--' + name, type=Path, required=True)
    main(p.parse_args())
