"""Reject stale captures when substantially different fixture inputs look identical."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def validate(directory):
    rows = [json.loads(line) for line in (directory / 'images.jsonl').read_text().splitlines()]
    if not rows:
        raise ValueError('No captures in manifest')
    images = [np.asarray(Image.open(directory / row['image_file']).convert('RGB'), dtype=np.float32)
              for row in rows]
    signatures = [hashlib.sha256(image.tobytes()).hexdigest() for image in images]
    differences = []
    for i, row in enumerate(rows):
        for j in range(i):
            color_change = np.linalg.norm(np.array(row['linear_rgb_proxy']) - rows[j]['linear_rgb_proxy'])
            if color_change > .2:
                differences.append(float(np.abs(images[i] - images[j]).mean()))
    if differences and max(differences) < 1.0:
        raise ValueError('Substantially different recipe colors produced stale or imperceptibly different images')
    return dict(ok=True, captures=len(rows), unique_images=len(set(signatures)),
                maximum_color_pair_mae=max(differences, default=None),
                cross_sample_check=bool(differences),
                scope='Capture variation check only; not optical or physical calibration')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    try:
        result = validate(args.directory)
    except (ValueError, OSError) as exc:
        result = dict(ok=False, error=str(exc))
        (args.directory / 'capture_check.json').write_text(json.dumps(result, indent=2) + '\n')
        raise SystemExit(str(exc))
    (args.directory / 'capture_check.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
