"""CPU replay of frozen example predictions; not independent model accuracy."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('Use a fresh verification output directory')
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    import numpy as np
    from PIL import Image
    import torch
    from temporal_pool import TemporalPool, validate_rgb

    manifest = json.loads((args.assets / 'manifest.json').read_text())
    for name, digest in manifest['files'].items():
        path = (args.assets / name).resolve()
        if not path.is_relative_to(args.assets.resolve()):
            raise ValueError('Manifest path outside asset directory')
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('Asset hash mismatch: ' + name)
    torch.set_num_threads(4)
    model = TemporalPool(args.assets / 'models/liquid-best.pt', device='cpu')
    checks = []
    for row in json.loads((args.assets / 'examples/examples.json').read_text())['examples']:
        current = np.array(Image.open(args.assets / row['current']).convert('RGB'))
        reference = np.array(Image.open(args.assets / row['reference']).convert('RGB'))
        for rgb, key in [(current, 'source_rgb_sha256'), (reference, 'reference_rgb_sha256')]:
            if hashlib.sha256(rgb.tobytes()).hexdigest() != row[key]:
                raise ValueError('Example RGB identity mismatch')
        result = model.predict(current, reference)
        differences = {}
        for key in ['raw_mask', 'component_mask']:
            expected = np.array(Image.open(args.assets / row['expected_' + key])) > 0
            differences[key] = int(((result[key] > 0) != expected).sum())
        if any(differences.values()) or result['presence_gate_passed'] != row['expected_presence_gate']:
            raise ValueError('Frozen example mask/gate differs: ' + str(row['frame']))
        error = abs(result['presence'] - row['expected_presence'])
        if error > 1e-4:
            raise ValueError('Frozen example presence differs')
        checks.append(dict(frame=row['frame'], mask_pixel_differences=differences, presence_abs_difference=error))
    for invalid in [np.zeros((480, 640, 3), np.float32), np.zeros((240, 320, 3), np.uint8)]:
        try:
            validate_rgb(invalid)
        except ValueError:
            continue
        raise ValueError('Invalid input accepted')
    report = dict(schema='temporal-handoff-check/v1', device='cpu', GPU_used=False,
                  torch=torch.__version__, asset_hashes_verified=len(manifest['files']),
                  examples=checks, invalid_shape_and_dtype_rejected=True,
                  real_accuracy=None, measurement_QA_executed=False)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
