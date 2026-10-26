"""Portable inference for the frozen paired-difference liquid candidate.

This exports model proposals only. Reference qualification, same-frame vial
association and measurement QA remain explicit downstream responsibilities.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from .train_pair_masks import ANCHOR, ROI, build_model, features
except ImportError:
    from train_pair_masks import ANCHOR, ROI, build_model, features


CHECKPOINT_SHA256 = '6714fe32191d8eba230aa87246f38a25c2906c619d60d7ba30c9e60205352eef'
HELPER_SHA256 = 'a1bd2e6c2dc80d35743049ace3e37f30fe4c0c8f1cd9aa8614867c90081fd31c'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_rgb(rgb):
    if not isinstance(rgb, np.ndarray) or rgb.shape != (480, 640, 3) or rgb.dtype != np.uint8:
        raise ValueError('Expected original RGB uint8 pixels at 640x480; do not resize or mirror implicitly')


class TemporalPool:
    """Load once; predict from explicitly supplied current/reference RGB pairs."""

    def __init__(self, checkpoint, device='cpu'):
        import torch
        if sha(checkpoint) != CHECKPOINT_SHA256:
            raise ValueError('Checkpoint does not match the frozen temporal candidate')
        if sha(Path(__file__).with_name('train_pair_masks.py')) != HELPER_SHA256:
            raise ValueError('Architecture helper differs from the executed frozen helper')
        self.device = torch.device(device)
        self.model = build_model(6).to(self.device).eval()
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        self.model.load_state_dict(state['model_state'])

    def predict(self, current, reference):
        import cv2
        import torch
        validate_rgb(current)
        validate_rgb(reference)
        x0, y0, x1, y1 = ROI
        a0, b0, a1, b1 = ANCHOR

        def inputs(rgb):
            crop = rgb[y0:y1, x0:x1].transpose(2, 0, 1).copy()
            tensor = torch.from_numpy(crop).to(self.device, dtype=torch.float32)[None] / 255
            median = np.median(rgb[b0:b1, a0:a1], axis=(0, 1))
            anchor = torch.tensor(median, device=self.device, dtype=torch.float32)[None] / 255
            return tensor, anchor

        cur, ca = inputs(current)
        ref, ra = inputs(reference)
        with torch.inference_mode():
            logits, presence = self.model(features(cur, ref, ca, ra, 'paired_diff_gain'))
        raw = np.zeros((480, 640), dtype=np.uint8)
        raw[y0:y1, x0:x1] = (logits[0].sigmoid().cpu().numpy() >= .5).astype(np.uint8)
        probability = float(presence.sigmoid()[0])
        count, labels, stats, _ = cv2.connectedComponentsWithStats(raw)
        component = np.zeros_like(raw)
        box = None
        if count > 1:
            k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            component[labels == k] = 1
            x, y, w, h, _ = map(int, stats[k])
            box = [x, y, x + w, y + h]
        accepted = probability >= .5 and box is not None
        return dict(raw_mask=raw, component_mask=component,
                    proposal_mask=component if accepted else np.zeros_like(raw),
                    presence=probability, presence_gate_passed=accepted,
                    proposal_box=box if accepted else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'current', 'reference', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--device', default='cpu', help='Use cuda:0 only on the approved GPU host')
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('Use a fresh output directory; previous evidence is preserved')
    current = np.array(Image.open(args.current).convert('RGB'), copy=True)
    reference = np.array(Image.open(args.reference).convert('RGB'), copy=True)
    import torch
    torch.set_num_threads(4)
    predictor = TemporalPool(args.checkpoint, args.device)
    result = predictor.predict(current, reference)
    args.out.mkdir(parents=True, exist_ok=False)
    masks = {}
    for key in ('raw_mask', 'component_mask', 'proposal_mask'):
        mask = result.pop(key)
        target = args.out / (key + '.png')
        Image.fromarray(mask * 255).save(target)
        masks[key] = dict(path=target.name, sha256=sha(target), pixels=int(mask.sum()))
    report = dict(schema='temporal-pool-proposal/v1', **result, masks=masks,
                  checkpoint_sha256=CHECKPOINT_SHA256, device=str(predictor.device),
                  roi=ROI, anchor=ANCHOR,
                  current=dict(file_sha256=sha(args.current), rgb_sha256=hashlib.sha256(current.tobytes()).hexdigest()),
                  reference=dict(file_sha256=sha(args.reference), rgb_sha256=hashlib.sha256(reference.tobytes()).hexdigest(),
                                 status='unreviewed_not_certified_empty'),
                  measurement_QA_executed=False, recipe_decision=None,
                  status='model_proposal_only' if result['presence_gate_passed'] else 'no_model_proposal')
    (args.out / 'observation.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(output=str(args.out), presence=result['presence'], status=report['status'])))


if __name__ == '__main__':
    main()
