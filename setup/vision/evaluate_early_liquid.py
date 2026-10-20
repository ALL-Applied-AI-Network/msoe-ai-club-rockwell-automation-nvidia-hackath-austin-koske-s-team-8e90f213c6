"""Evaluate new architecture pilots on Run 4 and short supplied targets only."""
import argparse
import json
from pathlib import Path

import cv2

from audit_liquid_specialists import development_rows, targets_for, summarize, gallery
from evaluate_liquid_specialists import SpecialistPair
from liquid_specialists_data import read, write, sha
from train_early_liquid import ARMS, build_resnet, validate


class SizedYOLO:
    def __init__(self, model, imgsz):
        self.model = model; self.imgsz = imgsz

    def predict(self, images, **kwargs):
        return self.model.predict(images, **{**kwargs, 'imgsz': self.imgsz})


class EarlyPair(SpecialistPair):
    def __init__(self, baseline, root, arm, device):
        import torch
        from ultralytics import YOLO
        torch.set_num_threads(4)
        self.torch = torch; self.device = device; self.root = root; self.arm = arm
        self.roi = [0, 0, 640, 480]
        self.baseline = YOLO(baseline)
        self.metadata = read(root/'models'/arm/'model.json')
        self.dino = arm == 'fasterrcnn-resnet50'  # Shared TorchVision tensor-inference path.
        if self.dino:
            self.candidate = build_resnet(root)
            state = torch.load(root/'models'/arm/'best.pt', map_location='cpu', weights_only=True)
            self.candidate.load_state_dict(state['model_state'])
            self.candidate.to(device).eval()
        else:
            model = YOLO(root/'models'/arm/'best.pt')
            if model.names != {0: 'colored_liquid'}:
                raise ValueError('Candidate class contract changed')
            self.candidate = SizedYOLO(model, ARMS[arm]['imgsz'])
        self.candidate_title = arm+' | liquid only, fixed vial detector'
        self.inference_frames = 0; self.inference_s = 0.


def main(a):
    if a.out.exists():
        raise ValueError('Use a fresh output directory')
    plan = validate(a.experiment)
    data, sources = development_rows(a.source_experiment)
    original = read(a.previous_audit/'comparison.json')
    if (original['development_run'] != 4 or original['run5_images_used']
            or original['experiment_plan_sha256'] != sha(a.source_experiment/'experiment-plan.json')):
        raise ValueError('Previous audit provenance changed')
    if sha(a.baseline) != original['baseline_checkpoint_sha256']:
        raise ValueError('Baseline checkpoint changed')
    a.out.mkdir(parents=True)
    predictions = {name: read(a.previous_audit/(name+'-predictions.json')) for name in original['models']}
    expected = [r['source_id'] for r in sources]
    expected_targets = [targets_for(data, r) for r in sources]
    for rows in predictions.values():
        if [r['source_id'] for r in rows] != expected or [r['targets'] for r in rows] != expected_targets:
            raise ValueError('Previous audit image/target inventory differs')
    checkpoint_hashes = {}
    for arm in a.arms:
        model = read(a.experiment/'models'/arm/'model.json')
        digest = sha(a.experiment/'models'/arm/'best.pt')
        if (model['checkpoint_sha256'] != digest or model['excluded_runs'] != [5]
                or model['experiment_plan_sha256'] != sha(a.experiment/'experiment-plan.json')
                or model['preparation_sha256'] != sha(a.experiment/'preparation.json')
                or model['data_provenance_sha256'] != plan['data_provenance_sha256']):
            raise ValueError('Frozen model ancestry changed')
        checkpoint_hashes[arm] = digest
        pair = EarlyPair(a.baseline, a.experiment, arm, a.device)
        rows = []; baseline = []
        for offset in range(0, len(sources), 16):
            batch = sources[offset:offset+16]
            rgb = [cv2.cvtColor(cv2.imread(str(data/r['image'])), cv2.COLOR_BGR2RGB) for r in batch]
            result = pair.predict(rgb, .001)
            for i, source in enumerate(batch):
                common = dict(source_id=source['source_id'], targets=targets_for(data, source))
                rows.append(dict(**common, predictions=[p for p in result[1][i] if p['class_id'] == 1]))
                baseline.append(dict(**common, predictions=[p for p in result[0][i] if p['class_id'] == 1]))
        if baseline != predictions['Deployed']:
            raise ValueError('Baseline predictions differ from original audit')
        predictions[arm] = rows
        write(a.out/(arm+'-predictions.json'), rows)
        print(json.dumps(dict(arm=arm, **summarize(rows))), flush=True)
        del pair
    gallery(data, sources, predictions, a.out, short_only=True)
    write(a.out/'comparison.json', dict(schema_version=1, kind='early_liquid_run4_architecture_audit',
        experiment_plan_sha256=sha(a.experiment/'experiment-plan.json'),
        previous_audit_sha256=sha(a.previous_audit/'comparison.json'), checkpoint_hashes=checkpoint_hashes,
        evaluation_script_sha256=sha(Path(__file__)), development_run=4, run5_images_used=False,
        detector_promoted=False, measurement_qa_changed=False,
        models={name: summarize(rows) for name, rows in predictions.items()}, limits=plan['limits']))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('experiment', 'source-experiment', 'previous-audit', 'baseline', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--arms', nargs='+', choices=list(ARMS), default=list(ARMS))
    main(p.parse_args())
