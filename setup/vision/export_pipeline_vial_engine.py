"""Export the frozen vial model at the existing full-pipeline input shape."""
import argparse
from pathlib import Path
import shutil
import time
from liquid_specialists_data import read, write, sha
from probe_prompted_liquid import BASELINE_SHA


def main(a):
    if a.out.exists():
        raise ValueError('Use a fresh output directory')
    if sha(a.model) != BASELINE_SHA:
        raise ValueError('Unexpected vial checkpoint')
    from ultralytics import YOLO
    import torch
    import tensorrt
    import ultralytics
    a.out.mkdir(parents=True)
    shutil.copy2(__file__, a.out/Path(__file__).name)
    target = a.out/'vial-source.pt'
    shutil.copy2(a.model, target)
    write(a.out/'plan.json', {'source_model_sha256': BASELINE_SHA, 'input_hw': [640, 640],
        'precision': 'fp16', 'batch': 1, 'dynamic': False, 'embedded_nms': False,
        'changes_training_or_thresholds': False, 'run5_used': False})
    t = time.perf_counter()
    engine = Path(YOLO(str(target)).export(format='engine', imgsz=640, half=True,
        batch=1, dynamic=False, workspace=2, simplify=False, device=0, verbose=False))
    write(a.out/'completed.json', {'engine': engine.name, 'engine_sha256': sha(engine),
        'source_model_sha256': BASELINE_SHA, 'input_hw': [640, 640], 'precision': 'fp16',
        'seconds': time.perf_counter()-t, 'gpu': torch.cuda.get_device_name(0),
        'torch': torch.__version__, 'tensorrt': tensorrt.__version__, 'ultralytics': ultralytics.__version__,
        'artifacts': {p.name: sha(p) for p in a.out.iterdir() if p.is_file()}})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    main(p.parse_args())
