"""Small, credential-free environment checks. GPU operations belong in a Slurm job."""
import argparse
import importlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', choices=['data', 'cosmos', 'torch'], default='data')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    names = {
        'data': ['numpy', 'pandas', 'scipy', 'matplotlib', 'PIL', 'av', 'cv2', 'skimage', 'sklearn', 'colour'],
        'cosmos': ['torch', 'transformers', 'accelerate', 'huggingface_hub', 'safetensors', 'av'],
        'torch': ['torch'],
    }[args.profile]
    report = {'python': sys.version, 'architecture': platform.machine(), 'profile': args.profile,
              'checks': {}, 'ok': True}
    for name in names:
        try:
            module = importlib.import_module(name)
            report['checks'][name] = {'ok': True, 'version': getattr(module, '__version__', 'unknown')}
        except Exception as exc:
            report['checks'][name] = {'ok': False, 'error': str(exc)}
            report['ok'] = False
    if args.gpu:
        try:
            import torch
            assert torch.cuda.is_available(), 'CUDA is unavailable'
            x = torch.arange(1024, device='cuda', dtype=torch.float32)
            assert x.sum().item() == 523776.0
            report['gpu'] = {'name': torch.cuda.get_device_name(0),
                             'capability': list(torch.cuda.get_device_capability(0)),
                             'torch_cuda': torch.version.cuda, 'tensor_check': 'passed'}
        except Exception as exc:
            report['gpu'] = {'error': str(exc)}
            report['ok'] = False
    if args.profile == 'cosmos' and report['checks'].get('transformers', {}).get('ok'):
        import transformers
        report['cosmos_classes'] = {name: hasattr(transformers, name) for name in
                                   ['Cosmos3OmniForConditionalGeneration', 'Cosmos3EdgeForConditionalGeneration']}
    encoded = json.dumps(report, indent=2)
    print(encoded)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded + '\n', encoding='utf-8')
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
