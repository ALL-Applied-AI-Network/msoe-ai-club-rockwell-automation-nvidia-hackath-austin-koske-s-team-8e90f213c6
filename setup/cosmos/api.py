"""Hosted NVIDIA inference client. Uses only Python's standard library."""
import argparse
import base64
import json
import mimetypes
import os
from pathlib import Path
import urllib.error
import urllib.request


def build_payload(image, model, prompt, max_tokens):
    content = [{'type': 'text', 'text': prompt}]
    if image:
        mime = mimetypes.guess_type(str(image))[0]
        if mime not in {'image/jpeg', 'image/png', 'image/webp'}:
            raise ValueError('Use a JPEG, PNG or WebP image.')
        if image.stat().st_size > 4 * 1024 * 1024:
            raise ValueError('Use an image smaller than 4 MiB for the onboarding request.')
        encoded = base64.b64encode(image.read_bytes()).decode('ascii')
        content.append({'type': 'image_url', 'image_url': {'url': f'data:{mime};base64,{encoded}'}})
    return {'model': model, 'messages': [{'role': 'user', 'content': content}],
            'max_tokens': max_tokens, 'temperature': 0, 'stream': False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=Path)
    parser.add_argument('--model', default='nvidia/cosmos3-nano-reasoner')
    parser.add_argument('--prompt', default='Describe the visible objects and colors. Do not infer hidden process conditions.')
    parser.add_argument('--max-tokens', type=int, default=96)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    payload = build_payload(args.image, args.model, args.prompt, args.max_tokens)
    if args.dry_run:
        print(json.dumps({'ok': True, 'model': args.model, 'image_attached': bool(args.image),
                          'request_bytes': len(json.dumps(payload)), 'network_request': False}))
        return 0
    key = os.environ.get('NVIDIA_API_KEY')
    if not key:
        parser.exit(2, 'Set NVIDIA_API_KEY in your own shell, or use --dry-run.\n')
    request = urllib.request.Request('https://integrate.api.nvidia.com/v1/chat/completions',
                                     data=json.dumps(payload).encode(), method='POST',
                                     headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        parser.exit(1, f'NVIDIA endpoint returned HTTP {exc.code}; check model access and the model card.\n')
    except urllib.error.URLError:
        parser.exit(1, 'Could not reach the NVIDIA endpoint. Check the network connection.\n')
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(result['choices'][0]['message']['content'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
