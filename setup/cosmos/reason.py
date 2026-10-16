"""Local Cosmos Reasoner: small image test, explicit model and provenance."""
import argparse
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--model', default=os.getenv('COSMOS_MODEL', 'nvidia/Cosmos3-Edge'))
    parser.add_argument('--revision')
    parser.add_argument('--prompt', default='Describe the visible objects and colors. Do not infer hidden process conditions.')
    parser.add_argument('--max-tokens', type=int, default=96)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    revisions = {'nvidia/Cosmos3-Edge': '344d602b128d1bbdacb43b08d0a3626f46343e29',
                 'nvidia/Cosmos3-Nano': 'e59a53c25979a090fa8706c9acc0c254a6e89b92'}
    args.revision = args.revision or os.getenv('COSMOS_REVISION') or revisions.get(args.model)
    if not args.image.is_file():
        parser.error('Input image does not exist.')
    import torch
    import transformers
    from transformers import AutoProcessor
    if not torch.cuda.is_available():
        raise SystemExit('A CUDA GPU is required. On Rosie submit the Slurm job.')
    if args.model == 'nvidia/Cosmos3-Edge':
        cls = getattr(transformers, 'Cosmos3EdgeForConditionalGeneration', None)
        if cls is None:
            raise SystemExit('Edge needs a newer Transformers source integration; use the tested Nano profile.')
    elif args.model in {'nvidia/Cosmos3-Nano', 'nvidia/Cosmos3-Super'}:
        cls = transformers.Cosmos3OmniForConditionalGeneration
    else:
        raise SystemExit('Select an explicitly supported Cosmos3 model.')
    started = time.monotonic()
    processor = AutoProcessor.from_pretrained(args.model, revision=args.revision)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = cls.from_pretrained(args.model, revision=args.revision, dtype=dtype,
                               device_map='auto', attn_implementation='eager')
    messages = [{'role': 'user', 'content': [
        {'type': 'image', 'path': str(args.image.resolve())},
        {'type': 'text', 'text': args.prompt}]}]
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                            return_dict=True, return_tensors='pt').to(model.device, dtype)
    with torch.inference_mode():
        generated = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_tokens)
    trimmed = [output[len(source):] for source, output in zip(inputs.input_ids, generated)]
    text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    if '</think>' in text:
        text = text.split('</think>', 1)[1].strip()
    result = {'model': args.model, 'revision': args.revision, 'transformers': transformers.__version__,
              'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0),
              'image': str(args.image), 'prompt': args.prompt, 'response': text,
              'seconds_including_model_load': time.monotonic() - started}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
