"""Experimental event explanation; VLM output never enters the action coordinator."""
import argparse
import json
from pathlib import Path
import shutil
import time

from liquid_specialists_data import read, write, sha

MODEL = 'nvidia/Cosmos3-Edge'
REVISION = '344d602b128d1bbdacb43b08d0a3626f46343e29'
PROMPT = '''Review this short vial-filling clip. Return a concise JSON object with exactly:
{"visible_summary": "one sentence describing only visible events",
 "quality_assessment": "unavailable",
 "limitations": ["missing evidence"],
 "operator_message": "a short explanation for a human reviewer"}.
The intended recipe, acceptable reference, delivered volumes, approved mask labels,
physical vial identity and authoritative completion signal are unavailable.
You cannot determine recipe correctness, contamination, cause, acceptance, or final
dispensing completion from appearance alone. A pause may be between ingredients.
Do not invent recipe IDs, numerical measurements, diagnoses or actions already taken.
Telemetry, when supplied, is provisional image-based evidence, not approved truth.
Do not command equipment, recommend release, or claim a confirmed defect.
Return only the JSON, without reasoning or markdown.'''


def parse_review(text):
    """Only syntactically valid, correctly abstaining explanations can be displayed."""
    if '</think>' in text:
        text = text.split('</think>', 1)[1].strip()
    stripped = text.strip()
    if stripped.startswith('```'):
        stripped = stripped.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    try:
        value = json.loads(stripped)
    except (ValueError, TypeError):
        return None, 'invalid_json'
    keys = {'visible_summary', 'quality_assessment', 'limitations', 'operator_message'}
    if not isinstance(value, dict) or set(value) != keys:
        return None, 'wrong_schema'
    if value['quality_assessment'] != 'unavailable':
        return None, 'unsupported_quality_verdict'
    if not all(isinstance(value[k], str) and value[k].strip() for k in ('visible_summary', 'operator_message')):
        return None, 'missing_explanation'
    if not isinstance(value['limitations'], list) or not value['limitations'] or not all(isinstance(s, str) and s.strip() for s in value['limitations']):
        return None, 'missing_limitations'
    return value, None


def load_clip(path, count):
    """Decode explicitly with PyAV and retain sampled indices/PTS for the VLM."""
    import av
    import numpy as np
    import hashlib
    with av.open(str(path)) as c:
        stream = c.streams.video[0]
        fps = float(stream.average_rate)
        frames = [(f.to_ndarray(format='rgb24'), f.pts, str(f.time_base)) for f in c.decode(video=0)]
    if not 1 <= count <= len(frames):
        raise ValueError('Invalid frame count')
    indices = np.linspace(0, len(frames)-1, count).astype(int).tolist()
    selected = np.stack([frames[i][0] for i in indices])
    metadata = {'total_num_frames': len(frames), 'fps': fps, 'frames_indices': indices,
                'video_backend': 'pyav', 'width': selected.shape[2], 'height': selected.shape[1]}
    trace = [{'clip_frame': i, 'pts': frames[i][1], 'time_base': frames[i][2],
              'rgb_sha256': hashlib.sha256(frames[i][0].tobytes()).hexdigest()} for i in indices]
    return selected, metadata, trace


def main(a):
    import torch
    import transformers
    if a.out.exists():
        raise ValueError('Use a fresh output directory')
    manifest = read(a.cases / 'manifest.json')
    if manifest['run'] != 4 or manifest['run5_used'] or manifest['quality_truth_available']:
        raise ValueError('Unexpected experimental scope')
    for case in manifest['cases']:
        for name, digest in case['artifacts'].items():
            if sha(a.cases / case['id'] / name) != digest:
                raise ValueError('Event evidence changed')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    a.out.mkdir(parents=True)
    shutil.copy2(__file__, a.out / Path(__file__).name)
    write(a.out / 'plan.json', {'model': MODEL, 'revision': REVISION, 'frames_per_clip': a.frames,
                               'max_new_tokens': a.max_tokens, 'prompt': PROMPT,
                               'manifest_sha256': sha(a.cases/'manifest.json'), 'run5_used': False,
                               'action_coordinator_connected': False, 'hardware_actions': 0})
    transformers.set_seed(19)
    t0 = time.perf_counter()
    processor = transformers.AutoProcessor.from_pretrained(MODEL, revision=REVISION)
    model = transformers.Cosmos3EdgeForConditionalGeneration.from_pretrained(
        MODEL, revision=REVISION, dtype=torch.bfloat16, device_map='cuda:0', attn_implementation='sdpa')
    model.eval()
    load_seconds = time.perf_counter()-t0
    records = []
    for case in manifest['cases'][:a.limit]:
        pixels, metadata, input_trace = load_clip(a.cases/case['video'], a.frames)
        for mode in ('video_only', 'with_telemetry'):
            prompt = PROMPT
            if mode == 'with_telemetry':
                ctx = read(a.cases / case['context'])
                # Compact exact telemetry; do not invent missing process state.
                ctx['sample_count'] = len(ctx['samples'])
                ctx['qa_passing_samples'] = sum(s['sample_qa_passes'] for s in ctx['samples'])
                ctx['image_phases'] = sorted({s['image_phase'] for s in ctx['samples']})
                del ctx['samples']
                prompt += '\nRecorded context:\n'+json.dumps(ctx, sort_keys=True)
            messages = [{'role': 'user', 'content': [
                {'type': 'video', 'video': pixels},
                {'type': 'text', 'text': prompt}]}]
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors='pt',
                processor_kwargs={'do_sample_frames': False, 'video_metadata': [metadata]},
                enable_thinking=False).to(model.device)
            with torch.inference_mode():
                generated = model.generate(**inputs, do_sample=False, max_new_tokens=a.max_tokens)
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-started
            trimmed = generated[:, inputs['input_ids'].shape[1]:]
            text = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
            parsed, error = parse_review(text)
            record = {'case': case['id'], 'mode': mode, 'response': text,
                      'parsed': parsed, 'contract_error': error,
                      'semantic_review_status': 'pending_human_review',
                      'seconds': elapsed, 'input_tokens': inputs['input_ids'].shape[1],
                      'output_tokens': trimmed.shape[1], 'generation_hit_token_limit': trimmed.shape[1] >= a.max_tokens,
                      'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                      'sampled_input_frames': input_trace,
                      'video_sha256': sha(a.cases/case['video']), 'prompt': prompt,
                      'policy_assessment': 'unavailable', 'hardware_command': 'none'}
            records.append(record)
            write(a.out/f'{case["id"]}-{mode}.json', record)
            print(json.dumps({k: record[k] for k in ('case', 'mode', 'seconds', 'contract_error')}), flush=True)
            del inputs, generated, trimmed
    write(a.out/'completed.json', {'model': MODEL, 'revision': REVISION, 'torch': torch.__version__,
        'transformers': transformers.__version__, 'gpu': torch.cuda.get_device_name(0),
        'load_seconds': load_seconds, 'responses': len(records),
        'contract_passes': sum(r['contract_error'] is None for r in records),
        'quality_accuracy': None, 'semantic_accuracy': None, 'hardware_actions': 0,
        'artifacts': {p.name: sha(p) for p in a.out.iterdir() if p.is_file()}})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--frames', type=int, default=8)
    p.add_argument('--max-tokens', type=int, default=512)
    p.add_argument('--limit', type=int, default=6)
    main(p.parse_args())
