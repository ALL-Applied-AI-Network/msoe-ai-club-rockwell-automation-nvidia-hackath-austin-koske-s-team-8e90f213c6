"""Build a portable, offline visible-pool review packet from frozen model drafts.

This creates evaluation annotations only. Nothing here trains a model or approves
machine suggestions without an explicit human save.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pinned(root, relative, expected):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or sha(path) != expected:
        raise ValueError(f"Source integrity check failed: {relative}")
    return path


def build(probe, context, out):
    probe, context, out = map(Path, (probe, context, out))
    drafts = json.loads((probe / 'review-drafts.json').read_text())
    plan = json.loads((probe / 'plan.json').read_text())
    completed = json.loads((probe / 'completed.json').read_text())
    for name, digest in completed['artifacts'].items():
        pinned(probe, name, digest)
    if plan['source_run'] != 4 or plan['run5_used'] or plan['approved_labels']:
        raise ValueError('Expected unreviewed, evaluation-only Run 4 drafts')
    data = dict(schema='visible-pool-review-packet/v1', source_run=4,
                source_video_sha256=plan['source_video_sha256'], width=640,
                height=480, primary_mode=plan['primary_draft_mode'],
                training_eligible=False, tasks=[], cases=plan['cycles'])
    for task in drafts['tasks']:
        task = dict(task)
        image = pinned(probe, task['image'], task['image_sha256'])
        if task['review_status'] != 'unreviewed' or task['source_video_sha256'] != data['source_video_sha256']:
            raise ValueError('Draft source or review status differs')
        task['image_data'] = 'data:image/png;base64,' + base64.b64encode(image.read_bytes()).decode()
        task['drafts'] = [dict(d) for d in task['drafts']]
        for draft in task['drafts']:
            mask = pinned(probe, draft['mask'], draft['mask_sha256'])
            draft['mask_data'] = 'data:image/png;base64,' + base64.b64encode(mask.read_bytes()).decode()
        data['tasks'].append(task)
    data['packet_id'] = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    out.mkdir(parents=True, exist_ok=False)
    (out / 'clips').mkdir()
    context_hashes = json.loads((context / 'package-hashes.json').read_text())
    # Older packet records hashes under `files`; accept the documented flat form too.
    context_hashes = context_hashes.get('files', context_hashes)
    clips = {}
    for case in data['cases']:
        name = case['clip_relative_path']
        expected = context_hashes[name]
        if isinstance(expected, dict):
            expected = expected['sha256']
        path = pinned(context, name, expected)
        shutil.copy2(path, out / name)
        mapping = context / 'clips' / (case['case_id'] + '-source-frames.json')
        pinned(context, 'clips/' + mapping.name, context_hashes['clips/' + mapping.name])
        shutil.copy2(mapping, out / 'clips' / mapping.name)
        clips[name] = expected
    source = Path(__file__).parent / 'static'
    html = (source / 'mask_review.html').read_text(encoding='utf-8')
    payload = json.dumps(data, separators=(',', ':')).replace('<', '\\u003c')
    html = html.replace('__PACKET__', payload).replace('__SCRIPT__', (source / 'mask_review.js').read_text(encoding='utf-8'))
    (out / 'review.html').write_text(html, encoding='utf-8')
    manifest = dict(packet_id=data['packet_id'], frames=len(data['tasks']),
                    cases=len(data['cases']), source_run=4, training_eligible=False,
                    approved_labels=False, source_video_sha256=data['source_video_sha256'],
                    primary_draft_mode=data['primary_mode'], clips=clips,
                    builder_sha256=sha(__file__))
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (out / 'README.txt').write_text(
        'VISIBLE LIQUID REVIEW — EVALUATION ONLY\n\n'
        'Open review.html in Chrome or Edge. No server, install, or GPU is required.\n'
        'Enter your name. Inspect the original image before showing a machine draft.\n'
        'Empty: no visible pooled liquid in the filling vial. A stream alone is not a pool.\n'
        'Pool: outline only the visible liquid body; exclude glass, holder, label, glare and stream.\n'
        'Uncertain: use when visibility or the liquid boundary cannot be decided; explain why.\n'
        'Use bottom zoom, draw a polygon, or load a draft and paint/erase corrections.\n'
        'Save is an explicit human decision. Model probability is not label accuracy.\n'
        'Review the first five frames together to agree on conventions.\n\n'
        'Onset: review endpoint stills, then save last definitely empty / first definitely pooled.\n'
        'This is an interval, not an exact first-visible frame unless the native frames are adjacent.\n'
        'Clips provide context only. Never label from browser playback seconds or assumed 30 fps.\n'
        'A reference frame is not certified empty simply because a temporal model predicts nothing.\n\n'
        'Download your JSON regularly. Browser storage is only a convenience; do not rely on it.\n'
        'Imports require this packet identity and matching history; divergent histories are rejected.\n'
        'Keep each reviewer export intact. Verification examples are not team-approved labels.\n'
        'These real Run 4 labels are reserved for evaluation, never synthetic-only model training.\n'
        'No Run 5 images are included. No approved labels are supplied with this packet.\n', encoding='utf-8')
    hashes = {str(p.relative_to(out)).replace('\\', '/'): sha(p) for p in out.rglob('*') if p.is_file()}
    (out / 'package-hashes.json').write_text(json.dumps(hashes, indent=2) + '\n')
    archive = out.with_suffix('.zip')
    if archive.exists():
        raise FileExistsError(archive)
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as z:
        for p in out.rglob('*'):
            if p.is_file():
                z.write(p, out.name + '/' + str(p.relative_to(out)))
    print(json.dumps(dict(**manifest, zip=str(archive), zip_sha256=sha(archive))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--context', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    build(args.probe, args.context, args.out)
