"""Paired development replay: frozen raw Run 4 evidence vs pre-inference correction.

Uses completed raw inference, never labels or synthetic images. QA availability
and paired mask agreement are diagnostics, not detection/mask accuracy.
"""
import argparse
import ast
from collections import Counter
import json
from pathlib import Path
import shutil
import sys

import av
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from live_twin_observer import NativeModels, Pipeline
from liquid_specialists_data import read, write, sha
from render_liquid_pipeline import pixel_hash
from stage2_camera_reference import PATCHES, anchor_health
from replay.config import load_config
from replay.geometry import decode_mask


def preprocess(rgb, seconds, calibration):
    if seconds < 5:
        return None, dict(status='unavailable', reason='startup_reference_not_yet_available')
    anchors = []
    for key in calibration['selected']:
        x, y, r, b = PATCHES[key]
        anchors.append(np.median(rgb[y:b, x:r].reshape(-1, 3), axis=0))
    healthy, gain, spread = anchor_health(anchors, calibration['reference_rgb'], calibration['maximum_disagreement'])
    meta = dict(status='available' if healthy else 'unavailable',
                reason=None if healthy else 'unhealthy_background_reference',
                anchors=np.asarray(anchors).tolist(), gain=gain.tolist() if gain is not None else None,
                disagreement=spread, clipped_output_fraction=None)
    if not healthy:
        return None, meta
    corrected = rgb.astype(float)/gain
    meta['clipped_output_fraction'] = float(((corrected < 0) | (corrected > 255)).any(axis=2).mean())
    return np.clip(np.rint(corrected), 0, 255).astype(np.uint8), meta


def counts(row):
    return dict(qa_pass=int(row['measurement']['measurement_valid']),
                liquid_proposed=int(any(p['class_id'] == 1 for p in row['predictions'])),
                mask_nonempty=int(bool(row['measurement']['geometry']['mask_pixels'])))


def main(a):
    if a.out.exists():
        raise ValueError('Preserve existing evidence; choose a fresh output directory')
    plan = read(a.baseline/'plan.json')
    done = read(a.baseline/'completed.json')
    raw_dir = a.baseline/'pytorch-pass0'
    if sha(raw_dir/'completed.json') != done['runs']['pytorch-pass0']:
        raise ValueError('Raw completion identity changed')
    marker = read(raw_dir/'completed.json')
    if sha(raw_dir/'frames.jsonl') != marker['artifacts']['frames.jsonl']:
        raise ValueError('Raw inference evidence changed')
    if a.source.name != 'vial_fill_video_4.mp4' or sha(a.source) != plan['source_sha256']:
        raise ValueError('Expected exact original Run 4 video')
    if sha(a.calibration) != plan['camera_calibration_sha256']:
        raise ValueError('Frozen camera calibration changed')
    repo = Path(__file__).resolve().parents[2]
    for name, digest in plan['qa_code_sha256'].items():
        if sha(repo/'replay'/name) != digest:
            raise ValueError('QA implementation changed: '+name)
    def runtime_ast(path):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        # The benchmark CLI's descriptive metadata was corrected after its run.
        # Require identical imports, constants and inference class definitions.
        tree.body = [n for n in tree.body if not isinstance(n, ast.If)
                     and not (isinstance(n, ast.FunctionDef) and n.name == 'main')]
        return ast.dump(tree, include_attributes=False)
    for name, digest in plan['helpers'].items():
        if sha(Path(__file__).with_name(name)) != digest:
            if name != 'benchmark_liquid_pipeline.py' or sha(a.baseline/name) != digest or runtime_ast(a.baseline/name) != runtime_ast(Path(__file__).with_name(name)):
                raise ValueError('Frozen pipeline helper changed: '+name)
    config = load_config(repo/'replay/examples/bootstrap-vision.json')
    if config.to_dict() != plan['qa_config']:
        raise ValueError('Frozen QA settings changed')
    tools = Path.home()/'ai-club-tools'
    baseline = tools/'results/crop-clahe-detector-20261019/baseline-best.pt'
    experiment = tools/'results/early-liquid-20261020'
    sam = tools/'results/prompted-liquid-assets-20261020/sam2.1_t.pt'
    paths = dict(vial=baseline, liquid=experiment/'models/fasterrcnn-resnet50/best.pt', sam=sam)
    if {k: sha(v) for k, v in paths.items()} != plan['frozen_models']:
        raise ValueError('Frozen model identity changed')
    rows = [json.loads(s) for s in (raw_dir/'frames.jsonl').read_text().splitlines()]
    raw = {r['source_frame']: r for r in rows}
    if len(raw) != 1581:
        raise ValueError('Expected the completed full Run 4 sampling grid')
    calibration = read(a.calibration)
    a.out.mkdir(parents=True)
    shutil.copy2(__file__, a.out/'executed-helper.py')
    shutil.copy2(a.calibration, a.out/'calibration.json')
    (a.out/'helpers').mkdir()
    helpers = list(plan['helpers'])+['live_twin_observer.py']
    for name in helpers:
        shutil.copy2(Path(__file__).with_name(name), a.out/'helpers'/name)
    write(a.out/'plan.json', dict(kind='real_run4_preinference_camera_correction_development',
        source_sha256=plan['source_sha256'], baseline_frames_sha256=sha(raw_dir/'frames.jsonl'),
        baseline_completion_sha256=sha(a.baseline/'completed.json'), frozen_models=plan['frozen_models'],
        calibration_sha256=sha(a.calibration), helper_sha256=sha(__file__),
        current_helpers_sha256={name: sha(a.out/'helpers'/name) for name in helpers},
        baseline_runtime_ast_verified=True,
        raw_baseline_reused=True, candidate_predictions_or_masks_reused=False,
        sampling='same 1581 native frames, every sixth decoded frame; 5 Hz',
        reference='frozen first-five-second Run 4 background profile, available from 5 s',
        transform='rounded uint8 sRGB channel division before all models and QA',
        unavailable_policy='no candidate inference; reset all candidate temporal state',
        thresholds_changed=False, qa_code_sha256=plan['qa_code_sha256'],
        run5_used=False, training=False, hardware_actions=0,
        accuracy=None, approved_onset_labels=0, approved_masks=0))
    models = NativeModels(baseline, experiment, sam)
    pipeline = Pipeline(models, 'pytorch', config, plan['frozen_models'], calibration)
    totals = Counter(); raw_all = Counter(); raw_paired = Counter(); candidate = Counter()
    disagreements = []; ious = []; native_count = 0
    with av.open(str(a.source)) as c, (a.out/'frames.jsonl').open('w', encoding='utf-8') as log:
        for index, f in enumerate(c.decode(video=0)):
            native_count += 1
            if index not in raw:
                continue
            previous = raw[index]; rgb = f.to_ndarray(format='rgb24')
            digest = pixel_hash(rgb)
            if digest != previous['source_rgb_sha256'] or f.pts != previous['pts'] or str(f.time_base) != previous['time_base']:
                raise ValueError('Source RGB/native clock differs from raw baseline')
            corrected, camera = preprocess(rgb, previous['seconds'], calibration)
            totals['samples'] += 1; raw_all.update(counts(previous))
            output = dict(source_frame=index, pts=f.pts, time_base=str(f.time_base),
                          seconds=previous['seconds'], raw_rgb_sha256=digest,
                          camera_preprocessing=camera, processed_rgb_sha256=None, model_output=None,
                          hardware_action='none')
            if corrected is None:
                totals[camera['reason']] += 1
                pipeline = Pipeline(models, 'pytorch', config, plan['frozen_models'], calibration)
            else:
                row = pipeline.process(corrected, index, f.pts, f.time_base)
                # Pipeline's camera feature is a second correction on processed RGB.
                # Exclude it: this trial compares upstream correction only.
                row.pop('camera_feature')
                row.pop('recipe_decision')
                output.update(processed_rgb_sha256=pixel_hash(corrected), model_output=row)
                totals['paired_available'] += 1; raw_paired.update(counts(previous)); candidate.update(counts(row))
                if counts(previous)['qa_pass'] != counts(row)['qa_pass']:
                    disagreements.append(dict(source_frame=index, seconds=previous['seconds'],
                        raw_qa=bool(counts(previous)['qa_pass']), corrected_qa=bool(counts(row)['qa_pass'])))
                left = decode_mask(previous['measurement']['geometry']['mask']).astype(bool)
                right = decode_mask(row['measurement']['geometry']['mask']).astype(bool)
                union = int((left | right).sum())
                if union:
                    ious.append(float((left & right).sum()/union))
            log.write(json.dumps(output, allow_nan=False)+'\n')
            if totals['samples'] % 100 == 0:
                log.flush(); print(dict(totals), flush=True)
    if native_count != 9482 or totals['samples'] != len(raw):
        raise ValueError('Incomplete source decode')
    summary = dict(native_decoded_frames=native_count, counts=dict(totals),
        raw_all=dict(raw_all), raw_on_camera_eligible_frames=dict(raw_paired),
        corrected_on_camera_eligible_frames=dict(candidate), qa_disagreements=disagreements,
        paired_nonempty_union_mask_iou=dict(samples=len(ious), median=float(np.median(ious)) if ious else None,
            minimum=min(ious) if ious else None),
        accuracy=None, onset_delay=None, mask_accuracy=None,
        limits=['Inspected Run 4 development recording; no injected shifts in this trial.',
                'Model-output agreement and QA availability are not accuracy.',
                'Camera-gate unavailability remains in total denominator; no raw fallback.',
                'Fixed Run 4 background patches are not a portable commissioned camera profile.',
                'No action/recipe policy or reference was commissioned.'])
    write(a.out/'comparison.json', summary)
    if sha(a.source) != plan['source_sha256']:
        raise ValueError('Source changed during evaluation')
    write(a.out/'completed.json', dict(samples=len(raw), hardware_actions=0, run5_used=False,
        artifacts={p.name: sha(p) for p in a.out.iterdir() if p.is_file()}))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'baseline', 'calibration', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    main(p.parse_args())
