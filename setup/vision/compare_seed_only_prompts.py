"""Additional paired arms: point-only SAM or a short causal prompt-box bridge.

Reuses all completed detector/association/box+point results; no new fits or
repeated completed GPU evaluations. This remains an inspected Run 4 pilot.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import av
import cv2

from compare_pool_prompts import run_sam, measure, render, statistics, verify_video
from render_liquid_pipeline import FrameAdapter, pixel_hash
from replay.config import load_config
from replay.geometry import encode_mask, decode_mask
from replay.measurement import Extractor
from liquid_specialists_data import read, write, sha
from probe_prompted_liquid import SAM2_SHA
from pool_prompt_policy import CausalPromptBridge, BRIDGE_POLICY


def main(a):
    if a.out.exists():
        raise ValueError('Use a fresh output directory')
    previous = read(a.previous/'comparison.json'); prior_plan = read(a.previous/'plan.json')
    if (prior_plan['source_run'] != 4 or prior_plan['run5_used'] or
            sha(a.source) != prior_plan['source_sha256'] or sha(a.sam) != SAM2_SHA):
        raise ValueError('Frozen source/model identity differs')
    for name, digest in previous['artifacts'].items():
        if sha(a.previous/name) != digest:
            raise ValueError('Previous artifact changed: '+name)
    repo = Path(__file__).resolve().parents[2]
    for name, digest in prior_plan['qa_code_sha256'].items():
        if sha(repo/'replay'/name) != digest:
            raise ValueError('QA code changed')
    config = load_config(repo/'replay/examples/bootstrap-vision.json')
    if config.to_dict() != prior_plan['qa_config']:
        raise ValueError('QA configuration changed')
    rows = [json.loads(line) for line in (a.previous/'frames.jsonl').read_text().splitlines()]
    if [r['source_frame'] for r in rows] != list(range(0, 1800, 6)):
        raise ValueError('Sample inventory changed')
    a.out.mkdir(parents=True)
    arm = 'seed_only' if a.mode == 'points_only' else 'causal_box'
    plan = dict(schema_version=1, kind='additional_sam_prompt_arm', source_run=4, arm=arm,
        source=str(a.source), source_sha256=sha(a.source), models=prior_plan['models'],
        prior_comparison_sha256=sha(a.previous/'comparison.json'),
        difference=('Identical guided seed/negative points, omit box from SAM input on every eligible frame.'
                    if arm == 'seed_only' else 'Guided boxes/points plus short causal box bridge on area jumps.'),
        bridge_policy=BRIDGE_POLICY if arm == 'causal_box' else None,
        original_detector_predictions_reused=True, all_300_frames_tested=True,
        policy=prior_plan['policy'], qa_config=prior_plan['qa_config'],
        qa_code_sha256=prior_plan['qa_code_sha256'], run5_used=False, future_frames_used=False,
        scripts={name: sha(Path(__file__).with_name(name)) for name in
                 ('compare_seed_only_prompts.py', 'compare_pool_prompts.py', 'pool_prompt_policy.py', 'render_liquid_pipeline.py')},
        limits=prior_plan['limits']+[
            'Additional arms added after inspecting preceding prompt experiments on this same clip.',
            'Causal arm reuses only recent box geometry with fresh current seed/detection; no SAM mask propagation.',
            'This is iterative development, not independent evidence of generalization.'])
    write(a.out/'plan.json', plan)
    for name in plan['scripts']:
        (a.out/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    import torch
    from ultralytics import SAM
    torch.set_num_threads(4)
    sam = SAM(str(a.sam)); adapter = FrameAdapter(); bridge = CausalPromptBridge()
    extractor = Extractor(config, detector=adapter); extractor.geometry.segmenter = adapter
    videos = [a.out/'before-after.mp4', a.out/'early-fill-before-after.mp4']
    writers = [cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*'mp4v'), 5., (1280, 1312)) for p in videos]
    if not all(w.isOpened() for w in writers):
        raise RuntimeError('Video encoder failed')
    totals = Counter(); flags = Counter(); count = 0; tall_frames = []; bridge_modes = Counter()
    try:
        with av.open(str(a.source)) as container, (a.out/'frames.jsonl').open('w', encoding='utf-8') as log:
            for index, frame in enumerate(container.decode(video=0)):
                if index >= 1800:
                    break
                if index % 6:
                    continue
                row = rows[count]; rgb = frame.to_ndarray(format='rgb24')
                if (pixel_hash(rgb) != row['source_rgb_sha256'] or frame.pts != row['pts'] or
                        str(frame.time_base) != row['time_base']):
                    raise ValueError('Current decoded source differs from cached evidence')
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                policy = row['prompt_policy']
                points, labels = policy['points'], policy['point_labels']
                box_prompts = []; bridge_evidence = None
                if arm == 'causal_box':
                    bridge_evidence = bridge.update(policy, index, row['seconds'],
                        row['variants']['original']['measurement']['motion'])
                    box_prompts, points, labels = (bridge_evidence[k] for k in ('boxes', 'points', 'labels'))
                    bridge_modes.update([bridge_evidence['mode']])
                masks, qualities = run_sam(sam, bgr, box_prompts, points, labels, a.device)
                # Predictions are immutable in the original render; the paired
                # comparison retains their source indices in prompt decisions.
                predictions = a.original_predictions[count]['predictions']
                confidence = ([] if policy['selected_prediction_index'] is None else
                              [predictions[policy['selected_prediction_index']]['confidence']])
                measured = measure(extractor, adapter, rgb, predictions, masks, qualities, confidence,
                                   prior_plan['models'], arm)
                variant = dict(boxes=box_prompts, points=points, labels=labels, masks=masks, qualities=qualities,
                               measurement=measured, status=policy['status'] if bridge_evidence is None else bridge_evidence['mode'])
                original = row['variants']['original']
                original = {**original, 'masks': [decode_mask(m, rgb.shape[:2]).astype(bool)
                                                 for m in original['raw_masks']]}
                variants = {'original': original, arm: variant}
                stats = statistics(variant, policy['vial']); totals.update(stats)
                if stats['masks_taller_than_70pct_vial']:
                    tall_frames.append(index)
                flags.update(measured['quality_flags'])
                rendered = render(bgr, variants, index, row['seconds'],
                    arms=('original', arm),
                    titles=('BEFORE / original pipeline', 'AFTER / pool seed + points, no SAM box' if arm == 'seed_only'
                            else 'AFTER / guided pool + causal prompt bridge'))
                writers[0].write(rendered)
                if 10 <= row['seconds'] < 22:
                    writers[1].write(rendered)
                if index in (330, 402, 414, 420, 450, 480, 540, 600):
                    if not cv2.imwrite(str(a.out/f'before-after-frame-{index:04d}.png'), rendered):
                        raise RuntimeError('Poster encoding failed')
                exported = {k: v for k, v in variant.items() if k != 'masks'}
                exported['raw_masks'] = [encode_mask(m) for m in masks]
                log.write(json.dumps(dict(source_frame=index, pts=frame.pts, time_base=str(frame.time_base),
                    seconds=row['seconds'], source_rgb_sha256=row['source_rgb_sha256'],
                    prompt_policy=policy, bridge=bridge_evidence, variant=exported), allow_nan=False)+'\n')
                count += 1
                if count % 50 == 0:
                    print(json.dumps(dict(frames=count, arm=arm, counts=dict(totals))), flush=True)
    finally:
        for writer in writers:
            writer.release()
    if count != 300:
        raise ValueError('Incomplete source inventory')
    full_count, snapshots = verify_video(videos[0], 300, width=1280)
    early_count, _ = verify_video(videos[1], 60, width=1280)
    storyboard = cv2.vconcat([cv2.hconcat(snapshots[n:n+2]) for n in (0, 2, 4)])
    if not cv2.imwrite(str(a.out/'decoded-storyboard.png'), storyboard):
        raise RuntimeError('Storyboard encoding failed')
    report = dict(schema_version=1, plan_sha256=sha(a.out/'plan.json'),
        source_pixels_and_native_pts_verified=count, fully_decoded_frames=dict(full=full_count, early=early_count),
        arms={**previous['arms'], arm: dict(totals)}, additional_arm=arm, tall_mask_frames=tall_frames,
        additional_arm_qa_flags=dict(flags), bridge_modes=dict(bridge_modes), qa_code_and_config_unchanged=True,
        artifacts={p.name: sha(p) for p in a.out.iterdir() if p.is_file()}, limits=plan['limits'])
    write(a.out/'comparison.json', report)
    print(json.dumps(report['arms'], indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'previous', 'original', 'sam', 'out'):
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--mode', choices=('points_only', 'causal_box'), default='points_only')
    args = parser.parse_args()
    manifest = read(args.original/'manifest.json')
    if sha(args.original/'frames.jsonl') != manifest['artifacts']['frames.jsonl']:
        raise ValueError('Original detector prediction cache changed')
    if sha(args.original/'manifest.json') != read(args.previous/'plan.json')['previous_manifest_sha256']:
        raise ValueError('Original prediction ancestry differs')
    args.original_predictions = [json.loads(line) for line in (args.original/'frames.jsonl').read_text().splitlines()]
    main(args)
