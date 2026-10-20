"""Paired Run 4 prompt experiment using verified existing detector predictions."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import av
import cv2
import numpy as np

from render_liquid_pipeline import FrameAdapter, pixel_hash, overlay, inset, panel, label
from replay.config import load_config
from replay.geometry import encode_mask, decode_mask
from replay.measurement import Extractor
from liquid_specialists_data import read, write, sha
from probe_prompted_liquid import SAM2_SHA
from pool_prompt_policy import pool_prompts, POLICY


ARMS = ('original', 'association', 'guided')
TITLES = ('A  ORIGINAL / strict association', 'B  ASSOCIATION / tolerant + clipped',
          'C  POOL GUIDANCE / localized box + points')


def run_sam(model, bgr, boxes, points, labels, device):
    if not boxes and not points:
        return [], []
    result = model.predict(bgr, bboxes=boxes or None, points=points or None, labels=labels or None,
                           device=device, conf=0., imgsz=1024, verbose=False, save=False)[0]
    masks = [] if result.masks is None else [m.astype(bool) for m in result.masks.data.cpu().numpy()]
    quality = [] if result.masks is None else result.boxes.conf.cpu().tolist()
    if len(masks) != (len(boxes) if boxes else len(points)):
        raise ValueError('Expected one raw SAM mask per prompt')
    return masks, quality


def measure(extractor, adapter, rgb, predictions, masks, qualities, confidences, models, arm):
    proposals = []
    for mask, quality, confidence in zip(masks, qualities, confidences):
        yy, xx = np.where(mask)
        if not len(xx):
            continue
        proposals.append(dict(class_id=1, box=[int(xx.min()), int(yy.min()), int(xx.max()+1), int(yy.max()+1)],
                              confidence=confidence, candidate_mask=encode_mask(mask), sam_mask_quality=quality))
    adapter.set(rgb, predictions, proposals)
    measured = extractor.measure(rgb)
    measured['geometry'].update(model_sha256=models['liquid'], vial_model_sha256=models['vial'],
        segmentation_model_sha256=models['sam'], liquid_proposal_method='sam2_'+arm,
        mask_probability_threshold=None, mask_logit_threshold=0.)
    return measured


def render(bgr, variants, source_frame, seconds, arms=ARMS, titles=TITLES):
    canvas = np.full((1312, 640*len(arms), 3), (13, 17, 25), np.uint8)
    label(canvas, 'EARLY FILL / SAME SOURCE, DETECTOR AND SAM CHECKPOINTS', (20, 29), size=.86)
    label(canvas, f'Run 4 | native frame {source_frame} | PTS {seconds:.1f}s | unchanged per-frame QA | no future prompts',
          (20, 58), size=.62)
    for col, (arm, title) in enumerate(zip(arms, titles)):
        v = variants[arm]; raw = bgr.copy()
        for mask in v['masks']:
            raw = overlay(raw, mask, (230, 90, 220))
        for box in v['boxes']:
            x, y, r, b = [int(x) for x in box]
            cv2.rectangle(raw, (x, y), (r, b), (70, 230, 100), 1)
        for points, labels in zip(v['points'], v['labels']):
            for (px, py), target in zip(points, labels):
                cv2.circle(raw, (int(px), int(py)), 3, (80, 255, 80) if target else (60, 80, 255), -1)
        m = v['measurement']; pixels = m['geometry']['mask_pixels']
        sampled = overlay(bgr, decode_mask(m['geometry']['mask'], bgr.shape[:2]), (50, 240, 245))
        lines = [f'Boxes {len(v["boxes"])} | point groups {len(v["points"])} | raw masks {len(v["masks"])}',
                 'Magenta = raw SAM; green/red points = positive/negative', v['status']]
        upper = panel(inset(raw), title, lines, (93, 36, 103))
        flags = m['quality_flags']
        reason = ' | '.join(flags) if flags else 'No per-frame flags; full-window QA not evaluated'
        lower = panel(inset(sampled), 'SAMPLE PASSES' if m['measurement_valid'] else 'SAMPLE UNAVAILABLE',
                      [f'Final sample pixels: {pixels} | yellow = actual sampling mask',
                       reason[:104], reason[104:208]],
                      (39, 99, 63) if m['measurement_valid'] else (49, 61, 127))
        canvas[76:664, col*640:(col+1)*640] = upper
        canvas[664:1252, col*640:(col+1)*640] = lower
    label(canvas, 'Guided prompts use existing HSV thresholds; pale/clear liquid remains untested.',
          (20, 1277), size=.56)
    label(canvas, 'Development comparison. No reviewed onset/mask truth, approved labels, full-window QA or deployed changes.',
          (20, 1300), (175, 184, 197), .51)
    return canvas


def statistics(variant, vial):
    masks = variant['masks']; tall = 0
    if vial:
        height = vial[3]-vial[1]
        for mask in masks:
            yy, xx = np.where(mask)
            if len(yy) and yy.max()-yy.min()+1 > .7*height:
                tall += 1
    return dict(prompted=int(bool(variant['boxes'] or variant['points'])), raw_masks=len(masks),
                masks_taller_than_70pct_vial=tall,
                nonempty_sample=int(variant['measurement']['geometry']['mask_pixels'] > 0),
                per_frame_pass=int(variant['measurement']['measurement_valid']))


def verify_video(path, expected, width=1920):
    count = 0; snapshots = []
    with av.open(str(path)) as container:
        if container.streams.video[0].average_rate != 5:
            raise ValueError('Playback rate changed')
        for i, frame in enumerate(container.decode(video=0)):
            if (frame.width, frame.height) != (width, 1312):
                raise ValueError('Output geometry changed')
            if i in (55, 70, 75, 80, 90, 100):
                snapshots.append(cv2.resize(frame.to_ndarray(format='bgr24'), (960, 656)))
            count += 1
    if count != expected:
        raise ValueError('Incomplete decoded video')
    return count, snapshots


def main(a):
    if a.out.exists():
        raise ValueError('Use a fresh output directory')
    old = read(a.previous/'manifest.json')
    if old['source_run'] != 4 or old['run5_used'] or sha(a.source) != old['source_sha256']:
        raise ValueError('Source identity differs')
    if sha(a.sam) != SAM2_SHA or old['models']['sam'] != SAM2_SHA:
        raise ValueError('SAM checkpoint differs')
    for name, digest in old['artifacts'].items():
        if sha(a.previous/name) != digest:
            raise ValueError('Previous artifact changed: '+name)
    repo = Path(__file__).resolve().parents[2]
    for name, digest in old['qa_code_sha256'].items():
        if sha(repo/'replay'/name) != digest:
            raise ValueError('Existing QA code changed')
    config = load_config(repo/'replay/examples/bootstrap-vision.json')
    if config.to_dict() != old['qa_config']:
        raise ValueError('QA configuration differs')
    cached = [json.loads(line) for line in (a.previous/'frames.jsonl').read_text().splitlines()]
    if [r['source_frame'] for r in cached] != list(range(0, 1800, 6)):
        raise ValueError('Cached sample inventory differs')
    a.out.mkdir(parents=True)
    plan = dict(schema_version=1, source=str(a.source), source_sha256=sha(a.source),
        previous_manifest_sha256=sha(a.previous/'manifest.json'), source_run=4, run5_used=False,
        interval='same first 60 seconds, native frames 0:1800:6', arms=list(ARMS), policy=POLICY,
        models=old['models'], detector_inference_reused=True, sam_temporal_propagation=False,
        future_frames_used=False, qa_config=old['qa_config'], qa_code_sha256=old['qa_code_sha256'],
        scripts={name: sha(Path(__file__).with_name(name)) for name in
                 ('compare_pool_prompts.py', 'pool_prompt_policy.py', 'render_liquid_pipeline.py')},
        limits=['Same inspected Run 4 development clip; camera-specific prompt hypotheses, not a blind test.',
                'Raw detector proposals and original masks reused after file/pixel/PTS verification.',
                'Guided arm uses largest basal colored component and smallest associated box containing its deepest point.',
                'Color seed absence is unresolved, not proof of an empty vial; pale/clear liquid not validated.',
                'Prompt search changes only; original raw SAM masks retained and existing QA rules unchanged.',
                'Tall-mask geometry counts are not segmentation errors or approved mask accuracy.',
                'No reviewed physical onset or mask truth, annotation decisions, new fits or detector promotion.'])
    write(a.out/'plan.json', plan)
    for name in plan['scripts']:
        (a.out/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    import torch
    from ultralytics import SAM
    torch.set_num_threads(4)
    sam = SAM(str(a.sam))
    adapters = {name: FrameAdapter() for name in ARMS[1:]}
    extractors = {name: Extractor(config, detector=adapters[name]) for name in ARMS[1:]}
    for name in extractors:
        extractors[name].geometry.segmenter = adapters[name]
    video = a.out/'prompt-comparison.mp4'; early = a.out/'early-fill-comparison.mp4'
    writers = [cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*'mp4v'), 5., (1920, 1312)) for p in (video, early)]
    if not all(w.isOpened() for w in writers):
        raise RuntimeError('Video encoder failed')
    totals = {name: Counter() for name in ARMS}; flags = {name: Counter() for name in ARMS}
    statuses = Counter(); processed = 0; start = time.perf_counter()
    try:
        with av.open(str(a.source)) as container, (a.out/'frames.jsonl').open('w', encoding='utf-8') as log:
            for index, frame in enumerate(container.decode(video=0)):
                if index >= 1800:
                    break
                if index % 6:
                    continue
                original = cached[processed]; rgb = frame.to_ndarray(format='rgb24')
                if (pixel_hash(rgb) != original['source_rgb_sha256'] or frame.pts != original['pts']
                        or str(frame.time_base) != original['time_base']):
                    raise ValueError('Fresh decoded source differs from cached inference')
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                predictions = original['predictions']; policy = pool_prompts(rgb, predictions)
                variants = dict(original=dict(boxes=original['prompt_boxes'], points=[], labels=[],
                    masks=[decode_mask(m, rgb.shape[:2]).astype(bool) for m in original['raw_masks']],
                    qualities=original['sam_mask_quality'], measurement=original['measurement'], status='Archived original pipeline'))
                boxes = dict(association=policy['association_boxes'], guided=policy['guided_boxes'])
                confidences = dict(association=[predictions[d['prediction_index']]['confidence'] for d in policy['association'] if d.get('accepted')],
                                   guided=[] if policy['selected_prediction_index'] is None else
                                   [predictions[policy['selected_prediction_index']]['confidence']])
                for arm in ARMS[1:]:
                    points = policy['points'] if arm == 'guided' else []
                    labels = policy['point_labels'] if arm == 'guided' else []
                    masks, qualities = run_sam(sam, bgr, boxes[arm], points, labels, a.device)
                    measured = measure(extractors[arm], adapters[arm], rgb, predictions, masks, qualities,
                                       confidences[arm], old['models'], arm)
                    variants[arm] = dict(boxes=boxes[arm], points=points, labels=labels, masks=masks,
                                         qualities=qualities, measurement=measured,
                                         status=policy['status'] if arm == 'guided' else 'Expanded association; all accepted prompts retained')
                for arm, variant in variants.items():
                    totals[arm].update(statistics(variant, policy['vial']))
                    flags[arm].update(variant['measurement']['quality_flags'])
                statuses.update([policy['status']])
                output = render(bgr, variants, index, original['seconds'])
                writers[0].write(output)
                if 10 <= original['seconds'] < 22:
                    writers[1].write(output)
                if index in (330, 420, 450, 480, 540, 600):
                    if not cv2.imwrite(str(a.out/f'comparison-frame-{index:04d}.png'), output):
                        raise RuntimeError('Poster encoding failed')
                export = {name: {**{k: v for k, v in variant.items() if k != 'masks'},
                                 'raw_masks': [encode_mask(m) for m in variant['masks']]}
                          for name, variant in variants.items()}
                log.write(json.dumps(dict(source_frame=index, pts=frame.pts, time_base=str(frame.time_base),
                    seconds=original['seconds'], source_rgb_sha256=original['source_rgb_sha256'],
                    prompt_policy=policy, variants=export), allow_nan=False)+'\n')
                processed += 1
                if processed % 50 == 0:
                    print(json.dumps(dict(frames=processed, elapsed_s=round(time.perf_counter()-start, 1))), flush=True)
    finally:
        for writer in writers:
            writer.release()
    if processed != 300:
        raise ValueError('Input sample incomplete')
    decoded, snapshots = verify_video(video, 300)
    early_decoded, _ = verify_video(early, 60)
    sheet = cv2.vconcat([cv2.hconcat(snapshots[n:n+2]) for n in (0, 2, 4)])
    if not cv2.imwrite(str(a.out/'decoded-storyboard.png'), sheet):
        raise RuntimeError('Storyboard encoding failed')
    report = dict(schema_version=1, kind='run4_pool_prompt_comparison', plan_sha256=sha(a.out/'plan.json'),
        source_pixels_and_native_pts_verified=processed, fully_decoded_frames=dict(full=decoded, early=early_decoded),
        arms={name: dict(totals[name]) for name in ARMS}, qa_flags={name: dict(flags[name]) for name in ARMS},
        guidance_status=dict(statuses), qa_code_and_config_unchanged=True, source_labels_used=False,
        artifacts={p.name: sha(p) for p in a.out.iterdir() if p.is_file()}, limits=plan['limits'])
    write(a.out/'comparison.json', report)
    print(json.dumps(report['arms'], indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'previous', 'sam', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    main(parser.parse_args())
