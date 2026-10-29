"""Full causal Run 4 pipeline with opt-in TensorRT vial inference.

Both arms execute every model and the same CPU measurement logic. No predictions
or masks are reused to claim a speedup. Deployed replay behavior is untouched.
"""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import shutil
import sys
import time

import av
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from liquid_specialists_data import read, write, sha
from probe_prompted_liquid import BASELINE_SHA, SAM2_SHA
from train_early_liquid import build_resnet
from compare_pool_prompts import run_sam, measure
from pool_prompt_policy import pool_prompts, CausalPromptBridge, POLICY, BRIDGE_POLICY
from render_liquid_pipeline import FrameAdapter, pixel_hash
from stage2_camera_reference import PATCHES, anchor_health, correction, feature
from stage2_guard import GuardedEvidence
from replay.config import load_config
from replay.measurement import Extractor
from replay.detection import FillDetector
from replay.geometry import encode_mask, decode_mask
from replay.observations import build_detected_observations


VIAL_OPTIONS = dict(imgsz=640, rect=False, conf=.35, iou=.45, max_det=300,
    device='cuda:0', quantize=None, augment=False, agnostic_nms=False, verbose=False, save=False)


class Models:
    def __init__(self, baseline, engine, experiment, sam):
        import torch
        from ultralytics import YOLO, SAM
        self.torch = torch
        torch.set_num_threads(4)
        cv2.setNumThreads(4)
        self.vials = {'pytorch': YOLO(str(baseline)), 'tensorrt': YOLO(str(engine), task='detect')}
        self.liquid = build_resnet(experiment)
        state = torch.load(experiment/'models/fasterrcnn-resnet50/best.pt', map_location='cpu', weights_only=True)
        self.liquid.load_state_dict(state['model_state'])
        self.liquid.to('cuda:0').eval()
        self.sam = SAM(str(sam))

    def vial(self, bgr, backend):
        result = self.vials[backend].predict([bgr], **VIAL_OPTIONS)[0]
        return [dict(class_id=0, box=p[:4].tolist(), confidence=float(p[4]))
                for p in result.boxes.data.cpu().numpy() if int(p[5]) == 0]

    def liquid_predictions(self, rgb):
        from torchvision.ops import nms
        torch = self.torch
        with torch.no_grad():
            tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().to('cuda:0')/255
            native = self.liquid([tensor])[0]
        boxes, scores = native['boxes'].cpu(), native['scores'].cpu()
        keep = scores >= .35
        boxes, scores = boxes[keep], scores[keep]
        selected = nms(boxes, scores, .45)[:300]
        return [dict(class_id=1, box=boxes[i].tolist(), confidence=float(scores[i])) for i in selected]


class Pipeline:
    def __init__(self, models, backend, config, pins, calibration):
        self.models, self.backend, self.pins = models, backend, pins
        self.adapter = FrameAdapter()
        self.extractor = Extractor(config, detector=self.adapter)
        self.extractor.geometry.segmenter = self.adapter
        self.bridge = CausalPromptBridge()
        self.fill = FillDetector(config, 30, 4)
        self.guard = GuardedEvidence()
        self.calibration = calibration
        self.features = []

    def tick(self):
        self.models.torch.cuda.synchronize()
        return time.perf_counter()

    def process(self, rgb, index, pts, time_base):
        t0 = self.tick()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        digest = pixel_hash(rgb)
        seconds = float(pts*time_base)
        predictions = self.models.vial(bgr, self.backend)
        t1 = self.tick()
        predictions += self.models.liquid_predictions(rgb)
        t2 = self.tick()
        policy = pool_prompts(rgb, predictions)
        thumb = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (32, 24)).astype(float)[10:23, 11:21]
        previous = self.extractor.previous_thumb
        motion = float(np.abs(thumb-previous).mean()) if previous is not None else None
        guidance = self.bridge.update(policy, index, seconds, motion)
        t3 = self.tick()
        masks, qualities = run_sam(self.models.sam, bgr, guidance['boxes'], guidance['points'], guidance['labels'], 'cuda:0')
        t4 = self.tick()
        confidence = [] if policy['selected_prediction_index'] is None else [predictions[policy['selected_prediction_index']]['confidence']]
        measured = measure(self.extractor, self.adapter, rgb, predictions, masks, qualities, confidence, self.pins, 'causal_box')
        f = dict(measured, frame_index=index, video_s=seconds, source_rgb_sha256=digest)
        state = self.fill.update(f)
        f['fill_detection'] = state
        self.features.append(f)
        # The per-frame QA/color extractor is unchanged. Normalized features are
        # a separate experimental Stage 2 output, never a replacement QA decision.
        camera = dict(status='unavailable', reason='startup_reference_not_yet_available', lab=None)
        if seconds >= 5.:
            anchors = []
            for key in self.calibration['selected']:
                x, y, r, b = PATCHES[key]
                anchors.append(np.median(rgb[y:b, x:r].reshape(-1, 3), axis=0))
            healthy, gain, spread = anchor_health(anchors, self.calibration['reference_rgb'], self.calibration['maximum_disagreement'])
            mask = decode_mask(measured['geometry']['mask'], rgb.shape[:2]).astype(bool)
            lab = None
            if healthy and mask.any():
                corrected, _, _ = correction(rgb[mask], anchors, self.calibration['reference_rgb'], 'srgb_gain')
                normalized, _ = feature(corrected)
                lab = normalized.tolist() if normalized is not None else None
            camera = dict(status='available' if healthy and lab is not None else 'unavailable',
                healthy=healthy, gain=gain.tolist() if gain is not None else None, disagreement=spread, lab=lab)
        decision = self.guard.update(dict(session_id='run4-pipeline-benchmark',
            vial_id=state['detection_id'], recipe_key=None, source_time_s=seconds,
            decision_time_s=seconds, geometry_status='unreviewed_model_proposals',
            vial_identity_unambiguous=False, measurement_valid=measured['measurement_valid']))
        t5 = self.tick()
        row = dict(source_frame=index, pts=pts, time_base=str(time_base), seconds=seconds,
            source_rgb_sha256=digest, predictions=predictions, prompt_policy=policy, bridge=guidance,
            raw_masks=[encode_mask(m) for m in masks], sam_qualities=qualities,
            measurement=measured, fill_detection=state, camera_feature=camera, recipe_decision=decision,
            hardware_command='none')
        # Timed boundary ends after output mask conversion, before disk output.
        t6 = self.tick()
        row['timing_ms'] = dict(vial=(t1-t0)*1000, liquid=(t2-t1)*1000,
            prompting=(t3-t2)*1000, sam=(t4-t3)*1000, measurement_and_state=(t5-t4)*1000,
            result_encoding=(t6-t5)*1000, processing=(t6-t0)*1000,
            sam_prompted=bool(guidance['boxes'] or guidance['points']))
        return row


def main(a):
    import torch
    import tensorrt
    import ultralytics
    if a.out.exists():
        raise ValueError('Use a fresh directory; preserve interrupted evidence')
    done = read(a.evidence/'completed.json')
    prior = read(a.evidence/'plan.json')
    if a.source.name != 'vial_fill_video_4.mp4' or sha(a.source) != prior['source_sha256']:
        raise ValueError('Wrong Run 4 source')
    if sha(a.evidence/'frames.jsonl') != done['artifacts']['frames.jsonl']:
        raise ValueError('Frozen source evidence changed')
    metadata = read(a.experiment/'models/fasterrcnn-resnet50/model.json')
    if metadata['excluded_runs'] != [5] or sha(a.experiment/'models/fasterrcnn-resnet50/best.pt') != prior['frozen_models']['liquid']:
        raise ValueError('Liquid checkpoint differs')
    if sha(a.baseline) != BASELINE_SHA or sha(a.sam) != SAM2_SHA:
        raise ValueError('Original weights differ')
    engine_done = read(a.engine.parent/'completed.json')
    if sha(a.engine) != engine_done['engine_sha256'] or engine_done['input_hw'] != [640, 640] or engine_done['source_model_sha256'] != BASELINE_SHA:
        raise ValueError('Wrong TensorRT engine ancestry or shape')
    repo = Path(__file__).resolve().parents[2]
    config = load_config(repo/'replay/examples/bootstrap-vision.json')
    if config.to_dict() != prior['qa_config']:
        raise ValueError('Measurement QA differs')
    for name, digest in prior['qa_code_sha256'].items():
        if sha(repo/'replay'/name) != digest:
            raise ValueError('Replay measurement implementation changed')
    if sha(a.calibration) != 'f0f3228b0a9a6f883c6142b3c54a54496457188fd2bb53ba805930b738b7d7ed':
        raise ValueError('Camera calibration bytes differ')
    calibration = read(a.calibration)
    if calibration['selected'] != ['x192-y224', 'x112-y224', 'x112-y160'] or calibration['maximum_disagreement'] != .05:
        raise ValueError('Frozen camera profile differs')
    originals = [json.loads(s) for s in (a.evidence/'frames.jsonl').read_text().splitlines()]
    if a.limit_samples:
        originals = originals[:a.limit_samples]
    wanted = {r['source_frame']: r for r in originals}
    a.out.mkdir(parents=True)
    helpers = ['benchmark_liquid_pipeline.py', 'compare_pool_prompts.py', 'pool_prompt_policy.py',
               'render_liquid_pipeline.py', 'stage2_camera_reference.py', 'stage2_guard.py', 'stage2_policy.py',
               'train_early_liquid.py']
    for name in helpers:
        shutil.copy2(Path(__file__).with_name(name), a.out/name)
    plan = dict(schema_version=1, kind='full_pipeline_backend_benchmark', source_run=4,
        source_sha256=prior['source_sha256'], source_evidence_sha256=sha(a.evidence/'frames.jsonl'),
        frozen_models=prior['frozen_models'], engine_sha256=sha(a.engine), engine_input_hw=[640, 640],
        camera_calibration_sha256=sha(a.calibration), vial_options=VIAL_OPTIONS,
        qa_config=config.to_dict(), qa_code_sha256=prior['qa_code_sha256'],
        prompt_policy=POLICY, bridge_policy=BRIDGE_POLICY, samples_per_arm=len(originals), passes=a.passes,
        order=['pytorch', 'tensorrt'] if a.passes == 1 else ['pytorch', 'tensorrt', 'tensorrt', 'pytorch'],
        warmup_frames=[0, 324, 390, 720, 1692, 1800, 2094, 3444, 4512, 5010],
        cached_predictions_or_masks_reused=False, sam_runs_when_prompted=True, run5_used=False, hardware_actions=0,
        quality_accuracy=None, independent_onset_labels_available=False,
        timing_scope='decode/convert + live vial, liquid, SAM, prompts, raw QA, fill state, camera feature and unavailable recipe gate; excludes disk/video rendering, capture/network and final per-recording aggregation',
        helpers={n:sha(a.out/n) for n in helpers})
    write(a.out/'plan.json', plan)
    load_started = time.perf_counter()
    models = Models(a.baseline, a.engine, a.experiment, a.sam)
    load_seconds = time.perf_counter()-load_started
    warm_frames = {}
    with av.open(str(a.source)) as c:
        for index, frame in enumerate(c.decode(video=0)):
            if index in plan['warmup_frames']:
                warm_frames[index] = (frame.to_ndarray(format='rgb24'), frame.pts, frame.time_base)
            if len(warm_frames) == len(plan['warmup_frames']):
                break
    for backend in ('pytorch', 'tensorrt'):
        warm = Pipeline(models, backend, config, prior['frozen_models'], calibration)
        for _ in range(2):
            warm = Pipeline(models, backend, config, prior['frozen_models'], calibration)
            for index, (rgb, pts, tb) in sorted(warm_frames.items()):
                warm.process(rgb, index, pts, tb)
    summary = {}
    for repeat in range(a.passes):
        for backend in (('pytorch', 'tensorrt') if repeat % 2 == 0 else ('tensorrt', 'pytorch')):
            folder = a.out/f'{backend}-pass{repeat}'
            folder.mkdir()
            pipeline = Pipeline(models, backend, config, prior['frozen_models'], calibration)
            counts = Counter()
            started = time.perf_counter()
            with av.open(str(a.source)) as c, (folder/'frames.jsonl').open('w', encoding='utf-8') as log:
                if c.streams.video[0].average_rate != 30:
                    raise ValueError('Unexpected source frame rate')
                iterator = iter(enumerate(c.decode(video=0)))
                while counts['samples'] < len(originals):
                    decode_start = time.perf_counter()
                    while True:
                        index, frame = next(iterator)
                        if index in wanted:
                            break
                    rgb = frame.to_ndarray(format='rgb24')
                    decode_ms = (time.perf_counter()-decode_start)*1000
                    expected = wanted[index]
                    if pixel_hash(rgb) != expected['source_rgb_sha256'] or frame.pts != expected['pts'] or str(frame.time_base) != expected['time_base']:
                        raise ValueError('Source sample pixel identity or PTS differs')
                    row = pipeline.process(rgb, index, frame.pts, frame.time_base)
                    row['timing_ms']['decode_and_rgb'] = decode_ms
                    row['timing_ms']['decode_plus_processing'] = decode_ms+row['timing_ms']['processing']
                    log.write(json.dumps(row, allow_nan=False)+'\n')
                    counts.update(samples=1, qa_pass=int(row['measurement']['measurement_valid']),
                        nonempty_mask=int(row['measurement']['geometry']['mask_pixels'] > 0),
                        prompted=int(row['timing_ms']['sam_prompted']))
                    if counts['samples'] % 300 == 0:
                        print(json.dumps(dict(backend=backend, repeat=repeat, frame=index,
                            samples=counts['samples'], elapsed_s=round(time.perf_counter()-started, 1))), flush=True)
            observations = build_detected_observations(pipeline.features, pipeline.fill.completed, config, 30, 4)
            write(folder/'observations.json', observations)
            write(folder/'events.json', pipeline.fill.events)
            write(folder/'completed.json', dict(counts=dict(counts), completed_fill_candidates=len(observations),
                window_qa_pass=sum(o['measurement_valid'] for o in observations),
                elapsed_including_disk_s=time.perf_counter()-started,
                open_candidate_at_end=copy.deepcopy(pipeline.fill.active),
                artifacts={p.name:sha(p) for p in folder.iterdir() if p.is_file()}))
            summary[folder.name] = sha(folder/'completed.json')
    write(a.out/'completed.json', dict(load_seconds=load_seconds, passes=a.passes,
        samples_per_arm=len(originals), runs=summary, gpu=torch.cuda.get_device_name(0),
        torch=torch.__version__, tensorrt=tensorrt.__version__, ultralytics=ultralytics.__version__,
        helper_sha256=plan['helpers'], plan_sha256=sha(a.out/'plan.json'),
        run5_used=False, hardware_actions=0, promoted=False))
    print(json.dumps({'completed': list(summary), 'samples_per_arm': len(originals)}, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'evidence', 'experiment', 'baseline', 'sam', 'engine', 'calibration', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--passes', type=int, choices=[1, 2], default=2)
    p.add_argument('--limit-samples', type=int)
    main(p.parse_args())
