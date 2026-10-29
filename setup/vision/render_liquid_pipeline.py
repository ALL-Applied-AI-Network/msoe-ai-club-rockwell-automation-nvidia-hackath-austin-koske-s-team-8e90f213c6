"""Offline Run 4 video: detector -> current-frame SAM2 -> existing sampling gates.

This adapter is experimental. It does not alter replay defaults or certify a
measurement window, recipe, fluid mask or first-visible-liquid timestamp.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import av
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from replay.config import load_config
from replay.geometry import encode_mask, decode_mask
from replay.measurement import Extractor
from evaluate_early_liquid import EarlyPair
from liquid_specialists_data import sha, write, read
from probe_prompted_liquid import prompts_from_predictions, SAM2_SHA, BASELINE_SHA


def pixel_hash(rgb):
    return hashlib.sha256(rgb.tobytes()).hexdigest()


class FrameAdapter:
    """Require exact current pixels when reusing this frame's GPU proposals."""
    def set(self, rgb, predictions, masks):
        self.digest = pixel_hash(rgb)
        self.predictions = copy.deepcopy(predictions)
        for p in self.predictions:
            p['box'] = [int(v) for v in p['box']]
        self.masks = copy.deepcopy(masks)

    def check(self, rgb):
        if pixel_hash(rgb) != self.digest:
            raise ValueError('Refusing stale frame proposals')

    def detect(self, rgb):
        self.check(rgb)
        return copy.deepcopy(self.predictions)

    def propose(self, rgb, vial):
        self.check(rgb)
        return copy.deepcopy(self.masks)


def label(canvas, text, xy, color=(230, 235, 242), size=.55):
    cv2.putText(canvas, text, xy, cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)


def overlay(image, mask, color):
    result = image.copy()
    selected = mask.astype(bool)
    result[selected] = (.5*result[selected] + .5*np.array(color)).astype(np.uint8)
    return result


def inset(image):
    # Display-only magnification; inference always receives untouched 640x480.
    result = image.copy()
    crop = cv2.resize(image[230:460, 250:390], (168, 276))
    result[194:470, 462:630] = crop
    cv2.rectangle(result, (461, 193), (631, 471), (220, 220, 220), 1)
    cv2.rectangle(result, (460, 174), (639, 193), (18, 23, 33), -1)
    label(result, 'Station zoom (display)', (465, 188), size=.38)
    return result


def panel(image, title, lines, color):
    canvas = np.full((588, 640, 3), (18, 23, 33), np.uint8)
    cv2.rectangle(canvas, (0, 0), (640, 37), color, -1)
    label(canvas, title, (14, 25), (255, 255, 255), .61)
    canvas[38:518] = image
    for i, line in enumerate(lines[:3]):
        label(canvas, line, (12, 539+i*18), size=.44)
    return canvas


def compose(bgr, predictions, prompts, masks, qualities, measurement, index, seconds):
    detection = bgr.copy()
    for p in predictions:
        x, y, r, b = [int(v) for v in p['box']]
        color = (240, 170, 55) if p['class_id'] == 0 else (90, 230, 90)
        cv2.rectangle(detection, (x, y), (r, b), color, 2)
        name = 'vial' if p['class_id'] == 0 else 'liquid'
        label(detection, f'{name} {p["confidence"]:.2f}', (x, max(16, y-5)), color, .45)
    raw = bgr.copy()
    for i, mask in enumerate(masks):
        raw = overlay(raw, mask, (235, 90+40*(i % 3), 220))
    for box in prompts:
        x, y, r, b = [int(v) for v in box]
        cv2.rectangle(raw, (x, y), (r, b), (235, 200, 250), 1)
    geometry = measurement['geometry']
    final = decode_mask(geometry['mask'], bgr.shape[:2])
    sampled = overlay(bgr, final, (50, 240, 245))
    flags = measurement['quality_flags']
    valid = measurement['measurement_valid']
    state = 'FRAME SAMPLE PASSES' if valid else 'FRAME SAMPLE UNAVAILABLE'
    panes = [
        panel(bgr, '1  SOURCE / Run 4',
              [f'Decoded frame {index}  |  native PTS {seconds:.3f}s',
               'First 60 seconds; every sixth native frame; playback 5 fps',
               'No source labels, recipe IDs or future frames used'], (65, 63, 51)),
        panel(detection, '2  DETECT / fixed vial + Faster R-CNN liquid',
              [f'Liquid proposals: {sum(p["class_id"] == 1 for p in predictions)}  |  confidence >= 0.35',
               f'Station segment: {geometry["track_id"]}  |  age: {geometry["track_age"]} sampled frames',
               'Blue: baseline vial; green: experimental liquid proposal'], (99, 65, 25)),
        panel(inset(raw), '3  REFINE / SAM2.1 raw masks',
              [f'Current-frame liquid-box prompts: {len(prompts)}  |  raw masks: {len(masks)}',
               'Magenta: prompt-conditioned foreground; not fluid confirmation',
               'Independent frames; no temporal propagation or future prompts'], (93, 36, 103)),
        panel(inset(sampled), '4  SAMPLE / existing per-frame QA gates',
              [f'{state}  |  final pixels: {geometry["mask_pixels"]}',
               'Yellow: sampled interior after existing geometry / HSV / clipping',
               'Full-window, timing and recipe/product QA are not run'], (39, 99, 63) if valid else (49, 61, 127))]
    canvas = np.full((1328, 1280, 3), (13, 17, 25), np.uint8)
    label(canvas, 'LIQUID PIPELINE  /  experimental offline application', (22, 29), size=.77)
    label(canvas, 'SOURCE  >  VIAL + LIQUID DETECTION  >  SAM MASK  >  CONSERVATIVE SAMPLING', (22, 56), size=.57)
    canvas[72:660, :640] = panes[0]; canvas[72:660, 640:] = panes[1]
    canvas[660:1248, :640] = panes[2]; canvas[660:1248, 640:] = panes[3]
    reason = ' | '.join(flags) if flags else 'No per-frame flags; measurement window still not evaluated'
    label(canvas, 'QA FLAGS: '+reason[:145], (17, 1271), size=.43)
    if len(reason) > 145:
        label(canvas, reason[145:290], (17, 1289), size=.43)
    label(canvas, 'Detections and masks are proposals. No reviewed onset or mask truth; no model promoted.',
          (17, 1312), (172, 180, 195), .49)
    return canvas


def main(a):
    if a.out.exists():
        raise ValueError('Use a fresh output directory')
    if sha(a.sam) != SAM2_SHA or sha(a.baseline) != BASELINE_SHA:
        raise ValueError('Pinned model identity differs')
    checkpoint = a.experiment/'models/fasterrcnn-resnet50/best.pt'
    metadata = read(checkpoint.parent/'model.json')
    if sha(checkpoint) != metadata['checkpoint_sha256'] or metadata['excluded_runs'] != [5]:
        raise ValueError('Candidate model provenance differs')
    if a.source.name != 'vial_fill_video_4.mp4':
        raise ValueError('This demonstration is restricted to Run 4')
    from ultralytics import SAM
    a.out.mkdir(parents=True)
    config_path = Path(__file__).resolve().parents[2]/'replay/examples/bootstrap-vision.json'
    config = load_config(config_path)
    adapter = FrameAdapter()
    extractor = Extractor(config, detector=adapter)
    extractor.geometry.segmenter = adapter
    pair = EarlyPair(a.baseline, a.experiment, 'fasterrcnn-resnet50', a.device)
    sam = SAM(str(a.sam))
    (a.out/'executed-render.py').write_bytes(Path(__file__).read_bytes())
    writer = cv2.VideoWriter(str(a.out/'liquid-pipeline.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 5., (1280, 1328))
    if not writer.isOpened():
        raise RuntimeError('Video encoder failed')
    counts = dict(frames=0, liquid_detection_frames=0, prompted_frames=0,
                  raw_masks=0, final_mask_frames=0, per_frame_sampling_passes=0)
    flag_counts = {}; posters = []; start = time.perf_counter()
    try:
        with av.open(str(a.source)) as container, (a.out/'frames.jsonl').open('w', encoding='utf-8') as log:
            stream = container.streams.video[0]
            if stream.average_rate != 30:
                raise ValueError('Expected native 30 fps source')
            first_time = None
            for index, frame in enumerate(container.decode(stream)):
                if frame.pts is None:
                    raise ValueError('Missing native PTS')
                seconds = float(frame.pts*frame.time_base)
                if first_time is None:
                    first_time = seconds
                if seconds-first_time >= 60:
                    break
                if index % 6:
                    continue
                rgb = frame.to_ndarray(format='rgb24')
                if rgb.shape != (480, 640, 3):
                    raise ValueError('Expected unscaled 640x480 input')
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                predictions = pair.predict([rgb], .35)[1][0]
                vial, prompt_groups = prompts_from_predictions(predictions)
                prompts = prompt_groups['liquid_box']
                masks = []; qualities = []; proposals = []
                if prompts:
                    result = sam.predict(bgr, bboxes=prompts, device=a.device, conf=0., imgsz=1024,
                                         verbose=False, save=False)[0]
                    if result.masks is not None:
                        masks = [m.astype(bool) for m in result.masks.data.cpu().numpy()]
                        qualities = result.boxes.conf.cpu().tolist()
                    if len(masks) != len(prompts):
                        raise ValueError('Expected one SAM mask per prompt')
                    for box, mask, quality in zip(prompts, masks, qualities):
                        ys, xs = np.where(mask)
                        if not len(xs):
                            continue
                        source = next(p for p in predictions if p['class_id'] == 1 and p['box'] == box)
                        proposals.append(dict(class_id=1, box=[int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)],
                            confidence=source['confidence'], sam_mask_quality=quality,
                            candidate_mask=encode_mask(mask), mask_target_kind='sam2_raw_outline'))
                adapter.set(rgb, predictions, proposals)
                measurement = extractor.measure(rgb)
                # Correct the generic segmenter metadata in this offline adapter;
                # no replay code, saved configuration or reference bank is changed.
                measurement['geometry'].update(model_sha256=metadata['checkpoint_sha256'],
                    vial_model_sha256=BASELINE_SHA, segmentation_model_sha256=SAM2_SHA,
                    liquid_proposal_method='sam2_current_frame_liquid_box',
                    mask_probability_threshold=None, mask_logit_threshold=0.)
                row = dict(source_frame=index, pts=frame.pts, time_base=str(frame.time_base), seconds=seconds,
                    source_rgb_sha256=pixel_hash(rgb), predictions=predictions, prompt_boxes=prompts,
                    sam_mask_quality=qualities, raw_masks=[encode_mask(m) for m in masks], measurement=measurement)
                log.write(json.dumps(row, allow_nan=False)+'\n')
                rendered = compose(bgr, predictions, prompts, masks, qualities, measurement, index, seconds)
                writer.write(rendered)
                if index in (0, 300, 420, 540, 600, 900):
                    name = f'pipeline-frame-{index:04d}.png'
                    if not cv2.imwrite(str(a.out/name), rendered):
                        raise RuntimeError('Poster encoding failed')
                    posters.append(name)
                counts['frames'] += 1
                counts['liquid_detection_frames'] += int(any(p['class_id'] == 1 for p in predictions))
                counts['prompted_frames'] += int(bool(prompts)); counts['raw_masks'] += len(masks)
                counts['final_mask_frames'] += int(measurement['geometry']['mask_pixels'] > 0)
                counts['per_frame_sampling_passes'] += int(measurement['measurement_valid'])
                for flag in measurement['quality_flags']:
                    flag_counts[flag] = flag_counts.get(flag, 0)+1
                if counts['frames'] % 25 == 0:
                    print(json.dumps(dict(**counts, elapsed_s=round(time.perf_counter()-start, 1))), flush=True)
    finally:
        writer.release()
    decoded = 0
    with av.open(str(a.out/'liquid-pipeline.mp4')) as check:
        for frame in check.decode(video=0):
            if (frame.width, frame.height) != (1280, 1328):
                raise ValueError('Output geometry differs')
            decoded += 1
    if decoded != counts['frames'] or decoded != 300:
        raise ValueError('Output frame inventory incomplete')
    write(a.out/'manifest.json', dict(schema_version=1, source=str(a.source), source_sha256=sha(a.source),
        source_run=4, interval='first 60 seconds by native PTS', stride=6, output_fps=5,
        source_labels_used=False, run5_used=False, temporal_sam_propagation=False, future_frames_used=False,
        detector_confidence=.35, detector_nms_iou=.45, sam_image_size=1024, sam_quality_floor=0.,
        models=dict(vial=BASELINE_SHA, liquid=metadata['checkpoint_sha256'], sam=SAM2_SHA),
        script_sha256=sha(Path(__file__)), qa_config=config.to_dict(), qa_config_fingerprint=config.fingerprint,
        qa_code_sha256={name: sha(Path(__file__).resolve().parents[2]/'replay'/name)
                        for name in ('geometry.py', 'measurement.py', 'config.py')},
        counts=counts, flag_counts=flag_counts, fully_decoded_output_frames=decoded, posters=posters,
        artifacts={p.name: sha(p) for p in a.out.iterdir() if p.is_file()},
        limits=['Run 4 development demonstration; no reviewed onset or pixel-mask truth.',
                'SAM masks condition on current liquid detections and cannot recover a missing detector prompt.',
                'Existing per-frame sampling gates run through an offline SAM adapter; no window/recipe/product QA.',
                'No reference bank reused, labels changed, detector promoted or app defaults changed.',
                'Integer box conversion is confined to the existing geometry adapter; raw detector boxes retained.',
                'Station segments are causal geometry IDs, not organizer or physical vial identities.']))
    print(json.dumps(dict(completed=True, **counts, fully_decoded=decoded)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'experiment', 'baseline', 'sam', 'out'):
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--device', default='cuda:0')
    main(parser.parse_args())
