"""Run 4 SAM2 prompt probe: raw proposals, never labels or approved fluid masks."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from audit_liquid_specialists import development_rows, targets_for
from liquid_specialists_data import read, write, sha


SAM2_SHA = '7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69'
BASELINE_SHA = '10856c18a8a5c485e832adcc6b841b739e726a969dc043fe8db64c49163bb361'
ARMS = ('liquid_box', 'vial_box', 'lower_vial_box')


def prompts_from_predictions(predictions):
    """Use current-frame detector boxes; source labels never become prompts."""
    # Same active station profile as Config.stream_center_roi / VialGeometry.
    vials = [p['box'] for p in predictions if p['class_id'] == 0 and p['confidence'] >= .35
             and p['box'][0] < 306.5 < p['box'][2] and p['box'][3]-p['box'][1] > 180]
    if len(vials) != 1:
        return None, {name: [] for name in ARMS}
    vial = vials[0]; x, y, r, b = vial; w = r-x; h = b-y
    liquid = []
    for p in predictions:
        if p['class_id'] != 1 or p['confidence'] < .35:
            continue
        a = p['box']; area = (a[2]-a[0])*(a[3]-a[1])
        inside = max(0, min(a[2], r)-max(a[0], x))*max(0, min(a[3], b)-max(a[1], y))
        if area > 0 and inside/area >= .9:
            liquid.append(a)
    return vial, dict(liquid_box=liquid, vial_box=[vial],
                     lower_vial_box=[[x+.1*w, y+.55*h, r-.1*w, b]])


def mask_diagnostics(mask, vial):
    pixels = int(mask.sum())
    if vial is None or not pixels:
        return dict(pixels=pixels, fraction_outside_vial=None, fraction_above_lower_search=None)
    yy, xx = np.nonzero(mask)
    x, y, r, b = vial
    inside = (xx >= x) & (xx < r) & (yy >= y) & (yy < b)
    return dict(pixels=pixels, fraction_outside_vial=float((~inside).mean()),
                fraction_above_lower_search=float((yy < y+.55*(b-y)).mean()))


def main(a):
    if a.out.exists():
        raise ValueError('Use a fresh probe directory')
    if sha(a.checkpoint) != SAM2_SHA or sha(a.baseline) != BASELINE_SHA:
        raise ValueError('Pinned checkpoint changed')
    data, sources = development_rows(a.experiment)
    short = [r for r in sources if any(t['box'][3]-t['box'][1] <= 64 for t in targets_for(data, r))]
    negative = sorted([r for r in sources if not targets_for(data, r)], key=lambda r: r['source_id'])[:12]
    selected = sorted(short+negative, key=lambda r: r['source_id'])
    if len(short) != 18 or len(negative) != 12:
        raise ValueError('Frozen diagnostic selection changed')
    a.out.mkdir(parents=True)
    plan = dict(schema_version=1, kind='sam2_run4_prompt_probe', source_run=4,
        source_experiment_plan_sha256=sha(a.experiment/'experiment-plan.json'),
        model='SAM2.1 Hiera Tiny, official Meta checkpoint via Ultralytics image predictor',
        checkpoint_sha256=SAM2_SHA, baseline_checkpoint_sha256=BASELINE_SHA,
        checkpoint_source='https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt',
        script_sha256=sha(Path(__file__)), arms=list(ARMS), sam_imgsz=1024,
        sam_quality_score_floor=0., detector_confidence=.35, detector_nms_iou=.45,
        selection='All 18 Run4 supplied liquid targets <=64px high + first 12 chronological provisional-negative images',
        samples=[{k: r[k] for k in ('source_id', 'image_sha256', 'actual_label_sha256')} for r in selected],
        labels_used_as_prompts=False, run5_used=False, temporal_memory=False,
        labels_changed=False, detector_promoted=False, measurement_qa_changed=False,
        limits=['Raw prompt-conditioned masks, not semantic fluid-presence decisions or approved measurement masks.',
                'A box prompt can produce foreground on glass without visible fluid.',
                'No reviewed mask truth: pixel counts/geometry are diagnostics, not segmentation accuracy.',
                'No reviewed first-visible-liquid timing: no onset/delay measurement.',
                'Label-absence cases are provisional negatives, not confirmed-empty vials.',
                'Still images only; no video propagation, future prompts, training or pseudo-label export.'])
    write(a.out/'probe-plan.json', plan)
    (a.out/'executed-probe.py').write_bytes(Path(__file__).read_bytes())
    from ultralytics import YOLO, SAM
    import torch, ultralytics
    torch.set_num_threads(4)
    detector = YOLO(a.baseline); sam = SAM(str(a.checkpoint))
    reports = []; sheet_rows = []
    (a.out/'masks').mkdir(); (a.out/'overlays').mkdir()
    for source in selected:
        bgr = cv2.imread(str(data/source['image']))
        if bgr is None or bgr.shape != (480, 640, 3):
            raise ValueError('Image geometry changed')
        result = detector.predict(bgr, imgsz=640, rect=False, conf=.35, iou=.45,
                                  device=a.device, verbose=False)[0]
        predictions = [dict(class_id=int(p[5]), box=p[:4].tolist(), confidence=float(p[4]))
                       for p in result.boxes.data.cpu().numpy()]
        vial, prompts = prompts_from_predictions(predictions)
        report = dict(source_id=source['source_id'], image_sha256=source['image_sha256'],
                      provisional_liquid_target_count=len(targets_for(data, source)),
                      vial_box=vial, prompts=prompts, arms={})
        panels = []
        for arm in ['Source RGB', *ARMS]:
            image = bgr.copy()
            label = arm
            if arm in ARMS:
                union = np.zeros((480, 640), np.uint8); qualities = []; files = []
                if prompts[arm]:
                    raw = sam.predict(bgr, bboxes=prompts[arm], device=a.device, conf=0.,
                                      imgsz=1024, verbose=False, save=False)[0]
                    if raw.masks is not None:
                        masks = raw.masks.data.cpu().numpy()
                        if masks.shape[1:] != (480, 640) or not np.isfinite(masks).all():
                            raise ValueError('SAM output geometry/values changed')
                        qualities = raw.boxes.conf.cpu().tolist()
                        for n, mask in enumerate(masks):
                            mask = (mask > 0).astype(np.uint8); union |= mask
                            name = f'masks/{source["source_id"]}-{arm}-{n}.png'
                            if not cv2.imwrite(str(a.out/name), mask*255):
                                raise IOError('Mask export failed')
                            files.append(dict(file=name, sha256=sha(a.out/name), **mask_diagnostics(mask, vial)))
                stats = mask_diagnostics(union, vial)
                report['arms'][arm] = dict(prompt_count=len(prompts[arm]), masks=files,
                    model_mask_quality_scores=qualities, **stats)
                color = np.zeros_like(image); color[:, :, 1] = 230
                image[union.astype(bool)] = (.5*image[union.astype(bool)]+.5*color[union.astype(bool)]).astype(np.uint8)
                for box in prompts[arm]:
                    box = np.rint(box).astype(int)
                    cv2.rectangle(image, tuple(box[:2]), tuple(box[2:]), (255, 170, 70), 1)
                label += f' | {stats["pixels"]}px'
            header = np.zeros((40, 640, 3), np.uint8)
            cv2.putText(header, label, (8, 27), 0, .64, (255, 255, 255), 1)
            panels.append(cv2.resize(np.vstack([header, image]), (320, 260)))
        panel = np.hstack(panels)
        filename = 'overlays/'+source['source_id']+'.png'
        if not cv2.imwrite(str(a.out/filename), panel):
            raise IOError('Overlay export failed')
        report['overlay'] = dict(file=filename, sha256=sha(a.out/filename))
        reports.append(report); sheet_rows.append(panel)
        print(source['source_id'], {name: report['arms'][name]['pixels'] for name in ARMS}, flush=True)
    # Small pages preserve readable source pixels and include every case.
    sheets = []
    for offset in range(0, len(sheet_rows), 6):
        name = f'contact-sheet-{offset//6+1}.png'
        if not cv2.imwrite(str(a.out/name), np.vstack(sheet_rows[offset:offset+6])):
            raise IOError('Contact sheet export failed')
        sheets.append(dict(file=name, sha256=sha(a.out/name)))
    write(a.out/'comparison.json', dict(**plan, runtime=dict(torch=torch.__version__, ultralytics=ultralytics.__version__,
        gpu=torch.cuda.get_device_name(0)), cases=reports, contact_sheets=sheets))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('experiment', 'baseline', 'checkpoint', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    main(p.parse_args())
