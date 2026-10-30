"""Frozen-pipeline nuisance probes on real Run 4 and cached Isaac RGB.

No fitting, renderer edits, label adjudication, QA changes or deployment.
The domains are unpaired scenes. Metrics describe availability and repeatability,
not segmentation/defect accuracy or a causal estimate of the domain gap.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time

import av
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'setup/vision'))

CONDITIONS = ['raw', 'exposure_085', 'exposure_115', 'green_085', 'blur_sigma12', 'jpeg_q40']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rgb_hash(rgb):
    return hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()


def transform(rgb, condition):
    if condition == 'raw':
        return rgb.copy()
    if condition in ('exposure_085', 'exposure_115'):
        return np.clip(rgb.astype(np.float32)*(.85 if condition.endswith('085') else 1.15), 0, 255).astype(np.uint8)
    if condition == 'green_085':
        return np.clip(rgb.astype(np.float32)*np.array([1., .85, 1.], dtype=np.float32), 0, 255).astype(np.uint8)
    if condition == 'blur_sigma12':
        return cv2.GaussianBlur(rgb, (7, 7), 1.2)
    if condition == 'jpeg_q40':
        ok, encoded = cv2.imencode('.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 40])
        if not ok:
            raise RuntimeError('JPEG encoding failed')
        return cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    raise ValueError(condition)


def select_real(rows):
    """Three fixed sequence windows; selection uses row order, never predictions."""
    n = 37
    if len(rows) < 3*n:
        raise ValueError('Real source too short for predeclared windows')
    groups = []
    for number, fraction in enumerate((.10, .45, .80)):
        start = int((len(rows)-n)*fraction)
        groups.append((f'real-window-{number+1}', rows[start:start+n]))
    selected = [r['source_frame'] for _, group in groups for r in group]
    if len(selected) != len(set(selected)):
        raise ValueError('Real windows overlap')
    return groups


def sources(tools, out):
    real_root = tools/'results/pipeline-acceleration-20261020/full-run4-v1'
    evidence = real_root/'pytorch-pass0/frames.jsonl'
    prior = read(real_root/'plan.json')
    done = read(real_root/'completed.json')
    # The original benchmark's top-level marker pins every per-arm file.
    if 'artifacts' in done and 'pytorch-pass0/frames.jsonl' in done['artifacts']:
        if sha(evidence) != done['artifacts']['pytorch-pass0/frames.jsonl']:
            raise ValueError('Real evidence differs from its completion pin')
    all_rows = [json.loads(line) for line in evidence.read_text(encoding='utf-8').splitlines()]
    groups = select_real(all_rows)
    data = ROOT/'datasets/dataset2/csi-vial-fill-dataset'
    matches = list(data.rglob('vial_fill_video_4.mp4'))
    if len(matches) != 1 or sha(matches[0]) != prior['source_sha256']:
        raise ValueError('Expected the pinned original Run 4 video')
    requested = {r['source_frame']: (group, r) for group, rows in groups for r in rows}
    samples = []
    with av.open(str(matches[0])) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index not in requested:
                continue
            group, old = requested[index]
            rgb = frame.to_ndarray(format='rgb24')
            if rgb_hash(rgb) != old['source_rgb_sha256'] or frame.pts != old['pts'] or Fraction(frame.time_base) != Fraction(old['time_base']):
                raise ValueError('Original real RGB/clock differs at frame '+str(index))
            samples.append(dict(domain='real_run4', group=group, source_frame=index, pts=frame.pts,
                time_base=str(frame.time_base), rgb=rgb, source_rgb_sha256=rgb_hash(rgb)))
    if len(samples) != len(requested):
        raise ValueError('Real source decode incomplete')
    from replay.operator.prerender import PrerenderBundle
    cache = PrerenderBundle(tools/'results/operator-prerender-20261021/cache-v1')
    for old in cache.manifest['samples']:
        rgb = cv2.cvtColor(cv2.imread(str(cache.file(old['png']))), cv2.COLOR_BGR2RGB)
        if rgb_hash(rgb) != old['rgb_sha256']:
            raise ValueError('Cached simulation RGB identity differs')
        samples.append(dict(domain='isaac_cached', group=old['segment_id'], source_frame=old['video_frame'],
            pts=old['pts'], time_base=old['time_base'], rgb=rgb, source_rgb_sha256=rgb_hash(rgb)))
    for s in samples:
        name=f"{s['domain']}-{s['source_frame']:06}.png"
        s['source_png'] = name
        cv2.imwrite(str(out/'source-rgb'/name), cv2.cvtColor(s['rgb'], cv2.COLOR_RGB2BGR))
    identity = dict(real_video=str(matches[0]), real_video_sha256=sha(matches[0]),
        real_evidence_sha256=sha(evidence), real_plan_sha256=sha(real_root/'plan.json'),
        isaac_cache_manifest_sha256=sha(cache.root/'manifest.json'), isaac_video_sha256=sha(cache.video),
        simulated_model_outputs_reused=False, real_model_outputs_reused=False,
        selection='Real: 37 consecutive 5 Hz samples starting at 10%, 45%, 80% of available-window range; Isaac: all 111 cached samples, grouped by source segment.',
        real_samples=sum(s['domain']=='real_run4' for s in samples), isaac_samples=sum(s['domain']=='isaac_cached' for s in samples))
    return samples, identity, prior


def summarize(rows):
    return dict(attempts=len(rows), vial_available=sum(r['prompt_policy']['vial'] is not None for r in rows),
        liquid_proposal_available=sum(any(p['class_id']==1 for p in r['predictions']) for r in rows),
        nonempty_sam_masks=sum(r['raw_sam_union_pixels']>0 for r in rows),
        qa_pass=sum(r['measurement']['measurement_valid'] for r in rows),
        unavailable_flags=dict(Counter(flag for r in rows for flag in r['measurement']['quality_flags'])),
        mean_processing_ms=float(np.mean([r['timing_ms']['processing'] for r in rows])),
        p95_processing_ms=float(np.percentile([r['timing_ms']['processing'] for r in rows],95)))


def analyze(records, out, samples):
    from replay.geometry import decode_mask
    from replay.measurement import delta_e
    groups = defaultdict(list)
    for r in records:
        groups[(r['domain'], r['condition'])].append(r)
    conditions=read(out/'plan.json')['conditions']
    summaries = {}
    for domain in ('real_run4', 'isaac_cached'):
        base = groups[(domain, 'raw')]
        index = {r['source_frame']: r for r in base}
        summaries[domain] = {}
        for condition in conditions:
            rows = groups[(domain, condition)]
            current = summarize(rows)
            pairs = [(index[r['source_frame']], r) for r in rows]
            both_qa = [(a,b) for a,b in pairs if a['measurement']['measurement_valid'] and b['measurement']['measurement_valid']]
            base_qa = sum(a['measurement']['measurement_valid'] for a,b in pairs)
            overlaps = []
            for a,b in pairs:
                if a['raw_sam_union_pixels']:
                    aa=decode_mask(a['raw_sam_union'],(480,640)).astype(bool)
                    bb=decode_mask(b['raw_sam_union'],(480,640)).astype(bool)
                    overlaps.append(float(np.logical_and(aa,bb).sum()/np.logical_or(aa,bb).sum()))
            colors = [float(delta_e(a['measurement']['lab'],b['measurement']['lab'])) for a,b in both_qa]
            current.update(raw_qa_pass=base_qa, retained_raw_qa=len(both_qa),
                lost_raw_qa=base_qa-len(both_qa), gained_qa=sum(not a['measurement']['measurement_valid'] and b['measurement']['measurement_valid'] for a,b in pairs),
                raw_qa_retention=len(both_qa)/base_qa if base_qa else None,
                raw_mask_comparison_count=len(overlaps), mean_raw_mask_self_iou=float(np.mean(overlaps)) if overlaps else None,
                paired_usable_color_count=len(colors), median_paired_color_de2000=float(np.median(colors)) if colors else None,
                p95_paired_color_de2000=float(np.percentile(colors,95)) if colors else None)
            summaries[domain][condition] = current
    active_plan=read(out/'plan.json')
    synthetic_arm=active_plan['arm']!='legacy_real_baseline'
    report = dict(schema_version=1, kind='real_and_simulated_frozen_pipeline_nuisance_probe',arm=active_plan['arm'],
        summary=summaries, attempts=len(records), training=False, run5_used=False, hardware_commands=0,
        mask_accuracy=None, onset_accuracy=None, real_defect_accuracy=None, sim_to_real_gap_closed=False,
        limits=['Domains have different scenes, fills and capture histories; raw between-domain rates are not a controlled accuracy comparison.',
            'Three real windows and three cached simulated segments contain correlated frames; counts are not independent trials.',
            'Self-IoU measures prediction repeatability against raw predictions, not correctness against human masks.',
            'Perturbations operate on encoded RGB, not calibrated radiometry or measured physical camera changes.',
            'Synthetic arm: only CSI synthetic fine-tuning; generic COCO/SAM pretraining and earlier real-informed prompt/camera development are disclosed. Legacy arm: real-trained comparator, not an eligible synthetic-only solution.',
            'The sources are already inspected development evidence. Fresh source-disjoint physical evaluation and reviewed truth remain required.',
            'No physical-vial identity, acceptable recipe references or final-fixture inspection truth is provided by this probe.'])
    write(out/'comparison.json',report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    label_map=dict(zip(CONDITIONS,['Raw','Exposure ×0.85','Exposure ×1.15','Green ×0.85','Blur σ1.2','JPEG q40']))
    labels=[label_map[c] for c in conditions]
    metrics=[('qa_pass','QA passes / 111 attempts',111),('raw_qa_retention','Retention of raw QA passes',1),('mean_raw_mask_self_iou','Mask repeatability vs raw prediction',1)]
    for ax,(key,title,vmax) in zip(axes,metrics):
        grid=np.array([[summaries[d][c][key] if summaries[d][c][key] is not None else np.nan for d in ('real_run4','isaac_cached')] for c in conditions])
        ax.imshow(grid,vmin=0,vmax=vmax,cmap='YlGnBu',aspect='auto')
        ax.set_xticks([0,1],['Real Run 4','Isaac RGB']);ax.set_yticks(range(len(labels)),labels);ax.set_title(title,fontsize=10)
        for (i,j),value in np.ndenumerate(grid):
            ax.text(j,i,'N/A' if np.isnan(value) else f'{value:.0f}' if key=='qa_pass' else f'{value:.2f}',ha='center',va='center',color='white' if value>vmax*.6 else '#142B40')
    title='Synthetic-trained YOLO + Faster R-CNN + SAM2.1' if active_plan['arm']=='synthetic_fasterrcnn' else 'Synthetic-trained joint YOLO + SAM2.1' if synthetic_arm else 'Legacy YOLO + Faster R-CNN + SAM2.1'
    fig.suptitle(title+' | actual inference on every input',fontsize=12)
    fig.text(.5,.025,'Development robustness only. Unpaired scenes; mask self-IoU is not accuracy. No training or QA changes.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,.93));fig.savefig(out/'robustness-scorecard.png',dpi=170);plt.close(fig)
    # Diagnostic examples selected after measurement; not a scored favorable subset.
    from PIL import Image,ImageDraw,ImageFont
    show_conditions=[c for c in ('raw','green_085','jpeg_q40') if c in conditions]
    canvas=Image.new('RGB',(1280,360*len(show_conditions)),'#142B40');draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',20)
    lookup={(s['domain'],s['source_frame']):s for s in samples};selected=[]
    for col,domain in enumerate(('real_run4','isaac_cached')):
        available=[r for r in groups[(domain,'raw')] if r['raw_sam_union_pixels']]
        chosen=max(available,key=lambda r:r['raw_sam_union_pixels']) if available else groups[(domain,'raw')][len(base)//2]
        for row,condition in enumerate(show_conditions):
            result=next(r for r in groups[(domain,condition)] if r['source_frame']==chosen['source_frame'])
            sample=lookup[(domain,chosen['source_frame'])];rgb=transform(sample['rgb'],condition)
            mask=decode_mask(result['raw_sam_union'],(480,640)).astype(bool)
            overlay=rgb.copy();overlay[mask]=(.55*overlay[mask]+.45*np.array([255,180,40])).astype(np.uint8)
            im=Image.fromarray(overlay).resize((640,320));canvas.paste(im,(col*640,row*360+40))
            draw.text((col*640+12,row*360+8),f'{domain} | frame {chosen["source_frame"]} | {condition} | QA {result["measurement"]["measurement_valid"]}',fill='white',font=font)
            selected.append(dict(domain=domain,frame=chosen['source_frame'],condition=condition))
    canvas.save(out/'diagnostic-examples.png');write(out/'diagnostic-example-selection.json',dict(selection='Posthoc largest raw predicted mask per domain, solely for visualization; displayed images resized for layout.',examples=selected))
    return report


def main(a):
    os.environ['YOLO_OFFLINE']='true';os.environ['YOLO_AUTOINSTALL']='false'
    if a.out.exists():
        raise ValueError('Use a fresh experiment output; preserve partial runs')
    a.out.mkdir(parents=True);(a.out/'source-rgb').mkdir();(a.out/'helpers').mkdir()
    helpers=[Path(__file__),ROOT/'setup/vision/benchmark_liquid_pipeline.py',ROOT/'setup/vision/live_twin_observer.py',ROOT/'setup/vision/compare_pool_prompts.py',ROOT/'setup/vision/pool_prompt_policy.py',ROOT/'replay/config.py',ROOT/'replay/measurement.py',ROOT/'replay/geometry.py']
    if a.basal_policy:
        helpers += [ROOT/'setup/vision/vial_basal_policy.py',ROOT/'setup/vision/target_synthetic_transfer.py']
    if a.synthetic_rcnn and read(a.synthetic_rcnn/'frozen-model.json').get('input_view'):
        helpers += [ROOT/'setup/vision/vial_input_view.py',ROOT/'setup/vision/vial_basal_policy.py',ROOT/'setup/vision/target_synthetic_transfer.py']
    helpers=list(dict.fromkeys(helpers))
    helper_pins={str(p.relative_to(ROOT)):sha(p) for p in helpers}
    for p in helpers:
        shutil.copy2(p,a.out/'helpers'/p.name)
    samples,identity,prior=sources(a.tools,a.out)
    experiment=a.tools/'results/early-liquid-20261020'
    baseline=a.tools/'results/crop-clahe-detector-20261019/baseline-best.pt'
    sam=a.tools/'results/prompted-liquid-assets-20261020/sam2.1_t.pt'
    model_paths=dict(vial=baseline,liquid=experiment/'models/fasterrcnn-resnet50/best.pt',sam=sam)
    pins={k:sha(p) for k,p in model_paths.items()}
    if pins!=prior['frozen_models']:
        raise ValueError('Frozen model identities differ')
    calibration_path=a.tools/'results/stage2-refinement-20261020/camera-reference-v2/calibration.json'
    if sha(calibration_path)!=prior['camera_calibration_sha256']:
        raise ValueError('Frozen calibration identity differs')
    from replay.config import load_config
    config=load_config(ROOT/'replay/examples/bootstrap-vision.json')
    if config.to_dict()!=prior['qa_config']:
        raise ValueError('Frozen QA configuration differs')
    for name,expected in prior['qa_code_sha256'].items():
        if sha(ROOT/'replay'/name)!=expected:
            raise ValueError('Frozen QA helper differs: '+name)
    synthetic_metadata=None
    synthetic_checkpoint=None
    if a.arm!='legacy_real_baseline':
        synthetic_checkpoint=a.tools/'results/real-synthetic-detector-20261019/models/sim-only/checkpoint.pt'
        synthetic_metadata=read(synthetic_checkpoint.with_name('model.json'))
        if (sha(synthetic_checkpoint)!=synthetic_metadata['checkpoint_sha256'] or synthetic_metadata['train_runs']
            or synthetic_metadata['development_runs'] or synthetic_metadata['excluded_real_runs']!=[1,2,3,4,5]):
            raise ValueError('Synthetic-only model provenance differs')
        pins=dict(vial=sha(synthetic_checkpoint),liquid=sha(synthetic_checkpoint),sam=pins['sam'])
    rcnn_meta=None
    if a.arm=='synthetic_fasterrcnn':
        if a.synthetic_rcnn is None:
            raise ValueError('Specify the frozen synthetic Faster R-CNN root')
        rcnn_meta=read(a.synthetic_rcnn/'frozen-model.json')
        if rcnn_meta['training_real_runs'] or rcnn_meta['selection_real_runs'] or sha(a.synthetic_rcnn/'final.pt')!=rcnn_meta['checkpoint_sha256']:
            raise ValueError('Synthetic Faster R-CNN boundary differs')
        if not (a.synthetic_rcnn/'completed.json').exists():
            raise ValueError('Synthetic fit/evaluation must finish before pipeline comparison')
        pins['liquid']=rcnn_meta['checkpoint_sha256']
    basal_policy=read(a.basal_policy) if a.basal_policy else None
    if basal_policy and (basal_policy['real_calibration_runs'] or basal_policy['run5_used'] or basal_policy['vial_checkpoint_sha256']!=pins['vial']):
        raise ValueError('Basal association provenance differs')
    plan=dict(schema_version=1,prepared='2026-10-22',kind='predeclared_real_sim_nuisance_probe',arm=a.arm,conditions=a.conditions,
        perturbations='Multiplicative encoded-RGB gains 0.85/1.15; green-only 0.85; Gaussian 7x7 sigma1.2; JPEG quality40. Identity geometry, unchanged labels.',
        source_identity=identity,models=pins,qa_config=config.to_dict(),helper_sha256=helper_pins,
        camera_calibration_sha256=sha(calibration_path),training=False,run5_used=False,synthetic_metadata=synthetic_metadata,synthetic_fasterrcnn=rcnn_meta,
        model_results_reused=False,renderer_changed=False,thresholds_changed=False,
        association_policy_changed=basal_policy is not None,basal_policy=basal_policy,
        basal_policy_sha256=sha(a.basal_policy) if a.basal_policy else None,
        ordered_samples=[{k:v for k,v in s.items() if k!='rgb'} for s in samples],
        scoring='All attempted samples; within-source raw pairing. No human accuracy claim; no independent-trial intervals.')
    write(a.out/'plan.json',plan)
    from live_twin_observer import NativeModels
    from benchmark_liquid_pipeline import Pipeline
    from replay.geometry import decode_mask,encode_mask
    if a.arm=='synthetic_fasterrcnn':
        from train_early_liquid import build_resnet
        import torch
        from ultralytics import YOLO,SAM

        class SyntheticFasterModels(NativeModels):
            def __init__(self):
                self.torch=torch;torch.set_num_threads(4);cv2.setNumThreads(4)
                self.vials={'pytorch':YOLO(str(synthetic_checkpoint))}
                self.liquid=build_resnet(a.synthetic_rcnn)
                self.liquid.load_state_dict(torch.load(a.synthetic_rcnn/'final.pt',map_location='cpu',weights_only=True)['model_state'])
                self.liquid.to('cuda:0').eval();self.sam=SAM(str(sam))
            def vial(self,bgr,backend):
                result=super().vial(bgr,backend)
                self.view_vials=list(result);self.view_hash=rgb_hash(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
                return result
            def liquid_predictions(self,rgb):
                view=rcnn_meta.get('input_view')
                if not view:return super().liquid_predictions(rgb)
                if self.view_hash!=rgb_hash(rgb):raise ValueError('Input-view vial is not from this RGB')
                from vial_input_view import make_view,map_box
                pixels,affine=make_view(rgb,self.view_vials,view)
                result=super().liquid_predictions(pixels)
                return [dict(p,box=box) for p in result if (box:=map_box(p['box'],affine,inverse=True)) is not None]

        models=SyntheticFasterModels()
    elif synthetic_checkpoint is None:
        models=NativeModels(baseline,experiment,sam)
    else:
        import torch
        from ultralytics import YOLO,SAM
        from benchmark_liquid_pipeline import VIAL_OPTIONS

        class SyntheticModels:
            """One synthetic-trained two-class detector, then generic SAM.

            Cached class outputs belong to this exact RGB call, not a previous
            frame or a previous experiment. This is an experimental architecture
            alternative, not a synthetic-trained Faster R-CNN claim.
            """
            def __init__(self):
                self.torch=torch;torch.set_num_threads(4);cv2.setNumThreads(4)
                self.detector=YOLO(str(synthetic_checkpoint));self.sam=SAM(str(sam));self.last=None

            def vial(self,bgr,backend):
                result=self.detector.predict([bgr],**VIAL_OPTIONS)[0]
                proposals=[dict(class_id=int(p[5]),box=p[:4].tolist(),confidence=float(p[4])) for p in result.boxes.data.cpu().numpy()]
                self.last=(rgb_hash(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)),[p for p in proposals if p['class_id']==1])
                return [p for p in proposals if p['class_id']==0]

            def liquid_predictions(self,rgb):
                if self.last is None or self.last[0]!=rgb_hash(rgb):
                    raise ValueError('Joint-detector output is not from the current RGB frame')
                result=self.last[1];self.last=None
                return result

        models=SyntheticModels()
    if basal_policy is not None:
        from vial_basal_policy import filter_predictions

        class BasalModels:
            def __init__(self,native):self.native=native;self.current_vials=None;self.current_hash=None
            def __getattr__(self,key):return getattr(self.native,key)
            def vial(self,bgr,backend):
                result=self.native.vial(bgr,backend)
                self.current_vials=list(result);self.current_hash=rgb_hash(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
                return result
            def liquid_predictions(self,rgb):
                if self.current_hash!=rgb_hash(rgb):raise ValueError('Basal vial association is not from this RGB')
                self.raw_liquid_predictions=self.native.liquid_predictions(rgb)
                return filter_predictions(self.raw_liquid_predictions,self.current_vials,basal_policy)

        models=BasalModels(models)
    write(a.out/'runtime.json',dict(python=sys.version,device=models.torch.cuda.get_device_name(),versions={n:importlib.metadata.version(n) for n in ['torch','torchvision','ultralytics','numpy','av','opencv-python']},warmup_samples=6))
    warm=Pipeline(models,'pytorch',config,pins,read(calibration_path))
    for s in samples[:6]:
        warm.process(s['rgb'],s['source_frame'],s['pts'],Fraction(s['time_base']))
    del warm
    records=[]
    with (a.out/'frames.jsonl').open('x',encoding='utf-8') as log:
        for condition in a.conditions:
            previous=None;pipeline=None;started=time.perf_counter()
            for s in samples:
                key=(s['domain'],s['group'])
                if key!=previous:
                    pipeline=Pipeline(models,'pytorch',config,pins,read(calibration_path));previous=key
                rgb=transform(s['rgb'],condition)
                row=pipeline.process(rgb,s['source_frame'],s['pts'],Fraction(s['time_base']))
                pipeline.features=pipeline.features[-3:]
                mask=np.zeros((480,640),bool)
                for encoded in row['raw_masks']:
                    mask|=decode_mask(encoded,(480,640)).astype(bool)
                row.update(domain=s['domain'],group=s['group'],condition=condition,
                    original_rgb_sha256=s['source_rgb_sha256'],raw_sam_union_pixels=int(mask.sum()),raw_sam_union=encode_mask(mask))
                if basal_policy is not None:row['pre_basal_liquid_predictions']=models.raw_liquid_predictions
                log.write(json.dumps(row,allow_nan=False)+'\n');records.append(row)
            log.flush()
            print(json.dumps(dict(condition=condition,attempts=len(samples),elapsed_s=round(time.perf_counter()-started,2),qa={domain:sum(r['measurement']['measurement_valid'] for r in records if r['condition']==condition and r['domain']==domain) for domain in ('real_run4','isaac_cached')})),flush=True)
    report=analyze(records,a.out,samples)
    for name,expected in helper_pins.items():
        if sha(ROOT/name)!=expected:
            raise ValueError('Source changed during experiment: '+name)
    write(a.out/'completed.json',dict(attempts=len(records),frames_per_domain=identity['real_samples'],conditions=len(a.conditions),training=False,run5_used=False,qa_changed=False,app_defaults_changed=False,hardware_actions=0,plan_sha256=sha(a.out/'plan.json'),artifacts={p.name:sha(p) for p in a.out.iterdir() if p.is_file()}))
    print(json.dumps(report['summary'],indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tools',type=Path,default=Path.home()/'ai-club-tools')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--arm',choices=['legacy_real_baseline','original_sim_only','synthetic_fasterrcnn'],default='original_sim_only')
    p.add_argument('--conditions',nargs='+',choices=CONDITIONS,default=CONDITIONS)
    p.add_argument('--synthetic-rcnn',type=Path)
    p.add_argument('--basal-policy',type=Path,help='Optional frozen synthetic-calibrated association ablation; app defaults and QA stay unchanged')
    main(p.parse_args())
