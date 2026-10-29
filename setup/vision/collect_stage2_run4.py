"""Extend frozen Stage 1 to full Run 4; measure action-evidence availability.

No recipe/quality labels, fitting, control interface, or changed replay gates.
The first 300 detector/SAM results are verified and reused, never rerun on GPU.
"""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import sys
import time

import av
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from compare_pool_prompts import run_sam, measure
from evaluate_early_liquid import EarlyPair
from liquid_specialists_data import read, write, sha
from pool_prompt_policy import pool_prompts, CausalPromptBridge, POLICY, BRIDGE_POLICY
from probe_prompted_liquid import SAM2_SHA, BASELINE_SHA
from render_liquid_pipeline import FrameAdapter, pixel_hash
from replay.config import load_config
from replay.detection import FillDetector
from replay.geometry import encode_mask, decode_mask
from replay.measurement import Extractor
from replay.observations import build_detected_observations


def main(a):
    if a.out.exists():
        raise ValueError('Use a fresh output directory; preserve partial evidence')
    prior=read(a.previous/'plan.json'); report=read(a.previous/'comparison.json')
    for name in ('frames.jsonl','plan.json'):
        if sha(a.previous/name)!=report['artifacts'][name]:
            raise ValueError('Prior artifact changed: '+name)
    original=read(a.original/'manifest.json')
    if sha(a.original/'frames.jsonl')!=original['artifacts']['frames.jsonl']:
        raise ValueError('Original detector cache changed')
    if a.source.name!='vial_fill_video_4.mp4' or sha(a.source)!=prior['source_sha256']:
        raise ValueError('Only the pinned development recording is allowed')
    if sha(a.sam)!=SAM2_SHA or sha(a.baseline)!=BASELINE_SHA:
        raise ValueError('Pinned weights changed')
    model=read(a.experiment/'models/fasterrcnn-resnet50/model.json')
    if sha(a.experiment/'models/fasterrcnn-resnet50/best.pt')!=prior['models']['liquid'] or model['excluded_runs']!=[5]:
        raise ValueError('Liquid checkpoint ancestry differs')
    repo=Path(__file__).resolve().parents[2]
    for name,digest in prior['qa_code_sha256'].items():
        if sha(repo/'replay'/name)!=digest:raise ValueError('Replay QA changed')
    config=load_config(repo/'replay/examples/bootstrap-vision.json')
    if config.to_dict()!=prior['qa_config']:raise ValueError('QA config changed')
    previous=[json.loads(s) for s in (a.previous/'frames.jsonl').read_text().splitlines()]
    originals=[json.loads(s) for s in (a.original/'frames.jsonl').read_text().splitlines()]
    if len(previous)!=300 or len(originals)!=300:raise ValueError('Prefix inventory differs')
    a.out.mkdir(parents=True)
    plan=dict(schema_version=1,kind='stage2_real_evidence_availability',source_run=4,
        source=str(a.source),source_sha256=prior['source_sha256'],stride=6,
        frozen_models=prior['models'],qa_config=config.to_dict(),qa_code_sha256=prior['qa_code_sha256'],
        prompt_policy=POLICY,bridge_policy=BRIDGE_POLICY,prefix_reused_samples=300,
        prefix_rows_sha256=sha(a.previous/'frames.jsonl'),source_labels_used=False,
        run5_used=False,recipe_reference_available=False,hardware_connected=False,
        limits=['Inspected development recording, no independent test.',
                'Image fill candidates are not reviewed identities/timing or product quality.',
                'Unknown recipe and no approved reference: all real recipe decisions must abstain.',
                'Per-frame sample QA and completed-window QA are distinct.'])
    scripts=('collect_stage2_run4.py','pool_prompt_policy.py','compare_pool_prompts.py','render_liquid_pipeline.py')
    plan['helper_sha256']={n:sha(Path(__file__).with_name(n)) for n in scripts}
    write(a.out/'plan.json',plan)
    for n in scripts:(a.out/n).write_bytes(Path(__file__).with_name(n).read_bytes())
    adapter=FrameAdapter();extractor=Extractor(config,detector=adapter);extractor.geometry.segmenter=adapter
    bridge=CausalPromptBridge();detector=FillDetector(config,30,4)
    pair=sam=None;features=[];timing=[];counts=Counter();flags=Counter();source_count=0
    started=time.perf_counter()
    with av.open(str(a.source)) as container,(a.out/'frames.jsonl').open('w',encoding='utf-8') as log:
        stream=container.streams.video[0]
        if stream.average_rate!=30:raise ValueError('Expected 30 fps')
        for index,frame in enumerate(container.decode(video=0)):
            source_count+=1
            if index%6:continue
            t=time.perf_counter();rgb=frame.to_ndarray(format='rgb24');seconds=float(frame.pts*frame.time_base)
            if frame.pts is None or rgb.shape!=(480,640,3):raise ValueError('Source shape or native PTS missing')
            bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
            reused=index<1800
            if reused:
                saved=previous[index//6]
                if (saved['source_frame']!=index or saved['source_rgb_sha256']!=pixel_hash(rgb)
                    or saved['pts']!=frame.pts or saved['time_base']!=str(frame.time_base)):
                    raise ValueError('Prefix source pixels/PTS differ')
                predictions=originals[index//6]['predictions']
            else:
                if pair is None:
                    from ultralytics import SAM
                    import torch
                    torch.set_num_threads(4)
                    pair=EarlyPair(a.baseline,a.experiment,'fasterrcnn-resnet50',a.device)
                    sam=SAM(str(a.sam))
                predictions=pair.predict([rgb],.35)[1][0]
            detected_at=time.perf_counter()
            policy=pool_prompts(rgb,predictions)
            thumb=cv2.resize(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),(32,24)).astype(float)[10:23,11:21]
            motion=float(np.abs(thumb-extractor.previous_thumb).mean()) if extractor.previous_thumb is not None else None
            guidance=bridge.update(policy,index,seconds,motion)
            if reused:
                if policy!=saved['prompt_policy'] or guidance!=saved['bridge']:raise ValueError('Prefix policy differs')
                masks=[decode_mask(m,rgb.shape[:2]).astype(bool) for m in saved['variant']['raw_masks']]
                qualities=saved['variant']['qualities']
            else:
                masks,qualities=run_sam(sam,bgr,guidance['boxes'],guidance['points'],guidance['labels'],a.device)
            segmented_at=time.perf_counter()
            confidence=[] if policy['selected_prediction_index'] is None else [predictions[policy['selected_prediction_index']]['confidence']]
            measured=measure(extractor,adapter,rgb,predictions,masks,qualities,confidence,prior['models'],'causal_box')
            if reused and measured!=saved['variant']['measurement']:raise ValueError('Prefix measurements differ')
            feature=dict(measured,frame_index=index,video_s=seconds,source_rgb_sha256=pixel_hash(rgb))
            feature['fill_detection']=detector.update(feature)
            features.append(feature)
            row=dict(source_frame=index,pts=frame.pts,time_base=str(frame.time_base),seconds=seconds,
                source_rgb_sha256=feature['source_rgb_sha256'],predictions=predictions,prompt_policy=policy,
                bridge=guidance,raw_masks=[encode_mask(m) for m in masks],measurement=measured,
                fill_detection=feature['fill_detection'],prefix_reused=reused)
            log.write(json.dumps(row,allow_nan=False)+'\n')
            counts['sampled_frames']+=1;counts['prefix_reused']+=int(reused)
            counts['new_gpu_frames']+=int(not reused);counts['sample_qa_pass']+=int(measured['measurement_valid'])
            counts['nonempty_final_mask']+=int(measured['geometry']['mask_pixels']>0)
            flags.update(measured['quality_flags'])
            if not reused:
                timing.append(dict(frame=index,detector_ms=1000*(detected_at-t),
                    sam_ms=1000*(segmented_at-detected_at),total_sample_ms=1000*(time.perf_counter()-t),
                    sam_prompted=bool(guidance['boxes'] or guidance['points'])))
            if index%300==0:print(json.dumps(dict(frame=index,source_s=seconds,counts=dict(counts),elapsed_s=round(time.perf_counter()-started,1))),flush=True)
    observations=build_detected_observations(features,detector.completed,config,30,4)
    write(a.out/'observations.json',observations);write(a.out/'events.json',detector.events)
    write(a.out/'timing.json',timing)
    durations={}
    for key in ('detector_ms','sam_ms','total_sample_ms'):
        v=[r[key] for r in timing[1:]] # exclude one model-loading sample; retained in raw timing
        durations[key]={q:float(np.percentile(v,p)) for q,p in [('p50',50),('p95',95)]}
    output=dict(schema_version=1,source_run=4,source_decoded_frames=source_count,counts=dict(counts),
        expected_sampled_frames=(source_count+5)//6,completed_fill_candidates=len(observations),
        open_fill_candidate_at_eof=copy.deepcopy(detector.active),
        window_qa_pass=sum(o['measurement_valid'] for o in observations),
        sample_qa_flag_counts=dict(flags),timing_excluding_first_new_sample=durations,
        wall_elapsed_s=time.perf_counter()-started,recipe_assessments_available=0,
        validated_defect_labels=0,run5_used=False,hardware_actions=0,
        plan_sha256=sha(a.out/'plan.json'),limits=plan['limits'])
    if counts['prefix_reused']!=300 or counts['sampled_frames']!=output['expected_sampled_frames']:
        raise ValueError('Incomplete sample inventory')
    output['artifacts']={p.name:sha(p) for p in a.out.iterdir() if p.is_file()}
    write(a.out/'completed.json',output)
    print(json.dumps(output,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('source','previous','original','experiment','baseline','sam','out'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--device',default='cuda:0');main(p.parse_args())
