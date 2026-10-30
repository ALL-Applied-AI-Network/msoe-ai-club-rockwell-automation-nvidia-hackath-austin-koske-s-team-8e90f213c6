"""Actual frozen-model replay on camera-corrected saved simulation RGB.

This is an adaptive development follow-up, not a new live-render or held-out test.
Private composition and geometric label masks are not read by this observer.
"""
import argparse
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path
import shutil
import sys
import time

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from live_twin_observer import NativeModels,Pipeline
from replay.config import load_config
from replay.detection import FillDetector
from render_liquid_pipeline import pixel_hash
from liquid_specialists_data import read,write,sha
from twin_diagnostics import DiagnosticSuite
from twin_camera_preprocess import anchors,normalize


def main(a):
    if a.out.exists():raise ValueError('Use a fresh replay output directory')
    done=read(a.source/'completed.json')
    if done['status']!='completed':raise ValueError('Original live run is incomplete')
    original=[json.loads(s) for s in (a.source/'observer/frames.jsonl').read_text().splitlines()]
    episodes=read(a.source/'observer/completed.json')['episodes']
    roles={e['episode_id']:e['reference'] for e in episodes}
    tools=Path.home()/'ai-club-tools'
    model_root=tools/'results/early-liquid-20261020'
    pins=read(a.source/'observer/models.json')['models']
    baseline=tools/'results/crop-clahe-detector-20261019/baseline-best.pt'
    sam=tools/'results/prompted-liquid-assets-20261020/sam2.1_t.pt'
    liquid=model_root/'models/fasterrcnn-resnet50/best.pt'
    for name,path in [('vial',baseline),('liquid',liquid),('sam',sam)]:
        if sha(path)!=pins[name]:raise ValueError('Frozen model changed')
    a.out.mkdir(parents=True);(a.out/'helpers').mkdir()
    here=Path(__file__).resolve().parent;repo=here.parents[1]
    for name in ['replay_twin_preprocessing.py','twin_camera_preprocess.py','twin_diagnostics.py','live_twin_observer.py']:
        shutil.copy2(here/name,a.out/'helpers'/name)
    launch=read(a.source/'launch.json')
    # Inference dependencies must still match the original frozen study.
    for rel,expected in launch['dependency_sha256'].items():
        if sha(repo/rel)!=expected:raise ValueError('Shared inference dependency changed: '+rel)
    write(a.out/'plan.json',dict(kind='adaptive_full_frame_camera_correction_replay',
        source_frames_sha256=sha(a.source/'observer/frames.jsonl'),source_completion_sha256=sha(a.source/'completed.json'),
        model_pins=pins,private_truth_read=False,new_live_render=False,training=False,
        reference='first prior nominal RGB, three frozen background patches',
        guarded=True,raw_and_processed_rgb_identity_recorded_separately=True,
        helpers_sha256={p.name:sha(p) for p in (a.out/'helpers').iterdir()},
        frozen_shared_dependencies=launch['dependency_sha256'],hardware_actions=0,run5_used=False))
    models=NativeModels(baseline,model_root,sam)
    config=load_config(repo/'replay/examples/bootstrap-vision.json')
    calibration=read(tools/'results/stage2-refinement-20261020/camera-reference-v2/calibration.json')
    reference=None;suite=DiagnosticSuite();last=None;counts=Counter();summaries=[];pipeline=None
    with (a.out/'frames.jsonl').open('w') as log:
        for previous in original:
            h=previous['public_source'];eid=h['episode_id'];is_reference=roles[eid]
            if eid!=last:
                if last is not None:summaries.append(dict(episode_id=last,counts=dict(counts),image_events=pipeline.fill.events))
                pipeline=Pipeline(models,'pytorch',config,pins,calibration)
                pipeline.fill=FillDetector(config,30,'synthetic-'+eid)
                suite.reset();counts=Counter();last=eid
            filename=f'{eid}-{h["frame_index"]:05d}.png'
            rgb=np.array(Image.open(a.source/'simulation/rgb'/filename).convert('RGB'))
            if pixel_hash(rgb)!=h['rgb_sha256']:raise ValueError('Saved source RGB differs from live evidence')
            if reference is None:
                if not is_reference or h['frame_index']!=0:raise ValueError('Background reference was not available before test cases')
                reference=anchors(rgb)
            started=time.perf_counter();corrected,camera=normalize(rgb,reference)
            output=dict(public_source=h,raw_source_rgb_sha256=h['rgb_sha256'],camera_preprocessing=camera,
                processed_rgb_sha256=None,model_output=None,diagnostic_suite=None,
                operational_recipe_decision=dict(status='unavailable',reason='uncommissioned_camera_identity_geometry_reference'),
                hardware_action='none')
            counts['frames']+=1
            if corrected is None:
                counts['blocked_camera']+=1
                # Break consecutive evidence; a later frame cannot bridge a blocked interval.
                for state in suite.states.values():state['count']=0
            else:
                row=pipeline.process(corrected,h['frame_index']*6,h['frame_index'],Fraction(1,5))
                row.pop('recipe_decision') # Recording placeholder is not this source's evidence decision.
                diagnostic=suite.update(h,row,corrected,is_reference)
                output.update(processed_rgb_sha256=pixel_hash(corrected),model_output=row,diagnostic_suite=diagnostic)
                counts['processed']+=1;counts['qa_pass']+=int(row['measurement']['measurement_valid'])
                counts['sam_prompted']+=int(row['timing_ms']['sam_prompted'])
            output['replay_wall_s']=time.perf_counter()-started
            log.write(json.dumps(output,allow_nan=False)+'\n');log.flush()
            if counts['frames']%10==0:print(eid,dict(counts),flush=True)
    if last is not None:summaries.append(dict(episode_id=last,counts=dict(counts),image_events=pipeline.fill.events))
    write(a.out/'reference.json',dict(reference_background=reference.tolist(),diagnostics=suite.export()))
    write(a.out/'completed.json',dict(episodes=summaries,frames=len(original),new_live_render=False,
        hardware_actions=0,private_truth_read=False,frames_sha256=sha(a.out/'frames.jsonl')))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
