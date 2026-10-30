"""Run frozen models on live loopback Isaac RGB, without private simulation truth.

The real evidence gate keeps model geometry unreviewed. A separate provisional
color comparison is diagnostic only and never emits a flush or release command.
"""
import argparse
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path
import socket
import sys
import time

import cv2
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from benchmark_liquid_pipeline import Models, Pipeline, VIAL_OPTIONS
from train_early_liquid import build_resnet
from liquid_specialists_data import read,write,sha
from probe_prompted_liquid import BASELINE_SHA,SAM2_SHA
from replay.config import load_config
from replay.measurement import delta_e
from replay.geometry import decode_mask
from replay.detection import FillDetector
from stage2_guard import GuardedEvidence
from twin_contract import receive,send,validate_frame
from twin_diagnostics import DiagnosticSuite,SETTINGS as DIAGNOSTIC_SETTINGS


class NativeModels(Models):
    def __init__(self,baseline,experiment,sam):
        import torch
        from ultralytics import YOLO,SAM
        self.torch=torch
        torch.set_num_threads(4);cv2.setNumThreads(4)
        self.vials={'pytorch':YOLO(str(baseline))}
        self.liquid=build_resnet(experiment)
        self.liquid.load_state_dict(torch.load(experiment/'models/fasterrcnn-resnet50/best.pt',map_location='cpu',weights_only=True)['model_state'])
        self.liquid.to('cuda:0').eval()
        self.sam=SAM(str(sam))


class ProvisionalComparison:
    """Frozen clean-render model measurements, never geometric ground truth.

    Gaps are unavailable, not interpolated over missing intervals. This diagnostic
    cannot override the operational geometry gate or command an intervention.
    """
    def __init__(self):
        self.reference={};self.count=0;self.warned=False;self.phase=None

    def reset(self):self.count=0;self.warned=False;self.phase=None

    def update(self,header,row,reference):
        m=row['measurement'];key=(header['phase_key'],round(sum(header['metered_ml']),6))
        if self.phase!=header['phase_key']:self.count=0
        self.phase=header['phase_key']
        lab=m.get('lab')
        valid=bool(m['measurement_valid'] and lab is not None)
        if reference:
            if valid:self.reference[key]=list(lab)
            return dict(status='building_reference',delta_e2000=None,warning=False,reference_knots=len(self.reference))
        expected=self.reference.get(key)
        if not valid or expected is None:
            self.count=0
            return dict(status='unavailable',reason='measurement_qa_failed' if not valid else 'missing_eligible_reference',
                        delta_e2000=None,warning=self.warned)
        score=float(delta_e(lab,expected));self.count=self.count+1 if score>5. else 0
        self.warned=self.warned or self.count>=3
        return dict(status='provisional_review' if self.warned else 'within_simulated_reference',
                    delta_e2000=score,consecutive_deviations=self.count,warning=self.warned)


def render(rgb,row,header,diagnostic):
    canvas=np.full((720,1280,3),(20,25,34),np.uint8)
    image=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
    mask=decode_mask(row['measurement']['geometry']['mask'],rgb.shape[:2]).astype(bool)
    image[mask]=(.5*image[mask]+.5*np.array([80,230,240])).astype(np.uint8)
    for p in row['predictions']:
        x,y,r,b=map(int,p['box']);color=(240,170,55) if p['class_id']==0 else (70,230,80)
        cv2.rectangle(image,(x,y),(r,b),color,1)
    canvas[105:585,:640]=image
    canvas[105:585,640:1280]=cv2.resize(image[40:460,240:390],(640,480))
    def line(text,y,color=(230,235,240),size=.65):cv2.putText(canvas,text,(22,y),cv2.FONT_HERSHEY_SIMPLEX,size,color,1,cv2.LINE_AA)
    line('LIVE DIGITAL TWIN / ACTUAL FROZEN VISION PIPELINE',35,size=.85)
    line(f'{header["episode_id"]} | simulation {header["sim_time_s"]:.1f}s | phase {header["phase_key"]} | metered {sum(header["metered_ml"]):.1f} mL',73)
    line(f'Image sample QA: {"passes" if row["measurement"]["measurement_valid"] else "unavailable"} | {row["measurement"]["geometry"]["mask_pixels"]} sampling pixels',620)
    score=diagnostic.get('delta_e2000');value='unavailable' if score is None else f'{score:.2f}'
    line(f'Synthetic-only color diagnostic: {diagnostic["status"]} | dE {value}',651)
    line('Operational recipe gate: unavailable / uncommissioned evidence. No hardware actions.',680,size=.57)
    line('Procedural liquid + hypothetical carryover; not calibrated fluid physics or real defect accuracy.',706,size=.53)
    return canvas


def main(a):
    if a.out.exists():raise ValueError('Use a fresh observer output directory')
    if sha(a.baseline)!=BASELINE_SHA or sha(a.sam)!=SAM2_SHA:raise ValueError('Frozen model differs')
    meta=read(a.experiment/'models/fasterrcnn-resnet50/model.json')
    if sha(a.experiment/'models/fasterrcnn-resnet50/best.pt')!=meta['checkpoint_sha256']:raise ValueError('Liquid checkpoint changed')
    pins=dict(vial=BASELINE_SHA,liquid=meta['checkpoint_sha256'],sam=SAM2_SHA)
    a.out.mkdir(parents=True)
    write(a.out/'models.json',dict(models=pins,backend='pytorch',vial_options=VIAL_OPTIONS,
        geometry_gate='unchanged: unreviewed_model_proposals',private_truth_received=False))
    models=NativeModels(a.baseline,a.experiment,a.sam)
    config=load_config(Path(__file__).resolve().parents[2]/'replay/examples/bootstrap-vision.json')
    calibration=read(a.calibration)
    comparisons=ProvisionalComparison();summaries=[]
    suite=DiagnosticSuite() if a.diagnostic_suite else None
    if suite:write(a.out/'diagnostic-settings.json',DIAGNOSTIC_SETTINGS)
    with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as listener:
        listener.bind(('127.0.0.1',0));listener.listen(1);listener.settimeout(180.)
        write(a.out/'ready.json',dict(host='127.0.0.1',port=listener.getsockname()[1],transport='length_prefixed_RGB_request_ack',protocol_version=1))
        print('OBSERVER READY',listener.getsockname(),flush=True)
        conn,_=listener.accept()
        with conn, (a.out/'frames.jsonl').open('w',encoding='utf-8') as log:
            conn.settimeout(180.)
            previous=None;pipeline=None;reference=False;episode=None;counts=None;writer=None;live_guard=None
            while True:
                header,pixels=receive(conn)
                if header['type']=='start_episode':
                    if pipeline is not None:raise ValueError('Previous episode not ended')
                    episode=header['episode_id'];reference=header['reference']
                    comparisons.reset();previous=None;counts=Counter()
                    if suite:suite.reset()
                    pipeline=Pipeline(models,'pytorch',config,pins,calibration)
                    pipeline.fill=FillDetector(config,30,'synthetic-'+episode)
                    live_guard=GuardedEvidence()
                    writer=cv2.VideoWriter(str(a.out/(episode+'.mp4')),cv2.VideoWriter_fourcc(*'mp4v'),5.,(1280,720))
                    if not writer.isOpened():raise RuntimeError('Video writer failed')
                    send(conn,dict(type='episode_ready',episode_id=episode));continue
                if header['type']=='end_episode':
                    writer.release();writer=None
                    summaries.append(dict(episode_id=episode,reference=reference,counts=dict(counts),
                        provisional_warning=comparisons.warned,reference_knots=len(comparisons.reference),
                        image_events=pipeline.fill.events,image_completed_fills=pipeline.fill.completed))
                    write(a.out/'progress.json',summaries);pipeline=None
                    send(conn,dict(type='episode_ended',episode_id=episode));continue
                if header['type']=='finish':
                    if pipeline is not None:raise ValueError('Cannot finish an open episode')
                    send(conn,dict(type='finished'));break
                if pipeline is None:raise ValueError('Frame outside an episode')
                validate_frame(header,pixels,previous)
                if header['episode_id']!=episode:raise ValueError('Episode source mismatch')
                rgb=np.frombuffer(pixels,dtype=np.uint8).reshape(480,640,3).copy()
                t0=time.perf_counter()
                # Existing pipeline is defined at 30 Hz native indices / 5 Hz samples.
                index=header['frame_index']*6
                row=pipeline.process(rgb,index,header['frame_index'],Fraction(1,5))
                diagnostic=comparisons.update(header,row,reference)
                # Replace the recording benchmark's unknown-context placeholder
                # with this live source's actual declared process metadata. None
                # of the nominal-rendered reference grants commissioning approval.
                key=(header['phase_key'],round(sum(header['metered_ml']),6))
                context=dict(session_id=header['session_id'],vial_id=header['episode_id'],
                    recipe_key=header['recipe_key'],reference_recipe_key=header['recipe_key'],
                    source_time_s=header['sim_time_s'],decision_time_s=header['sim_time_s'],
                    fill_started_s=0.,reference_available_s=-1.,context_available_s=0.,
                    process_state_source_s=header['sim_time_s'],phase_key=header['phase_key'],
                    process_phase_key=header['phase_key'],expected_phase_key=header['phase_key'],
                    reference_id='nominal-rendered-reference' if key in comparisons.reference else None,
                    measurement_profile='synthetic-raw-lab',reference_measurement_profile='synthetic-raw-lab',
                    camera_reference_key='uncommissioned-FillCam',reference_camera_key=None,
                    camera_health_valid=False,vial_identity_unambiguous=False,
                    geometry_status='unreviewed_model_proposals',measurement_valid=row['measurement']['measurement_valid'],
                    observed_lab=row['measurement']['lab'],expected_lab=comparisons.reference.get(key))
                row['recipe_decision']=live_guard.update(context)
                row.update(public_source=header,synthetic_color_diagnostic=diagnostic,
                    declared_recipe_context=context,wall_inference_s=time.perf_counter()-t0,simulated_control_command='none')
                if suite:row['diagnostic_suite']=suite.update(header,row,rgb,reference)
                log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
                writer.write(render(rgb,row,header,diagnostic))
                counts['frames']+=1;counts['sample_qa_pass']+=int(row['measurement']['measurement_valid'])
                counts['sam_prompted']+=int(row['timing_ms']['sam_prompted'])
                counts['vial_predictions']+=sum(p['class_id']==0 for p in row['predictions'])
                counts['liquid_predictions']+=sum(p['class_id']==1 for p in row['predictions'])
                counts['operational_'+row['recipe_decision']['status']]+=1
                send(conn,dict(type='frame_ack',episode_id=episode,frame_index=header['frame_index'],
                    rgb_sha256=header['rgb_sha256'],sample_qa_valid=row['measurement']['measurement_valid'],
                    provisional_warning=diagnostic['warning'],simulated_control_command='none'))
                previous=header
                print('FRAME',episode,header['frame_index'],dict(counts),flush=True)
    write(a.out/'reference.json',dict(status='synthetic_development_only',
        knots=[dict(phase=k[0],metered_total_ml=k[1],lab=v) for k,v in comparisons.reference.items()]))
    write(a.out/'completed.json',dict(episodes=summaries,hardware_actions=0,
        private_truth_received=False,operational_geometry_approved=False,reference_knots=len(comparisons.reference)))
    if suite:write(a.out/'diagnostic-reference.json',suite.export())


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['out','baseline','experiment','sam','calibration']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--diagnostic-suite',action='store_true')
    main(p.parse_args())
