"""Continuous inspection station. Sources publish frames; operators review evidence."""
import argparse
from collections import OrderedDict, deque
from copy import deepcopy
from datetime import datetime, timezone
from fractions import Fraction
from http.server import ThreadingHTTPServer
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import threading
import time
from urllib.parse import parse_qs, urlsplit
import uuid

from .frame_input import IncomingFrame
from .server import Handler, STATIC, byte_range
from .store import Conflict, text
from .prerender import PrerenderBundle, PlaybackClock

ROOT=Path(__file__).resolve().parents[2]
SIM_RECIPE='synthetic-10-10-10-order-0-1-2'


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


class InspectionStation:
    def __init__(self,out,source_id='local-fillcam',kind='isaac',record_frames=False):
        self.out=Path(out);self.out.mkdir(parents=True,exist_ok=False)
        for folder in ('evidence','helpers','frames'):(self.out/folder).mkdir()
        self.lock=threading.RLock();self.processing=threading.Lock();self.stop=threading.Event()
        self.processor=None;self.child=None;self.error=None;self.source_error=None
        self.bundle=None;self.playback=None;self.requested_mode='isaac' if kind=='isaac' else 'frames'
        self.active_mode=self.requested_mode;self.source_generation=0;self.switching=False
        self.source_id=source_id;self.kind=kind;self.source_paused=False;self.record_frames=record_frames
        self.session_id='inspection-'+uuid.uuid4().hex[:12]
        self.frame_token=secrets.token_urlsafe(32);write(self.out/'frame-input-token.json',dict(token=self.frame_token))
        self.last=None;self.last_received=None;self.last_source=None;self.active_key=None;self.current_id=None
        self.cases=OrderedDict();self.images=OrderedDict();self.samples=OrderedDict();self.captures={};self.events=[];self.requests={};self.request_results={}
        self.context=dict(recipe_id=SIM_RECIPE if kind=='isaac' else '',batch_id='',
            origin='simulated_job_configuration' if kind=='isaac' else 'unknown')
        self.counts=dict(frames=0,qa_pass=0,sequence_gaps=0,streams=0)
        self.frame_times=deque(maxlen=30);self.reference=None
        self.event('session_started',dict(source_id=source_id,kind=kind,hardware_commands=0))

    def event(self,kind,payload,actor=None,source=None):
        event=dict(seq=len(self.events)+1,kind=kind,payload=payload,actor=actor,
            recorded_at_utc=datetime.now(timezone.utc).isoformat(),source=source or (self.last['source'] if self.last else None))
        self.events.append(event)
        with (self.out/'operator-events.jsonl').open('a',encoding='utf-8') as log:
            log.write(json.dumps(event,allow_nan=False)+'\n');log.flush();os.fsync(log.fileno())
        return event

    def new_case(self,key,source):
        if self.current_id:self.cases[self.current_id]['in_view']=False
        cid='obs-'+uuid.uuid4().hex[:10]
        case=dict(case_id=cid,source_key=key,source_start=source,source_last=source,
            recipe_id=self.context['recipe_id'],batch_id=self.context['batch_id'],context_origin=self.context['origin'],
            physical_vial_id='',identity_confirmed=False,revision=1,in_view=True,
            fill_complete=False,completion_origin=None,endpoint=None,
            frames=0,qa_pass=0,comparisons=0,max_delta_e2000=None,diagnostic_warning=False,
            review='none',review_id=None,review_reason=None,review_cycle=0,actions=[],resolution=None,
            evidence=None,last_measurement=None,last_diagnostic=None,_samples=deque(maxlen=3),_consecutive=0,_phase=None)
        self.cases[cid]=case;self.current_id=cid;self.active_key=key
        self.event('observation_started',dict(case_id=cid,identity='source_window_not_confirmed_physical_vial'),source=source)
        return case

    def freeze(self,case,latest):
        key=latest['image_key'];stem=case['case_id']+'-'+str(case['review_cycle'])
        images=self.images.get(key)
        if images is None:raise Conflict('Evidence expired; wait for a current frame')
        paths={}
        for kind,png in images.items():
            path=Path('evidence')/(stem+'-'+kind+'.png');(self.out/path).write_bytes(png);paths[kind]=path.as_posix()
        case['evidence']=dict(paths=paths,source=deepcopy(latest['source']),measurement=deepcopy(latest['measurement']),
            diagnostic=deepcopy(latest['diagnostic']),image_key=key)
        write(self.out/'evidence'/(stem+'.json'),case['evidence'])

    def request_review(self,case,reason,origin,actor=None,capture=None):
        if case['review'] in ('requested','acknowledged'):raise Conflict('Review is already open')
        case.update(review='requested',review_id=uuid.uuid4().hex,review_reason=reason,
                    review_cycle=case['review_cycle']+1,resolution=None,revision=case['revision']+1)
        if capture:case['evidence']=deepcopy(capture)
        else:self.freeze(case,self.last)
        self.event('review_requested',dict(case_id=case['case_id'],review_id=case['review_id'],origin=origin,reason=reason),actor,source=case['evidence']['source'])

    def diagnostic(self,case,row,source):
        from replay.measurement import delta_e
        measurement=row['measurement'];telemetry=source['telemetry']
        result=dict(status='unavailable',reason='No qualified progress/reference for this source',delta_e2000=None,
                    reference_lab=None,warning=case['diagnostic_warning'],threshold_de2000=5.,consecutive=0)
        if self.kind!='isaac' or not telemetry or not self.reference:
            case['_consecutive']=0;return result
        if case['recipe_id']!=self.reference['recipe_key'] or telemetry['recipe_key']!=case['recipe_id']:
            case['_consecutive']=0;result['reason']='Intended recipe is unknown or has no matching reference';return result
        phase=telemetry['phase_key']
        if case['_phase']!=phase:case['_consecutive']=0
        case['_phase']=phase
        expected=self.reference['knots'].get((phase,round(sum(telemetry['metered_ml']),6)))
        if not measurement['measurement_valid'] or expected is None:
            case['_consecutive']=0;result['reason']='Measurement QA failed' if not measurement['measurement_valid'] else 'No reference at this declared progress';return result
        score=delta_e(measurement['lab'],expected)
        case['_consecutive']=case['_consecutive']+1 if score>5. else 0
        case['diagnostic_warning']=case['diagnostic_warning'] or case['_consecutive']>=3
        return dict(status='appearance_deviation' if case['diagnostic_warning'] else 'outside_development_band_pending' if score>5. else 'within_development_band',
            reason='Uncommissioned rendered reference; recipe acceptance unavailable',delta_e2000=score,
            reference_lab=expected,warning=case['diagnostic_warning'],threshold_de2000=5.,consecutive=case['_consecutive'])

    def accept(self,frame,row,images,generation=None):
        """Only the internal model processor can supply a measurement row."""
        source=deepcopy(frame.metadata)
        if row['source_rgb_sha256']!=source['rgb_sha256']:raise ValueError('Inference/source pixels differ')
        with self.lock:
            if generation is not None and generation!=self.source_generation:raise Conflict('Source changed during processing')
            previous=self.last_source
            if previous and source['stream_id']==previous['stream_id']:
                if source['sequence']<=previous['sequence'] or frame.seconds<=float(previous['pts']*Fraction(previous['time_base'])):
                    raise Conflict('Duplicate, backward or stale source frame')
                gap=source['sequence']!=previous['sequence']+1 or frame.seconds-float(previous['pts']*Fraction(previous['time_base']))>.400001
            else:gap=False;self.counts['streams']+=1
            if self.switching:
                self.switching=False;self.source_error=None;self.event('source_mode_active',dict(mode=self.active_mode,generation=self.source_generation),source=source)
            if gap:self.counts['sequence_gaps']+=1
            segment=source['segment_id'] or row['fill_detection'].get('detection_id') or 'unassigned'
            key=(source['stream_id'],segment)
            case=self.cases.get(self.current_id)
            if not case or key!=self.active_key:case=self.new_case(key,source)
            if gap:case['_consecutive']=0
            diagnostic=self.diagnostic(case,row,source)
            image_key=uuid.uuid4().hex
            self.images[image_key]=images
            while len(self.images)>48:self.images.popitem(last=False)
            measurement=deepcopy(row['measurement'])
            self.last=dict(image_key=image_key,case_id=case['case_id'],source=source,seconds=frame.seconds,
                measurement=measurement,diagnostic=diagnostic,inference_ms=row['inference_ms'],
                predictions=row['predictions'],image_phase=row['fill_detection']['phase'])
            self.samples[image_key]=self.last
            while len(self.samples)>48:self.samples.popitem(last=False)
            case.update(source_last=source,last_measurement=measurement,last_diagnostic=diagnostic)
            case['frames']+=1;case['qa_pass']+=int(measurement['measurement_valid'])
            case['_samples'].append(dict(source=source,measurement=measurement))
            if diagnostic['delta_e2000'] is not None:
                case['comparisons']+=1;case['max_delta_e2000']=max(case['max_delta_e2000'] or 0,diagnostic['delta_e2000'])
            self.last_source=source;self.last_received=time.monotonic();self.frame_times.append(self.last_received)
            self.counts['frames']+=1;self.counts['qa_pass']+=int(measurement['measurement_valid'])
            if diagnostic['warning'] and case['review']=='none':
                self.request_review(case,'Appearance differs from the configured development reference','actual_model_color_diagnostic')
            record=dict(source=source,case_id=case['case_id'],model_output=row,diagnostic=diagnostic)
            with (self.out/'frames.jsonl').open('a',encoding='utf-8') as log:log.write(json.dumps(record,allow_nan=False)+'\n')
            if self.record_frames:
                for kind,png in images.items():(self.out/'frames'/(f'{self.counts["frames"]:08d}-{kind}.png')).write_bytes(png)
            return dict(accepted_sequence=source['sequence'],rgb_sha256=source['rgb_sha256'],case_id=case['case_id'],hardware_commands=0)

    def command(self,body):
        actor=text(body.get('actor'),'Operator name',maximum=80)
        request_id=text(body.get('request_id'),'Request identity',maximum=80)
        command=body.get('command');digest=json.dumps(body,sort_keys=True,allow_nan=False)
        with self.lock:
            if request_id in self.requests:
                if self.requests[request_id]!=digest:raise Conflict('Request identity was reused')
                return {**self.request_results[request_id], 'duplicate':True}
            extra={}
            if command in ('source_pause','source_stop','source_mode'):
                if self.kind!='isaac':raise Conflict('Physical/external feeds have no simulator controls')
                if command=='source_mode':
                    mode=body.get('mode')
                    if mode not in ('isaac','prerender'):raise ValueError('Choose live Isaac or pre-rendered real-time input')
                    if mode=='prerender' and self.bundle is None:raise Conflict('Prepare the video cache before selecting pre-rendered input')
                    self.requested_mode=mode;self.source_generation+=1;self.switching=True;self.playback=None
                    self.source_paused=False;self.source_error=None;self.frame_times.clear()
                    self.event('source_mode_requested',dict(mode=mode,generation=self.source_generation),actor)
                elif command=='source_stop':
                    self.stop.set();self.event('test_source_stop_requested',{},actor)
                else:
                    if type(body.get('paused')) is not bool:raise ValueError('Explicit source pause state required')
                    self.source_paused=body['paused']
                    if self.playback:self.playback.pause(self.source_paused)
                    self.event('test_source_pause_changed',dict(paused=self.source_paused),actor)
            else:
                cid=body.get('case_id');case=self.cases.get(cid)
                if not case:raise Conflict('Select an existing observation')
                if type(body.get('expected_revision')) is not int or body['expected_revision']!=case['revision']:
                    raise Conflict('This observation changed; reopen the action')
                if command=='capture':
                    if case['review'] in ('requested','acknowledged'):raise Conflict('This observation already has an open review')
                    sample=self.samples.get(body.get('image_key'))
                    if not sample or sample['case_id']!=cid:raise Conflict('That source sample expired; capture a current frame')
                    capture_id='capture-'+uuid.uuid4().hex
                    holder=dict(case_id=capture_id,review_cycle=0)
                    self.freeze(holder,sample)
                    self.captures[capture_id]=dict(case_id=cid,evidence=holder['evidence'])
                    extra=dict(capture_id=capture_id,evidence=holder['evidence'])
                    self.event('review_frame_captured',dict(case_id=cid,capture_id=capture_id),actor,source=sample['source'])
                elif command=='context':
                    recipe=text(body.get('recipe_id'),'Intended recipe',required=False,maximum=100)
                    batch=text(body.get('batch_id'),'Batch',required=False,maximum=100)
                    physical=text(body.get('physical_vial_id'),'Physical vial',required=False,maximum=100)
                    confirmed=body.get('identity_confirmed')
                    if type(confirmed) is not bool or (confirmed and not physical):raise ValueError('Confirmed vial association requires its identifier')
                    note=text(body.get('note'),'Context source')
                    case.update(recipe_id=recipe,batch_id=batch,physical_vial_id=physical,identity_confirmed=confirmed,context_origin='manual_operator')
                    case['_consecutive']=0
                    if body.get('apply_next') is True:self.context=dict(recipe_id=recipe,batch_id=batch,origin='manual_operator')
                    self.event('context_recorded',dict(case_id=cid,recipe_id=recipe,batch_id=batch,physical_vial_id=physical,identity_confirmed=confirmed,note=note,applies_to_next=body.get('apply_next') is True),actor)
                elif command=='new_vial':
                    if cid!=self.current_id or not self.fresh():raise Conflict('New vial boundary requires the current fresh observation')
                    case=self.new_case(self.active_key,self.last['source'])
                    self.event('manual_vial_boundary',dict(case_id=case['case_id'],note=text(body.get('note'),'Boundary note')),actor)
                elif command=='request_review':
                    capture=self.captures.get(body.get('capture_id'))
                    if capture:
                        if capture['case_id']!=cid:raise Conflict('Captured evidence belongs to another observation')
                    elif cid!=self.current_id or not self.fresh():raise Conflict('Manual review requires captured evidence or a fresh current frame')
                    self.request_review(case,text(body.get('note'),'Review reason'),'manual_operator',actor,capture=capture['evidence'] if capture else None)
                elif command=='complete':
                    if cid!=self.current_id or not self.fresh():raise Conflict('Completion capture needs a fresh current frame')
                    if case['fill_complete']:raise Conflict('Completion already recorded')
                    if body.get('confirmed') is not True:raise ValueError('Confirm actual observed completion')
                    samples=list(case['_samples']);valid=len(samples)==3 and all(s['measurement']['measurement_valid'] for s in samples)
                    tracks={s['measurement']['geometry']['track_id'] for s in samples}
                    times=[float(s['source']['pts']*Fraction(s['source']['time_base'])) for s in samples]
                    valid=valid and len(tracks)==1 and None not in tracks and all(0<b-a<=.400001 for a,b in zip(times,times[1:]))
                    from replay.measurement import delta_e
                    import numpy as np
                    lab=np.median([s['measurement']['lab'] for s in samples],axis=0).tolist() if valid else None
                    valid=valid and max(delta_e(s['measurement']['lab'],lab) for s in samples)<=5.
                    case.update(fill_complete=True,completion_origin='manual_operator',endpoint=dict(measurement_valid=bool(valid),lab=lab if valid else None,
                        source=self.last['source'],policy='operator_last_three_samples/v1',recipe_acceptance='unavailable'))
                    self.event('fill_completion_recorded',dict(case_id=cid,endpoint=case['endpoint']),actor)
                elif command=='acknowledge':
                    if case['review']!='requested':raise Conflict('Review is no longer awaiting acknowledgement')
                    case['review']='acknowledged';self.event('review_acknowledged',dict(case_id=cid,review_id=case['review_id']),actor)
                elif command=='action':
                    if case['review']!='acknowledged':raise Conflict('Acknowledge this review first')
                    action=body.get('action')
                    if action not in ('reviewed_evidence','set_aside_for_review','notified_supervisor','approved_procedure','other'):raise ValueError('Choose a supported human action')
                    if body.get('confirmed') is not True:raise ValueError('Record only an action actually performed')
                    procedure=text(body.get('procedure_id'),'Procedure/receipt',required=action=='approved_procedure',maximum=150)
                    item=dict(action=action,note=text(body.get('note'),'Action details'),actor=actor,cycle=case['review_cycle'],procedure_id=procedure,hardware_command_sent=False)
                    case['actions'].append(item);self.event('operator_action_recorded',dict(case_id=cid,**item),actor)
                elif command=='close':
                    if case['review']!='acknowledged' or not any(a['cycle']==case['review_cycle'] for a in case['actions']):raise Conflict('Acknowledge and record a response before closing follow-up')
                    case.update(review='closed',resolution=dict(note=text(body.get('note'),'Follow-up summary'),actor=actor,product_release_authorized=False))
                    self.event('follow_up_closed',dict(case_id=cid,**case['resolution']),actor)
                else:raise ValueError('Unknown operator command')
                if command!='capture':case['revision']+=1
            self.requests[request_id]=digest
            result=dict(duplicate=False,case_id=self.current_id if command in ('source_pause','source_stop','source_mode','new_vial') else cid,**extra)
            self.request_results[request_id]=result
            return result

    def fresh(self):
        same_loop=not self.playback or (self.last_source and self.last_source['stream_id']==self.playback.stream_id)
        timely=not self.playback or (self.last and self.playback.seconds()-self.last['seconds']<=3)
        return bool(self.last_received is not None and time.monotonic()-self.last_received<=3 and not self.error and not self.source_error and not self.source_paused and not self.stop.is_set() and not self.switching and same_loop and timely)

    @staticmethod
    def public_case(case):return {k:deepcopy(v) for k,v in case.items() if not k.startswith('_')}

    @staticmethod
    def dashboard_measurement(measurement):
        # Masks are fetched as exact PNGs; polling must not resend their RLE for
        # every historical case. Full model rows remain in logs and exports.
        result={k:deepcopy(v) for k,v in measurement.items() if k!='geometry'}
        result['geometry']={k:deepcopy(v) for k,v in measurement['geometry'].items()
            if k in ('mask_pixels','verification_status','quality_flags')}
        return result

    @classmethod
    def dashboard_case(cls,case):
        result={k:deepcopy(v) for k,v in case.items() if not k.startswith('_') and
            k not in ('source_start','source_last','last_measurement','last_diagnostic','endpoint','evidence')}
        result['evidence']=None if not case['evidence'] else dict(source=deepcopy(case['evidence']['source']),
            measurement=cls.dashboard_measurement(case['evidence']['measurement']),diagnostic=deepcopy(case['evidence']['diagnostic']))
        return result

    def snapshot(self,compact=False):
        with self.lock:
            age=None if self.last_received is None else time.monotonic()-self.last_received
            status='loading' if self.processor is None else 'waiting' if age is None else 'live' if self.fresh() else 'stale'
            if self.source_paused:status='paused'
            if self.switching:status='switching'
            if self.stop.is_set():status='stopped'
            if self.error or self.source_error:status='error'
            recent_ids=set(list(self.cases)[-100:])
            visible_cases=[c for c in self.cases.values() if c['case_id'] in recent_ids or c['review'] in ('requested','acknowledged')]
            latest=deepcopy(self.last) if not compact or not self.last else {
                **{k:deepcopy(v) for k,v in self.last.items() if k!='measurement'},
                'measurement':self.dashboard_measurement(self.last['measurement'])}
            return dict(session_id=self.session_id,version=self.counts['frames']+len(self.events),mode='continuous_inspection',source=dict(source_id=self.source_id,kind=self.kind,status=status,
                age_s=age,paused=self.source_paused,error=self.error or self.source_error,ingest_fps=(len(self.frame_times)-1)/(self.frame_times[-1]-self.frame_times[0]) if len(self.frame_times)>1 else 0,
                delivery_mode=self.active_mode,requested_mode=self.requested_mode,generation=self.source_generation,
                available_modes=['isaac']+(['prerender'] if self.bundle else []) if self.kind=='isaac' else ['frames']),
                playback=None if not self.playback else dict(self.playback.snapshot(),video_url='/api/video.mp4',fps=30,
                    analysis_lag_s=max(0.,self.playback.seconds()-self.last['seconds']) if self.last and self.last['source']['stream_id']==self.playback.stream_id else None),
                latest=latest,current_id=self.current_id,context=deepcopy(self.context),counts=dict(self.counts),
                cases=[self.dashboard_case(c) if compact else self.public_case(c) for c in visible_cases],events=deepcopy(self.events[-25:]),
                open_reviews=sum(c['review'] in ('requested','acknowledged') for c in self.cases.values()),
                reference=None if not self.reference else {k:v for k,v in self.reference.items() if k!='knots'},
                operational_recipe_assessment='unavailable',hardware_commands_sent=0,geometry_status='unreviewed_model_proposals')


class ModelProcessor:
    def __init__(self,station):
        sys.path.insert(0,str(ROOT/'setup/vision'))
        from live_twin_observer import NativeModels
        from probe_prompted_liquid import BASELINE_SHA,SAM2_SHA
        from replay.config import load_config
        tools=Path(os.environ.get('AI_CLUB_HOME',str(Path.home()/'ai-club-tools')))
        baseline=tools/'results/crop-clahe-detector-20261019/baseline-best.pt'
        experiment=tools/'results/early-liquid-20261020';sam=tools/'results/prompted-liquid-assets-20261020/sam2.1_t.pt'
        liquid=experiment/'models/fasterrcnn-resnet50/best.pt'
        pins=dict(vial=BASELINE_SHA,liquid=json.loads(liquid.with_name('model.json').read_text())['checkpoint_sha256'],sam=SAM2_SHA)
        if any(sha(path)!=pins[key] for path,key in [(baseline,'vial'),(liquid,'liquid'),(sam,'sam')]):raise ValueError('Frozen model hash differs')
        write(station.out/'models.json',pins)
        self.models=NativeModels(baseline,experiment,sam);self.pins=pins
        self.config=load_config(ROOT/'replay/examples/bootstrap-vision.json')
        self.calibration=json.loads((tools/'results/stage2-refinement-20261020/camera-reference-v2/calibration.json').read_text())
        self.pipeline=None;self.key=None;self.previous_index=None

    def process(self,station,frame,generation=None):
        import cv2
        import numpy as np
        from benchmark_liquid_pipeline import Pipeline
        from replay.geometry import decode_mask
        key=(frame.metadata['stream_id'],frame.metadata['segment_id'])
        index=round(frame.seconds*30)
        if key!=self.key or self.pipeline is None:
            self.pipeline=Pipeline(self.models,'pytorch',self.config,self.pins,self.calibration);self.key=key;self.previous_index=None
        if self.previous_index is not None and index<=self.previous_index:raise ValueError('Frames are faster than the current 30 Hz detector clock; sample the input')
        start=time.perf_counter()
        row=self.pipeline.process(frame.rgb,index,frame.metadata['pts'],Fraction(frame.metadata['time_base']))
        self.pipeline.features=self.pipeline.features[-3:]
        row['inference_ms']=(time.perf_counter()-start)*1000;self.previous_index=index
        bgr=cv2.cvtColor(frame.rgb,cv2.COLOR_RGB2BGR);overlay=bgr.copy()
        mask=decode_mask(row['measurement']['geometry']['mask'],(480,640)).astype(bool)
        overlay[mask]=(.55*overlay[mask]+.45*np.array([65,225,220])).astype(np.uint8)
        box=row['measurement']['geometry'].get('vial_box')
        if box:cv2.rectangle(overlay,tuple(map(int,box[:2])),tuple(map(int,box[2:])),(235,160,55),1)
        images={}
        for kind,data in [('source',bgr),('overlay',overlay),('mask',mask.astype(np.uint8)*255)]:
            ok,encoded=cv2.imencode('.png',data)
            if not ok:raise RuntimeError('Could not encode source-aligned evidence')
            images[kind]=encoded.tobytes()
        return station.accept(frame,row,images,generation)


class StationServer(ThreadingHTTPServer):
    daemon_threads=True
    def __init__(self,address,station):
        self.station=station;self.csrf=secrets.token_urlsafe(32)
        super().__init__(address,StationHandler)


class StationHandler(Handler):
    def producer(self):return self.valid_host() and secrets.compare_digest(self.headers.get('X-Frame-Token',''),self.server.station.frame_token)

    def generation(self):return int(self.headers.get('X-Source-Generation','0'))

    def video(self):
        bundle=self.server.station.bundle
        if not bundle:return self.respond(dict(error='Pre-rendered input is unavailable'),404)
        size=bundle.video.stat().st_size
        try:start,end,partial=byte_range(self.headers.get('Range'),size)
        except ValueError:
            self.send_response(416);self.send_header('Content-Range',f'bytes */{size}');self.send_header('Content-Length','0');self.end_headers();return
        self.send_response(206 if partial else 200);self.send_header('Content-Type','video/mp4')
        self.send_header('Content-Length',str(end-start+1));self.send_header('Accept-Ranges','bytes')
        self.send_header('Cache-Control','no-store')
        if partial:self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
        self.end_headers()
        if self.command=='HEAD':return
        with bundle.video.open('rb') as source:
            source.seek(start);remaining=end-start+1
            while remaining:
                block=source.read(min(262144,remaining))
                if not block:break
                self.wfile.write(block);remaining-=len(block)

    def do_GET(self):
        try:
            if not self.valid_host():return self.respond(dict(error='Use the local station URL'),403)
            parsed=urlsplit(self.path);station=self.server.station
            assets={'/':('station.html','text/html; charset=utf-8'),'/station.js':('station.js','text/javascript; charset=utf-8'),
                '/station.css':('station.css','text/css; charset=utf-8'),'/operator.css':('operator.css','text/css; charset=utf-8')}
            if parsed.path in assets:
                name,mime=assets[parsed.path];return self.respond((STATIC/name).read_bytes(),mime=mime)
            if parsed.path=='/favicon.ico':return self.respond(b'',204,'image/x-icon')
            if parsed.path=='/api/session':return self.respond(dict(csrf_token=self.server.csrf,state=station.snapshot(compact=True)))
            if parsed.path=='/api/state':return self.respond(station.snapshot(compact=True))
            if parsed.path=='/api/video.mp4':return self.video()
            if parsed.path=='/api/source':
                if not self.producer():return self.respond(dict(error='Frame producer token required'),403)
                with station.lock:control=dict(paused=station.source_paused,stop=station.stop.is_set() or self.generation()!=station.source_generation,
                    ready=station.processor is not None,playback=None if not station.playback else station.playback.snapshot())
                return self.respond(control)
            if parsed.path=='/api/export.json':
                with station.lock:data=dict(state=station.snapshot(),cases=[station.public_case(c) for c in station.cases.values()],events=list(station.events),product_release_authorized=False)
                return self.respond(data,filename='inspection-session.json')
            if parsed.path=='/api/image':
                query=parse_qs(parsed.query);kind=query.get('kind',['overlay'])[0]
                if kind not in ('source','overlay','mask'):raise ValueError('Unknown view')
                with station.lock:
                    if 'capture_id' in query:
                        capture=station.captures.get(query['capture_id'][0]);evidence=capture['evidence'] if capture else None
                        data=(station.out/evidence['paths'][kind]).read_bytes() if evidence else None
                    elif 'case_id' in query:
                        case=station.cases.get(query['case_id'][0]);evidence=case.get('evidence') if case else None
                        data=(station.out/evidence['paths'][kind]).read_bytes() if evidence else None
                    else:data=station.images.get(query.get('key',[''])[0],{}).get(kind)
                if data is None:return self.respond(dict(error='Evidence unavailable'),404)
                return self.respond(data,mime='image/png')
            return self.respond(dict(error='Not found'),404)
        except (ValueError,KeyError,TypeError) as exc:self.respond(dict(error=str(exc)),400)

    def do_POST(self):
        station=self.server.station
        try:
            if not self.valid_host():return self.respond(dict(error='Use the local station URL'),403)
            if self.path=='/api/playback':
                if not self.producer():return self.respond(dict(error='Frame producer token required'),403)
                if self.generation()!=station.source_generation or station.active_mode!='prerender' or not station.bundle:
                    return self.respond(dict(error='Playback source changed'),409)
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=1000:raise ValueError('Expected a playback stream identity')
                body=json.loads(self.rfile.read(size))
                with station.lock:
                    if self.generation()!=station.source_generation:raise Conflict('Playback source changed')
                    station.playback=PlaybackClock(body['stream_id'],station.bundle.duration)
                    station.playback.pause(station.source_paused)
                    station.event('prerender_loop_started',dict(stream_id=body['stream_id'],speed=1.,cached_predictions=False))
                return self.respond(station.playback.snapshot(),201)
            if self.path=='/api/frames':
                self.close_connection=True # Do not reuse a connection with an early-rejected PNG body.
                if not self.producer():return self.respond(dict(error='Frame producer token required'),403)
                generation=self.generation() if station.kind=='isaac' else None
                if generation is not None and generation!=station.source_generation:return self.respond(dict(error='Frame source changed'),409)
                if station.processor is None:return self.respond(dict(error='Models are still loading'),503)
                if self.headers.get('Content-Type')!='image/png':return self.respond(dict(error='Expected PNG frame'),415)
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=2*1024*1024:raise ValueError('Frame size outside local input limits')
                metadata=self.headers.get('X-Frame-Metadata','')
                if len(metadata)>4000:raise ValueError('Frame metadata too large')
                if not station.processing.acquire(blocking=False):return self.respond(dict(error='Inference busy; wait for the previous acknowledgement'),429)
                try:
                    frame=IncomingFrame.decode(json.loads(metadata),self.rfile.read(size),station.source_id,station.kind)
                    with station.lock:
                        old=station.last_source
                        if old and old['stream_id']==frame.metadata['stream_id'] and (frame.metadata['sequence']<=old['sequence'] or frame.seconds<=float(old['pts']*Fraction(old['time_base']))):raise Conflict('Duplicate/backward frame')
                    result=station.processor.process(station,frame,generation) if station.kind=='isaac' else station.processor.process(station,frame)
                finally:station.processing.release()
                return self.respond(result,201)
            if self.path!='/api/command':return self.respond(dict(error='Not found'),404)
            if self.headers.get('Origin') not in (None,'http://'+self.headers.get('Host','')):return self.respond(dict(error='Origin mismatch'),403)
            if not secrets.compare_digest(self.headers.get('X-Operator-CSRF',''),self.server.csrf):return self.respond(dict(error='Operator token required'),403)
            if self.headers.get('Content-Type','').split(';')[0]!='application/json':return self.respond(dict(error='JSON required'),415)
            size=int(self.headers.get('Content-Length','0'))
            if not 0<size<=16384:raise ValueError('Expected a small operator command')
            body=json.loads(self.rfile.read(size))
            if not isinstance(body,dict):raise ValueError('Expected command object')
            return self.respond(station.command(body),201)
        except Conflict as exc:self.respond(dict(error=str(exc)),409)
        except (ValueError,TypeError,KeyError,ZeroDivisionError) as exc:self.respond(dict(error=str(exc)),400)
        except Exception as exc:
            station.error=str(exc);self.respond(dict(error='Processing failed; evidence unavailable'),500)


def source_manager(station,args,port):
    """Retire only our producer on mode changes; keep models, cases and journal."""
    while not station.stop.is_set():
        with station.lock:
            generation=station.source_generation;mode=station.requested_mode
            station.active_mode=mode;station.switching=True
        folder=station.out/f'source-{generation:03d}-{mode}'
        with (station.out/f'source-{generation:03d}-{mode}.log').open('w') as log:
            if mode=='isaac':
                command=['cmd.exe','/d','/c',str(args.isaac_python),'-u',str(Path(__file__).with_name('station_simulator.py')),
                    '--url',f'http://127.0.0.1:{port}','--token-file',str(station.out/'frame-input-token.json'),
                    '--out',str(folder),'--source-id',station.source_id,'--generation',str(generation),'--no-ros-env']
                if args.gui:command.append('--gui')
            else:
                command=[sys.executable,'-u',str(Path(__file__).with_name('station_prerender.py')),
                    '--url',f'http://127.0.0.1:{port}','--token-file',str(station.out/'frame-input-token.json'),
                    '--out',str(folder),'--source-id',station.source_id,'--generation',str(generation),'--cache',str(station.bundle.root)]
            station.child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            write(station.out/'pids.json',dict(operator=os.getpid(),source_launcher=station.child.pid,mode=mode,generation=generation))
            while not station.stop.wait(.1) and station.source_generation==generation:
                if station.child.poll() is not None:
                    station.source_error='Frame producer exited; see its source log or switch input mode.'
                    station.switching=False
                    while not station.stop.wait(.1) and station.source_generation==generation:pass
                    break
            try:station.child.wait(timeout=8)
            except subprocess.TimeoutExpired:
                subprocess.run(['taskkill','/PID',str(station.child.pid),'/T','/F'],capture_output=True)
                station.child.wait(timeout=5)


def bootstrap(station,args,port):
    try:
        processor=ModelProcessor(station)
        if station.stop.is_set():return
        if station.kind=='isaac' and args.reference_run:
            models=json.loads((args.reference_run/'models.json').read_text())
            if models!=processor.pins:raise ValueError('Development reference model identity differs')
            path=args.reference_run/'frames.jsonl';rows=[json.loads(s) for s in path.read_text().splitlines()]
            knots={}
            for row in rows:
                h=row['public_source'];m=row['measurement']
                if h['episode_id'].endswith('-reference') and m['measurement_valid']:
                    knots[(h['phase_key'],round(sum(h['metered_ml']),6))]=m['lab']
            station.reference=dict(kind='rendered_development_only',approved=False,recipe_key=SIM_RECIPE,
                source_sha256=sha(path),knots=knots,source=str(path))
        with station.lock:
            station.processor=processor;station.event('inference_ready',dict(models=processor.pins))
        if station.kind=='isaac':
            source_manager(station,args,port)
    except BaseException as exc:
        import traceback
        station.error=str(exc);write(station.out/'failed.json',dict(error=str(exc),traceback=traceback.format_exc()))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--port',type=int,default=8767);p.add_argument('--source',choices=['isaac','prerender','frames'],default='isaac')
    p.add_argument('--source-id',default='local-fillcam');p.add_argument('--gui',action='store_true');p.add_argument('--record-frames',action='store_true')
    tools=Path(os.environ.get('AI_CLUB_HOME',str(Path.home()/'ai-club-tools')))
    p.add_argument('--reference-run',type=Path,default=tools/'results/operator-isaac-20261021/live-v1')
    p.add_argument('--isaac-python',type=Path,default=Path('C:/Apps/IsaacSim/isaacsim/_build/windows-x86_64/release/python.bat'))
    p.add_argument('--prerender-cache',type=Path,default=tools/'results/operator-prerender-20261021/cache-v1')
    args=p.parse_args()
    if not 0<=args.port<=65535:p.error('Invalid port')
    if args.source=='isaac' and (os.name!='nt' or not args.isaac_python.is_file()):p.error('Native Windows Isaac runtime required')
    station=InspectionStation(args.out,args.source_id,'isaac' if args.source=='prerender' else args.source,args.record_frames)
    station.requested_mode=args.source;station.active_mode=args.source
    if (args.prerender_cache/'manifest.json').exists():station.bundle=PrerenderBundle(args.prerender_cache)
    if args.source=='prerender' and station.bundle is None:p.error('Prepare the pre-rendered input cache first')
    server=StationServer(('127.0.0.1',args.port),station)
    dependencies=list(Path(__file__).parent.glob('station*.py'))+[Path(__file__).with_name(n) for n in ('frame_input.py','twin_process.py','prerender.py','render_prerender.py','prepare_prerender_cache.py')]
    dependencies += [ROOT/'setup/vision'/n for n in ['live_twin_isaac.py','live_twin_observer.py','benchmark_liquid_pipeline.py','pool_prompt_policy.py']]
    hashes={}
    for file in dependencies:
        relative=file.relative_to(ROOT);target=station.out/'helpers'/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(file,target);hashes[str(relative)]=sha(file)
    write(station.out/'launch.json',dict(source=args.source,source_id=args.source_id,source_sha256=hashes,
        recipe_context=station.context,hardware_commands=0,run5_used=False,record_frames=args.record_frames))
    thread=threading.Thread(target=bootstrap,args=(station,args,server.server_port),daemon=True);thread.start()
    print(f'Live inspection station: http://127.0.0.1:{server.server_port}',flush=True)
    print(f'Frame adapter configuration: {station.out/"frame-input-token.json"}',flush=True)
    try:server.serve_forever(poll_interval=.25)
    except KeyboardInterrupt:pass
    finally:
        station.stop.set();thread.join(timeout=5)
        if station.child and station.child.poll() is None:subprocess.run(['taskkill','/PID',str(station.child.pid),'/T','/F'],capture_output=True)
        server.server_close()


if __name__=='__main__':main()
