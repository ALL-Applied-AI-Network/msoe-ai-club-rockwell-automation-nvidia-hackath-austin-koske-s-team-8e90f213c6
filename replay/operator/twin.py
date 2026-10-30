"""Interactive Isaac camera -> frozen inference -> operator response rehearsal."""
import argparse
from collections import OrderedDict
from copy import deepcopy
from datetime import datetime, timezone
from fractions import Fraction
from http.server import ThreadingHTTPServer
import hashlib
import json
import os
from pathlib import Path
import queue
import secrets
import shutil
import socket
import subprocess
import sys
import threading
import time
from urllib.parse import parse_qs, urlsplit
import uuid

from .server import Handler, STATIC
from .store import Conflict, text

ROOT=Path(__file__).resolve().parents[2]


class TwinSession:
    def __init__(self,out):
        self.out=Path(out);self.out.mkdir(parents=True,exist_ok=False)
        self.lock=threading.RLock();self.commands=queue.Queue();self.images=OrderedDict()
        self.requests={};self.events=[];self.episodes=[];self.latest=None;self.error=None
        self.session_id='operator-twin-'+uuid.uuid4().hex[:12]
        self.step='loading';self.review='none';self.review_id=None;self.actor=None
        self.cycle=1;self.current=None;self.connected=False;self.last_received=None
        self.stop_requested=threading.Event();self.child=None
        self.journal=(self.out/'operator-events.jsonl').open('a',encoding='utf-8')
        self.event('session_created',dict(mode='live_isaac_simulation',hardware_commands=0))

    def event(self,kind,payload,actor=None):
        row=dict(seq=len(self.events)+1,kind=kind,payload=payload,actor=actor,
            recorded_at_utc=datetime.now(timezone.utc).isoformat(),
            source=self.latest['source'] if self.latest else None)
        self.events.append(row);self.journal.write(json.dumps(row,allow_nan=False)+'\n');self.journal.flush()
        os.fsync(self.journal.fileno())

    def command(self,body):
        actor=text(body.get('actor'),'Operator name',maximum=80)
        request_id=text(body.get('request_id'),'Request identity',maximum=80)
        action=body.get('command');fingerprint=json.dumps(body,sort_keys=True,allow_nan=False)
        with self.lock:
            if request_id in self.requests:
                if self.requests[request_id]!=fingerprint:raise Conflict('Request identity was reused')
                return dict(duplicate=True)
            allowed={'reference':'ready_reference','test':'ready_test','flush':'awaiting_response',
                     'recovery':'ready_recovery','close':'recovery_complete','reset':'complete'}
            if self.error or not self.connected:raise Conflict('Simulator is unavailable')
            if action=='acknowledge':
                if self.review!='requested' or body.get('review_id')!=self.review_id:raise Conflict('Review request changed')
                self.review='acknowledged';self.event('review_acknowledged',dict(review_id=self.review_id),actor)
            elif action=='stop':
                if self.step.endswith('_running') or self.step=='flushing':raise Conflict('Wait for the current simulated operation to finish')
                self.step='stopping';self.commands.put(dict(type='stop'))
                self.event('simulation_stop_requested',{},actor)
            elif action in allowed:
                if self.step!=allowed[action] or body.get('expected_step')!=self.step:raise Conflict('Workflow changed; refresh before continuing')
                if action in ('flush','close'):
                    if self.review!='acknowledged':raise Conflict('Acknowledge the review first')
                    if body.get('confirmed') is not True:raise ValueError('Explicit simulation-only confirmation required')
                    note=text(body.get('note'),'Operator response / follow-up note')
                if action=='flush':
                    self.step='flushing';self.event('simulated_flush_requested',dict(note=note,request_id=request_id),actor)
                    self.commands.put(dict(type='flush',request_id=request_id,actor=actor))
                elif action=='close':
                    self.review='closed';self.step='complete'
                    self.event('follow_up_closed',dict(note=note,product_release_authorized=False),actor)
                elif action=='reset':
                    self.cycle+=1;self.step='ready_reference';self.review='none';self.review_id=None
                    self.latest=None;self.event('new_walkthrough',dict(cycle=self.cycle),actor)
                else:
                    self.step=action+'_running';self.current=f'cycle-{self.cycle:02d}-{action}'
                    self.event('episode_requested',dict(episode_id=self.current,role=action,recipe='10/10/10 mL; order 0,1,2'),actor)
                    self.commands.put(dict(type='run',role=action,episode_id=self.current,actor=actor))
            else:raise ValueError('Unknown simulation command')
            self.requests[request_id]=fingerprint
            return dict(duplicate=False,step=self.step)

    def snapshot(self):
        with self.lock:
            age=None if self.last_received is None else time.monotonic()-self.last_received
            return deepcopy(dict(session_id=self.session_id,mode='live_isaac_simulation',step=self.step,
                connected=self.connected,error=self.error,cycle=self.cycle,review=self.review,review_id=self.review_id,
                latest=self.latest,episodes=self.episodes,events=self.events[-15:],
                frame_age_s=age,hardware_commands_sent=0,operational_recipe_assessment='unavailable',
                geometry_status='unreviewed_model_proposals',references='rendered_development_only',
                physical_calibration=False,model_inference='actual_frozen_models_on_new_RGB',
                process='hypothesized 2 mL mixed compartment; 8 mL simulated prime to drain',
                native_runtime='Isaac Sim, PathTracing, 8 SPP, two subframes',
                journal_path=str(self.out/'operator-events.jsonl')))


class TwinServer(ThreadingHTTPServer):
    daemon_threads=True
    def __init__(self,address,session):
        self.session=session;self.csrf=secrets.token_urlsafe(32)
        super().__init__(address,TwinHandler)


class TwinHandler(Handler):
    def do_GET(self):
        try:
            if not self.valid_host():return self.respond(dict(error='Use the loopback URL'),403)
            parsed=urlsplit(self.path);session=self.server.session
            assets={'/':('twin.html','text/html; charset=utf-8'),'/twin.js':('twin.js','text/javascript; charset=utf-8'),
                    '/twin.css':('twin.css','text/css; charset=utf-8'),'/operator.css':('operator.css','text/css; charset=utf-8')}
            if parsed.path in assets:
                name,mime=assets[parsed.path];return self.respond((STATIC/name).read_bytes(),mime=mime)
            if parsed.path=='/favicon.ico':return self.respond(b'',204,'image/x-icon')
            if parsed.path=='/api/session':return self.respond(dict(csrf_token=self.server.csrf,state=session.snapshot()))
            if parsed.path=='/api/state':return self.respond(session.snapshot())
            if parsed.path=='/api/export.json':
                with session.lock:data=dict(state=session.snapshot(),events=list(session.events),source='simulation_only',product_release_authorized=False)
                return self.respond(data,filename='isaac-operator-session.json')
            if parsed.path=='/api/image':
                key=parse_qs(parsed.query).get('key',[''])[0]
                kind=parse_qs(parsed.query).get('kind',['overlay'])[0]
                if kind not in ('overlay','source','mask'):raise ValueError('Unknown image view')
                with session.lock:data=session.images.get(key,{}).get(kind)
                if data is None:return self.respond(dict(error='Frame expired; refresh the live sample'),404)
                return self.respond(data,mime='image/png')
            return self.respond(dict(error='Not found'),404)
        except (ValueError,KeyError,TypeError) as exc:self.respond(dict(error=str(exc)),400)

    def do_POST(self):
        try:
            if not self.valid_host() or self.headers.get('Origin') not in (None,'http://'+self.headers.get('Host','')):
                return self.respond(dict(error='Origin mismatch'),403)
            if not secrets.compare_digest(self.headers.get('X-Operator-CSRF',''),self.server.csrf):return self.respond(dict(error='Workspace token required'),403)
            if self.path!='/api/command':return self.respond(dict(error='Not found'),404)
            if self.headers.get('Content-Type','').split(';')[0]!='application/json':return self.respond(dict(error='JSON required'),415)
            size=int(self.headers.get('Content-Length','0'))
            if not 0<size<=16384:raise ValueError('Expected a small command')
            body=json.loads(self.rfile.read(size))
            if not isinstance(body,dict):raise ValueError('Expected an object')
            return self.respond(self.server.session.command(body),201)
        except Conflict as exc:self.respond(dict(error=str(exc)),409)
        except (ValueError,KeyError,TypeError) as exc:self.respond(dict(error=str(exc)),400)


def infer_episode(session,conn,command,models,config,pins,calibration,comparisons):
    import cv2
    import numpy as np
    from benchmark_liquid_pipeline import Pipeline
    from replay.detection import FillDetector
    from replay.geometry import decode_mask
    from twin_contract import receive,send,validate_frame
    from stage2_guard import GuardedEvidence
    role=command['role'];episode=command['episode_id']
    if role=='reference':comparisons.reference.clear()
    comparisons.reset();pipeline=Pipeline(models,'pytorch',config,pins,calibration)
    pipeline.fill=FillDetector(config,30,'synthetic-'+episode);guard=GuardedEvidence()
    previous=None;rows=[];started=time.monotonic()
    with (session.out/'frames.jsonl').open('a',encoding='utf-8') as log:
        while True:
            if session.stop_requested.is_set():raise InterruptedError('Application shutdown requested')
            header,pixels=receive(conn)
            if header['type']=='episode_done':
                if header['episode_id']!=episode or header['frames']!=len(rows) or not rows or not previous['dispense_complete']:
                    raise ValueError('Incomplete simulated fill')
                break
            validate_frame(header,pixels,previous)
            if header['episode_id']!=episode or header['session_id']!=session.session_id:raise ValueError('Simulation source changed')
            rgb=np.frombuffer(pixels,dtype=np.uint8).reshape(480,640,3).copy();start=time.perf_counter()
            row=pipeline.process(rgb,header['frame_index']*6,header['frame_index'],Fraction(1,5))
            diagnostic=comparisons.update(header,row,role=='reference')
            key=(header['phase_key'],round(sum(header['metered_ml']),6))
            context=dict(session_id=header['session_id'],vial_id=episode,recipe_key=header['recipe_key'],
                reference_recipe_key=header['recipe_key'],source_time_s=header['sim_time_s'],decision_time_s=header['sim_time_s'],
                fill_started_s=0.,reference_available_s=-1.,context_available_s=0.,process_state_source_s=header['sim_time_s'],
                phase_key=header['phase_key'],process_phase_key=header['phase_key'],expected_phase_key=header['phase_key'],
                reference_id='nominal-rendered-reference' if key in comparisons.reference else None,
                measurement_profile='synthetic-raw-lab',reference_measurement_profile='synthetic-raw-lab',
                camera_reference_key='uncommissioned-FillCam',reference_camera_key=None,camera_health_valid=False,
                vial_identity_unambiguous=False,geometry_status='unreviewed_model_proposals',
                measurement_valid=row['measurement']['measurement_valid'],observed_lab=row['measurement']['lab'],expected_lab=comparisons.reference.get(key))
            row.update(public_source=header,synthetic_color_diagnostic=diagnostic,recipe_decision=guard.update(context),
                declared_recipe_context=context,wall_inference_s=time.perf_counter()-start)
            log.write(json.dumps(row,allow_nan=False)+'\n');log.flush();rows.append(row)
            bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR);overlay=bgr.copy()
            mask=decode_mask(row['measurement']['geometry']['mask'],(480,640)).astype(bool)
            overlay[mask]=(.55*overlay[mask]+.45*np.array([65,225,220])).astype(np.uint8)
            box=row['measurement']['geometry'].get('vial_box')
            if box:cv2.rectangle(overlay,tuple(map(int,box[:2])),tuple(map(int,box[2:])),(235,160,55),1)
            images={}
            for name,data in [('source',bgr),('overlay',overlay),('mask',mask.astype(np.uint8)*255)]:
                ok,buffer=cv2.imencode('.png',data)
                if not ok:raise RuntimeError('Image encoding failed')
                images[name]=buffer.tobytes()
            image_key=episode+'-'+str(header['frame_index'])
            (session.out/'overlays'/(image_key+'.png')).write_bytes(images['overlay'])
            with session.lock:
                session.images[image_key]=images
                while len(session.images)>12:session.images.popitem(last=False)
                session.last_received=time.monotonic()
                session.latest=dict(image_key=image_key,source=header,role=role,
                    measurement_valid=row['measurement']['measurement_valid'],quality_flags=row['measurement']['quality_flags'],
                    mask_pixels=int(mask.sum()),diagnostic=diagnostic,operational_recipe_assessment=row['recipe_decision']['status'],
                    inference_ms=round(row['wall_inference_s']*1000,1),image_phase=row['fill_detection']['phase'])
                if role!='reference' and diagnostic['warning'] and session.review=='none':
                    session.review='requested';session.review_id=uuid.uuid4().hex
                    session.event('review_requested',dict(origin='actual_model_color_diagnostic',
                        review_id=session.review_id,delta_e2000=diagnostic.get('delta_e2000'),product_release_authorized=False))
            send(conn,dict(type='frame_ack',episode_id=episode,frame_index=header['frame_index'],rgb_sha256=header['rgb_sha256'],hardware_commands=0))
            previous=header
    scores=[r['synthetic_color_diagnostic']['delta_e2000'] for r in rows if r['synthetic_color_diagnostic']['delta_e2000'] is not None]
    summary=dict(episode_id=episode,role=role,frames=len(rows),qa_pass=sum(r['measurement']['measurement_valid'] for r in rows),
        comparisons=len(scores),max_delta_e2000=max(scores) if scores else None,warning=comparisons.warned,
        seconds_wall=time.monotonic()-started,operational_recipe_assessment='unavailable')
    with session.lock:
        session.episodes.append(summary)
        session.step={'reference':'ready_test','test':'awaiting_response','recovery':'recovery_complete'}[role]
        session.event('episode_completed',summary)
        if role=='test' and session.review=='none':
            # A planned human inspection is explicit and must not be reported as a model detection.
            session.review='requested';session.review_id=uuid.uuid4().hex
            session.event('review_requested',dict(origin='planned_manual_scenario_review',review_id=session.review_id,
                model_warning_observed=False,reason='Inspect the result before any simulated intervention'))
        (session.out/'state.json').write_text(json.dumps(session.snapshot(),indent=2))


def worker(session,a):
    child=None
    try:
        sys.path.insert(0,str(ROOT/'setup/vision'))
        from live_twin_observer import NativeModels, ProvisionalComparison
        from liquid_specialists_data import sha,read
        from probe_prompted_liquid import BASELINE_SHA,SAM2_SHA
        from replay.config import load_config
        from twin_contract import send,receive
        tools=Path(os.environ.get('AI_CLUB_HOME',str(Path.home()/'ai-club-tools')))
        baseline=tools/'results/crop-clahe-detector-20261019/baseline-best.pt'
        experiment=tools/'results/early-liquid-20261020';sam=tools/'results/prompted-liquid-assets-20261020/sam2.1_t.pt'
        liquid=experiment/'models/fasterrcnn-resnet50/best.pt';meta=read(liquid.with_name('model.json'))
        pins=dict(vial=BASELINE_SHA,liquid=meta['checkpoint_sha256'],sam=SAM2_SHA)
        if sha(baseline)!=pins['vial'] or sha(liquid)!=pins['liquid'] or sha(sam)!=pins['sam']:raise ValueError('Frozen model hash differs')
        (session.out/'models.json').write_text(json.dumps(pins,indent=2))
        models=NativeModels(baseline,experiment,sam)
        config=load_config(ROOT/'replay/examples/bootstrap-vision.json')
        calibration=read(tools/'results/stage2-refinement-20261020/camera-reference-v2/calibration.json')
        comparisons=ProvisionalComparison()
        if session.stop_requested.is_set():return
        with socket.socket() as listener, (session.out/'isaac.log').open('w') as log:
            listener.bind(('127.0.0.1',0));listener.listen(1);listener.settimeout(240)
            command=[str(a.isaac_python),'-u',str(Path(__file__).with_name('twin_simulator.py')),
                '--port',str(listener.getsockname()[1]),'--out',str(session.out/'simulation'),
                '--session',session.session_id,'--no-ros-env']
            if a.gui:command.append('--gui')
            child=subprocess.Popen(['cmd.exe','/d','/c',*command],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            session.child=child
            (session.out/'pids.json').write_text(json.dumps(dict(operator=os.getpid(),isaac_launcher=child.pid)))
            conn,_=listener.accept()
            with conn:
                conn.settimeout(180);ready,payload=receive(conn)
                if payload or ready!={'type':'simulator_ready','runtime':'native_isaac','hardware_commands':0}:raise ValueError('Unexpected simulator handshake')
                with session.lock:
                    session.connected=True;session.step='ready_reference';session.event('simulator_ready',dict(models=pins))
                while True:
                    try:command=session.commands.get(timeout=.5)
                    except queue.Empty:
                        if child.poll() is not None:raise RuntimeError('Isaac exited; see isaac.log')
                        continue
                    send(conn,{k:v for k,v in command.items() if k!='actor'})
                    if command['type']=='run':infer_episode(session,conn,command,models,config,pins,calibration,comparisons)
                    elif command['type']=='flush':
                        reply,payload=receive(conn)
                        if payload or reply.get('type')!='flush_done' or reply.get('request_id')!=command['request_id']:raise ValueError('Simulated procedure receipt missing')
                        with session.lock:
                            session.step='ready_recovery';session.event('simulated_flush_completed',reply,command['actor'])
                    elif command['type']=='stop':
                        reply,payload=receive(conn)
                        if reply!={'type':'stopped'} or payload:raise ValueError('Unexpected stop acknowledgement')
                        with session.lock:session.connected=False;session.step='stopped';session.event('simulator_stopped',{})
                        break
            child.wait(timeout=30)
            if child.returncode:raise RuntimeError('Isaac shutdown failed')
    except BaseException as exc:
        import traceback
        if session.stop_requested.is_set():
            with session.lock:
                session.connected=False;session.step='stopped';session.event('simulator_stopped',dict(reason='application_shutdown'))
            return
        with session.lock:
            session.error=str(exc);session.connected=False;session.step='failed';session.event('session_failed',dict(error=str(exc)))
        (session.out/'failed.json').write_text(json.dumps(dict(error=str(exc),traceback=traceback.format_exc()),indent=2))
        traceback.print_exc()
    finally:
        if child is not None and child.poll() is None:
            subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],capture_output=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--port',type=int,default=8767);p.add_argument('--gui',action='store_true')
    p.add_argument('--isaac-python',type=Path,default=Path('C:/Apps/IsaacSim/isaacsim/_build/windows-x86_64/release/python.bat'))
    a=p.parse_args()
    if os.name!='nt':p.error('Use native Windows nvidia-pilot-windows Python for the live GPU environment')
    if not a.isaac_python.is_file():p.error('Isaac launcher not found')
    if not 0<=a.port<=65535:p.error('Invalid port')
    session=TwinSession(a.out);(a.out/'overlays').mkdir();(a.out/'helpers').mkdir()
    dependencies=list(Path(__file__).parent.glob('twin*.py'))+list((ROOT/'setup/vision').glob('*twin*.py'))
    dependencies += [ROOT/'setup/vision'/n for n in ['benchmark_liquid_pipeline.py','stage2_guard.py','pool_prompt_policy.py']]
    hashes={}
    for file in dependencies:
        relative=file.relative_to(ROOT);target=a.out/'helpers'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(file,target);hashes[str(relative)]=hashlib.sha256(file.read_bytes()).hexdigest()
    (a.out/'launch.json').write_text(json.dumps(dict(python=sys.executable,isaac_launcher=str(a.isaac_python),gui=a.gui,
        source_sha256=hashes,hardware_commands=0,run5_used=False),indent=2))
    server=TwinServer(('127.0.0.1',a.port),session)
    thread=threading.Thread(target=worker,args=(session,a),daemon=True);thread.start()
    print(f'Isaac operator workspace: http://127.0.0.1:{server.server_port}',flush=True)
    print(f'Local simulation evidence: {a.out}',flush=True)
    try:server.serve_forever(poll_interval=.25)
    except KeyboardInterrupt:pass
    finally:
        session.stop_requested.set();session.commands.put(dict(type='stop'));thread.join(timeout=10)
        if thread.is_alive() and session.child is not None and session.child.poll() is None:
            subprocess.run(['taskkill','/PID',str(session.child.pid),'/T','/F'],capture_output=True)
            thread.join(timeout=5)
        server.server_close()
        if not thread.is_alive():session.journal.close()


if __name__=='__main__':main()
