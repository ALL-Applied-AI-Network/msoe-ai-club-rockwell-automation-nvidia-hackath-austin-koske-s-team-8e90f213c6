"""Native Isaac worker for operator-directed local simulation; no plant adapters."""
import argparse
import json
from pathlib import Path
import socket
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'setup/vision'));sys.path.insert(0,str(ROOT))
from twin_contract import default_plan, SharedPath, send, receive, digest
from replay.operator.twin_process import fill_states


def main(a):
    from isaacsim import SimulationApp
    app=SimulationApp(dict(headless=not a.gui,renderer='PathTracing',width=640,height=480,
        multi_gpu=False,extra_args=['--/app/extensions/registryEnabled=false',
        '--/telemetry/enableAnonymousData=false','--/crashreporter/enabled=false']))
    a.out.mkdir(parents=True,exist_ok=True)
    try:
        from PIL import Image
        from live_twin_isaac import Station, CAMERA
        from box_annotations import sha256
        from isaacsim.core.version import get_version
        assets=ROOT/'datasets/mixing_twin/assets/csi_digital_twin_v1'
        before={str(p.relative_to(assets)):sha256(p) for p in assets.rglob('*') if p.is_file()}
        plan=default_plan();station=Station(app,assets,8,2);path=None
        if a.gui:
            from omni.kit.viewport.utility import get_active_viewport
            viewport=get_active_viewport()
            if viewport is None:raise RuntimeError('Native viewport is unavailable')
            viewport.camera_path=CAMERA
        (a.out/'runtime.json').write_text(json.dumps(dict(isaac_version=list(get_version()),
            camera=station.camera,renderer='PathTracing',spp=8,subframes=2,gui=a.gui,
            asset_sha256=before,physics='hypothesized_2ml_mixed_path',run5_used=False),indent=2))
        (a.out/'rgb').mkdir();(a.out/'geometric-masks').mkdir()
        with socket.create_connection(('127.0.0.1',a.port),timeout=180) as conn, (a.out/'private-truth.jsonl').open('w') as log:
            conn.settimeout(.25)
            send(conn,dict(type='simulator_ready',runtime='native_isaac',hardware_commands=0))
            while app.is_running():
                # Peek before receive: partial framed messages must not be abandoned
                # on a short socket timeout while keeping the Kit UI responsive.
                import select
                if not select.select([conn],[],[],.05)[0]:app.update();continue
                conn.settimeout(180)
                command,payload=receive(conn)
                if payload:raise ValueError('Unexpected command pixels')
                if command['type']=='stop':send(conn,dict(type='stopped'));break
                if command['type']=='flush':
                    if path is None:raise ValueError('No current path to flush')
                    prior=list(path.inventory);result=path.flush(0,8.)
                    log.write(json.dumps(dict(kind='operator_flush',request_id=command['request_id'],
                        path_before_ml=prior,path_after_ml=list(path.inventory),result=result))+'\n');log.flush()
                    send(conn,dict(type='flush_done',request_id=command['request_id'],drained_ml=8.,
                        procedure='synthetic_8ml_tank0_prime_to_drain',physical_procedure_approved=False))
                    conn.settimeout(.25);continue
                if command['type']!='run':raise ValueError('Unknown simulation command')
                role=command['role'];episode=command['episode_id']
                if role=='reference':path=SharedPath(2.,plan['initial_fraction'])
                elif role=='test':path=SharedPath(2.,plan['prior_fraction'])
                elif role!='recovery' or path is None:raise ValueError('Missing retained recovery state')
                log.write(json.dumps(dict(kind='episode_start',episode_id=episode,role=role,
                    initial_path_components_ml=list(path.inventory),new_vial=True,
                    initialization='declared_scenario' if role!='recovery' else 'retained_after_operator_flush'))+'\n');log.flush()
                for state in fill_states(plan,path):
                    t0=time.perf_counter();rgb,mask,parameters=station.render(state,plan)
                    stem=f'{episode}-{state["frame_index"]:05d}'
                    Image.fromarray(rgb).save(a.out/'rgb'/(stem+'.png'))
                    Image.fromarray(mask.astype('uint8')*255).save(a.out/'geometric-masks'/(stem+'.png'))
                    header={k:state[k] for k in ['frame_index','sim_time_s','phase_key','recipe_key','metered_ml','dispense_complete']}
                    header.update(schema_version=1,type='frame',session_id=a.session,episode_id=episode,
                        rgb_sha256=digest(rgb.tobytes()),shape=[480,640,3])
                    send(conn,header,rgb.tobytes());reply,body=receive(conn)
                    if body or reply!={**{k:header[k] for k in ['episode_id','frame_index','rgb_sha256']},'type':'frame_ack','hardware_commands':0}:
                        raise ValueError('Source-aligned inference acknowledgement missing')
                    log.write(json.dumps(dict(kind='frame',episode_id=episode,state=state,
                        parameters=parameters,rgb_sha256=header['rgb_sha256'],rgb_file='rgb/'+stem+'.png',
                        mask_file='geometric-masks/'+stem+'.png',ack=reply,wall_cycle_s=time.perf_counter()-t0))+'\n');log.flush()
                send(conn,dict(type='episode_done',episode_id=episode,frames=state['frame_index']+1))
                conn.settimeout(.25)
            after={str(p.relative_to(assets)):sha256(p) for p in assets.rglob('*') if p.is_file()}
            if before!=after:raise RuntimeError('Source assets changed')
            (a.out/'stopped.json').write_text(json.dumps(dict(source_assets_unchanged=True,hardware_commands=0)))
    except BaseException as exc:
        (a.out/'failed.json').write_text(json.dumps(dict(error=str(exc),traceback=traceback.format_exc())))
        raise
    finally:app.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port',type=int,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--session',required=True);p.add_argument('--gui',action='store_true')
    p.add_argument('--no-ros-env',action='store_true')
    main(p.parse_args())
