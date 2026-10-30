"""Isaac frame producer for the source-neutral station input; no operator steps."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import traceback
from urllib.request import Request,urlopen
from urllib.error import HTTPError
import uuid

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'setup/vision'))
from replay.operator.frame_input import FramePublisher
from replay.operator.twin_process import fill_states
from twin_contract import SharedPath,default_plan


def main(a):
    from isaacsim import SimulationApp
    app=SimulationApp(dict(headless=not a.gui,renderer='PathTracing',width=640,height=480,multi_gpu=False,
        extra_args=['--/app/extensions/registryEnabled=false','--/telemetry/enableAnonymousData=false','--/crashreporter/enabled=false']))
    a.out.mkdir(parents=True,exist_ok=False)
    try:
        from PIL import Image
        from live_twin_isaac import Station,CAMERA
        from box_annotations import sha256
        from isaacsim.core.version import get_version
        assets=ROOT/'datasets/mixing_twin/assets/csi_digital_twin_v1'
        before={str(p.relative_to(assets)):sha256(p) for p in assets.rglob('*') if p.is_file()}
        station=Station(app,assets,8,2);plan=default_plan();publisher=FramePublisher(a.url,a.token_file,generation=a.generation)
        if a.gui:
            from omni.kit.viewport.utility import get_active_viewport
            viewport=get_active_viewport()
            if viewport is None:raise RuntimeError('Isaac viewport unavailable')
            viewport.camera_path=CAMERA
        (a.out/'runtime.json').write_text(json.dumps(dict(isaac_version=list(get_version()),camera=station.camera,
            asset_sha256=before,spp=8,subframes=2,private_scenario_pattern='independent nominal, residue, nominal scenes',
            hardware_commands=0,operator_commands_drive_simulation=False),indent=2))
        stream_id='isaac-'+uuid.uuid4().hex;sequence=0;number=0
        token=json.loads(a.token_file.read_text())['token']
        def control():
            request=Request(a.url+'/api/source',headers={'X-Frame-Token':token,'X-Source-Generation':str(a.generation)})
            with urlopen(request,timeout=10) as response:return json.load(response)
        with (a.out/'private-truth.jsonl').open('w') as log:
            while app.is_running():
                segment='window-'+str(number+1).zfill(6)
                fraction=plan['prior_fraction'] if number%3==1 else plan['initial_fraction']
                path=SharedPath(2.,fraction)
                for state in fill_states(plan,path):
                    status=control()
                    while status['paused'] and not status['stop']:
                        app.update();time.sleep(.05);status=control()
                    if status['stop']:return
                    rgb,mask,parameters=station.render(state,plan)
                    buffer=io.BytesIO();Image.fromarray(rgb).save(buffer,format='PNG')
                    metadata=dict(schema_version=1,source_id=a.source_id,stream_id=stream_id,kind='isaac',
                        sequence=sequence,pts=sequence,time_base='1/5',width=640,height=480,
                        rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),segment_id=segment,
                        telemetry={k:state[k] for k in ['phase_key','metered_ml','dispense_complete','recipe_key']})
                    while True:
                        try:reply=publisher.publish(metadata,buffer.getvalue());break
                        except HTTPError as exc:
                            if control()['stop']:return
                            if exc.code!=429:raise
                            time.sleep(.05)
                    if reply['accepted_sequence']!=sequence or reply['rgb_sha256']!=metadata['rgb_sha256']:
                        raise ValueError('Frame/inference acknowledgement differs')
                    log.write(json.dumps(dict(segment_id=segment,sequence=sequence,state=state,
                        initial_fraction=fraction,rgb_sha256=metadata['rgb_sha256'],geometric_mask_pixels=int(mask.sum()),ack=reply))+'\n');log.flush()
                    sequence+=1
                number+=1
    except BaseException as exc:
        (a.out/'failed.json').write_text(json.dumps(dict(error=str(exc),traceback=traceback.format_exc()),indent=2));raise
    finally:
        if 'before' in locals():
            after={str(p.relative_to(assets)):sha256(p) for p in assets.rglob('*') if p.is_file()}
            (a.out/'stopped.json').write_text(json.dumps(dict(source_assets_unchanged=before==after,hardware_commands=0)))
        app.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url',required=True);p.add_argument('--token-file',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--source-id',required=True)
    p.add_argument('--generation',type=int,default=0)
    p.add_argument('--gui',action='store_true');p.add_argument('--no-ros-env',action='store_true')
    main(p.parse_args())
