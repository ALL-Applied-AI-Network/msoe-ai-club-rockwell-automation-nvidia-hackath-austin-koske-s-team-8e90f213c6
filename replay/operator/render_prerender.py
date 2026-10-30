"""Render genuine 30 Hz Isaac RGB once; no inference, labels or plant connection."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'setup/vision'))
from replay.operator.twin_process import fill_states
from twin_contract import SharedPath,default_plan


def main(a):
    if a.out.exists():raise ValueError('Use a fresh pre-render directory')
    a.out.mkdir(parents=True);(a.out/'render-rgb').mkdir()
    from isaacsim import SimulationApp
    app=SimulationApp(dict(headless=True,renderer='PathTracing',width=640,height=480,multi_gpu=False,
        extra_args=['--/app/extensions/registryEnabled=false','--/telemetry/enableAnonymousData=false','--/crashreporter/enabled=false']))
    try:
        from PIL import Image
        from live_twin_isaac import Station
        from box_annotations import sha256
        from isaacsim.core.version import get_version
        assets=ROOT/'datasets/mixing_twin/assets/csi_digital_twin_v1'
        before={str(p.relative_to(assets)):sha256(p) for p in assets.rglob('*') if p.is_file()}
        station=Station(app,assets,8,2);plan=default_plan();index=0;start=time.monotonic()
        with (a.out/'render-manifest.jsonl').open('w') as log:
            for number in range(3):
                initial=plan['prior_fraction'] if number==1 else plan['initial_fraction']
                path=SharedPath(2.,initial)
                for state in fill_states(plan,path,1/30):
                    rgb,_,_=station.render(state,plan,labels=False)
                    image=a.out/'render-rgb'/f'{index:06d}.png';Image.fromarray(rgb).save(image)
                    log.write(json.dumps(dict(index=index,segment_id=f'window-{number+1:06d}',
                        local_frame=state['frame_index'],render_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                        telemetry={k:state[k] for k in ['phase_key','metered_ml','dispense_complete','recipe_key']}))+'\n');log.flush()
                    index+=1
                    if index%30==0:print('RENDERED',index,'frames',round(time.monotonic()-start,1),'seconds',flush=True)
        after={str(p.relative_to(assets)):sha256(p) for p in assets.rglob('*') if p.is_file()}
        if before!=after:raise RuntimeError('Source assets changed')
        (a.out/'render-completed.json').write_text(json.dumps(dict(frames=index,fps=30,isaac_version=list(get_version()),
            assets_sha256=before,source_assets_unchanged=True,wall_seconds=time.monotonic()-start,
            labels_generated=False,model_predictions_cached=False,hardware_commands=0),indent=2))
    finally:app.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--no-ros-env',action='store_true');main(p.parse_args())
