"""Matched cached-input and live-render compute benchmark; no fitting or devices.

Run against a pinned, isolated source/model bundle. Output paths must be fresh.
The measured station is distinct from the operator's active session.
"""
import argparse
from fractions import Fraction
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import uuid


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda:file.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def read(path):return json.loads(path.read_text())


def summary(values):
    import numpy as np
    return dict(count=len(values),p50=float(np.percentile(values,50)),
        p95=float(np.percentile(values,95)),maximum=float(max(values))) if values else None


def runtime():
    import cv2
    import torch
    versions={}
    for name in ('torch','torchvision','ultralytics','numpy','av','scikit-image','Pillow'):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:pass
    smi=shutil.which('nvidia-smi') or ('C:/Windows/System32/nvidia-smi.exe' if os.name=='nt' else 'nvidia-smi')
    gpu=subprocess.run([smi,'--query-gpu=name,utilization.gpu,power.draw','--format=csv,noheader'],capture_output=True,text=True)
    return dict(python=sys.version,platform=platform.platform(),machine=platform.machine(),versions=versions,
        opencv=cv2.__version__,torch_threads=torch.get_num_threads(),opencv_threads=cv2.getNumThreads(),
        cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),capability=list(torch.cuda.get_device_capability()),
        cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,gpu_status=gpu.stdout.strip())


def prepare(a):
    if a.out.exists():raise ValueError('Use a fresh benchmark output')
    a.out.mkdir(parents=True)
    repo=a.bundle/'repo';sys.path.insert(0,str(repo));sys.path.insert(0,str(repo/'setup/vision'))
    os.environ['AI_CLUB_HOME']=str(a.bundle/'tools')
    os.environ['MPLBACKEND']='Agg';os.environ['YOLO_OFFLINE']='true';os.environ['YOLO_AUTOINSTALL']='false'
    plan=read(a.bundle/'input-plan.json')
    for name,expected in plan['files'].items():
        if sha(a.bundle/name)!=expected:raise ValueError('Pinned bundle file changed: '+name)
    write(a.out/'input-identity.json',dict(plan_sha256=sha(a.bundle/'input-plan.json'),
        runner_sha256=sha(__file__),verified_files=len(plan['files']),mode=a.mode,
        bundle=str(a.bundle),training=False,run5_used=False,hardware_connections=0))
    return repo


def station(a):
    from replay.operator.station import InspectionStation,ModelProcessor,SIM_RECIPE
    instance=InspectionStation(a.out/'station','benchmark-fillcam','isaac',record_frames=False)
    started=time.perf_counter();processor=ModelProcessor(instance);instance.processor=processor
    load_ms=(time.perf_counter()-started)*1000
    reference=a.bundle/'tools/results/operator-isaac-20261021/live-v1/frames.jsonl';knots={}
    for row in [json.loads(s) for s in reference.read_text().splitlines()]:
        h=row['public_source'];m=row['measurement']
        if h['episode_id'].endswith('-reference') and m['measurement_valid']:
            knots[(h['phase_key'],round(sum(h['metered_ml']),6))]=m['lab']
    instance.reference=dict(kind='rendered_development_only',approved=False,recipe_key=SIM_RECIPE,
        source_sha256=sha(reference),knots=knots,source=str(reference))
    write(a.out/'runtime.json',runtime()|dict(model_load_ms=load_ms,model_sha256=processor.pins))
    return instance,processor


def cache_frames(a):
    import cv2
    from replay.operator.prerender import PrerenderBundle
    bundle=PrerenderBundle(a.bundle/'cache');samples=[]
    for sample in bundle.manifest['samples']:
        rgb=cv2.cvtColor(cv2.imread(str(bundle.root/sample['png'])),cv2.COLOR_BGR2RGB)
        if hashlib.sha256(rgb.tobytes()).hexdigest()!=sample['rgb_sha256']:raise ValueError('Input pixels differ')
        samples.append((sample,rgb))
    return bundle,samples


def frame_for(sample,rgb,stream,sequence):
    from replay.operator.frame_input import IncomingFrame
    return IncomingFrame(dict(schema_version=1,source_id='benchmark-fillcam',stream_id=stream,kind='isaac',
        sequence=sequence,pts=sample['pts'],time_base=sample['time_base'],width=640,height=480,
        rgb_sha256=sample['rgb_sha256'],segment_id=sample['segment_id'],telemetry=sample['telemetry']),rgb)


def cached(a):
    instance,processor=station(a);bundle,samples=cache_frames(a)
    measurements=[];start=time.monotonic();total=0
    with (a.out/'timings.jsonl').open('w') as log:
        passes=3 if a.mode=='offline' else 2 # Full warm-up, then two offline or one paced pass.
        for repetition in range(passes):
            warmup=repetition==0;stream='bench-'+uuid.uuid4().hex;clock=time.monotonic();index=0;last=-1
            while index<len(samples):
                if a.mode=='paced' and not warmup:
                    elapsed=time.monotonic()-clock
                    if elapsed>=bundle.duration:break
                    import bisect
                    due=bisect.bisect_right([s['seconds'] for s,_ in samples],elapsed)-1
                    if due<=last:time.sleep(.005);continue
                    index=due
                sample,rgb=samples[index];frame=frame_for(sample,rgb,stream,index)
                began=time.perf_counter();processor.process(instance,frame)
                elapsed_ms=(time.perf_counter()-began)*1000
                latest=instance.last
                record=dict(repetition=repetition,warmup=warmup,sample_index=index,
                    source_rgb_sha256=sample['rgb_sha256'],stream_id=stream,
                    source_seconds=sample['seconds'],station_wall_ms=elapsed_ms,
                    inference_ms=latest['inference_ms'],qa_valid=latest['measurement']['measurement_valid'],
                    mask_pixels=latest['measurement']['geometry']['mask_pixels'],
                    skipped_samples=index-last-1 if a.mode=='paced' and not warmup else 0,
                    source_to_result_ms=max(0.,(time.monotonic()-clock-sample['seconds'])*1000) if a.mode=='paced' and not warmup else None)
                log.write(json.dumps(record)+'\n');log.flush();total+=1
                if not warmup:measurements.append(record)
                last=index;index+=1
                if total%25==0:print(json.dumps(dict(mode=a.mode,processed=total,warmup=warmup,wall_s=round(time.monotonic()-start,1))),flush=True)
    rows=[json.loads(line) for line in (instance.out/'frames.jsonl').read_text().splitlines()]
    rows=[r for r,t in zip(rows,[json.loads(s) for s in (a.out/'timings.jsonl').read_text().splitlines()]) if not t['warmup']]
    result=dict(status='passed',mode=a.mode,warmup_samples=len(samples),measured_samples=len(measurements),
        selected_input_samples=len(samples),station_wall_ms=summary([m['station_wall_ms'] for m in measurements]),
        inference_ms=summary([m['inference_ms'] for m in measurements]),
        throughput_samples_s=len(measurements)/(sum(m['station_wall_ms'] for m in measurements)/1000),
        qa_valid=sum(m['qa_valid'] for m in measurements),samples_with_masks=sum(m['mask_pixels']>0 for m in measurements),
        prompted_samples=sum(r['model_output']['timing_ms']['sam_prompted'] for r in rows),
        components_ms={key:summary([r['model_output']['timing_ms'][key] for r in rows]) for key in
            ('vial','liquid','prompting','sam','measurement_and_state','result_encoding','processing')},
        source_to_result_ms=summary([m['source_to_result_ms'] for m in measurements if m['source_to_result_ms'] is not None]),
        skipped_input_samples=sum(m['skipped_samples'] for m in measurements),
        unprocessed_tail_samples=(len(samples)-1-measurements[-1]['sample_index']) if a.mode=='paced' else 0,
        model_sha256=processor.pins,source_video_sha256=bundle.manifest['video_sha256'],
        hardware_commands=0,cached_predictions=False,
        limits=['Station wall time includes actual inference, mask PNG encoding and local journal/log writes.',
            'Images are predecoded before timed processing; no browser, network or camera capture is included.',
            'Platform/runtime differences are recorded; this is not an architecture-only hardware comparison.',
            'Paced run skips to the latest scheduled input; source-to-result age excludes browser display.'])
    write(a.out/'completed.json',result);print(json.dumps(result,indent=2),flush=True)


def serve(a):
    from replay.operator.station import StationServer
    instance,processor=station(a);server=StationServer(('127.0.0.1',0),instance)
    write(a.out/'endpoint.json',dict(url='http://127.0.0.1:'+str(server.server_port),
        token_file=str(instance.out/'frame-input-token.json')))
    print('Loopback benchmark observer ready',server.server_port,flush=True)
    try:server.serve_forever(poll_interval=.1)
    finally:server.server_close()


def render(a,repo):
    from isaacsim import SimulationApp
    app=SimulationApp(dict(headless=True,renderer='PathTracing',width=640,height=480,multi_gpu=False,
        extra_args=['--/app/extensions/registryEnabled=false','--/telemetry/enableAnonymousData=false','--/crashreporter/enabled=false']))
    try:
        import numpy as np
        from PIL import Image
        from isaacsim.core.version import get_version
        # 6.0 and 6.1 label APIs differ. Only the benchmark's semantic tagging is
        # adapted; no source renderer, asset or liquid appearance is modified.
        compat=False
        try:from isaacsim.core.experimental.utils.semantics import add_labels
        except ImportError:
            import types
            from isaacsim.core.utils.semantics import add_update_semantics
            module=types.ModuleType('isaacsim.core.experimental.utils.semantics')
            module.add_labels=lambda prim,labels,taxonomy:add_update_semantics(prim,semanticLabel=labels[0],type_label=taxonomy)
            sys.modules[module.__name__]=module;compat=True
        from live_twin_isaac import Station
        from replay.operator.twin_process import fill_states
        from replay.operator.frame_input import FramePublisher
        from twin_contract import SharedPath,default_plan
        assets=repo/'datasets/mixing_twin/assets/csi_digital_twin_v1'
        before={str(p.relative_to(assets)):sha(p) for p in assets.rglob('*') if p.is_file()}
        native=Station(app,assets,8,2);plan=default_plan()
        endpoint=read(a.endpoint) if a.endpoint else None
        publisher=FramePublisher(endpoint['url'],Path(endpoint['token_file']),generation=0) if endpoint else None
        timings=[];stream='render-bench-'+uuid.uuid4().hex
        for index,state in enumerate(fill_states(plan,SharedPath(2.,plan['prior_fraction']))):
            began=time.perf_counter();rgb,_,_=native.render(state,plan,labels=False);render_ms=(time.perf_counter()-began)*1000
            digest=hashlib.sha256(rgb.tobytes()).hexdigest()
            path=a.out/f'{index:04d}.png';Image.fromarray(rgb).save(path)
            metadata=dict(schema_version=1,source_id='benchmark-fillcam',stream_id=stream,kind='isaac',sequence=index,
                pts=index,time_base='1/5',width=640,height=480,rgb_sha256=digest,segment_id='rendered-residue',
                telemetry={k:state[k] for k in ('phase_key','metered_ml','dispense_complete','recipe_key')})
            publish_ms=None
            if publisher:
                began=time.perf_counter();ack=publisher.publish(metadata,path.read_bytes());publish_ms=(time.perf_counter()-began)*1000
                assert ack['accepted_sequence']==index and ack['rgb_sha256']==digest
            timings.append(dict(sequence=index,rgb_sha256=digest,render_ms=render_ms,publish_inference_ms=publish_ms,
                step_ms=render_ms+(publish_ms or 0.),warmup=index<5))
            with (a.out/'timings.jsonl').open('a') as log:log.write(json.dumps(timings[-1])+'\n')
            if index%5==0:print('RENDER_BENCH',index,round(render_ms,1),publish_ms,flush=True)
        after={str(p.relative_to(assets)):sha(p) for p in assets.rglob('*') if p.is_file()};assert before==after
        measured=timings[5:]
        result=dict(status='passed',mode='render_plus_inference' if publisher else 'render_only',
            rendered_frames=len(timings),warmup_frames=5,measured_frames=len(measured),
            render_ms=summary([t['render_ms'] for t in measured]),
            publish_inference_ms=summary([t['publish_inference_ms'] for t in measured]) if publisher else None,
            step_ms=summary([t['step_ms'] for t in measured]),
            sequential_steps_s=len(measured)/(sum(t['step_ms'] for t in measured)/1000),
            isaac_version=list(get_version()),semantic_API_compatibility=compat,
            source_assets_sha256=before,source_assets_unchanged=True,hardware_commands=0,
            limits=['RGB-only path: 8 spp, two subframes, two settled captures per input frame; no private geometric-label captures.',
                'Sequential renderer waits for inference acknowledgement, matching the current native source behavior.',
                'Different Isaac versions/hardware may produce different RGB; this is a timing test, not image-quality equivalence.',
                'PNG persistence is outside render/step timing; network is loopback and excludes remote browser display.'])
        write(a.out/'completed.json',result);print(json.dumps(result,indent=2),flush=True)
    finally:app.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--mode',choices=('offline','paced','serve','render'),required=True)
    parser.add_argument('--endpoint',type=Path);parser.add_argument('--no-ros-env',action='store_true')
    args=parser.parse_args();repo=prepare(args)
    if args.mode=='render':render(args,repo)
    elif args.mode=='serve':serve(args)
    else:cached(args)


if __name__=='__main__':main()
