"""GB10 actual vial detection -> direct temporal mask -> unchanged QA benchmark.

Frozen development candidate only. Cached exact RGB inputs; no rendering,
transport or browser latency. Does not promote or change the operator app.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def write(p,v): Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def main(a):
    import fcntl
    import os
    a.out.mkdir(parents=True,exist_ok=False)
    write(a.out/'plan.json',dict(helper_sha256=sha(__file__),mode='paired_diff_gain',
        inputs='Same264 exact Run4 RGB frames/references as completed mask probe.',
        warmup_frames=30,passes=2,thresholds=dict(mask=.5,presence=.5,vial_conf=.35,vial_iou=.45),
        stages=['vial_detector','temporal_mask','unchanged_QA'],training=False,run5_used=False,
        excluded='Decode, rendering, network, API, browser, operator time. Cached inputs; no production FPS claim.'))
    write(a.out/'queued.json',dict(pid=os.getpid(),queued=time.time()))
    lock=(Path.home()/'ai-club-tools/spark-gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX)
    import av
    import numpy as np
    import torch
    import cv2
    from ultralytics import YOLO
    sys.path.insert(0,str(a.helper.parent));sys.path.insert(0,str(a.bundle/'repo'))
    from train_pair_masks import build_model
    from evaluate_pair_masks import infer,largest
    from replay.geometry import encode_mask
    from replay.measurement import Extractor
    from replay.config import load_config
    torch.set_num_threads(4);cv2.setNumThreads(4)
    meta=read(a.models/'paired_diff_gain/frozen-model.json')
    assert sha(a.models/'paired_diff_gain/best.pt')==meta['checkpoint_sha256']
    assert sha(a.helper)==meta['helper_sha256']
    assert not meta['training_real_runs'] and not meta['selection_real_runs'] and not meta['run5_used']
    runtime=read(a.probe/'runtime.json')
    for name,digest in runtime['qa_code_sha256'].items():assert sha(a.bundle/'repo/replay'/name)==digest
    vial_meta=read(a.vial.with_suffix('.json'))
    assert sha(a.vial)==vial_meta['checkpoint_sha256']==runtime['vial_model_sha256']
    assert not vial_meta['train_runs'] and not vial_meta['development_runs']
    plan=read(a.probe/'plan.json');assert sha(a.video)==plan['source_video_sha256']
    rows=[json.loads(s) for s in (a.probe/'frames.jsonl').read_text().splitlines()]
    wanted={r['source_frame']:r for r in rows};frames={}
    with av.open(str(a.video)) as container:
        for index,frame in enumerate(container.decode(video=0)):
            if index>max(wanted):break
            if index in wanted:
                rgb=frame.to_ndarray(format='rgb24');row=wanted[index]
                assert hashlib.sha256(rgb.tobytes()).hexdigest()==row['source_rgb_sha256']
                assert frame.pts==row['pts'] and str(frame.time_base)==row['time_base']
                frames[index]=rgb
    assert len(frames)==264
    model=build_model(6).cuda().eval()
    model.load_state_dict(torch.load(a.models/'paired_diff_gain/best.pt',map_location='cpu',weights_only=True)['model_state'])
    vial_model=YOLO(str(a.vial));config=load_config(a.config)
    assert config.to_dict()==runtime['qa_config']
    class Adapter:
        def set(self,rgb,predictions,proposals):
            self.digest=hashlib.sha256(rgb.tobytes()).hexdigest();self.predictions=copy.deepcopy(predictions);self.proposals=copy.deepcopy(proposals)
            for p in self.predictions:p['box']=[int(v) for v in p['box']]
        def check(self,rgb):
            if hashlib.sha256(rgb.tobytes()).hexdigest()!=self.digest:raise ValueError('Stale frame')
        def detect(self,rgb):self.check(rgb);return copy.deepcopy(self.predictions)
        def propose(self,rgb,vial):self.check(rgb);return copy.deepcopy(self.proposals)
    def new_extractor():
        adapter=Adapter();extractor=Extractor(config,detector=adapter);extractor.geometry.segmenter=adapter
        return adapter,extractor
    def run(row,adapter,extractor):
        rgb=frames[row['source_frame']];ref=frames[row['reference_frame']]
        torch.cuda.synchronize();start=time.perf_counter()
        native=vial_model.predict([rgb[:,:,::-1].copy()],imgsz=640,rect=False,conf=.35,iou=.45,max_det=300,device='cuda:0',verbose=False,save=False)[0]
        vials=[dict(class_id=0,box=p[:4].tolist(),confidence=float(p[4])) for p in native.boxes.data.cpu().numpy() if int(p[5])==0]
        torch.cuda.synchronize();vial_end=time.perf_counter()
        mask,p=infer(model,'paired_diff_gain',rgb,ref);component,box=largest(mask);accepted=box is not None and p>=.5
        proposal=[] if not accepted else [dict(class_id=1,box=box,confidence=p,candidate_mask=encode_mask(component))]
        predictions=vials+([] if not accepted else [dict(class_id=1,box=box,confidence=p)])
        torch.cuda.synchronize();mask_end=time.perf_counter()
        adapter.set(rgb,predictions,proposal);measurement=extractor.measure(rgb)
        end=time.perf_counter()
        measurement['geometry'].update(liquid_proposal_method='paired_diff_gain',
            segmentation_model_sha256=meta['checkpoint_sha256'],model_sha256=meta['checkpoint_sha256'],
            vial_model_sha256=runtime['vial_model_sha256'],mask_probability_threshold=.5)
        return dict(case_id=row['case_id'],source_frame=row['source_frame'],presence=p,
            presence_gate_passed=accepted,measurement=measurement,
            timing_ms=dict(vial_detector=(vial_end-start)*1000,temporal_mask=(mask_end-vial_end)*1000,
                unchanged_QA=(end-mask_end)*1000,total=(end-start)*1000))
    adapter,extractor=new_extractor()
    for row in rows[:30]:run(row,adapter,extractor)
    reports=[];all_records=[]
    for repeat in range(2):
        records=[];case=None
        for row in rows:
            if case!=row['case_id']:case=row['case_id'];adapter,extractor=new_extractor()
            result=run(row,adapter,extractor)
            result['qa_matches_completed_probe']=result['measurement']['measurement_valid']==row['modes']['paired_diff_gain']['measurement']['measurement_valid']
            result['final_mask_matches_completed_probe']=result['measurement']['geometry']['mask']==row['modes']['paired_diff_gain']['measurement']['geometry']['mask']
            result['color_matches_completed_probe']=result['measurement']['lab']==row['modes']['paired_diff_gain']['measurement']['lab']
            assert abs(result['presence']-row['modes']['paired_diff_gain']['presence'])<1e-4
            records.append(result)
        write(a.out/('pass-'+str(repeat+1)+'.json'),records);all_records+=records
        reports.append(dict(pass_id=repeat+1,frames=len(records),qa_pass=sum(r['measurement']['measurement_valid'] for r in records),
            qa_decision_differences=sum(not r['qa_matches_completed_probe'] for r in records),
            final_mask_differences=sum(not r['final_mask_matches_completed_probe'] for r in records),
            color_differences=sum(not r['color_matches_completed_probe'] for r in records)))
    timings={stage:dict(samples=len(all_records),median=float(np.median([r['timing_ms'][stage] for r in all_records])),
        p95=float(np.percentile([r['timing_ms'][stage] for r in all_records],95))) for stage in ['vial_detector','temporal_mask','unchanged_QA','total']}
    write(a.out/'comparison.json',dict(passes=reports,timings_ms=timings,device=torch.cuda.get_device_name(),torch=torch.__version__,
        liquid_model_sha256=meta['checkpoint_sha256'],vial_model_sha256=sha(a.vial),qa_config_unchanged=True,
        source_rgb_and_pts_verified=True,source_frames=264,measured_samples=len(all_records),sam_used=False,
        accuracy=None,onset_accuracy=None,app_changed=False,local_gpu_used=False))
    write(a.out/'completed.json',dict(artifacts={p.name:sha(p) for p in a.out.iterdir() if p.is_file()},run5_used=False,app_changed=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['models','helper','probe','bundle','video','vial','config','out']:p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
