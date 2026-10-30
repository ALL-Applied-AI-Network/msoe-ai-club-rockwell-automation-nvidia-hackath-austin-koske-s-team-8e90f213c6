"""Matched GB10 inference for frozen SAM-free fluid heads; no real fitting."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time


ARMS=['unet','dino_rgb','dino_paired']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def main(a):
    import fcntl
    import os
    import shutil
    if not (a.root/'completed.json').exists():raise ValueError('Both synthetic fits must finish first')
    frozen=read(a.root/'fitted-v1/all-models-frozen.json');training=read(a.root/'experiment-plan.json')
    assert sha(a.root/'experiment-plan.json')==frozen['plan_sha256']
    for name,digest in training['helpers'].items():assert sha(a.root/name)==digest
    for name,digest in training['backbone_files'].items():assert sha(a.root/'pretrained'/name)==digest
    assert sha(a.root/'training/manifest.json')==training['training_manifest_sha256']
    assert sha(a.root/'baseline/best.pt')==training['baseline_checkpoint_sha256']
    models={arm:dict(checkpoint_sha256=frozen['models'][mode]['checkpoint_sha256'],**{'mode':mode}) for arm,mode in [('dino_rgb','rgb'),('dino_paired','paired')]}
    models['unet']=dict(checkpoint_sha256=training['baseline_checkpoint_sha256'],mode='paired_diff_gain')
    for arm,mode in [('dino_rgb','rgb'),('dino_paired','paired')]:assert sha(a.root/'fitted-v1'/mode/'best.pt')==models[arm]['checkpoint_sha256']
    probe=a.v1/'mask-probe-v1';previous=read(probe/'plan.json');runtime=read(probe/'runtime.json')
    vial=a.v1/'checkpoint.pt';vmeta=read(vial.with_suffix('.json'))
    assert not vmeta['train_runs'] and not vmeta['development_runs'] and sha(vial)==vmeta['checkpoint_sha256']
    a.out.mkdir(parents=True,exist_ok=False);(a.out/'helpers').mkdir();(a.out/'frames').mkdir()
    helpers=[Path(__file__),a.root/'dinov3_liquid_mask.py',a.root/'train_pair_masks.py',a.v1/'evaluate_pair_masks.py',a.v1/'liquid_agreement_metrics.py']
    for p in helpers:shutil.copyfile(p,a.out/'helpers'/p.name)
    primary=max(['rgb','paired'],key=lambda m:frozen['models'][m]['synthetic_development']['selection_score'])
    plan=dict(schema='dinov3-direct-mask-comparison/v1',arms=ARMS,models=models,
        training_plan_sha256=frozen['plan_sha256'],all_models_frozen_sha256=sha(a.root/'fitted-v1/all-models-frozen.json'),
        source_video_sha256=previous['source_video_sha256'],source_run=4,run5_used=False,cycles=previous['cycles'],
        qa_config=runtime['qa_config'],qa_code_sha256=runtime['qa_code_sha256'],vial_sha256=sha(vial),
        mask_threshold=.5,presence_threshold=.5,frames=264,warmup_frames=264,measured_passes=2,
        primary_dino_mode=primary,primary_selection='Synthetic development selection score only, frozen before real inference',
        reference='First exact source frame of each predefined clip; unreviewed, not certified empty or same stationary vial.',
        input='Native112x192 station ROI at full source resolution; no resizing to224',
        timing='Fresh vial inference, liquid mask/presence and unchanged QA per arm/frame; GPU synchronized; excludes decode/transport/render/browser/serialization.',
        helper_sha256={p.name:sha(p) for p in helpers},training=False,SAM_used=False,app_changed=False,
        limits=['264 correlated Run4 frames across six previously inspected development clips; no blind test.',
                'QA availability is sampling-gate success, not mask, onset or quality accuracy.',
                'No human-approved real masks or first-visible-liquid times; real accuracy unavailable.',
                'DINO uses generic pretraining; U-Net starts random. Same synthetic CSI data, differing model families.',
                'Reference qualification and fixed station ROI remain deployment dependencies.'])
    write(a.out/'plan.json',plan);write(a.out/'queued.json',dict(pid=os.getpid(),time=time.time()))
    lock=(Path.home()/'ai-club-tools/spark-gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    import av
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from ultralytics import YOLO
    sys.path.insert(0,str(a.v1/'bundle/repo'));sys.path.insert(0,str(a.out/'helpers'))
    from dinov3_liquid_mask import DinoLiquidMask
    from train_pair_masks import build_model,ROI,ANCHOR
    from evaluate_pair_masks import infer,largest
    from replay.config import load_config
    from replay.geometry import encode_mask,decode_mask
    from replay.measurement import Extractor
    torch.set_num_threads(4);cv2.setNumThreads(4)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    for name,digest in plan['qa_code_sha256'].items():assert sha(a.v1/'bundle/repo/replay'/name)==digest
    config=load_config(a.base/'bundle-v1/repo/replay/examples/bootstrap-vision.json');assert config.to_dict()==plan['qa_config']
    source=a.v1/'bundle/evaluation/vial_fill_video_4.mp4';assert sha(source)==plan['source_video_sha256']
    rows=[json.loads(s) for s in (probe/'frames.jsonl').read_text().splitlines()];by_frame={r['source_frame']:r for r in rows};frames={}
    with av.open(str(source)) as container:
        for index,frame in enumerate(container.decode(video=0)):
            if index>max(by_frame):break
            if index not in by_frame:continue
            rgb=frame.to_ndarray(format='rgb24');row=by_frame[index]
            assert hashlib.sha256(rgb.tobytes()).hexdigest()==row['source_rgb_sha256']
            assert frame.pts==row['pts'] and str(frame.time_base)==row['time_base'];frames[index]=rgb
            Image.fromarray(rgb).save(a.out/'frames'/f'f{index:06d}.png')
    assert len(frames)==264
    networks={}
    networks['unet']=build_model(6).cuda().eval()
    networks['unet'].load_state_dict(torch.load(a.root/'baseline/best.pt',map_location='cpu',weights_only=True)['model_state'])
    for arm,mode in [('dino_rgb','rgb'),('dino_paired','paired')]:
        model=DinoLiquidMask(a.root/'pretrained',mode).cuda().eval()
        model.load_decoder(torch.load(a.root/'fitted-v1'/mode/'best.pt',map_location='cpu',weights_only=True)['decoder_state']);networks[arm]=model
    vial_model=YOLO(str(vial))
    write(a.out/'runtime.json',dict(torch=torch.__version__,transformers=__import__('transformers').__version__,
        device=torch.cuda.get_device_name(),opencv=cv2.__version__,source_RGB_and_native_PTS_verified=True,
        cudnn_deterministic=torch.backends.cudnn.deterministic,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,local_gpu_used=False,SAM_loaded=False))
    class Adapter:
        def set(self,rgb,predictions,proposals):
            self.digest=hashlib.sha256(rgb.tobytes()).hexdigest();self.predictions=copy.deepcopy(predictions);self.proposals=copy.deepcopy(proposals)
            for p in self.predictions:p['box']=[int(v) for v in p['box']]
        def check(self,rgb):
            if hashlib.sha256(rgb.tobytes()).hexdigest()!=self.digest:raise ValueError('Stale frame proposals')
        def detect(self,rgb):self.check(rgb);return copy.deepcopy(self.predictions)
        def propose(self,rgb,vial):self.check(rgb);return copy.deepcopy(self.proposals)
    def states():
        result={}
        for arm in ARMS:
            adapter=Adapter();extractor=Extractor(config,detector=adapter);extractor.geometry.segmenter=adapter;result[arm]=(adapter,extractor)
        return result
    def dino_infer(model,rgb,reference):
        x,y,r,b=ROI;aa,c,d,e=ANCHOR
        cur=torch.from_numpy(rgb[y:b,x:r].transpose(2,0,1).copy()).float().cuda()[None]/255
        ref=torch.from_numpy(reference[y:b,x:r].transpose(2,0,1).copy()).float().cuda()[None]/255
        ca=torch.tensor(np.median(rgb[c:e,aa:d],axis=(0,1)),dtype=torch.float32,device='cuda')[None]/255
        ra=torch.tensor(np.median(reference[c:e,aa:d],axis=(0,1)),dtype=torch.float32,device='cuda')[None]/255
        with torch.inference_mode():logits,p=model(cur,ref,ca,ra)
        mask=np.zeros((480,640),np.uint8);mask[y:b,x:r]=(logits[0].sigmoid().cpu().numpy()>=.5).astype(np.uint8)
        return mask,float(p.sigmoid()[0])
    def run(arm,row,state):
        adapter,extractor=state;rgb=frames[row['source_frame']];reference=frames[row['reference_frame']]
        torch.cuda.synchronize();start=time.perf_counter()
        native=vial_model.predict([rgb[:,:,::-1].copy()],imgsz=640,rect=False,conf=.35,iou=.45,max_det=300,device='cuda:0',verbose=False,save=False)[0]
        vials=[dict(class_id=0,box=p[:4].tolist(),confidence=float(p[4])) for p in native.boxes.data.cpu().numpy() if int(p[5])==0]
        torch.cuda.synchronize();vial_end=time.perf_counter()
        if arm=='unet':mask,p=infer(networks[arm],'paired_diff_gain',rgb,reference)
        else:mask,p=dino_infer(networks[arm],rgb,reference)
        component,box=largest(mask);accepted=box is not None and p>=.5
        liquids=[] if not accepted else [dict(class_id=1,box=box,confidence=p)]
        proposals=[] if not accepted else [dict(**liquids[0],candidate_mask=encode_mask(component))]
        torch.cuda.synchronize();liquid_end=time.perf_counter()
        adapter.set(rgb,vials+liquids,proposals);measurement=extractor.measure(rgb);end=time.perf_counter()
        measurement['geometry'].update(model_sha256=models[arm]['checkpoint_sha256'],vial_model_sha256=plan['vial_sha256'],
            segmentation_model_sha256=models[arm]['checkpoint_sha256'],liquid_proposal_method=arm)
        gated=component if accepted else np.zeros_like(mask)
        return dict(source_frame=row['source_frame'],case_id=row['case_id'],pts=row['pts'],time_base=row['time_base'],
            source_rgb_sha256=row['source_rgb_sha256'],reference_frame=row['reference_frame'],reference_rgb_sha256=row['reference_rgb_sha256'],
            vials=vials,liquid_predictions=liquids,raw_mask=encode_mask(mask),component_mask=encode_mask(component),gated_mask=encode_mask(gated),
            proposal_available=accepted,mask_pixels=int(mask.sum()),presence_score=p,measurement=measurement,
            baseline_presence_delta=abs(p-row['modes']['paired_diff_gain']['presence']) if arm=='unet' else None,
            timing_ms=dict(vial=(vial_end-start)*1000,liquid=(liquid_end-vial_end)*1000,qa=(end-liquid_end)*1000,total=(end-start)*1000))
    measured={arm:[] for arm in ARMS};first={};vial_checks=0
    for repeat in [-1,0,1]:
        state=states();case=None;records={arm:[] for arm in ARMS};started=time.perf_counter()
        order=ARMS if repeat!=1 else list(reversed(ARMS))
        with (a.out/('warmup.jsonl' if repeat<0 else f'pass-{repeat+1}.jsonl')).open('w') as log:
            for n,row in enumerate(rows):
                if case!=row['case_id']:state=states();case=row['case_id']
                current={arm:run(arm,row,state[arm]) for arm in order}
                assert current['unet']['vials']==current['dino_rgb']['vials']==current['dino_paired']['vials'];vial_checks+=1
                assert current['unet']['baseline_presence_delta']<1e-4
                for arm in ARMS:
                    records[arm].append(current[arm])
                    if repeat>=0:measured[arm].append(current[arm])
                log.write(json.dumps(dict(arms=current),allow_nan=False)+'\n')
                if n%66==65:print(json.dumps(dict(pass_id=repeat+1,frames=n+1,elapsed_s=time.perf_counter()-started)),flush=True)
        if repeat==0:first=records
    comparison={};repeats={};pairs={}
    for arm,records in measured.items():
        aa,bb=records[:264],records[264:]
        repeats[arm]={key:sum(x[key]!=y[key] for x,y in zip(aa,bb)) for key in ['gated_mask','component_mask','presence_score']}
        repeats[arm]['measurement_valid']=sum(x['measurement']['measurement_valid']!=y['measurement']['measurement_valid'] for x,y in zip(aa,bb))
        comparison[arm]=dict(source_frames=264,measured_samples=len(records),proposal_frames=sum(r['proposal_available'] for r in aa),
            qa_pass=sum(r['measurement']['measurement_valid'] for r in aa),
            latency_ms={key:dict(median=float(np.median([r['timing_ms'][key] for r in records])),p95=float(np.percentile([r['timing_ms'][key] for r in records],95))) for key in ['vial','liquid','qa','total']},
            per_clip={case:dict(frames=sum(r['case_id']==case for r in aa),proposals=sum(r['case_id']==case and r['proposal_available'] for r in aa),
                qa_pass=sum(r['case_id']==case and r['measurement']['measurement_valid'] for r in aa)) for case in sorted({r['case_id'] for r in aa})})
        if arm!='unet':
            base=first['unet'];pairs[arm]=dict(qa_gains=sum(not x['measurement']['measurement_valid'] and y['measurement']['measurement_valid'] for x,y in zip(base,aa)),
                qa_losses=sum(x['measurement']['measurement_valid'] and not y['measurement']['measurement_valid'] for x,y in zip(base,aa)))
    report=dict(arms=comparison,repeat_differences=repeats,versus_unet=pairs,identical_vial_frame_checks=vial_checks,
        baseline_reproduced_qa=comparison['unet']['qa_pass']==105,accuracy=None,onset_accuracy=None,SAM_used=False,training=False,
        run5_used=False,app_changed=False,limits=plan['limits'])
    write(a.out/'comparison.json',report)
    write(a.out/'completed.json',dict(artifacts={str(p.relative_to(a.out)):sha(p) for p in sorted(a.out.rglob('*')) if p.is_file()},
        local_gpu_used=False,run5_used=False,app_changed=False))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['root','v1','base','out']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    try:main(a)
    except BaseException as error:
        if a.out.is_dir():write(a.out/'failed.json',dict(type=type(error).__name__,error=str(error)))
        raise
