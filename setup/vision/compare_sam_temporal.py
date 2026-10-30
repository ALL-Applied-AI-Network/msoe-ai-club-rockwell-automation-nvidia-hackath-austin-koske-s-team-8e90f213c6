"""Matched GB10 comparison: legacy/synthetic detector-guided SAM vs temporal mask.

Practical system comparison, not a pure architecture ablation. No fitting,
checkpoint selection, threshold tuning, human label approval or app changes.
"""
import argparse
import copy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
import time


ARMS=['legacy_sam','synthetic_sam','temporal']
SAM_SHA='7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69'
LEGACY_SHA='9479fb13ed9c7ebd815c314e8219ba3b86c3ffa267f758f714a3058f9ac3a343'
SYNTH_SHA='07f5d23da43c35e4203d65105c2f8a020fb895dd74852802f415ce9c9d220efa'
TEMPORAL_SHA='6714fe32191d8eba230aa87246f38a25c2906c619d60d7ba30c9e60205352eef'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def main(a):
    import fcntl
    import os
    a.out.mkdir(parents=True,exist_ok=False);(a.out/'helpers').mkdir();(a.out/'draft-masks').mkdir()
    sources=[Path(__file__),a.v1/'train_pair_masks.py',a.v1/'evaluate_pair_masks.py',a.v1/'liquid_agreement_metrics.py',
        a.v1/'bundle/repo/setup/vision/pool_prompt_policy.py',a.v1/'bundle/repo/setup/vision/train_early_liquid.py']
    for p in sources:(a.out/'helpers'/p.name).write_bytes(p.read_bytes())
    benchmark=a.base/'bundle-v1';legacy=benchmark/'tools/results/early-liquid-20261020'
    synthetic=a.v2/'long-original-v1';vial=a.v1/'checkpoint.pt'
    checkpoints=dict(legacy_sam=legacy/'models/fasterrcnn-resnet50/best.pt',synthetic_sam=synthetic/'epoch-03.pt',
        temporal=a.v1/'masks-v1/paired_diff_gain/best.pt',sam=benchmark/'tools/results/prompted-liquid-assets-20261020/sam2.1_t.pt',vial=vial)
    for name,digest in [('legacy_sam',LEGACY_SHA),('synthetic_sam',SYNTH_SHA),('temporal',TEMPORAL_SHA),('sam',SAM_SHA)]:
        if sha(checkpoints[name])!=digest:raise ValueError('Frozen model differs: '+name)
    vmeta=read(vial.with_suffix('.json'));assert not vmeta['train_runs'] and not vmeta['development_runs']
    assert sha(vial)==vmeta['checkpoint_sha256']
    probe=a.v1/'mask-probe-v1';previous=read(probe/'plan.json');runtime=read(probe/'runtime.json')
    models={k:dict(file=str(p),sha256=sha(p)) for k,p in checkpoints.items()}
    plan=dict(schema='sam-temporal-comparison/v1',arms=ARMS,models=models,
        source_video_sha256=previous['source_video_sha256'],source_run=4,run5_used=False,
        cycles=previous['cycles'],qa_config=runtime['qa_config'],qa_code_sha256=runtime['qa_code_sha256'],
        helper_sha256={p.name:sha(p) for p in sources},frames=264,warmup='Full264-frame sequence for all arms.',passes=2,
        prompt='Existing pool_prompts plus CausalPromptBridge, unchanged thresholds; reset per clip.',
        temporal_reference='First exact source frame of each clip, unreviewed; never supplied to SAM.',
        legacy_training='Legacy R-CNN CSI Runs1-3 fitting / Run4 selection, generic COCO; generic SAM2.1. Comparator only.',
        synthetic_sam_training='Existing fixed epoch3 GB10 synthetic-only R-CNN control (614 images); generic COCO and SAM2.1.',
        temporal_training='Frozen epoch40 synthetic-only U-Net (144 images), random initialization.',
        same_vial='Shared synthetic YOLO11n checkpoint, actual fresh inference in each arm, outputs equality checked.',
        accuracy=None,onset_accuracy=None,training=False,app_changed=False,
        limits=['Practical system comparison; training corpora, initialization and checkpoint selection differ.',
            'Legacy comparator selected on inspected Run4; all clips are development evidence, not blind tests.',
            'QA availability, model agreement and tall-mask counts are not accuracy.',
            'Cached-input timing excludes decode, rendering, transport, browser and serialization.'])
    write(a.out/'plan.json',plan);write(a.out/'queued.json',dict(pid=os.getpid(),time=time.time()))
    lock=(Path.home()/'ai-club-tools/spark-gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX)
    import av
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from ultralytics import YOLO,SAM
    sys.path.insert(0,str(a.v1/'bundle/repo'))
    sys.path.insert(0,str(a.v1/'bundle/repo/setup/vision'))
    sys.path.insert(0,str(a.out/'helpers'))
    from train_pair_masks import build_model
    from evaluate_pair_masks import infer,largest
    from train_early_liquid import build_resnet
    from pool_prompt_policy import pool_prompts,CausalPromptBridge,POLICY,BRIDGE_POLICY
    from replay.config import load_config
    from replay.geometry import encode_mask,decode_mask
    from replay.measurement import Extractor
    torch.set_num_threads(4);cv2.setNumThreads(4)
    for name,digest in plan['qa_code_sha256'].items():assert sha(a.v1/'bundle/repo/replay'/name)==digest
    config=load_config(benchmark/'repo/replay/examples/bootstrap-vision.json');assert config.to_dict()==plan['qa_config']
    source=a.v1/'bundle/evaluation/vial_fill_video_4.mp4';assert sha(source)==plan['source_video_sha256']
    source_rows=[json.loads(s) for s in (probe/'frames.jsonl').read_text().splitlines()];by_frame={r['source_frame']:r for r in source_rows};frames={}
    with av.open(str(source)) as container:
        for index,frame in enumerate(container.decode(video=0)):
            if index>max(by_frame):break
            if index not in by_frame:continue
            rgb=frame.to_ndarray(format='rgb24');row=by_frame[index]
            assert hashlib.sha256(rgb.tobytes()).hexdigest()==row['source_rgb_sha256']
            assert frame.pts==row['pts'] and str(frame.time_base)==row['time_base'];frames[index]=rgb
    assert len(frames)==264
    networks={}
    for arm,root in [('legacy_sam',legacy),('synthetic_sam',synthetic)]:
        net=build_resnet(root);net.load_state_dict(torch.load(checkpoints[arm],map_location='cpu',weights_only=True)['model_state'])
        networks[arm]=net.cuda().eval()
    temporal=build_model(6).cuda().eval();temporal.load_state_dict(torch.load(checkpoints['temporal'],map_location='cpu',weights_only=True)['model_state'])
    vial_model=YOLO(str(vial));sam=SAM(str(checkpoints['sam']))
    write(a.out/'runtime.json',dict(torch=torch.__version__,device=torch.cuda.get_device_name(),opencv=cv2.__version__,
        policy=POLICY,bridge_policy=BRIDGE_POLICY,source_frames=264,source_RGB_and_native_PTS_verified=True,
        cudnn_deterministic=torch.backends.cudnn.deterministic,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,local_gpu_used=False))
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
            adapter=Adapter();extractor=Extractor(config,detector=adapter);extractor.geometry.segmenter=adapter
            result[arm]=(adapter,extractor,CausalPromptBridge())
        return result
    def run(arm,row,state):
        adapter,extractor,bridge=state;rgb=frames[row['source_frame']]
        torch.cuda.synchronize();start=time.perf_counter()
        pred=vial_model.predict([rgb[:,:,::-1].copy()],imgsz=640,rect=False,conf=.35,iou=.45,max_det=300,device='cuda:0',verbose=False,save=False)[0]
        vials=[dict(class_id=0,box=p[:4].tolist(),confidence=float(p[4])) for p in pred.boxes.data.cpu().numpy() if int(p[5])==0]
        torch.cuda.synchronize();vial_end=time.perf_counter();liquids=[];proposals=[];raw_masks=[];policy=None;bridge_info=None;p=None;qualities=[]
        detector_end=vial_end;sam_called=False
        if arm=='temporal':
            mask,p=infer(temporal,'paired_diff_gain',rgb,frames[row['reference_frame']]);component,box=largest(mask)
            accepted=box is not None and p>=.5
            if accepted:
                raw_masks=[component.astype(bool)];liquids=[dict(class_id=1,box=box,confidence=p)]
                proposals=[dict(**liquids[0],candidate_mask=encode_mask(component))]
            assert abs(p-row['modes']['paired_diff_gain']['presence'])<1e-4
        else:
            tensor=torch.from_numpy(rgb.copy()).permute(2,0,1).float().cuda()/255
            with torch.inference_mode():native=networks[arm]([tensor])[0]
            liquids=[dict(class_id=1,box=b.tolist(),confidence=float(s)) for b,s,c in zip(native['boxes'].cpu(),native['scores'].cpu(),native['labels'].cpu()) if int(c)==1 and float(s)>=.35]
            torch.cuda.synchronize();detector_end=time.perf_counter()
            policy=pool_prompts(rgb,vials+liquids)
            thumb=cv2.resize(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),(32,24)).astype(float)[10:23,11:21]
            motion=None if extractor.previous_thumb is None else float(np.abs(thumb-extractor.previous_thumb).mean())
            bridge_info=bridge.update(policy,row['source_frame'],float(row['pts']*Fraction(row['time_base'])),motion)
            boxes,points,labels=(bridge_info[k] for k in ['boxes','points','labels'])
            if boxes or points:
                sam_called=True
                result=sam.predict(rgb[:,:,::-1].copy(),bboxes=boxes or None,points=points or None,labels=labels or None,
                    device='cuda:0',conf=0.,imgsz=1024,verbose=False,save=False)[0]
                raw_masks=[] if result.masks is None else [m.astype(bool) for m in result.masks.data.cpu().numpy()]
                qualities=[] if result.masks is None else result.boxes.conf.cpu().tolist()
                if len(raw_masks)!=(len(boxes) if boxes else len(points)):raise ValueError('SAM prompt count differs')
                selected=(vials+liquids)[policy['selected_prediction_index']]['confidence']
                for mask,quality in zip(raw_masks,qualities):
                    yy,xx=np.where(mask)
                    if len(xx):proposals.append(dict(class_id=1,box=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)],
                        confidence=selected,candidate_mask=encode_mask(mask),sam_mask_quality=quality))
        torch.cuda.synchronize();liquid_end=time.perf_counter()
        adapter.set(rgb,vials+liquids,proposals);measurement=extractor.measure(rgb);end=time.perf_counter()
        measurement['geometry'].update(model_sha256=models[arm]['sha256'],vial_model_sha256=models['vial']['sha256'],
            segmentation_model_sha256=models[arm if arm=='temporal' else 'sam']['sha256'],liquid_proposal_method=arm)
        union=np.zeros((480,640),dtype=np.uint8)
        for mask in raw_masks:union|=mask.astype(np.uint8)
        vial_box=measurement['geometry']['vial_box'];ys,xs=np.where(union)
        tall=bool(len(ys) and vial_box and ys.max()-ys.min()+1>.7*(vial_box[3]-vial_box[1]))
        outside=None if not len(ys) or vial_box is None else float(((xs<vial_box[0])|(xs>=vial_box[2])|(ys<vial_box[1])|(ys>=vial_box[3])).mean())
        return dict(source_frame=row['source_frame'],case_id=row['case_id'],pts=row['pts'],time_base=row['time_base'],
            source_rgb_sha256=row['source_rgb_sha256'],vials=vials,liquid_predictions=liquids,raw_masks=[encode_mask(m) for m in raw_masks],
            raw_union=encode_mask(union),proposal_available=bool(proposals),raw_mask_pixels=int(union.sum()),tall_mask=tall,
            fraction_outside_vial=outside,presence_score=p,prompt_policy=policy,bridge=bridge_info,sam_called=sam_called,
            measurement=measurement,timing_ms=dict(vial=(vial_end-start)*1000,liquid_detector=(detector_end-vial_end)*1000,
                mask_and_prompt=(liquid_end-detector_end)*1000,qa=(end-liquid_end)*1000,total=(end-start)*1000))
    all_records={arm:[] for arm in ARMS};measured={arm:[] for arm in ARMS};vial_checks=0
    for repeat in [-1,0,1]:
        state=states();case=None;records={arm:[] for arm in ARMS};started=time.perf_counter()
        order=ARMS if repeat!=1 else list(reversed(ARMS))
        with (a.out/('warmup.jsonl' if repeat<0 else f'pass-{repeat+1}.jsonl')).open('w') as log:
            for n,row in enumerate(source_rows):
                if case!=row['case_id']:state=states();case=row['case_id']
                current={arm:run(arm,row,state[arm]) for arm in order}
                assert current['legacy_sam']['vials']==current['synthetic_sam']['vials']==current['temporal']['vials'],'Vial geometry differs'
                vial_checks+=1
                for arm in ARMS:
                    records[arm].append(current[arm])
                    if repeat>=0:measured[arm].append(current[arm])
                log.write(json.dumps(dict(arms=current),allow_nan=False)+'\n')
                if n%66==65:print(json.dumps(dict(pass_id=repeat+1,frames=n+1,elapsed_s=time.perf_counter()-started)),flush=True)
        if repeat==0:all_records=records
    comparison={};repeats={}
    for arm,records in measured.items():
        first=records[:264];second=records[264:]
        repeats[arm]={key:sum(a['measurement'][key]!=b['measurement'][key] for a,b in zip(first,second)) for key in ['measurement_valid','lab']}
        repeats[arm]['final_masks']=sum(a['measurement']['geometry']['mask']!=b['measurement']['geometry']['mask'] for a,b in zip(first,second))
        comparison[arm]=dict(source_frames=264,measured_samples=len(records),proposal_frames=sum(r['proposal_available'] for r in first),
            qa_pass=sum(r['measurement']['measurement_valid'] for r in first),tall_mask_frames=sum(r['tall_mask'] for r in first),
            sam_invoked_frames=sum(r['sam_called'] for r in first),
            latency_ms={key:dict(median=float(np.median([r['timing_ms'][key] for r in records])),p95=float(np.percentile([r['timing_ms'][key] for r in records],95))) for key in ['vial','liquid_detector','mask_and_prompt','qa','total']},
            invoked_latency_ms=None if not any(r['sam_called'] for r in records) else dict(samples=sum(r['sam_called'] for r in records),
                median=float(np.median([r['timing_ms']['total'] for r in records if r['sam_called']]))))
    paired=[]
    for i,row in enumerate(source_rows):
        masks={arm:decode_mask(all_records[arm][i]['raw_union']).astype(bool) for arm in ARMS}
        overlap={}
        for other in ARMS[:2]:
            a_mask,b_mask=masks[other],masks['temporal'];union=int((a_mask|b_mask).sum())
            overlap[other]=dict(both_nonempty=bool(a_mask.any() and b_mask.any()),proposal_disagreement=all_records[other][i]['proposal_available']!=all_records['temporal'][i]['proposal_available'],
                prediction_iou=float((a_mask&b_mask).sum()/union) if union else 1.)
        paired.append(dict(source_frame=row['source_frame'],case_id=row['case_id'],overlap=overlap))
    write(a.out/'paired-predictions.json',paired)
    # Existing72 exact images are the common review inventory, without generating human labels.
    drafts=read(probe/'review-drafts.json');new_tasks=[]
    for task in drafts['tasks']:
        idx=next(i for i,r in enumerate(source_rows) if r['source_frame']==task['frame_index'])
        out=copy.deepcopy(task);out['drafts']=[]
        for arm in ARMS:
            record=all_records[arm][idx];mask=decode_mask(record['raw_union'])
            rel=f'draft-masks/{task["sample_id"]}-{arm}.png';Image.fromarray(mask*255).save(a.out/rel)
            out['drafts'].append(dict(mode=arm,model_sha256=models[arm]['sha256'],segmentation_model_sha256=models[arm if arm=='temporal' else 'sam']['sha256'],
                mask=rel,mask_sha256=sha(a.out/rel),proposal_available=record['proposal_available'],
                presence_probability=record['presence_score'],status='machine_draft_not_approved'))
        new_tasks.append(out)
    write(a.out/'review-drafts.json',dict(tasks=new_tasks,source_video_sha256=plan['source_video_sha256'],approved_labels=False))
    # Same source frames and playback timeline for every arm.
    video=a.out/'sam-vs-temporal.mp4';writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),5,(1920,560))
    if not writer.isOpened():raise RuntimeError('Video writer failed')
    count=0
    try:
        for i,row in enumerate(source_rows):
            cycle=next(c for c in plan['cycles'] if c['case_id']==row['case_id'])
            if row['source_frame']>=cycle['source_end_frame_exclusive'] or (row['source_frame']-cycle['source_start_frame'])%6:continue
            canvas=np.full((560,1920,3),22,dtype=np.uint8)
            for col,arm in enumerate(ARMS):
                record=all_records[arm][i];rgb=frames[row['source_frame']].copy();mask=decode_mask(record['raw_union']).astype(bool)
                rgb[mask]=(.6*rgb[mask]+.4*np.array([45,255,150])).astype(np.uint8)
                canvas[65:545,col*640:(col+1)*640]=rgb[:,:,::-1]
                cv2.putText(canvas,arm+' | QA '+str(record['measurement']['measurement_valid']),(col*640+8,24),0,.6,(245,245,245),1)
                cv2.putText(canvas,f'{row["case_id"]} native f{row["source_frame"]} | UNREVIEWED',(col*640+8,48),0,.47,(220,220,220),1)
            writer.write(canvas);count+=1
            if row['source_frame']==cycle['provisional_anchor_frame']+30:cv2.imwrite(str(a.out/(row['case_id']+'-comparison.png')),canvas)
    finally:writer.release()
    with av.open(str(video)) as container:decoded=sum(1 for _ in container.decode(video=0))
    assert count==decoded==240
    report=dict(arms=comparison,repeat_differences=repeats,identical_vial_frame_checks=vial_checks,video_frames=decoded,video_fps=5,
        accuracy=None,onset_accuracy=None,approved_labels=False,training=False,run5_used=False,app_changed=False,
        temporal_reproduced_qa=sum(r['measurement']['measurement_valid'] for r in all_records['temporal'])==105,
        limits=plan['limits'])
    write(a.out/'comparison.json',report)
    write(a.out/'completed.json',dict(artifacts={p.name:sha(p) for p in a.out.iterdir() if p.is_file()},
        helpers={p.name:sha(p) for p in (a.out/'helpers').iterdir() if p.is_file()},draft_masks={p.name:sha(p) for p in (a.out/'draft-masks').iterdir() if p.is_file()},
        local_gpu_used=False,run5_used=False,app_changed=False))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['v1','v2','base','out']:p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
