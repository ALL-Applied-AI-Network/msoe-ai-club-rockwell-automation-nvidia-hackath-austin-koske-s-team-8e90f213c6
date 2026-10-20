"""Diagnostic liquid-only comparison with a fixed baseline vial detector."""
import argparse
import json
from pathlib import Path
import time

import av
import cv2
import numpy as np

from evaluate_crop_contrast import score,videos
from evaluate_epoch_augmentation import fill_clips
from liquid_specialists_data import ARMS,WEIGHTS,read,write,sha,BASELINE


class SpecialistPair:
    def __init__(self,baseline,root,arm,device):
        import torch
        from ultralytics import YOLO,RTDETR
        torch.set_num_threads(4)
        self.torch=torch;self.device='cuda:'+device if str(device).isdigit() else device;self.root=root;self.arm=arm;self.roi=[0,0,640,480]
        self.baseline=YOLO(baseline);path=root/'models'/arm
        self.metadata=read(path/'model.json');self.dino=arm=='dinov3-all'
        if self.dino:
            from dino_liquid import load
            self.candidate=load(root/'pretrained/dinov3-convnext-small',path/'best.pt',self.device)
        else:self.candidate=(RTDETR if arm=='rtdetr-all' else YOLO)(path/'best.pt')
        if not self.dino and self.candidate.names!={0:'colored_liquid'}:raise ValueError('Candidate is not the liquid-only class contract')
        self.candidate_title=arm+' | liquid only, fixed vial detector'
        self.inference_frames=0;self.inference_s=0.

    def predict(self,rgbs,confidence):
        from torchvision.ops import nms
        images=[cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR) for rgb in rgbs]
        start=time.perf_counter()
        if self.torch.cuda.is_available():self.torch.cuda.synchronize()
        old=self.baseline.predict(images,imgsz=640,rect=False,conf=confidence,iou=.45,max_det=300,
            device=self.device,quantize=None,augment=False,agnostic_nms=False,verbose=False,save=False)
        baseline=[[dict(class_id=int(p[5]),box=p[:4].tolist(),confidence=float(p[4])) for p in r.boxes.data.cpu().numpy()] for r in old]
        native=[]
        if self.dino:
            with self.torch.no_grad():
                for offset in range(0,len(rgbs),4):
                    tensors=[self.torch.from_numpy(rgb.copy()).permute(2,0,1).float().to(self.device)/255 for rgb in rgbs[offset:offset+4]]
                    outputs=self.candidate(tensors)
                    native.extend([dict(boxes=p['boxes'].cpu(),scores=p['scores'].cpu()) for p in outputs])
        else:
            for offset in range(0,len(images),16):
                outputs=self.candidate.predict(images[offset:offset+16],imgsz=640,rect=False,conf=confidence,iou=.45,max_det=300,
                    device=self.device,quantize=None,augment=False,agnostic_nms=False,verbose=False,save=False)
                native.extend([dict(boxes=r.boxes.xyxy.cpu(),scores=r.boxes.conf.cpu()) for r in outputs])
        result=[]
        for old,p in zip(baseline,native):
            keep=p['scores']>=confidence;boxes=p['boxes'][keep];scores=p['scores'][keep]
            # Explicit shared final NMS; RT-DETR is normally NMS-free.
            selected=nms(boxes,scores,.45)[:300]
            liquid=[dict(class_id=1,box=boxes[i].tolist(),confidence=float(scores[i])) for i in selected]
            result.append([p for p in old if p['class_id']==0]+liquid)
        if len(result)!=len(rgbs):raise ValueError('Incomplete candidate batch')
        if self.torch.cuda.is_available():self.torch.cuda.synchronize()
        self.inference_s+=time.perf_counter()-start;self.inference_frames+=len(rgbs)
        return [baseline,result]


def negative_diagnostics(out):
    rows=read(out/'scoring-predictions.json')
    negative=[r for r in rows if not any(t['class_id']==1 for t in r['targets'])]
    result=dict(provisional_negative_frames=len(negative),confirmed_empty=False,
        reason='Vial teacher target exists but liquid target absent; could include missing liquid annotations',
        counts={arm:sum(any(p['class_id']==1 and p['confidence']>=.35 for p in r[arm]) for r in negative)
                for arm in ('baseline','candidate')})
    write(out/'provisional-negative-diagnostics.json',result);return result


def main(a):
    if a.out.exists():raise ValueError('Use fresh evaluation output')
    root=a.experiment;plan=read(root/'experiment-plan.json')
    baseline_onnx=Path(__file__).resolve().parents[2]/'replay/models/vial-bootstrap.onnx'
    if sha(baseline_onnx)!=BASELINE or sha(a.baseline)!='10856c18a8a5c485e832adcc6b841b739e726a969dc043fe8db64c49163bb361':raise ValueError('Baseline changed')
    if plan['excluded_from_fit_and_selection']!=[5]:raise ValueError('Run 5 exclusion changed')
    for name,expected in WEIGHTS.items():
        if sha(root/'pretrained'/name)!=expected:raise ValueError('Backbone/initialization bytes changed')
    a.out.mkdir(parents=True);reports={};sources=None
    for arm in a.arms:
        path=root/'models'/arm;m=read(path/'model.json')
        if (m['arm']!=arm or m['family']!=ARMS[arm]['family'] or m['excluded_runs']!=[5]
                or m['experiment_plan_sha256']!=sha(root/'experiment-plan.json')
                or m['preparation_sha256']!=sha(root/'preparation.json')
                or sha(path/'best.pt')!=m['checkpoint_sha256']
                or m['data_provenance_sha256']!=plan['data'][ARMS[arm]['dataset']]['provenance_sha256']):
            raise ValueError('Frozen checkpoint ancestry differs')
        out=a.out/arm;out.mkdir();pair=SpecialistPair(a.baseline,root,arm,a.device)
        metrics=score(pair,a.teacher,out)
        if sources is None:sources=metrics['sources']
        if sources!=metrics['sources']:raise ValueError('Scoring inventory differs')
        # Vial predictions are copied from baseline, never scored as a model improvement.
        if metrics['baseline']['per_class'][0]!=metrics['candidate']['per_class'][0]:raise ValueError('Fixed vial proposals changed')
        negative=negative_diagnostics(out)
        video=videos(pair,a.data,a.cache,a.cache_sha256,BASELINE,out);fill=fill_clips(out);checks=[]
        for name,expected in [('run5-full-comparison.mp4',video['full_video_frames']),('run5-early-pours.mp4',video['early_video_frames']),('run5-two-fills.mp4',364)]:
            with av.open(str(out/name)) as movie:count=sum(1 for _ in movie.decode(video=0))
            if count!=expected:raise ValueError('Video decode failed')
            checks.append(dict(file=name,frames=count,sha256=sha(out/name)))
        reports[arm]=dict(model=m,teacher_agreement={k:v for k,v in metrics.items() if k!='sources'},
            video=video,full_fill_video=fill,decoded_videos=checks,provisional_negatives=negative,
            timing=dict(frames=pair.inference_frames,paired_inference_s=pair.inference_s,
                scope='Includes baseline plus candidate, conversion/NMS, mixed batch sizes; not standalone real-time latency'))
        write(out/'completed-arm.json',reports[arm]);write(a.out/'progress.json',dict(completed=list(reports)))
        del pair
    write(a.out/'comparison.json',dict(schema_version=1,kind='liquid_specialist_diagnostics',arms=reports,
        experiment_plan_sha256=sha(root/'experiment-plan.json'),fixed_vial_detector=True,
        detector_promoted=False,measurement_qa_changed=False,run5_fit_or_selection=False,
        report_primary_metric='Liquid AP50-95, not combined-class AP; vial branch copied from baseline',
        confidence_warning='0.35 is unchanged for diagnostics, but confidence is not calibrated across families',
        postprocessing='Shared final liquid NMS IoU0.45; RT-DETR native output is normally NMS-free',limits=plan['limits']))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('experiment','baseline','teacher','data','cache','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--cache-sha256',required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--arms',nargs='+',choices=list(ARMS),default=list(ARMS));main(p.parse_args())
