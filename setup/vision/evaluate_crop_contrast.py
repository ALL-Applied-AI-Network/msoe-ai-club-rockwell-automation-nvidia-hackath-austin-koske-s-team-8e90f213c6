"""Paired source-coordinate detector scoring and LOCAL Run 5 result videos."""
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import av
import cv2
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from replay.dataset import Catalog,DEFAULT_DATA,sha256
from replay.proposals import CachedImageDetector
from crop_contrast import enhance_crop,project_boxes,read_labels


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


class Pair:
    def __init__(self,baseline,candidate,roi,device):
        import torch
        from ultralytics import YOLO
        torch.set_num_threads(4)
        self.models=[YOLO(str(baseline)),YOLO(str(candidate))];self.roi=roi;self.device=device

    def predict(self,rgbs,confidence):
        results=[]
        for index,model in enumerate(self.models):
            images=[cv2.cvtColor(enhance_crop(rgb,self.roi) if index else rgb,cv2.COLOR_RGB2BGR) for rgb in rgbs]
            predictions=model.predict(images,imgsz=640,rect=False,conf=confidence,iou=.45,max_det=300,
                                      device=self.device,half=False,augment=False,agnostic_nms=False,verbose=False,save=False)
            batch=[]
            for prediction in predictions:
                boxes=[dict(class_id=int(row[5]),box=[float(v) for v in row[:4]],confidence=float(row[4]))
                       for row in prediction.boxes.data.cpu().numpy()]
                batch.append(project_boxes(boxes,self.roi) if index else boxes)
            results.append(batch)
        return results


def matched(predictions,targets):
    import torch
    from ultralytics.engine.validator import BaseValidator
    from ultralytics.utils.metrics import box_iou
    # The pinned library performs confidence-ordered one-to-one matching at
    # each IoU threshold; metrics are on FULL source boxes, never clipped GT.
    p=torch.tensor([p['box'] for p in predictions],dtype=torch.float32).reshape(-1,4)
    t=torch.tensor([t['box'] for t in targets],dtype=torch.float32).reshape(-1,4)
    return BaseValidator.match_predictions(SimpleNamespace(iouv=torch.linspace(.5,.95,10)),
        torch.tensor([p['class_id'] for p in predictions]),torch.tensor([t['class_id'] for t in targets]),box_iou(t,p)).numpy()


def score(pair,teacher,output):
    from ultralytics.utils.metrics import ap_per_class
    gallery={r['id']:r for r in read(teacher/'gallery.json') if r['camera']=='vial_fill'}
    rows=[r for r in read(teacher/'manifest.json') if r['camera']=='vial_fill' and r['video_id']=='5']
    selected=[r for r in rows if gallery[r['id']]['status']=='review_candidates' and gallery[r['id']]['boxes']]
    statistics=[[],[]];identities=[];outside=0;details=[]
    for offset in range(0,len(selected),32):
        batch=selected[offset:offset+32];rgbs=[];targets=[]
        for row in batch:
            image=teacher/'review-pack/vial_fill/images'/row['source'];label=teacher/'review-pack/vial_fill/labels'/(row['id']+'.txt')
            if sha256(image)!=row['source_sha256']:raise ValueError('Scoring source image changed')
            rgbs.append(cv2.cvtColor(cv2.imread(str(image)),cv2.COLOR_BGR2RGB))
            target=read_labels(label.read_text(encoding='utf-8'));targets.append(target)
            identities.append(dict(id=row['id'],image_sha256=sha256(image),label_sha256=sha256(label)))
            x,y,r,b=pair.roi
            outside+=sum(t['box'][0]<x or t['box'][1]<y or t['box'][2]>r or t['box'][3]>b for t in target)
        predicted=pair.predict(rgbs,.001)
        for j,row in enumerate(batch):
            details.append(dict(source_id=row['id'],targets=targets[j],baseline=predicted[0][j],candidate=predicted[1][j]))
            for arm in (0,1):
                p=predicted[arm][j]
                statistics[arm].append((matched(p,targets[j]),np.array([v['confidence'] for v in p]),
                                         np.array([v['class_id'] for v in p]),np.array([t['class_id'] for t in targets[j]])))
    scores=[]
    for stats in statistics:
        tp,conf,classes,target=[np.concatenate([row[i] for row in stats]) for i in range(4)]
        computed=ap_per_class(tp,conf,classes,target,plot=False,names={0:'vial',1:'colored_liquid'})
        per_class=[]
        for c,ap in zip(computed[6],computed[5]):
            keep=(classes==c)&(conf>=.35);true=int(tp[keep,0].sum());predicted=int(keep.sum());count=int((target==c).sum())
            per_class.append(dict(class_id=int(c),targets=count,ap50=float(ap[0]),ap50_95=float(ap.mean()),
                                  precision_at_035=true/predicted if predicted else 0.,recall_at_035=true/count if count else 0.,
                                  tp_at_035=true,fp_at_035=predicted-true,fn_at_035=count-true))
        scores.append(dict(map50=float(computed[5][:,0].mean()),map50_95=float(computed[5].mean()),per_class=per_class))
    report=dict(kind='paired_teacher_box_agreement',frames=len(selected),excluded_unknown_frames=len(rows)-len(selected),
                gt_boxes_outside_candidate_crop=outside,ground_truth_boxes_clipped_or_dropped=0,
                confidence_for_ap=.001,nms_iou=.45,baseline=scores[0],candidate=scores[1],sources=identities,
                limitations=['Unreviewed teacher boxes; not independent human accuracy.',
                             'Unknown/manual/empty records excluded, so not an absent-vial false-positive benchmark.',
                             'Both models use the same PyTorch backend, inference settings and full-source-coordinate matcher.',
                             'Run 5 excluded from fitting/checkpoint selection but already inspected previously.'])
    write(output/'teacher-agreement.json',report);write(output/'scoring-predictions.json',details)
    print(json.dumps({k:v for k,v in report.items() if k not in ('sources','limitations')}),flush=True)
    return report


def render(rgb,predictions,title):
    image=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
    for p in predictions:
        box=np.rint(p['box']).astype(int);color=(255,180,70) if p['class_id']==0 else (80,240,100)
        cv2.rectangle(image,tuple(box[:2]),tuple(box[2:]),color,2)
        text=f"{'vial' if p['class_id']==0 else 'liquid'} {p['confidence']:.2f}"
        cv2.putText(image,text,(max(0,box[0]-12),max(17,box[1]-6)),0,.46,color,1)
    header=np.zeros((40,640,3),np.uint8);cv2.putText(header,title,(8,27),0,.6,(255,255,255),1)
    return np.vstack([header,image])


def videos(pair,data,cache_path,cache_sha,baseline_model_sha,output):
    source=Catalog(data,include_annotations=False).runs[5]['cameras']['vial_fill']
    if source['fps']!=30:raise ValueError('This comparison requires the recorded 30 fps source')
    cache=CachedImageDetector(cache_path,cache_sha,baseline_model_sha,source['sha256'],5)
    original=read(cache_path)
    starts=[r['start_frame'] for r in original['detections'] if r['start_frame']>=60][:8]
    early={frame:(n,start) for n,start in enumerate(starts) for frame in range(start-15,start+106)}
    sampled=set(range(0,source['frames'],6));expected=sampled|set(early)
    counters=[Counter(),Counter()];full_count=early_count=decoded=processed=0;pending=[];tiles=[]
    with av.open(source['path']) as container,av.open(str(output/'run5-full-comparison.mp4'),'w') as full,av.open(str(output/'run5-early-pours.mp4'),'w') as short,(output/'video-predictions.jsonl').open('w',encoding='utf-8') as rows:
        container.streams.video[0].thread_type='AUTO'
        streams=[]
        for movie,rate in [(full,5),(short,30)]:
            stream=movie.add_stream('libx264',rate=rate);stream.width=1280;stream.height=560;stream.pix_fmt='yuv420p'
            stream.options={'crf':'23','preset':'veryfast'};streams.append(stream)
        def flush():
            nonlocal full_count,early_count,processed
            if not pending:return
            predictions=pair.predict([rgb for _,rgb in pending],.35)
            for j,(index,rgb) in enumerate(pending):
                arms=[predictions[0][j],predictions[1][j]]
                rows.write(json.dumps(dict(frame_index=index,mp4_s=index/30,source_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),baseline=arms[0],candidate=arms[1]))+'\n')
                top=np.hstack([render(rgb,arms[0],'CURRENT: raw full-frame detector'),render(rgb,arms[1],'CANDIDATE: cropped + L-CLAHE detector')])
                footer=np.zeros((40,1280,3),np.uint8)
                caption=f'Source frame {index} / {index/30:.2f}s | Same 0.35 confidence / 0.45 NMS | Overlays on ORIGINAL source pixels'
                if index in early:
                    number,start=early[index];caption+=f' | case {number+1}, {(index-start)/30:+.2f}s'
                cv2.putText(footer,caption,(6,25),0,.48,(255,255,255),1);image=np.vstack([top,footer])
                if index in sampled:
                    for arm in (0,1):
                        counters[arm]['samples_with_vial']+=any(p['class_id']==0 for p in arms[arm])
                        counters[arm]['samples_with_liquid']+=any(p['class_id']==1 for p in arms[arm])
                    encoded=av.VideoFrame.from_ndarray(image,format='bgr24');encoded.pts=full_count;encoded.time_base=Fraction(1,5)
                    for packet in streams[0].encode(encoded):full.mux(packet)
                    full_count+=1
                if index in early:
                    encoded=av.VideoFrame.from_ndarray(image,format='bgr24');encoded.pts=early_count;encoded.time_base=Fraction(1,30)
                    for packet in streams[1].encode(encoded):short.mux(packet)
                    early_count+=1
                    if index==early[index][1]+36:
                        cv2.imwrite(str(output/f'early-case{early[index][0]+1}.png'),image);tiles.append(cv2.resize(image,(640,280)))
                processed+=1
            pending.clear()
        for index,frame in enumerate(container.decode(video=0)):
            decoded+=1
            if index not in expected:continue
            if frame.pts is None or abs(float(frame.pts*frame.time_base)-source['start_pts_s']-index/30)>1/30:
                raise ValueError('Unsupported input presentation timestamp')
            rgb=frame.to_ndarray(format='rgb24')
            if index in sampled:cache.prepare_frame(index);cache.detect(rgb) # Exact baseline-source binding, no cached scores reused.
            pending.append((index,rgb))
            if len(pending)==32:flush()
        flush()
        for movie,stream in [(full,streams[0]),(short,streams[1])]:
            for packet in stream.encode():movie.mux(packet)
    if decoded!=source['frames'] or processed!=len(expected) or full_count!=len(sampled) or early_count!=len(early):
        raise ValueError('Incomplete paired video export')
    if tiles:cv2.imwrite(str(output/'early-contact-sheet.png'),np.vstack(tiles))
    report=dict(source_video_sha256=source['sha256'],source_frames_decoded=decoded,source_samples_verified=len(sampled),
                full_video_frames=full_count,full_video_fps=5,early_video_frames=early_count,early_video_fps=30,
                early_image_estimated_starts=starts,unique_source_frames_inferred=processed,
                baseline_presence=dict(counters[0]),candidate_presence=dict(counters[1]),
                presence_is_not_accuracy=True,early_starts_are_not_reviewed_flow_onsets=True,
                raw_source_pixels_preserved=True,no_recipe_color_qa_or_tracking_claim=True)
    write(output/'video-manifest.json',report);return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--teacher',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--cache-sha256',required=True)
    p.add_argument('--data',type=Path,default=DEFAULT_DATA);p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='0')
    args=p.parse_args();root=args.experiment;model=read(root/'model/model.json');plan=read(root/'experiment-plan.json')
    candidate=root/'model/best.pt'
    if sha256(candidate)!=model['checkpoint_sha256']:raise ValueError('Candidate checkpoint hash mismatch')
    if sha256(root/'experiment-plan.json')!=model['experiment_plan_sha256']:raise ValueError('Trial plan changed after training')
    if model['excluded_runs']!=[5] or plan['excluded_from_fit_and_selection']!=[5]:raise ValueError('Split provenance differs')
    args.out.mkdir(parents=True,exist_ok=False)
    pair=Pair(args.baseline,candidate,model['preprocessing']['roi'],args.device)
    metrics=score(pair,args.teacher,args.out)
    video=videos(pair,args.data,args.cache,args.cache_sha256,plan['baseline_onnx_sha256'],args.out)
    write(args.out/'comparison.json',dict(schema_version=1,kind='crop_contrast_detector_comparison',
          candidate=model,baseline_checkpoint_sha256=sha256(args.baseline),baseline_onnx_sha256=plan['baseline_onnx_sha256'],
          runtime_preprocessing_opencv=cv2.__version__,source_preprocessing_opencv=model['preprocessing']['opencv'],
          teacher_agreement={k:v for k,v in metrics.items() if k!='sources'},video=video,
          thresholds_unchanged=True,measurement_pipeline_changed=False,automatic_promotion=False,
          output_sha256={name:sha256(args.out/name) for name in ['run5-full-comparison.mp4','run5-early-pours.mp4','video-predictions.jsonl']}))
    print(json.dumps(video),flush=True)


if __name__=='__main__':main()
