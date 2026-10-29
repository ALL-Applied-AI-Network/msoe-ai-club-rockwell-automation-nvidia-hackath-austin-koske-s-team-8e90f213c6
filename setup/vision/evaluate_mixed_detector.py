"""Paired full-frame Run 5 diagnostic; original pixels and local MP4 outputs."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np

from evaluate_crop_contrast import Pair, score, videos
from train_mixed_detector import read, write, sha, BASELINE


class FullFramePair(Pair):
    candidate_title = 'CANDIDATE: real + synthetic, full-frame'

    def predict(self, rgbs, confidence):
        images=[cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR) for rgb in rgbs]
        output=[]
        for model in self.models:
            predictions=model.predict(images,imgsz=640,rect=False,conf=confidence,iou=.45,max_det=300,
                device=self.device,quantize=None,augment=False,agnostic_nms=False,verbose=False,save=False)
            output.append([[dict(class_id=int(row[5]),box=[float(v) for v in row[:4]],confidence=float(row[4]))
                for row in prediction.boxes.data.cpu().numpy()] for prediction in predictions])
        return output


def export_parity(root, output_path, checkpoint, exported):
    from ultralytics import YOLO
    data=read(root/'data/real/provenance.json')
    rows=[r for r in data['samples'] if r['split']=='val']
    rows=[rows[i] for i in np.linspace(0,len(rows)-1,16,dtype=int)]
    for row in rows:
        if sha(root/'data/real'/row['image'])!=row['image_sha256']:
            raise ValueError('Development parity image changed')
    images=[cv2.imread(str(root/'data/real'/r['image'])) for r in rows]
    models=[YOLO(str(path)) for path in [checkpoint,exported]]
    outputs=[m.predict(images,imgsz=640,rect=False,conf=.35,iou=.45,max_det=300,device='cpu',
        quantize=None,augment=False,agnostic_nms=False,verbose=False) for m in models]
    box_error=score_error=0.;count=0
    for left,right in zip(*outputs):
        a=sorted(left.boxes.data.cpu().numpy(),key=lambda r:(r[5],-r[4]))
        b=sorted(right.boxes.data.cpu().numpy(),key=lambda r:(r[5],-r[4]))
        if len(a)!=len(b):raise ValueError('CPU ONNX checkpoint parity count mismatch')
        for x,y in zip(a,b):
            if x[5]!=y[5]:raise ValueError('CPU ONNX class mismatch')
            box_error=max(box_error,float(np.abs(x[:4]-y[:4]).max()));score_error=max(score_error,float(abs(x[4]-y[4])))
            count+=1
    if box_error>1.1 or score_error>1e-4:raise ValueError('CPU export parity exceeded predeclared numeric tolerance')
    result=dict(passed=True,development_run=4,frames=len(rows),boxes=count,max_source_pixel_error=box_error,
        max_score_error=score_error,backend='PyTorch CPU vs ONNX Runtime CPU; no quantization',
        scope='Post-freeze export consistency only; does not select or tune checkpoints',
        checkpoint_sha256=sha(checkpoint),onnx_sha256=sha(exported),
        image_sha256=[r['image_sha256'] for r in rows])
    write(output_path,result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',type=Path,required=True);parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--teacher',type=Path,required=True);parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--cache',type=Path,required=True);parser.add_argument('--cache-sha256',required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--device',default='0')
    parser.add_argument('--include-sim-only',action='store_true')
    args=parser.parse_args();root=args.experiment;plan=read(root/'experiment-plan.json')
    baseline_export=Path(__file__).resolve().parents[2]/'replay/models/vial-bootstrap.onnx'
    if plan['baseline_onnx_sha256']!=BASELINE or sha(args.baseline)!=plan['baseline_checkpoint_sha256']:
        raise ValueError('Current baseline identity differs')
    if sha(baseline_export)!=BASELINE:
        raise ValueError('Published baseline ONNX changed')
    mixed=read(root/'models/mixed/model.json');control=read(root/'models/real-only/model.json')
    for expected_arm,model in [('mixed',mixed),('real-only',control)]:
        if model['arm']!=expected_arm:
            raise ValueError('Candidate/control identity changed')
        if model['experiment_plan_sha256']!=sha(root/'experiment-plan.json') or model['excluded_runs']!=[5]:
            raise ValueError('Training plan/exclusion changed')
        if sha(root/'models'/model['arm']/'best.pt')!=model['checkpoint_sha256']:
            raise ValueError('Candidate checkpoint changed')
        if sha(root/'models'/model['arm']/'detector.onnx')!=model['model_sha256']:
            raise ValueError('Candidate ONNX export changed')
    sim=None
    if args.include_sim_only:
        sim=read(root/'models/sim-only/model.json')
        if (sim['arm']!='sim-only' or sim['train_runs'] or sim['development_runs'] or
                sim['excluded_real_runs']!=[1,2,3,4,5] or sim['base_plan_sha256']!=sha(root/'experiment-plan.json') or
                sim['sim_only_plan_sha256']!=sha(root/'sim-only-plan.json')):
            raise ValueError('Sim-only fit/selection boundary changed')
        if (sha(root/'models/sim-only/checkpoint.pt')!=sim['checkpoint_sha256'] or
                sha(root/'models/sim-only/detector.onnx')!=sim['model_sha256']):
            raise ValueError('Sim-only model changed')
    args.out.mkdir(parents=True,exist_ok=False)
    parity=export_parity(root,args.out/'onnx-parity.json',root/'models/mixed/best.pt',root/'models/mixed/detector.onnx')
    baseline_parity=export_parity(root,args.out/'baseline-onnx-parity.json',args.baseline,baseline_export)
    control_dir=args.out/'control';control_dir.mkdir()
    control_parity=export_parity(root,control_dir/'onnx-parity.json',root/'models/real-only/best.pt',root/'models/real-only/detector.onnx')
    pair=FullFramePair(args.baseline,root/'models/mixed/best.pt',[0,0,640,480],args.device)
    metrics=score(pair,args.teacher,args.out)
    video=videos(pair,args.data,args.cache,args.cache_sha256,BASELINE,args.out)
    control_pair=FullFramePair(args.baseline,root/'models/real-only/best.pt',[0,0,640,480],args.device)
    control_metrics=score(control_pair,args.teacher,control_dir)
    if metrics['sources']!=control_metrics['sources']:
        raise ValueError('Scoring inventory changed between candidate/control comparisons')
    sim_result=None
    if sim:
        sim_dir=args.out/'sim-only';sim_dir.mkdir()
        sim_parity=export_parity(root,sim_dir/'onnx-parity.json',root/'models/sim-only/checkpoint.pt',root/'models/sim-only/detector.onnx')
        sim_pair=FullFramePair(args.baseline,root/'models/sim-only/checkpoint.pt',[0,0,640,480],args.device)
        sim_pair.candidate_title='SIM ONLY: fixed epoch, no real selection'
        sim_metrics=score(sim_pair,args.teacher,sim_dir)
        if sim_metrics['sources']!=metrics['sources']:raise ValueError('Sim-only scoring inventory changed')
        sim_video=videos(sim_pair,args.data,args.cache,args.cache_sha256,BASELINE,sim_dir)
        sim_result=dict(model=sim,onnx_parity=sim_parity,video=sim_video,
            teacher_agreement={k:v for k,v in sim_metrics.items() if k!='sources'},
            output_sha256={name:sha(sim_dir/name) for name in ['run5-full-comparison.mp4','run5-early-pours.mp4','video-predictions.jsonl']})
    write(args.out/'comparison.json',dict(schema_version=1,kind='real_plus_synthetic_full_frame_detector_comparison',
        candidate=mixed,matched_real_only_control=control,baseline_checkpoint_sha256=sha(args.baseline),
        baseline_onnx_sha256=BASELINE,experiment_plan_sha256=sha(root/'experiment-plan.json'),onnx_parity=parity,
        baseline_onnx_parity=baseline_parity,control_onnx_parity=control_parity,
        teacher_agreement={k:v for k,v in metrics.items() if k!='sources'},
        control_teacher_agreement={k:v for k,v in control_metrics.items() if k!='sources'},video=video,
        sim_only_transfer=sim_result,
        thresholds_unchanged=True,measurement_pipeline_changed=False,automatic_promotion=False,
        output_sha256={name:sha(args.out/name) for name in ['run5-full-comparison.mp4','run5-early-pours.mp4','video-predictions.jsonl']}))
    print(json.dumps(video),flush=True)


if __name__=='__main__':main()
