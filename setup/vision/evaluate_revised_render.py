"""Frozen revised-render transfer probes; original Run 5 pixels, local videos."""
import argparse
from pathlib import Path

from evaluate_mixed_detector import FullFramePair, export_parity, score, videos
from train_mixed_detector import read, write, sha, BASELINE, BASELINE_CHECKPOINT


def main(args):
    root=args.experiment;plan=read(root/'experiment-plan.json');previous=args.previous
    if sha(args.baseline)!=BASELINE_CHECKPOINT or plan['baseline_checkpoint_sha256']!=BASELINE_CHECKPOINT:
        raise ValueError('Current detector checkpoint changed')
    deployed=Path(__file__).resolve().parents[2]/'replay/models/vial-bootstrap.onnx'
    if sha(deployed)!=BASELINE or plan['baseline_onnx_sha256']!=BASELINE:
        raise ValueError('Published default detector changed')
    if sha(previous/'experiment-plan.json')!=plan['source_experiment_plan_sha256']:
        raise ValueError('Previous experiment identity changed')
    if plan['excluded_from_fit_and_selection']!=[5] or plan['unique_synthetic_images']!=18:
        raise ValueError('New split/unique-scene boundary changed')
    cached=read(previous/'evaluation/comparison.json')
    expected_sources=read(previous/'evaluation/teacher-agreement.json')['sources']
    if sha(previous/'models/real-only/best.pt')!=plan['cached_real_only_control']['checkpoint_sha256']:
        raise ValueError('Cached control changed')
    args.out.mkdir(parents=True,exist_ok=False)
    results={};inventories=[]
    for arm,title in [('mixed','REAL + REVISED RENDERS: 18 unique scenes'),
                      ('sim-only','REVISED SIM ONLY: 18 unique scenes, fixed epoch')]:
        model=read(root/f'models/{arm}/model.json')
        checkpoint=root/f'models/{arm}'/('best.pt' if arm=='mixed' else 'checkpoint.pt')
        if arm=='mixed':
            if model['excluded_runs']!=[5] or model['experiment_plan_sha256']!=sha(root/'experiment-plan.json'):
                raise ValueError('Mixed fit/selection boundary changed')
        elif (model['train_runs'] or model['development_runs'] or model['excluded_real_runs']!=[1,2,3,4,5] or
              model['sim_only_plan_sha256']!=sha(root/'sim-only-plan.json') or
              model['base_plan_sha256']!=sha(root/'experiment-plan.json')):
            raise ValueError('Sim-only fitting/selection boundary changed')
        if sha(checkpoint)!=model['checkpoint_sha256'] or sha(root/f'models/{arm}/detector.onnx')!=model['model_sha256']:
            raise ValueError('Frozen candidate changed')
        out=args.out/arm;out.mkdir()
        parity=export_parity(root,out/'onnx-parity.json',checkpoint,root/f'models/{arm}/detector.onnx')
        pair=FullFramePair(args.baseline,checkpoint,[0,0,640,480],args.device)
        pair.candidate_title=title
        metrics=score(pair,args.teacher,out)
        if metrics['sources']!=expected_sources:raise ValueError('Run 5 scoring sources/targets changed')
        video=videos(pair,args.data,args.cache,args.cache_sha256,BASELINE,out)
        results[arm]=dict(model=model,parity=parity,teacher_agreement={k:v for k,v in metrics.items() if k!='sources'},
            video=video,output_sha256={name:sha(out/name) for name in
                ['run5-full-comparison.mp4','run5-early-pours.mp4','video-predictions.jsonl']})
        inventories.append(out/'video-predictions.jsonl')
    import json
    from itertools import zip_longest
    count=0
    with inventories[0].open() as left,inventories[1].open() as right:
        for a,b in zip_longest(left,right):
            if a is None or b is None:raise ValueError('Paired video source counts differ')
            a,b=json.loads(a),json.loads(b)
            for key in ['frame_index','mp4_s','source_rgb_sha256','baseline']:
                if a[key]!=b[key]:raise ValueError('Paired original pixels/baseline predictions differ')
            count+=1
    # Fully decode all delivered MP4s, rather than trusting mux/header counts.
    import av
    decode_checks=[]
    for arm in results:
        for name,key in [('run5-full-comparison.mp4','full_video_frames'),('run5-early-pours.mp4','early_video_frames')]:
            path=args.out/arm/name
            with av.open(str(path)) as container:
                decoded=sum(1 for _ in container.decode(video=0))
            if decoded!=results[arm]['video'][key]:raise ValueError('Local MP4 decode incomplete')
            decode_checks.append(dict(path=str(path),frames_decoded=decoded,passed=True))
    write(args.out/'comparison.json',dict(schema_version=1,kind='18_revised_render_transfer_probe',
        plan_sha256=sha(root/'experiment-plan.json'),unique_rendered_scenes=18,
        baseline_checkpoint_sha256=sha(args.baseline),default_detector_sha256=BASELINE,
        revised=results,cached_previous=dict(source_comparison_sha256=sha(previous/'evaluation/comparison.json'),
            old_mixed=cached['teacher_agreement']['candidate'],
            real_only_control=cached['control_teacher_agreement']['candidate'],
            old_sim_only=cached['sim_only_transfer']['teacher_agreement']['candidate'],
            source_policy='Previously completed frozen models/diagnostics, not refitted under the new plan'),
        paired_original_video_sources_verified=count,video_decode_checks=decode_checks,
        thresholds_unchanged=True,measurement_pipeline_changed=False,automatic_promotion=False,
        limitations=plan['limits']))
    print('Revised transfer comparison, original-pixel source checks and complete local MP4 decoding passed')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['experiment','previous','baseline','teacher','data','cache','out']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--cache-sha256',required=True);parser.add_argument('--device',default='0')
    main(parser.parse_args())
