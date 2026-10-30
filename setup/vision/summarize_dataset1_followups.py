"""Recompute experiment evidence and export a diagnostic single-panel video."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess

from dataset1_fiveclass import CLASSES,read,sha,write
from train_dataset1_fiveclass import summarize
from experiment_dataset1_cap_state import CAP_LABEL,metrics as cap_metrics


def verify_bundle(root):
    manifest=read(root/'artifact-manifest.json')
    for name,digest in manifest.items():
        if sha(root/name)!=digest:raise ValueError('Artifact differs: '+str(root/name))
    return len(manifest)


def verify_predictions(root,normals=False):
    rows=[json.loads(s) for s in (root/'rendered-v1/manifest.jsonl').read_text().splitlines()];by_id={r['id']:r for r in rows}
    report=read(root/'comparison.json');verified=0;stratified={}
    for mode,m in report['arms'].items():
        file=root/('fitted-v1/'+mode+'/fresh-test-predictions.json' if mode.startswith('detail_') else 'baseline-fresh-v1/'+mode+'/predictions.json')
        predictions=read(file)
        for p in predictions:
            normal=p.get('variant')=='golden_as_current';scene=p['scene'].removesuffix('-normal') if normal else p['scene'];source=by_id[scene]
            expected=0 if normal else CLASSES.index(source['labels'][p['slot']]);key='golden_sha256' if normal else 'image_sha256'
            if p['truth']!=expected or p['source_sha256']!=source[key] or p['box_xyxy']!=source['boxes'][p['slot']]:raise ValueError('Source/slot/label join differs')
            if p['prediction']!=max(range(5),key=lambda j:p['probabilities'][j]):raise ValueError('Prediction differs from probabilities')
            verified+=1
        recalculated=summarize(predictions,m['occupied_defect_diagnostic']['threshold'])
        for key,value in recalculated.items():
            if m[key]!=value:raise ValueError('Reported metric differs: '+mode+'/'+key)
        if normals:
            strata={}
            for name in ('dense','sparse','normal'):
                selected=[p for p in predictions if (p.get('variant')=='golden_as_current' if name=='normal' else p.get('variant')=='inspection' and by_id[p['scene']]['nuisance']['density']==name)]
                strata[name]=summarize(selected,m['occupied_defect_diagnostic']['threshold'])
            stratified[mode]=strata
    return verified,stratified


def run(base,detail,visibility,normals,cap,confirmation,out):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image,ImageDraw,ImageFont
    out.mkdir(exist_ok=False)
    verified_artifacts={str(root):verify_bundle(root) for root in (base,detail,visibility,normals,cap,confirmation)}
    count1,_=verify_predictions(detail);count2,strata=verify_predictions(normals,True)
    comparison=read(detail/'comparison.json');normal_comparison=read(normals/'comparison.json')
    policies=read(normals/'operating-points-results.json');visibility_report=read(visibility.parent/'summary-v2/report.json')
    previous_policy=read(detail/'operating-points-frozen.json')['modes']['detail_paired']['thresholds']['0.95']
    previous_predictions=read(normals/'baseline-fresh-v1/previous_detail_paired/predictions.json')
    normal_ablation=dict(previous_without_normal_current=summarize(previous_predictions,previous_policy)['occupied_defect_diagnostic'],
        with_normal_current=policies['results']['detail_paired']['policies']['0.95'],
        note='Same fresh targets and architecture; each threshold frozen for pooled95% occupied-defect development recall. Not a guarantee of95% recall for each defect class.')
    cap_result=read(cap/'fitted-v1/results.json');cap_predictions=read(cap/'fitted-v1/test-predictions.json')
    sources={r['id']:r for r in [json.loads(s) for s in (normals/'rendered-v1/manifest.jsonl').read_text().splitlines()]}
    for p in cap_predictions:
        normal=p['variant']=='golden_as_current';source=sources[p['scene'].removesuffix('-normal') if normal else p['scene']]
        truth=0 if normal else CLASSES.index(source['labels'][p['slot']]);key='golden_sha256' if normal else 'image_sha256'
        if p['source_truth']!=truth or p['truth']!=CAP_LABEL[truth] or p['source_sha256']!=source[key] or p['box_xyxy']!=source['boxes'][p['slot']]:raise ValueError('Cap source/label semantics differ')
        if p['prediction']!=max(range(4),key=lambda j:p['probabilities'][j]):raise ValueError('Cap argmax differs')
    if cap_metrics(cap_predictions)!=cap_result['metrics']:raise ValueError('Cap metric recomputation differs')
    stress_verified=0
    for folder in ('reference-stress-v1','gain-correction-v1'):
        cases=read(cap/folder/'completed.json')['cases']
        for name,result in cases.items():
            records=read(cap/folder/(name+'-predictions.json'))
            for p in records:
                normal=p['variant']=='golden_as_current';source=sources[p['scene'].removesuffix('-normal') if normal else p['scene']]
                truth=0 if normal else CLASSES.index(source['labels'][p['slot']]);key='golden_sha256' if normal else 'image_sha256'
                if p['source_truth']!=truth or p['truth']!=CAP_LABEL[truth] or p['source_sha256']!=source[key] or p['box_xyxy']!=source['boxes'][p['slot']]:raise ValueError('Stress parent source/label join differs')
                if p['prediction']!=max(range(4),key=lambda j:p['probabilities'][j]):raise ValueError('Stress argmax differs')
                stress_verified+=1
            if cap_metrics(records)!=result['metrics']:raise ValueError('Stress metric recomputation differs')
            for policy,threshold in read(cap/'fitted-v1/frozen.json')['thresholds'].items():
                if cap_metrics(records,threshold)['cap_warning']!=result['policies'][policy]:raise ValueError('Stress policy recomputation differs')
    confirmation_result=read(confirmation/'results.json');confirmed=0
    confirm_sources={r['id']:r for r in [json.loads(s) for s in (confirmation/'rendered-v1/manifest.jsonl').read_text().splitlines()]}
    for name,result in confirmation_result['cases'].items():
        predictions=read(confirmation/(name+'-predictions.json'))
        for p in predictions:
            normal=p['variant']=='golden_as_current';source=confirm_sources[p['scene'].removesuffix('-normal') if normal else p['scene']]
            truth=0 if normal else CLASSES.index(source['labels'][p['slot']]);key='golden_sha256' if normal else 'image_sha256'
            if p['source_truth']!=truth or p['truth']!=CAP_LABEL[truth] or p['source_sha256']!=source[key] or p['box_xyxy']!=source['boxes'][p['slot']]:raise ValueError('Fresh confirmation source differs')
            if p['prediction']!=max(range(4),key=lambda j:p['probabilities'][j]):raise ValueError('Fresh confirmation argmax differs')
            confirmed+=1
        if cap_metrics(predictions)!=result['metrics']:raise ValueError('Fresh confirmation metrics differ')
    for root in (detail,normals):
        frozen=read(root/'fitted-v1/all-checkpoints-frozen.json');first_source=min(p.stat().st_mtime for p in (root/'rendered-v1').glob('*.png'))
        if frozen['utc']>first_source:raise ValueError('Checkpoint freeze followed test rendering')
        policy=read(root/'operating-points-frozen.json')
        first_prediction=min(p.stat().st_mtime for p in (root/'fitted-v1').glob('*/fresh-test-predictions.json'))
        if policy['frozen_utc']>=first_prediction:raise ValueError('Operating policy freeze followed test prediction')
    damage=[r['metrics']['64'] for r in visibility_report['cases'] if r['case']['geometric_class']=='damaged_vial']
    caps=[r['metrics']['64'] for r in visibility_report['cases'] if r['case']['geometric_class']!='damaged_vial']
    optical=dict(damage_cases=len(damage),cap_control_cases=len(caps),
        damage_change_mae_range=[min(r['change_mae'] for r in damage),max(r['change_mae'] for r in damage)],
        cap_change_mae_range=[min(r['change_mae'] for r in caps),max(r['change_mae'] for r in caps)],
        damage_pixels_above_noise_range=[min(r['changed_pixels_above_noise'] for r in damage),max(r['changed_pixels_above_noise'] for r in damage)],
        cap_pixels_above_noise_range=[min(r['changed_pixels_above_noise'] for r in caps),max(r['changed_pixels_above_noise'] for r in caps)],
        repeated_golden_noise_mae_max=max(r['repeat_noise_mae'] for r in damage+caps),pixels_per_crop=4096,
        difference_threshold_floor_digital_levels=2,physical_sensor_noise_modeled=False,
        repeated_unchanged_scene_render_can_reuse_accumulation=True,
        controls_are_synthetic=True,real_damage_visibility_proven=False)
    report=dict(schema='dataset1-followup-summary/v1',verified_artifacts=verified_artifacts,
        verified_fiveclass_test_predictions=count1+count2,verified_cap_test_predictions=len(cap_predictions),verified_confirmation_predictions=confirmed,verified_stress_parent_source_joins=stress_verified,
        detail_fresh_test=comparison,normal_coverage_fresh_test=normal_comparison,normal_strata=strata,
        normal_warning_policies=policies,matched_normal_ablation=normal_ablation,cap_state=cap_result,cap_reference_stress=read(cap/'reference-stress-v1/completed.json'),optical_identifiability=optical,
        authored_pose_coverage=read(normals/'authored-pose-audit.json'),
        cap_gain_correction=read(cap/'gain-correction-v1/completed.json'),cap_fresh_confirmation=confirmation_result,
        real_accuracy=None,real_training_runs=[],run5_used=False,body_damage_solved=False,
        full_dataset1_task_solved=False,app_defaults_changed=False,hardware_commands=0,
        limits=['Same CAD/camera family; synthetic held-out results are not real transfer accuracy.',
            'Damage proxy has nominal cap pose; cap-state label does not approve body/seal/product quality.',
            'Paired models require a qualified real golden that has not been captured.',
            'Normal and sparse scene strata share12 nuisance families; targets are correlated.',
            'Original S-body principal axes are81–84 degrees from vertical; generated canonical bodies cover only0–30. Authored lying-S geometry is not validated by these scores.',
            'New architectures are exploratory hypotheses motivated by earlier inspected synthetic failures.'])
    write(out/'report.json',report)
    # Matched architecture evidence, preserving distinct class scopes.
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    names=list(comparison['arms']);values=[comparison['arms'][n]['accuracy']*100 for n in names]
    axes[0].barh(names,values,color=['#2f9ba2' if 'paired' in n else '#597ca0' for n in names]);axes[0].set_xlim(0,100)
    axes[0].set_xlabel('Exact five-class accuracy (%)');axes[0].set_title('Fresh720 synthetic targets /12 families')
    for i,value in enumerate(values):axes[0].text(value+1,i,f'{value:.1f}%',va='center')
    for name in ('target_paired','detail_paired'):
        p=read(detail/'operating-points-results.json')['results'][name]['policies']
        xy=[(v['false_alarm_rate']*100,(1-sum(r['missed'] for r in v['per_class'].values())/sum(r['targets'] for r in v['per_class'].values()))*100) for v in p.values()]
        axes[1].plot([x for x,y in xy],[y for x,y in xy],'o-',label=name)
    axes[1].set(xlim=(0,102),ylim=(70,101),xlabel='Acceptable-target false alarms (%)',ylabel='Occupied-defect recall (%)',title='Thresholds frozen on synthetic development')
    axes[1].legend();axes[1].grid(alpha=.2);fig.suptitle('Reference input helps; five-class warning performance remains poor')
    fig.savefig(out/'architecture-and-warning-comparison.png',dpi=150);plt.close(fig)
    # One panel, actual cap-model predictions. Red marks errors against synthetic
    # truth and is explicitly a diagnostic overlay, never a product verdict.
    frames=out/'cap-diagnostic-frames';frames.mkdir();fontfile='/usr/share/fonts/truetype/lato/Lato-Regular.ttf'
    font=ImageFont.truetype(fontfile,23);small=ImageFont.truetype(fontfile,18)
    storyboard=[];frame=0
    for row in sources.values():
        for normal in (True,False):
            scene=row['id']+('-normal' if normal else '');pred=[p for p in cap_predictions if p['scene']==scene]
            key='golden' if normal else 'image';path=normals/'rendered-v1'/row[key]
            if sha(path)!=row[key+'_sha256']:raise ValueError('Video source differs')
            image=Image.open(path).convert('RGB');draw=ImageDraw.Draw(image)
            for p in pred:
                l,t,r,b=p['box_xyxy'];x,y=(l+r)/2,(t+b)/2
                color='#ff786c' if p['prediction']!=p['truth'] else '#4de1cf' if p['prediction']!=3 else '#b8c7cf'
                draw.ellipse((x-3,y-3,x+3,y+3),outline=color,width=2)
            correct=sum(p['prediction']==p['truth'] for p in pred);canvas=Image.new('RGB',(1280,1120),'#112539');draw=ImageDraw.Draw(canvas)
            draw.text((16,10),'SYNTHETIC CAP-STATE DIAGNOSTIC | body condition unassessed',font=font,fill='white')
            kind='all-good current image' if normal else row['nuisance']['density']+' defect current image'
            draw.text((16,44),f'{row["id"]} | {kind} | exact cap state {correct}/60',font=font,fill='#bfe5e9')
            draw.text((16,78),'Red = disagreement with synthetic truth; cyan = occupied; gray = empty proposal.',font=small,fill='#e4c890')
            canvas.paste(image.resize((1280,960)),(0,115))
            draw.text((16,1088),'Discrete rendered images, 1 second each. No real accuracy, live latency, seal or product-quality claim.',font=small,fill='#bdcdd5')
            target=frames/f'frame-{frame:03d}.png';canvas.save(target);storyboard.append(dict(frame=frame,scene=scene,source_sha256=row[key+'_sha256'],correct=correct,targets=60,png_sha256=sha(target)));frame+=1
    video=out/'synthetic-cap-state-diagnostic.mp4'
    subprocess.run(['ffmpeg','-v','error','-framerate','1','-i',str(frames/'frame-%03d.png'),'-vf','fps=24','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p',str(video)],check=True)
    process=subprocess.run(['ffmpeg','-v','error','-i',str(video),'-f','framehash','-'],check=True,stdout=subprocess.PIPE,text=True)
    decoded=sum(bool(line.strip()) and not line.startswith('#') for line in process.stdout.splitlines())
    if decoded!=24*frame:raise ValueError('Video decode count differs')
    write(out/'video-storyboard.json',dict(frames=storyboard,decoded_frames=decoded,video_sha256=sha(video),fps=24,source_images=frame,body_condition='unassessed'))
    write(out/'completed.json',dict(report_sha256=sha(out/'report.json'),verified_predictions=count1+count2+len(cap_predictions)+confirmed+stress_verified,decoded_video_frames=decoded,helper_sha256=sha(__file__)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('base','detail','visibility','normals','cap','confirmation','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.base,a.detail,a.visibility,a.normals,a.cap,a.confirmation,a.out)
