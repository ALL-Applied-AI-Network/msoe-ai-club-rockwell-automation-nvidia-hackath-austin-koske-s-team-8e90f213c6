"""Real-image paired camera-nuisance experiment; frozen Stage 1 geometry.

Runs 1-3 choose fixed anchor patches. Run 4 measures invariance, not liquid color
accuracy. No recipe, quality label, source annotation or Run 5 pixels are used.
"""
import argparse,json,time
from collections import defaultdict
from pathlib import Path
import sys
import av,cv2,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from stage2_camera_reference import PATCHES,METHODS,linearize,encode,correction,feature,perturb,ReferenceHealth,anchor_health
from replay.geometry import decode_mask
from replay.measurement import delta_e
from liquid_specialists_data import read,write,sha
from render_liquid_pipeline import pixel_hash


CONDITIONS={
 'nominal':dict(gain=[1,1,1],domain='linear'),
 'linear_green_plus12pct':dict(gain=[1,1.12,1],domain='linear'),
 'linear_white_balance':dict(gain=[1.1,.94,.88],domain='linear'),
 'linear_exposure_minus30pct':dict(gain=[.7,.7,.7],domain='linear'),
 'linear_exposure_plus30pct':dict(gain=[1.3,1.3,1.3],domain='linear'),
 'srgb_green_plus10pct':dict(gain=[1,1.1,1],domain='srgb'),
 'srgb_black_offset3':dict(gain=[1,1,1],offset=3.,domain='srgb'),
 'one_anchor_shadow':dict(gain=[1,1,1],domain='linear',anchor_shadow=True),
 'two_anchor_shadow':dict(gain=[1,1,1],domain='linear',two_anchor_shadow=True),
 'liquid_only_shift_strong':dict(gain=[.6,1.35,.85],domain='linear',liquid_only=True),
 'liquid_only_color_shift':dict(gain=[.75,1.15,1],domain='linear',liquid_only=True)}


def anchors(rgb):
    return {name:np.median(rgb[y:b,x:r].reshape(-1,3),axis=0).tolist() for name,(x,y,r,b) in PATCHES.items()}


def main(a):
    if a.out.exists():raise ValueError('Use fresh output directory')
    completed=read(a.real/'completed.json');plan_real=read(a.real/'plan.json')
    if sha(a.real/'frames.jsonl')!=completed['artifacts']['frames.jsonl']:raise ValueError('Evidence changed')
    a.out.mkdir(parents=True)
    plan=dict(schema_version=1,kind='real_pixels_camera_invariance_not_quality_accuracy',patch_grid=PATCHES,
        anchor_selection_runs=[1,2,3],development_run=4,calibration='Run4 first five seconds, before liquid sampling',
        train_stride=30,methods=METHODS,conditions=CONDITIONS,reference_health=ReferenceHealth().__dict__,
        sample_selection='All 849 frozen Run4 QA-passing samples, plus first five seconds for background baseline',
        frozen_geometry=True,rerun_detector_or_sam=False,run5_used=False,source_labels_changed=False,
        limits=['Conditional on original unreviewed masks; no detector robustness or product-color accuracy claim.',
                'Perturbations are controlled camera nuisances, not measured plant deviations.',
                'Background patches are static scene anchors, not certified color standards.',
                'Paired error compares each method against its own same-frame unperturbed feature.'])
    plan.update(health_calibration='Common sRGB anchor spread, Runs1-3 p99 plus 0.005; bounded [0.01,0.05]',
        counterfactual_quantization='8-bit RGB; 0.4 RGB count independent noise on altered conditions',
        nuisance_noise_seed=20261021)
    write(a.out/'plan.json',plan);training=[];sources={}
    for run in (1,2,3):
        path=a.data/f'vial_fill/videos/vial_fill_video_{run}.mp4';sources[str(run)]=sha(path)
        with av.open(str(path)) as c:
            for index,frame in enumerate(c.decode(video=0)):
                if index%30:continue
                rgb=frame.to_ndarray(format='rgb24')
                if rgb.shape!=(480,640,3):raise ValueError('Unexpected source shape')
                training.append(dict(run=run,frame=index,pts=frame.pts,time_base=str(frame.time_base),
                    source_rgb_sha256=pixel_hash(rgb),anchors=anchors(rgb)))
        print(json.dumps(dict(background_run_finished=run,total_anchor_samples=len(training))),flush=True)
    ranking=[]
    for name in PATCHES:
        scores=[]
        for run in (1,2,3):
            values=np.array([r['anchors'][name] for r in training if r['run']==run])
            median=np.median(values,axis=0)
            # Stable non-dark, non-clipped patches; no liquid target enters this score.
            score=float(np.percentile(np.abs(values/np.maximum(median,1)-1),95))
            score+=float(np.mean((values<25)|(values>240)))
            scores.append(score)
        ranking.append(dict(patch=name,score=max(scores),per_run_score=scores))
    ranking.sort(key=lambda x:x['score']);chosen=[r['patch'] for r in ranking[:3]]
    training_spreads=[]
    for run in (1,2,3):
        run_rows=[r for r in training if r['run']==run]
        startup=np.median([[r['anchors'][k] for k in chosen] for r in run_rows[:5]],axis=0)
        for r in run_rows:
            _,_,spread=anchor_health(np.array([r['anchors'][k] for k in chosen]),startup,1.)
            if spread is not None:training_spreads.append(spread)
    maximum_disagreement=float(np.clip(np.percentile(training_spreads,99)+.005,.01,.05))
    write(a.out/'anchor-training.json',training);write(a.out/'anchor-selection.json',dict(ranking=ranking,chosen=chosen,source_sha256=sources,maximum_disagreement=maximum_disagreement,training_p99_spread=float(np.percentile(training_spreads,99))))
    print(json.dumps(dict(selected_anchors=chosen,selection_scores=ranking[:3])),flush=True)
    rows=[json.loads(s) for s in (a.real/'frames.jsonl').read_text().splitlines()]
    path=a.data/'vial_fill/videos/vial_fill_video_4.mp4'
    if sha(path)!=plan_real['source_sha256']:raise ValueError('Wrong development recording')
    baseline=[];paired=[];verified=0;feature_frames=0;health=ReferenceHealth();reference=None
    with av.open(str(path)) as c:
        for index,frame in enumerate(c.decode(video=0)):
            if index%6:continue
            row=rows[index//6]
            if row['seconds']>=5 and not row['measurement']['measurement_valid']:continue
            rgb=frame.to_ndarray(format='rgb24')
            if pixel_hash(rgb)!=row['source_rgb_sha256'] or frame.pts!=row['pts'] or str(frame.time_base)!=row['time_base']:
                raise ValueError('Frozen frame source differs')
            verified+=1
            patch=anchors(rgb);current=np.array([patch[k] for k in chosen])
            if row['seconds']<5:
                baseline.append(current);continue
            if reference is None:reference=np.median(baseline,axis=0)
            m=row['measurement'];mask=decode_mask(m['geometry']['mask'],rgb.shape[:2]).astype(bool)
            pixels=rgb[mask].astype(float);raw_lab,_=feature(pixels)
            if not np.allclose(raw_lab,m['lab'],atol=1e-10):raise ValueError('Source feature reconstruction differs')
            feature_frames+=1
            nominal={}
            for method in METHODS:
                corrected,gain,spread=correction(pixels,current,reference,method)
                nominal[method]=feature(corrected)[0] if corrected is not None else None
            for condition_index,(name,parameters) in enumerate(CONDITIONS.items()):
                args={k:v for k,v in parameters.items() if k in ('gain','offset','domain')}
                rng=np.random.default_rng(20261021+index*100+condition_index)
                def quantized(values,params):
                    transformed=perturb(values,**params)
                    if name!='nominal':transformed+=rng.normal(0,.4,transformed.shape)
                    return np.clip(np.rint(transformed),0,255)
                altered=quantized(pixels,args)
                changed=[]
                for patch_index,k in enumerate(chosen):
                    x,y,r,b=PATCHES[k];patch_pixels=rgb[y:b,x:r].reshape(-1,3)
                    adjusted=patch_pixels.astype(float) if parameters.get('liquid_only') else quantized(patch_pixels,args)
                    if (parameters.get('anchor_shadow') and patch_index==0) or (parameters.get('two_anchor_shadow') and patch_index<2):
                        adjusted=quantized(adjusted,dict(gain=[.8,.8,.8]))
                    changed.append(np.median(adjusted,axis=0))
                changed=np.array(changed)
                healthy,health_gain,health_spread=anchor_health(changed,reference,maximum_disagreement)
                for method in METHODS:
                    corrected,gain,spread=correction(altered,changed,reference,method)
                    lab,fraction=feature(corrected) if corrected is not None else (None,0.)
                    valid=lab is not None and nominal[method] is not None and fraction>=.6
                    # Health is evaluated in one common domain for every method.
                    distance=delta_e(lab,nominal[method]) if valid else None
                    paired.append(dict(frame=index,seconds=row['seconds'],condition=name,method=method,
                        feature_valid=valid,reference_healthy=healthy,patch_disagreement=health_spread,
                        camera_gain=None if gain is None else gain.tolist(),paired_de2000=distance))
    if feature_frames!=849 or len(baseline)!=25:raise ValueError('Unexpected real evidence inventory')
    write(a.out/'paired-results.json',paired);write(a.out/'calibration.json',dict(selected=chosen,reference_rgb=reference.tolist(),startup_frames=25,maximum_disagreement=maximum_disagreement))
    metrics={}
    for condition in CONDITIONS:
        metrics[condition]={}
        for method in METHODS:
            subset=[r for r in paired if r['condition']==condition and r['method']==method]
            distances=[r['paired_de2000'] for r in subset if r['feature_valid']]
            usable=[r['paired_de2000'] for r in subset if r['feature_valid'] and r['reference_healthy']]
            metrics[condition][method]=dict(frames=len(subset),valid=len(distances),healthy_valid=len(usable),
                median_de=float(np.median(distances)) if distances else None,p95_de=float(np.percentile(distances,95)) if distances else None,
                above5=sum(v>5 for v in distances),healthy_above5=sum(v>5 for v in usable),
                healthy_p95_de=float(np.percentile(usable,95)) if usable else None)
    fig,ax=plt.subplots(figsize=(13,5.5));names=[n for n in CONDITIONS if n not in ('nominal','liquid_only_color_shift','liquid_only_shift_strong')]
    x=np.arange(len(names))
    for i,method in enumerate(METHODS):ax.bar(x+(i-1)*.25,[metrics[n][method]['p95_de'] or 0 for n in names],.25,label=method)
    ax.set_xticks(x,[n.replace('_','\n') for n in names],fontsize=8);ax.set_ylabel('Same-frame paired dE2000, p95')
    ax.axhline(5,color='#a65a3a',ls='--',label='Engineering threshold 5');ax.legend(ncol=4)
    ax.set_title('Real Run4 pixels + frozen masks: camera-nuisance invariance, not color accuracy')
    ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(a.out/'camera-invariance.png',dpi=160);plt.close(fig)
    result=dict(schema_version=1,background_training_samples=len(training),verified_development_frames=verified,
        qa_passing_liquid_frames=feature_frames,paired_evaluations=len(paired),selected_anchors=chosen,
        metrics=metrics,run5_used=False,real_quality_labels=0,plan_sha256=sha(a.out/'plan.json'),
        source_sha256={**sources,'4':plan_real['source_sha256']},
        scripts={n:sha(Path(__file__).with_name(n)) for n in ('experiment_stage2_camera.py','stage2_camera_reference.py')},limits=plan['limits'])
    for n in result['scripts']:(a.out/n).write_bytes(Path(__file__).with_name(n).read_bytes())
    result['artifacts']={p.name:sha(p) for p in a.out.iterdir() if p.is_file()}
    write(a.out/'completed.json',result)
    print(json.dumps(metrics,indent=2),flush=True)


if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for n in ('data','real','out'):p.add_argument('--'+n,type=Path,required=True)
 main(p.parse_args())
