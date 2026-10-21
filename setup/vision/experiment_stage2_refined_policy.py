"""Paired Stage 2 front-end ablation with fresh nuisance seeds.

Analytic optical scenarios test decision behavior, not product quality. Camera
anchor locations/health calibration come from the separate real-pixel experiment.
"""
import argparse,itertools,json,time
from collections import Counter
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiment_stage2_policy import fixture,Policy,rgb_to_lab
from stage2_policy import EvidencePolicy
from stage2_guard import GuardedEvidence
from stage2_camera_reference import correction,anchor_health,perturb
from liquid_specialists_data import read,write,sha

CONDITIONS={
 'nominal':{},
 'linear_white_balance':dict(gain=[1.1,.94,.88],domain='linear'),
 'linear_green_plus12pct':dict(gain=[1,1.12,1],domain='linear'),
 'srgb_green_plus10pct':dict(gain=[1,1.1,1],domain='srgb'),
 'srgb_black_offset3':dict(gain=[1,1,1],offset=3,domain='srgb'),
 'anchor_shadow':dict(anchor_shadow=True),
 'known_stale_process_1s':dict(clock_offset=-1.,state_age=1.),
 'hidden_wrong_progress_1s':dict(clock_offset=-1.,state_age=0.)}
ARMS=('raw','srgb_gain','linear_gain','srgb_gain_guarded','linear_gain_guarded')


def main(a):
 if a.out.exists():raise ValueError('Use fresh output')
 camera=read(a.camera/'completed.json')
 for name in ('calibration.json','anchor-selection.json'):
  if sha(a.camera/name)!=camera['artifacts'][name]:raise ValueError('Camera calibration changed')
 calibration=read(a.camera/'calibration.json');anchors=np.array(calibration['reference_rgb'])
 tolerance=calibration['maximum_disagreement']
 a.out.mkdir(parents=True)
 plan=dict(schema_version=1,kind='refined_stage2_controlled_policy',recipe_profiles=12,donors=[[.7,.2,.1],[.1,.7,.2],[.2,.1,.7]],
     hold_up_ml=[0,.5,2],nuisance_seeds=[11,12],conditions=CONDITIONS,arms=ARMS,policy=Policy().to_dict(),
     camera_calibration_sha256=sha(a.camera/'calibration.json'),anchor_rgb=anchors.tolist(),
     anchor_median_noise_rgb_sd=.15,anchor_quantization=True,
     exact_simulated_phase_and_volume_available_except_declared_clock_conditions=True,
     limits=['Analytic fixture; real defect labels and physical action outcomes are unavailable.',
         'Same optical family generates expected/observed appearances; seeds differ from the preceding policy study.',
         'Guarded abstention is reported separately from warning rate; blocking a fault is not detecting it.',
         'Hidden incorrect progress with falsely fresh metadata is an intentionally undetectable input fault.'])
 scripts=('experiment_stage2_refined_policy.py','stage2_camera_reference.py','stage2_guard.py','stage2_policy.py','experiment_stage2_policy.py')
 plan['helper_sha256']={n:sha(Path(__file__).with_name(n)) for n in scripts}
 write(a.out/'plan.json',plan)
 for n in scripts:(a.out/n).write_bytes(Path(__file__).with_name(n).read_bytes())
 results=[];started=time.perf_counter();showcases=[]
 profiles=[(np.array(v,float),o) for v in ([10,10,10],[20,10,10]) for o in itertools.permutations(range(3))]
 for recipe,(volumes,order) in enumerate(profiles):
  scenarios=[(0.,np.array(plan['donors'][0]),'none')]+[(h,np.array(d),str(j)) for j,d in enumerate(plan['donors']) for h in (.5,2.)]
  for hold,donor,donor_id in scenarios:
   for seed in plan['nuisance_seeds']:
    original=fixture(recipe,volumes,order,donor,hold,seed,None)
    for condition,settings in CONDITIONS.items():
     rgb=original['observed_rgb'].copy();expected=original['expected_lab'].copy();times=original['times'];n=len(times)
     rng=np.random.default_rng(20261022+recipe*1000+seed)
     reference_pixels=np.tile(anchors[None,:,:],(n,1,1))
     params={k:v for k,v in settings.items() if k in ('gain','domain','offset')}
     if params:
      rgb=perturb(rgb,**params);reference_pixels=perturb(reference_pixels,**params)
     if settings.get('anchor_shadow'):
      reference_pixels[:,0]=perturb(reference_pixels[:,0],gain=[.8,.8,.8])
     reference_pixels=np.clip(np.rint(reference_pixels+rng.normal(0,.15,reference_pixels.shape)),0,255)
     rgb=np.clip(np.rint(rgb),0,255)
     if settings.get('clock_offset'):
      expected=np.stack([np.interp(times+settings['clock_offset'],times,expected[:,j]) for j in range(3)],axis=1)
     normalized={};health=[]
     for method in ('raw','srgb_gain','linear_gain'):
      values=[]
      for i in range(n):
       corrected,_,_=correction(rgb[i:i+1],reference_pixels[i],anchors,method)
       values.append(rgb_to_lab(corrected[0]) if corrected is not None else np.array([np.nan]*3))
       if method=='raw':health.append(anchor_health(reference_pixels[i],anchors,tolerance)[0])
      normalized[method]=np.array(values)
     ident=f'r{recipe}-d{donor_id}-h{hold}-seed{seed}-{condition}'
     for arm in ARMS:
      engine=GuardedEvidence() if arm.endswith('_guarded') else EvidencePolicy()
      method=arm.removesuffix('_guarded');first=None;comparisons=0;reasons=Counter();trace=[]
      for i,t in enumerate(times):
       sample=dict(session_id='controlled-refinement',vial_id=ident,source_time_s=float(t),decision_time_s=float(t+.1),
           fill_started_s=0.,recipe_key='sim-recipe',reference_recipe_key='sim-recipe',
           phase_key=original['phases'][i],expected_phase_key=original['phases'][i],process_phase_key=original['phases'][i],
           reference_id='pre-session-analytic-reference',reference_available_s=-1.,context_available_s=0.,
           measurement_profile='toy-camera',reference_measurement_profile='toy-camera',
           geometry_status='simulated',vial_identity_unambiguous=True,measurement_valid=bool(original['valid'][i]),
           observed_lab=normalized[method][i].tolist(),expected_lab=expected[i].tolist(),
           process_state_source_s=float(t-settings.get('state_age',0.)),camera_reference_key='fixed-patch-profile',
           reference_camera_key='fixed-patch-profile',camera_health_valid=bool(health[i]))
       r=engine.update(sample);comparisons+=int(r['delta_e2000'] is not None)
       if r['reason']:reasons.update([r['reason']])
       if r['warning_time_s'] is not None and first is None:first=r['warning_time_s']
       if recipe==0 and hold==0 and seed==11 and condition in ('srgb_green_plus10pct','known_stale_process_1s') and arm in ('raw','srgb_gain_guarded'):
        trace.append(dict(sample=sample,result=r))
      results.append(dict(case_id=ident,recipe=recipe,seed=seed,hold_up_ml=hold,donor_id=donor_id,condition=condition,arm=arm,
          warned=first is not None,warning_time_s=first,assessable_samples=comparisons,blocked_reasons=dict(reasons),
          latency_s=first-original['onset_s'] if first is not None else None))
      if trace:showcases.append(dict(case_id=ident,condition=condition,arm=arm,trace=trace))
  print(json.dumps(dict(recipe_finished=recipe+1,evaluations=len(results),elapsed_s=round(time.perf_counter()-started,1))),flush=True)
 metrics={}
 for condition in CONDITIONS:
  metrics[condition]={}
  for arm in ARMS:
   selected=[r for r in results if r['condition']==condition and r['arm']==arm]
   clean=[r for r in selected if r['hold_up_ml']==0];injected=[r for r in selected if r['hold_up_ml']>0]
   assessable=[r for r in injected if r['assessable_samples']]
   metrics[condition][arm]=dict(clean_episodes=len(clean),clean_false_holds=sum(r['warned'] for r in clean),
       injected_episodes=len(injected),injected_warned=sum(r['warned'] for r in injected),
       injected_episodes_assessable=len(assessable),clean_episodes_assessable=sum(r['assessable_samples']>0 for r in clean),
       fully_blocked_episodes=sum(r['assessable_samples']==0 for r in selected),
       median_injected_latency_s=float(np.median([r['latency_s'] for r in injected if r['warned']])) if any(r['warned'] for r in injected) else None)
 write(a.out/'episode-results.json',results);write(a.out/'showcases.json',showcases)
 fig,axes=plt.subplots(2,1,figsize=(13,8),sharex=True)
 x=np.arange(len(CONDITIONS))
 for j,arm in enumerate(('raw','srgb_gain_guarded','linear_gain_guarded')):
  null=[metrics[c][arm]['clean_false_holds']/24*100 for c in CONDITIONS]
  eligible=[metrics[c][arm]['clean_episodes_assessable']/24*100 for c in CONDITIONS]
  axes[0].bar(x+(j-1)*.25,null,.25,label=arm)
  axes[1].bar(x+(j-1)*.25,eligible,.25,label=arm)
 axes[0].set_ylabel('Null episodes flagged (%)');axes[0].legend(ncol=3,fontsize=9)
 axes[1].set_ylabel('Null episodes assessable (%)');axes[1].set_xticks(x,[c.replace('_','\n') for c in CONDITIONS],fontsize=8)
 for ax in axes:ax.set_ylim(0,110);ax.grid(axis='y',alpha=.2)
 fig.suptitle('SIMULATION: drift correction versus abstention; no real quality accuracy')
 fig.tight_layout();fig.savefig(a.out/'refined-policy.png',dpi=160);plt.close(fig)
 summary=dict(schema_version=1,episodes=len(results)//len(ARMS),policy_evaluations=len(results),metrics=metrics,
     real_quality_accuracy=None,hardware_actions=0,run5_used=False,plan_sha256=sha(a.out/'plan.json'),limits=plan['limits'])
 summary['artifacts']={p.name:sha(p) for p in a.out.iterdir() if p.is_file()}
 write(a.out/'completed.json',summary)
 print(json.dumps(metrics,indent=2),flush=True)


if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--camera',type=Path,required=True);p.add_argument('--out',type=Path,required=True);main(p.parse_args())
