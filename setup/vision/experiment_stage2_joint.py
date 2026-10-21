"""Integrated candidate on fresh analytic nuisance seeds; no plant validation."""
import argparse,itertools,json
from collections import Counter
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiment_stage2_policy import prefix_at_volume,appearance,schedule,Policy,rgb_to_lab
from experiment_stage2_progress import profiles_for
from stage2_trajectory import TrajectoryBank
from stage2_policy import EvidencePolicy
from stage2_guard import GuardedEvidence
from stage2_camera_reference import correction,perturb,anchor_health
from liquid_specialists_data import read,write,sha


def main(a):
 if a.out.exists():raise ValueError('Use fresh output')
 camera=read(a.camera/'completed.json');calibration=read(a.camera/'calibration.json')
 if sha(a.camera/'calibration.json')!=camera['artifacts']['calibration.json']:raise ValueError('Calibration changed')
 a.out.mkdir(parents=True)
 plan=dict(kind='joint_stage2_candidate_on_fresh_simulator_seeds',seeds=[31,32],recipe_profiles=12,
     flow_variation=.3,pause_s_range=[1,4],camera_gain_range=[.85,1.15],
     camera_domains=['linear','srgb'],measurement_dropout_probability=.05,one_anchor_occlusion_every_samples=50,
     hold_up_ml=[0,.5,2],donors=[[.7,.2,.1],[.1,.7,.2],[.2,.1,.7]],
     candidate='metered trajectory + srgb gain correction + source/camera guards + three samples',
     baseline='ungated clock template + raw camera Lab + three samples',policy=Policy().to_dict(),
     camera_calibration_sha256=sha(a.camera/'calibration.json'),
     limits=['New nuisance seeds within the same analytic optical/plug-flow family, not independent plant data.',
         'Recipe/phase/meter inputs are truthful simulated values; real source accuracy is unvalidated.',
         'Simulated warnings identify injected cases, not confirmed unacceptable product.',
         'No hardware action, measured flushing or approved label is produced.'])
 scripts=('experiment_stage2_joint.py','experiment_stage2_progress.py','stage2_trajectory.py','stage2_guard.py','stage2_camera_reference.py','stage2_policy.py')
 plan['helper_sha256']={n:sha(Path(__file__).with_name(n)) for n in scripts};write(a.out/'plan.json',plan)
 for n in scripts:(a.out/n).write_bytes(Path(__file__).with_name(n).read_bytes())
 recipes=[(np.array(v,float),o) for v in ([10,10,10],[20,10,10]) for o in itertools.permutations(range(3))]
 bank=TrajectoryBank([p for i,(v,o) in enumerate(recipes) for p in profiles_for(i,v,o)])
 anchors=np.array(calibration['reference_rgb']);tolerance=calibration['maximum_disagreement'];results=[];showcases=[]
 for recipe,(volumes,order) in enumerate(recipes):
  for seed in plan['seeds']:
   rng=np.random.default_rng(20261024+recipe*100+seed);rates=np.array([3.,2.,4.])*rng.uniform(.7,1.3,3)
   pauses=rng.uniform(1,4,2);stages=[];at=1.
   for phase,tank in enumerate(order):
    duration=volumes[tank]/rates[tank];stages.append((at,at+duration,tank,phase+1));at+=duration
    if phase<2:stages.append((at,at+pauses[phase],None,phase+1));at+=pauses[phase]
   stages.append((at,at+2,None,3));times=np.round(np.arange(0,at+2,.2),8);nominal,_,_=schedule(volumes,order)
   parts=[];clock=[];phases=[]
   for t in times:
    p=np.zeros(3);q=np.zeros(3);key=1
    for start,stop,tank,phase in stages:
     if start<=t<stop:key=phase
     if tank is not None:p[tank]+=min(max(t-start,0),stop-start)*rates[tank]
    for start,stop,tank,_ in nominal:
     if tank is not None:q[tank]+=min(max(t-start,0),stop-start)*np.array([3.,2.,4.])[tank]
    parts.append(p);clock.append(q);phases.append(str(key))
   parts=np.array(parts);clock_lab=np.array([rgb_to_lab(v) for v in appearance(np.array(clock))])
   gain=rng.uniform(.85,1.15,3);domain='linear' if seed==31 else 'srgb'
   noise=rng.normal(0,1.2,(len(times),3));dropout=rng.random(len(times))<.05
   patch=np.tile(anchors[None,:,:],(len(times),1,1));patch=perturb(patch,gain=gain,domain=domain)
   for i in range(len(times)):
    if i%50==25:patch[i,0]=perturb(patch[i,0],gain=[.8,.8,.8])
   patch=np.clip(np.rint(patch+rng.normal(0,.15,patch.shape)),0,255)
   scenarios=[(0.,np.array(plan['donors'][0]),'none')]+[(h,np.array(d),str(j)) for j,d in enumerate(plan['donors']) for h in (.5,2.)]
   for hold,donor,donor_id in scenarios:
    actual=[]
    for p in parts:
     old=min(p.sum(),hold);actual.append(p-prefix_at_volume(old,volumes,order)+old*donor)
    rgb=np.clip(appearance(np.array(actual))+noise,0,255);rgb=np.rint(perturb(rgb,gain=gain,domain=domain))
    identity=f'r{recipe}-seed{seed}-d{donor_id}-h{hold}'
    for arm in ('raw_clock','candidate'):
     engine=EvidencePolicy() if arm=='raw_clock' else GuardedEvidence()
     first=None;assessable=0;blocked=Counter();trace=[]
     for i,t in enumerate(times):
      if arm=='candidate':
       ref=bank.lookup(str(recipe),phases[i],'toy',parts[i].sum(),0.)
       expected=ref.get('expected_lab');corrected,_,_=correction(rgb[i:i+1],patch[i],anchors,'srgb_gain')
       observed=rgb_to_lab(corrected[0]) if corrected is not None else [np.nan]*3
      else:expected=clock_lab[i];observed=rgb_to_lab(rgb[i])
      healthy=anchor_health(patch[i],anchors,tolerance)[0]
      s=dict(session_id='joint',vial_id=identity,source_time_s=float(t),decision_time_s=float(t+.1),fill_started_s=0.,
          recipe_key=str(recipe),reference_recipe_key=str(recipe),phase_key=phases[i],expected_phase_key=phases[i],
          process_phase_key=phases[i],reference_id='frozen-simulated-trajectory',reference_available_s=-1.,context_available_s=0.,
          measurement_profile='toy',reference_measurement_profile='toy',geometry_status='simulated',
          vial_identity_unambiguous=True,measurement_valid=bool(parts[i].sum()>.15 and not dropout[i] and expected is not None),
          observed_lab=np.asarray(observed).tolist(),expected_lab=np.asarray(expected).tolist() if expected is not None else None,
          process_state_source_s=float(t),camera_reference_key='static-scene',reference_camera_key='static-scene',camera_health_valid=bool(healthy))
      r=engine.update(s);assessable+=int(r['delta_e2000'] is not None)
      if r['reason']:blocked.update([r['reason']])
      if r['warning_time_s'] is not None and first is None:first=r['warning_time_s']
      if recipe==0 and seed==31 and donor_id in ('none','1') and hold in (0.,2.):trace.append(dict(sample=s,result=r,meter_total_ml=float(parts[i].sum())))
     results.append(dict(case_id=identity,recipe=recipe,seed=seed,hold_up_ml=hold,donor_id=donor_id,arm=arm,
         warned=first is not None,warning_time_s=first,assessable_samples=assessable,blocked_reasons=dict(blocked)))
     if trace:showcases.append(dict(case_id=identity,arm=arm,trace=trace))
 metrics={}
 for arm in ('raw_clock','candidate'):
  selected=[r for r in results if r['arm']==arm];clean=[r for r in selected if r['hold_up_ml']==0];bad=[r for r in selected if r['hold_up_ml']>0]
  reasons=Counter()
  for r in selected:reasons.update(r['blocked_reasons'])
  metrics[arm]=dict(clean_episodes=len(clean),clean_warned=sum(r['warned'] for r in clean),
      injected_episodes=len(bad),injected_warned=sum(r['warned'] for r in bad),
      assessable_episodes=sum(r['assessable_samples']>0 for r in selected),blocked_sample_reasons=dict(reasons),
      by_hold_up_ml={str(h):dict(episodes=len([r for r in bad if r['hold_up_ml']==h]),warned=sum(r['warned'] for r in bad if r['hold_up_ml']==h)) for h in (.5,2.)})
 write(a.out/'episode-results.json',results);write(a.out/'showcases.json',showcases)
 result=dict(schema_version=1,episodes=len(results)//2,policy_evaluations=len(results),metrics=metrics,
     real_quality_accuracy=None,run5_used=False,hardware_actions=0,plan_sha256=sha(a.out/'plan.json'),limits=plan['limits'])
 result['artifacts']={p.name:sha(p) for p in a.out.iterdir() if p.is_file()};write(a.out/'completed.json',result)
 print(json.dumps(metrics,indent=2))


if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--camera',type=Path,required=True);p.add_argument('--out',type=Path,required=True);main(p.parse_args())
