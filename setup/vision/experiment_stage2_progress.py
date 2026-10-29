"""Compare clock, phase-endpoint and metered-progression recipe references.

Physical flow duration varies while each ingredient's metered dose stays fixed.
All appearances/references are analytic toy values, not confirmed plant recipes.
"""
import argparse,itertools,json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiment_stage2_policy import prefix_at_volume,appearance,schedule,Policy,rgb_to_lab
from stage2_policy import EvidencePolicy
from stage2_trajectory import TrajectoryBank
from liquid_specialists_data import read,write,sha


def profiles_for(recipe,volumes,order):
    profiles=[];before=0
    for phase,tank in enumerate(order):
        progress=np.linspace(before,before+volumes[tank],81)
        parts=np.array([prefix_at_volume(v,volumes,order) for v in progress])
        labs=[rgb_to_lab(v).tolist() for v in appearance(parts)]
        profiles.append(dict(recipe_key=str(recipe),phase_key=str(phase+1),camera_profile='toy',
            cumulative_ml=progress.tolist(),lab=labs,available_s=-1.,reference_id=f'sim-{recipe}-{phase+1}',reference_status='simulated'))
        before+=volumes[tank]
    return profiles


def main(a):
 if a.out.exists():raise ValueError('Use fresh output')
 a.out.mkdir(parents=True)
 plan=dict(kind='process_progress_reference_ablation',flow_rate_variation=[0,.2,.4],pause_s_range=[1,4],
     profiles=12,seeds=[21,22],hold_up_ml=[0,2],donors=[[.7,.2,.1],[.1,.7,.2],[.2,.1,.7]],
     strategies=['clock_template','phase_endpoint','metered_bank'],reference_knots_per_phase=81,
     policy=Policy().to_dict(),reference_type='simulated_analytic_not_commissioned',
     time_is_not_used_for_metered_reference_lookup=True,run5_used=False,
     limits=['References and samples share an analytic optical family; no real quality accuracy.',
             'Metered/phase cues are ideal simulated process-state inputs.',
             'Phase-start/color uncertainty and plant logging accuracy remain to be measured.'])
 scripts=('experiment_stage2_progress.py','stage2_trajectory.py','stage2_policy.py','experiment_stage2_policy.py')
 plan['helper_sha256']={n:sha(Path(__file__).with_name(n)) for n in scripts}
 write(a.out/'plan.json',plan)
 for n in scripts:(a.out/n).write_bytes(Path(__file__).with_name(n).read_bytes())
 recipes=[(np.array(v,float),o) for v in ([10,10,10],[20,10,10]) for o in itertools.permutations(range(3))]
 profiles=[p for i,(volumes,order) in enumerate(recipes) for p in profiles_for(i,volumes,order)]
 bank=TrajectoryBank(profiles);write(a.out/'simulated-reference-bank.json',profiles);results=[]
 for recipe,(volumes,order) in enumerate(recipes):
  for seed in plan['seeds']:
   for variation in plan['flow_rate_variation']:
    rng=np.random.default_rng(20261023+recipe*100+seed)
    rates=np.array([3.,2.,4.])*rng.uniform(1-variation,1+variation,3)
    pauses=rng.uniform(1,4,2) if variation else np.array([2.,2.])
    stages=[];at=1.;prior_total=0.
    for phase,tank in enumerate(order):
     duration=volumes[tank]/rates[tank]
     stages.append((at,at+duration,tank,phase+1));at+=duration
     if phase<2:stages.append((at,at+pauses[phase],None,phase+1));at+=pauses[phase]
    end=at;stages.append((at,at+2,None,3));times=np.round(np.arange(0,at+2,.2),8)
    nominal,_,_=schedule(volumes,order)
    all_parts=[];phase_keys=[];clock_parts=[]
    for t in times:
     parts=np.zeros(3);current=1;clock=np.zeros(3)
     for start,stop,tank,phase in stages:
      if start<=t<stop:current=phase
      if tank is not None:parts[tank]+=min(max(t-start,0),stop-start)*rates[tank]
     for start,stop,tank,_ in nominal:
      if tank is not None:clock[tank]+=min(max(t-start,0),stop-start)*np.array([3.,2.,4.])[tank]
     all_parts.append(parts);clock_parts.append(clock);phase_keys.append(str(current))
    all_parts=np.array(all_parts);clock_lab=np.array([rgb_to_lab(v) for v in appearance(np.array(clock_parts))])
    noise=rng.normal(0,1.2,(len(times),3));gain=rng.uniform(.97,1.03);bias=rng.uniform(-1,1,3)
    scenarios=[(0.,np.array(plan['donors'][0]),'none')]+[(2.,np.array(d),str(j)) for j,d in enumerate(plan['donors'])]
    for hold,donor,donor_id in scenarios:
     actual=[]
     for parts in all_parts:
      old=min(parts.sum(),hold)
      actual.append(parts-prefix_at_volume(old,volumes,order)+old*donor)
     observed=np.clip(appearance(np.array(actual))*gain+bias+noise,0,255)
     observed_lab=np.array([rgb_to_lab(v) for v in observed])
     for strategy in plan['strategies']:
      engine=EvidencePolicy();first=None;assessable=0
      for i,t in enumerate(times):
       phase=phase_keys[i];progress=float(all_parts[i].sum())
       if strategy=='clock_template':expected=clock_lab[i]
       elif strategy=='phase_endpoint':
        cumulative=float(sum(volumes[order[k]] for k in range(int(phase))))
        expected=bank.lookup(str(recipe),phase,'toy',cumulative,0.)['expected_lab']
       else:
        ref=bank.lookup(str(recipe),phase,'toy',progress,0.)
        expected=ref.get('expected_lab')
       s=dict(session_id='progress-test',vial_id='v1',source_time_s=float(t),decision_time_s=float(t+.1),
           fill_started_s=0.,recipe_key=str(recipe),reference_recipe_key=str(recipe),phase_key=phase,expected_phase_key=phase,
           reference_id='frozen-sim-profile',reference_available_s=-1.,context_available_s=0.,measurement_profile='toy',
           reference_measurement_profile='toy',geometry_status='simulated',vial_identity_unambiguous=True,
           measurement_valid=progress>.15 and expected is not None,observed_lab=observed_lab[i].tolist(),expected_lab=expected)
       r=engine.update(s);assessable+=int(r['delta_e2000'] is not None)
       if r['warning_time_s'] is not None and first is None:first=r['warning_time_s']
      results.append(dict(recipe=recipe,seed=seed,flow_variation=variation,hold_up_ml=hold,donor=donor_id,strategy=strategy,
          warned=first is not None,warning_time_s=first,assessable_samples=assessable))
 metrics={}
 for variation in plan['flow_rate_variation']:
  metrics[str(variation)]={}
  for strategy in plan['strategies']:
   subset=[r for r in results if r['flow_variation']==variation and r['strategy']==strategy]
   clean=[r for r in subset if r['hold_up_ml']==0];bad=[r for r in subset if r['hold_up_ml']>0]
   metrics[str(variation)][strategy]=dict(clean_episodes=len(clean),clean_warned=sum(r['warned'] for r in clean),
       injected_episodes=len(bad),injected_warned=sum(r['warned'] for r in bad),
       assessable_episodes=sum(r['assessable_samples']>0 for r in subset))
 write(a.out/'episode-results.json',results)
 result=dict(schema_version=1,episodes=len(results)//3,policy_evaluations=len(results),metrics=metrics,
     real_quality_accuracy=None,run5_used=False,hardware_actions=0,plan_sha256=sha(a.out/'plan.json'),limits=plan['limits'])
 result['artifacts']={p.name:sha(p) for p in a.out.iterdir() if p.is_file()};write(a.out/'completed.json',result)
 print(json.dumps(metrics,indent=2))


if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);main(p.parse_args())
