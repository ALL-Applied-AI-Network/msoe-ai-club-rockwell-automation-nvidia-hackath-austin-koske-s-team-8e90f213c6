"""Controlled decision-policy simulation, NOT real carryover validation.

An analytic three-channel absorbance toy provides known recipe-prefix colors.
No images, training data, fitted physics, product labels, or hardware commands
are generated. Real Run 4 contributes ONLY a frozen QA-availability schedule.
"""
import argparse
from collections import defaultdict
import itertools
import json
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from replay.measurement import rgb_to_lab
from liquid_specialists_data import read,write,sha
from stage2_policy import Policy,EvidencePolicy,ActionCoordinator


ABSORBANCE=np.array([[.1,1.25,1.35],[1.35,.9,.15],[1.15,.1,1.4]])
FLOWS=np.array([3.,2.,4.])
ARMS={'final_reference_every_frame':(5.,3),'phase_reference_single_frame':(5.,1),
      'phase_reference_3_frames':(5.,3),'completed_fill_only':(5.,3),
      'phase_reference_threshold10':(10.,3),'phase_reference_threshold15':(15.,3)}


def prefix_at_volume(total,volumes,order):
    parts=np.zeros(3);remaining=total
    for tank in order:
        parts[tank]=min(max(remaining,0),volumes[tank]);remaining-=parts[tank]
    return parts


def schedule(volumes,order):
    stages=[];at=1.
    for i,tank in enumerate(order):
        end=at+volumes[tank]/FLOWS[tank]
        stages.append((at,end,tank,f'ingredient-{i+1}'));at=end
        if i<2:
            stages.append((at,at+2.,None,f'pause-{i+1}'));at+=2.
    end=at
    stages.append((at,at+2.,None,'dispense_complete'));return stages,end,at+2.


def appearance(parts):
    total=parts.sum(axis=-1,keepdims=True)
    fraction=parts/np.maximum(total,1e-9)
    bg=np.array([220.,225.,220.])/255
    bg=np.where(bg<=.04045,bg/12.92,((bg+.055)/1.055)**2.4)
    linear=bg*np.exp(-1.7*(fraction@ABSORBANCE.T))
    srgb=np.where(linear<=.0031308,linear*12.92,1.055*np.maximum(linear,0)**(1/2.4)-.055)
    return np.clip(srgb*255,0,255)


def fixture(recipe_index,volumes,order,donor,hold,seed,availability):
    stages,end,departure=schedule(volumes,order);times=np.round(np.arange(0,departure,.2),8)
    planned=[];phases=[]
    for t in times:
        volume=np.zeros(3);phase='prefill'
        for a,b,tank,name in stages:
            if a<=t<b:phase=name
            if tank is not None:volume[tank]+=min(max(t-a,0),b-a)*FLOWS[tank]
        planned.append(volume);phases.append(phase)
    planned=np.array(planned);actual=[]
    for parts in planned:
        old=min(parts.sum(),hold)
        actual.append(parts-prefix_at_volume(old,volumes,order)+old*donor)
    expected_rgb=appearance(planned);observed_rgb=appearance(np.array(actual))
    rng=np.random.default_rng(20261020+recipe_index*1000+seed)
    gain=rng.uniform(.97,1.03);bias=rng.uniform(-1.,1.,3)
    noise=rng.normal(0,1.2,observed_rgb.shape)
    # Single-frame camera nuisance is NOT a carryover event.
    impulse=rng.random(len(times))<.01
    noise[impulse]+=rng.normal(0,12,(int(impulse.sum()),3))
    observed_rgb=np.clip(observed_rgb*gain+bias+noise,0,255)
    valid=planned.sum(axis=1)>.15
    if availability is not None:
        lookup=np.minimum((np.arange(len(times))/max(1,len(times)-1)*(len(availability)-1)).astype(int),len(availability)-1)
        valid &= np.array(availability,dtype=bool)[lookup]
    return dict(times=times,phases=phases,valid=valid,expected_rgb=expected_rgb,observed_rgb=observed_rgb,
        expected_lab=np.array([rgb_to_lab(v) for v in expected_rgb]),observed_lab=np.array([rgb_to_lab(v) for v in observed_rgb]),
        final_lab=rgb_to_lab(appearance(volumes)),end_s=end,departure_s=departure,
        onset_s=float(times[np.flatnonzero(planned.sum(axis=1)>.15)[0]]),volume=planned.sum(axis=1))


def evaluate(case,arm,config,case_id,retain=False):
    engine=EvidencePolicy(config);first=None;logs=[];comparisons=0;last=None
    for i,t in enumerate(case['times']):
        if arm=='completed_fill_only' and t<case['end_s']+.5:continue
        expected=case['final_lab'] if arm in ('completed_fill_only','final_reference_every_frame') else case['expected_lab'][i]
        sample=dict(session_id='simulation',vial_id=case_id,source_time_s=float(t),decision_time_s=float(t+.1),
            fill_started_s=0.,recipe_key='simulated-recipe',reference_recipe_key='simulated-recipe',
            phase_key=case['phases'][i],expected_phase_key=case['phases'][i],
            reference_id='analytic-reference-before-session',reference_available_s=-1.,context_available_s=0.,
            measurement_profile='toy-camera',reference_measurement_profile='toy-camera',
            geometry_status='simulated',vial_identity_unambiguous=True,measurement_valid=bool(case['valid'][i]),
            observed_lab=case['observed_lab'][i].tolist(),expected_lab=expected.tolist())
        result=engine.update(sample);last=result
        comparisons+=int(result['delta_e2000'] is not None)
        if result['warning_time_s'] is not None and first is None:first=result['warning_time_s']
        if retain:logs.append(dict(sample=sample,result=result))
    return dict(warned=first is not None,first_warning_s=first,
        latency_after_injected_onset_s=first-case['onset_s'] if first is not None else None,
        warning_lead_before_scheduled_departure_s=case['departure_s']-first if first is not None else None,
        comparison_samples=comparisons,last_status=last['status'] if last else 'unavailable',trace=logs)


def aggregate(records):
    out={}
    for (qa,arm),group in itertools.groupby(sorted(records,key=lambda r:(r['qa_mode'],r['arm'])),key=lambda r:(r['qa_mode'],r['arm'])):
        rows=list(group);clean=[r for r in rows if r['hold_up_ml']==0];positive=[r for r in rows if r['hold_up_ml']>0]
        out.setdefault(qa,{})[arm]=dict(episodes=len(rows),clean_episodes=len(clean),injected_episodes=len(positive),
            clean_false_holds=sum(r['warned'] for r in clean),clean_false_hold_fraction=np.mean([r['warned'] for r in clean]).item(),
            injected_cases_warned=sum(r['warned'] for r in positive),injected_warning_fraction=np.mean([r['warned'] for r in positive]).item(),
            median_injected_warning_latency_s=float(np.median([r['latency_after_injected_onset_s'] for r in positive if r['warned']])),
            median_warning_lead_s=float(np.median([r['warning_lead_before_scheduled_departure_s'] for r in positive if r['warned']])),
            fully_unassessable_episodes=sum(r['comparison_samples']==0 for r in rows),
            by_hold_up_ml={str(h):dict(episodes=len([r for r in positive if r['hold_up_ml']==h]),
                warned=sum(r['warned'] for r in positive if r['hold_up_ml']==h)) for h in [.5,1.,2.,5.]})
    return out


def guard_experiment():
    changes=[('unknown_recipe',dict(recipe_key=None)),('unreviewed_geometry',dict(geometry_status='unreviewed_model_proposals')),
        ('ambiguous_identity',dict(vial_identity_unambiguous=False)),('future_reference',dict(reference_available_s=1.)),
        ('future_recipe',dict(context_available_s=20.)),('wrong_recipe_reference',dict(reference_recipe_key='other')),
        ('wrong_profile',dict(reference_measurement_profile='other')),('unknown_phase',dict(phase_key=None)),
        ('wrong_phase_reference',dict(expected_phase_key='wrong')),('missing_reference_identity',dict(reference_id=None)),
        ('invalid_color',dict(observed_lab=[None,0,0])),('qa_failure',dict(measurement_valid=False)),
        ('stale_sample',dict(decision_time_s=100.))]
    results=[]
    for name,change in changes:
        e=EvidencePolicy();warnings=0
        for i in range(5):
            t=1+i*.2
            s=dict(session_id='guard',vial_id='v1',source_time_s=t,decision_time_s=t+.1,fill_started_s=0.,
                recipe_key='a',reference_recipe_key='a',phase_key='1',expected_phase_key='1',
                reference_id='known',reference_available_s=-1.,context_available_s=0.,
                measurement_profile='m',reference_measurement_profile='m',geometry_status='simulated',
                vial_identity_unambiguous=True,measurement_valid=True,observed_lab=[40,40,0],expected_lab=[40,0,0])
            r=e.update(dict(s,**change));warnings+=int(r['warning_time_s'] is not None)
        results.append(dict(case=name,frames=5,warnings=warnings,last_reason=r['reason']))
    return results


def main(a):
    if a.out.exists():raise ValueError('Use a fresh output directory')
    real=read(a.real/'completed.json')
    for name in ('observations.json','frames.jsonl'):
        if sha(a.real/name)!=real['artifacts'][name]:raise ValueError('Run4 evidence changed')
    observations=read(a.real/'observations.json')
    rows=[json.loads(s) for s in (a.real/'frames.jsonl').read_text().splitlines()]
    schedules=[[r['measurement']['measurement_valid'] for r in rows if o['start_frame']<=r['source_frame']<o['departure_frame']] for o in observations]
    a.out.mkdir(parents=True)
    plan=dict(schema_version=1,kind='analytic_policy_simulation_not_product_validation',seed=20261020,
        recipe_profiles=12,donor_compositions=[[.7,.2,.1],[.1,.7,.2],[.2,.1,.7]],hold_up_ml=[0,.5,1,2,5],
        repeats_per_recipe_and_fault=2,qa_modes=['ideal','run4_availability'],
        arms={arm:Policy(threshold_de2000=threshold,consecutive_samples=n).to_dict() for arm,(threshold,n) in ARMS.items()},
        absorbance_matrix=ABSORBANCE.tolist(),flow_ml_per_s=FLOWS.tolist(),ingredient_pause_s=2.,
        nuisance=dict(global_gain=[.97,1.03],channel_bias=[-1,1],rgb_noise_sd=1.2,impulse_probability=.01,impulse_rgb_sd=12),
        same_noise_paired_across_fault_severities=True,real_qa_schedule_interpolated_by_relative_cycle_position=True,
        real_qa_input_sha256=sha(a.real/'frames.jsonl'),expected_phase_and_delivered_volume='perfect simulated process-state inputs',
        simulated_processing_latency_s=.1,no_training=True,no_real_recipe_labels=True,no_run5_data=True,
        limits=['Analytic three-channel optical toy, not fitted or validated fluid physics.',
                'Injected hold-up is a known simulated perturbation, not a product-defect label.',
                'Noise, flow rates, holds and recovery counts are engineering fixtures, not plant estimates.',
                'Same analytic model generates references and samples; real domain mismatch is not measured.',
                'Real QA scheduling is transplanted by normalized cycle position, not real timed pump synchronization.',
                'No hardware commands, approved reference bank or inference accuracy is evaluated.'])
    plan['scripts']={n:sha(Path(__file__).with_name(n)) for n in ('experiment_stage2_policy.py','stage2_policy.py')}
    write(a.out/'plan.json',plan)
    for n in plan['scripts']:(a.out/n).write_bytes(Path(__file__).with_name(n).read_bytes())
    results=[];showcase=None;started=time.perf_counter()
    profiles=[(np.array(vol,float),order) for vol in ([10,10,10],[20,10,10]) for order in itertools.permutations(range(3))]
    for recipe,(volumes,order) in enumerate(profiles):
        perturbations=[(0.,np.array([.7,.2,.1]),'none')]+[(h,np.array(d),str(j)) for j,d in enumerate(plan['donor_compositions']) for h in [.5,1.,2.,5.]]
        for hold,donor,donor_id in perturbations:
            for seed in range(2):
                for qa in plan['qa_modes']:
                    availability=None if qa=='ideal' else schedules[recipe%len(schedules)]
                    case=fixture(recipe,volumes,order,donor,hold,seed,availability)
                    ident=f'r{recipe}-d{donor_id}-h{hold}-seed{seed}-{qa}'
                    for arm,(threshold,n) in ARMS.items():
                        keep=recipe==0 and seed==0 and qa=='ideal' and hold in (0.,2.) and donor_id in ('none','1') and arm in ('phase_reference_3_frames','final_reference_every_frame')
                        result=evaluate(case,arm,Policy(threshold_de2000=threshold,consecutive_samples=n),ident,keep)
                        trace=result.pop('trace')
                        results.append(dict(case_id=ident,recipe_profile=recipe,hold_up_ml=hold,donor_id=donor_id,seed=seed,
                            qa_mode=qa,arm=arm,**result))
                        if keep:
                            write(a.out/(ident+'-'+arm+'.json'),dict(times=case['times'].tolist(),
                                expected_rgb=case['expected_rgb'].tolist(),observed_rgb=case['observed_rgb'].tolist(),
                                end_s=case['end_s'],departure_s=case['departure_s'],trace=trace))
        print(json.dumps(dict(recipe_profiles_finished=recipe+1,policy_evaluations=len(results),elapsed_s=round(time.perf_counter()-started,1))),flush=True)
    write(a.out/'episode-results.json',results)
    groups=aggregate(results);guards=guard_experiment();write(a.out/'guard-results.json',guards)
    coordinator=ActionCoordinator('sim-loop')
    coordinator.complete('v1','recipe-b','review_required',20,session_id='sim-loop')
    coordinator.complete('v2','recipe-b','review_required',40,session_id='sim-loop')
    coordinator.acknowledge_flush(42,'SIMULATED_OPERATOR','SIMULATED_FLUSH_RECEIPT',session_id='sim-loop')
    coordinator.complete('v3','recipe-b','within_expected_band',62,session_id='sim-loop')
    coordinator.complete('v4','recipe-b','within_expected_band',82,session_id='sim-loop')
    write(a.out/'action-loop.json',dict(source='scripted_action_contract_fixture_not_plant_execution',events=coordinator.events))
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    selected=['final_reference_every_frame','phase_reference_single_frame','phase_reference_3_frames','completed_fill_only']
    names=['Final color\nevery frame','Phase reference\none sample','Phase reference\n3 samples','Completed fill\nonly']
    for i,qa in enumerate(plan['qa_modes']):
        clean=[groups[qa][arm]['clean_false_hold_fraction']*100 for arm in selected]
        fault=[groups[qa][arm]['injected_warning_fraction']*100 for arm in selected]
        x=np.arange(len(selected));axes[i].bar(x-.17,clean,.34,label='Null-fixture false holds',color='#c66056')
        axes[i].bar(x+.17,fault,.34,label='Injected-case warnings',color='#287e96')
        axes[i].set_xticks(x,names,fontsize=9);axes[i].set_ylim(0,110);axes[i].set_ylabel('Scenario percentage')
        axes[i].set_title('Ideal sample availability' if qa=='ideal' else 'With Run 4 QA availability schedule')
        axes[i].grid(axis='y',alpha=.2)
    axes[0].legend(fontsize=9);fig.suptitle('SIMULATION ONLY: decision-policy comparison; not real defect accuracy',fontsize=13)
    fig.tight_layout();fig.savefig(a.out/'policy-comparison.png',dpi=160);plt.close(fig)
    output=dict(schema_version=1,simulated_episodes=len(results)//len(ARMS),policy_evaluations=len(results),
        recipe_profiles=len(profiles),metrics=groups,input_guard_cases=len(guards),
        guarded_frames=sum(r['frames'] for r in guards),guarded_warnings=sum(r['warnings'] for r in guards),
        simulated_action_sequence=[r['action'] for r in coordinator.events],hardware_actions=0,
        real_product_defect_accuracy=None,elapsed_s=time.perf_counter()-started,
        plan_sha256=sha(a.out/'plan.json'),limits=plan['limits'])
    output['artifacts']={p.name:sha(p) for p in a.out.iterdir() if p.is_file()}
    write(a.out/'completed.json',output)
    print(json.dumps(output,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--real',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
