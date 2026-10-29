"""Fixed perturbations of recipe-reference timing, optics and input latency."""
import argparse,itertools,json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiment_stage2_policy import fixture,evaluate,Policy,read,write,sha,rgb_to_lab
from stage2_policy import EvidencePolicy


def main(a):
    if a.out.exists():raise ValueError('Use a fresh directory')
    baseline=read(a.simulation/'completed.json')
    if sha(a.simulation/'plan.json')!=baseline['plan_sha256']:raise ValueError('Simulation plan changed')
    # This is a new, declared stress experiment; do not change completed evidence.
    a.out.mkdir(parents=True)
    stresses=['nominal','reference_clock_plus_1s','unreported_green_gain_10pct','unreported_optical_density_plus_10pct']
    plan=dict(kind='controlled_reference_domain_stress',stresses=stresses,recipes=12,repeats=2,
        hold_up_ml=[0,2],qa_mode='ideal',policy=Policy().to_dict(),new_data_used=False,
        training_or_threshold_selection=False,hardware_connected=False,
        hypothesis='An accurate phase/progress reference and photometric commissioning are necessary.',
        simulation_plan_sha256=sha(a.simulation/'plan.json'),script_sha256=sha(Path(__file__)))
    write(a.out/'plan.json',plan);results=[]
    profiles=[(np.array(v,float),o) for v in ([10,10,10],[20,10,10]) for o in itertools.permutations(range(3))]
    for index,(volumes,order) in enumerate(profiles):
        for seed in range(2):
            for hold in [0.,2.]:
                for stress in stresses:
                    case=fixture(index,volumes,order,np.array([.1,.7,.2]),hold,seed,None)
                    if stress=='reference_clock_plus_1s':
                        case['expected_lab']=np.stack([np.interp(case['times']+1,case['times'],case['expected_lab'][:,j]) for j in range(3)],axis=1)
                    elif stress=='unreported_green_gain_10pct':
                        case['observed_rgb'][:,1]=np.minimum(255,case['observed_rgb'][:,1]*1.1)
                        case['observed_lab']=np.array([rgb_to_lab(v) for v in case['observed_rgb']])
                    elif stress=='unreported_optical_density_plus_10pct':
                        # Perturb transmittance relative to the fixed background in linear light.
                        x=case['expected_rgb']/255.;bg=np.array([220.,225.,220.])/255.
                        linear=np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)
                        background=np.where(bg<=.04045,bg/12.92,((bg+.055)/1.055)**2.4)
                        linear=background*np.maximum(linear/background,1e-6)**1.1
                        expected=np.where(linear<=.0031308,linear*12.92,1.055*linear**(1/2.4)-.055)*255
                        case['expected_lab']=np.array([rgb_to_lab(v) for v in expected])
                    r=evaluate(case,'phase_reference_3_frames',Policy(),f'stress-{index}-{seed}-{hold}');r.pop('trace')
                    results.append(dict(stress=stress,recipe_profile=index,seed=seed,hold_up_ml=hold,**r))
    metrics={}
    for name in stresses:
        clean=[r for r in results if r['stress']==name and r['hold_up_ml']==0]
        bad=[r for r in results if r['stress']==name and r['hold_up_ml']>0]
        metrics[name]=dict(clean_episodes=len(clean),clean_false_holds=sum(r['warned'] for r in clean),
            injected_episodes=len(bad),injected_warned=sum(r['warned'] for r in bad))
    write(a.out/'episodes.json',results)
    write(a.out/'completed.json',dict(schema_version=1,policy_evaluations=len(results),metrics=metrics,
        real_quality_accuracy=None,limits=['Analytic toy only; perturbation sizes are not measured plant uncertainty.',
            'Systematic optical changes are intentionally NOT announced as profile changes, to expose silent drift risk.',
            'One-second clock error distorts the expected progression; no fitted process-state estimator is claimed.'],
        plan_sha256=sha(a.out/'plan.json'),artifacts={'episodes.json':sha(a.out/'episodes.json')}))
    print(json.dumps(metrics,indent=2))


if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--simulation',required=True,type=Path);p.add_argument('--out',required=True,type=Path);main(p.parse_args())
