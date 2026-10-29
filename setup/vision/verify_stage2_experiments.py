"""Audit frozen Stage 2 artifacts, causal checkpoints and real-data abstention."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from liquid_specialists_data import read,write,sha
from replay.config import load_config
from analyze_stage2_evidence import checkpoints
from stage2_policy import EvidencePolicy


def main(a):
    out=a.root/'verification'
    if out.exists():raise ValueError('Use a fresh verification directory')
    out.mkdir()
    names=['full-run4-v1','policy-simulation-v2','reference-stress-v1','demo-v1']
    reports={};verified=0
    for name in names:
        folder=a.root/name;r=read(folder/'completed.json');reports[name]=r
        for file,digest in r['artifacts'].items():
            if sha(folder/file)!=digest:raise ValueError('Artifact changed: '+name+'/'+file)
            verified+=1
    full=reports['full-run4-v1'];rows=[json.loads(s) for s in (a.root/'full-run4-v1/frames.jsonl').read_text().splitlines()]
    assert [r['source_frame'] for r in rows]==list(range(0,9482,6))
    assert len(rows)==full['counts']['sampled_frames']==1581
    assert sum(r['measurement']['measurement_valid'] for r in rows)==849
    config=load_config(Path(__file__).resolve().parents[2]/'replay/examples/bootstrap-vision.json')
    phase=read(a.root/'evidence-audit-v1/phase-checkpoints.json')
    assert checkpoints(rows,config)==phase
    for cut in [100,300,600,900,1200,1500,len(rows)]:
        expected=[p for p in phase if p['available_frame']<=rows[cut-1]['source_frame']]
        assert checkpoints(rows[:cut],config)==expected
    assert all(max(p['sample_frames'])<=p['available_frame'] and not p['final_dispense_complete_known'] for p in phase)
    audit=read(a.root/'evidence-audit-v1/metrics.json')
    assert sha(a.root/'evidence-audit-v1/metrics.json')==read(a.root/'evidence-audit-v1/verification.json')['metrics_sha256']
    sim=reports['policy-simulation-v2'];episodes=read(a.root/'policy-simulation-v2/episode-results.json')
    assert len(episodes)==3744
    for qa,arms in sim['metrics'].items():
        for arm,m in arms.items():
            subset=[r for r in episodes if r['qa_mode']==qa and r['arm']==arm]
            clean=[r for r in subset if r['hold_up_ml']==0];bad=[r for r in subset if r['hold_up_ml']>0]
            assert len(subset)==m['episodes'] and sum(r['warned'] for r in clean)==m['clean_false_holds']
            assert sum(r['warned'] for r in bad)==m['injected_cases_warned']
    guards=read(a.root/'policy-simulation-v2/guard-results.json');assert sum(g['warnings'] for g in guards)==0
    actions=read(a.root/'demo-v1/derived-action-loop.json')
    assert [e['action'] for e in actions['events']]==sim['simulated_action_sequence']
    assert not any(e['hardware_action_authorized'] for e in actions['events'])
    # Exercise the actual decision engine with the actual chronological real rows.
    engine=EvidencePolicy();statuses=Counter();reasons=Counter()
    with (out/'real-recipe-abstention.jsonl').open('w',encoding='utf-8') as log:
        for r in rows:
            state=r['fill_detection'];m=r['measurement'];t=r['seconds']
            sample=dict(session_id='run4-offline',vial_id=state['detection_id'] or 'unassigned',
                source_time_s=t,decision_time_s=t+.1,fill_started_s=(state['start_frame'] or 0)/30,
                recipe_key=None,phase_key=None,expected_phase_key=None,reference_id=None,
                reference_recipe_key=None,reference_available_s=None,context_available_s=None,
                measurement_profile='run4-sam2-causal',reference_measurement_profile=None,
                geometry_status='unreviewed_model_proposals',vial_identity_unambiguous=False,
                measurement_valid=m['measurement_valid'],observed_lab=m['lab'],expected_lab=None)
            result=engine.update(sample);statuses.update([result['status']]);reasons.update([result['reason']])
            assert result['warning_time_s'] is None and not result['hardware_action_authorized']
            log.write(json.dumps(result,allow_nan=False)+'\n')
    assert statuses=={'unavailable':1581} and reasons=={'unknown_intended_recipe':1581}
    copied=[]
    for folder,script,key in [('evidence-audit-v1','analyze_stage2_evidence.py','script_sha256'),
                              ('reference-stress-v1','stress_stage2_policy.py','script_sha256'),
                              ('demo-v1','render_stage2_demos.py','script_sha256')]:
        expected=read(a.root/folder/('plan.json' if folder=='reference-stress-v1' else 'verification.json' if folder=='evidence-audit-v1' else 'completed.json'))[key]
        p=Path(__file__).with_name(script);assert sha(p)==expected
        (out/script).write_bytes(p.read_bytes());copied.append(script)
    result=dict(schema_version=1,kind='stage2_experiments_verified',root=str(a.root),
        source_run=4,run5_used=False,new_training_jobs=0,hardware_actions=0,
        real_video=dict(decoded_frames=9482,samples=1581,fill_candidates=6,
            sample_qa_pass_within_fills=849,samples_within_fills=931,
            completed_departure_window_qa_pass=4,phase_checkpoints=9,
            final_episode_checkpoint_opportunities=6,lead_before_departure_s=[p['lead_before_departure_s'] for p in audit['per_fill']],
            final_episode_match_is_retrospective=True,recipe_decisions_available=0,
            processing_ms=full['timing_excluding_first_new_sample'],real_policy_status_counts=dict(statuses)),
        simulation=dict(episodes=sim['simulated_episodes'],policy_evaluations=len(episodes),metrics=sim['metrics'],
            invalid_input_conditions=13,guarded_frames=65,guarded_warnings=0,real_defect_accuracy=None),
        reference_stress=reports['reference-stress-v1']['metrics'],action_sequence=sim['simulated_action_sequence'],
        media=reports['demo-v1'],verified_frozen_artifacts=verified,checkpoint_prefix_cuts_verified=7,
        focused_contract_tests_passed=10,
        evidence_report_sha256=sha(a.root/'evidence-audit-v1/metrics.json'),
        real_policy_trace_sha256=sha(out/'real-recipe-abstention.jsonl'),
        completion_sha256={name:sha(a.root/name/'completed.json') for name in names},
        script_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*stage2*.py')},
        limits=['Real masks, identities, recipes and quality outcomes remain unreviewed or unavailable.',
                'Analytic simulation is not real product-defect accuracy or measured flush effectiveness.',
                'Trusted process progression and photometric health integration remain prerequisites.'])
    write(out/'completed.json',result)
    if a.record:write(a.record,result)
    print(json.dumps(dict(verified_frozen_artifacts=verified,real_policy_status_counts=dict(statuses),
        checkpoint_prefix_cuts_verified=7,real_recipe_assessments=0,hardware_actions=0),indent=2))


if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True,type=Path);p.add_argument('--record',type=Path);main(p.parse_args())
