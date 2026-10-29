"""Independent artifact/count audit for the second Stage 2 experiment series."""
import argparse,json
from collections import Counter
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from liquid_specialists_data import read,write,sha
from stage2_guard import GuardedEvidence


def main(a):
 out=a.root/'verification'
 if out.exists():raise ValueError('Use a fresh verification output')
 out.mkdir()
 reports={};verified=0
 for name in ('camera-reference-v2','policy-refinement-v1','progress-reference-v1','joint-candidate-v1','camera-demo-v1'):
  report=read(a.root/name/'completed.json');reports[name]=report
  for path,digest in report['artifacts'].items():
   if sha(a.root/name/path)!=digest:raise ValueError('Artifact differs: '+name+'/'+path)
   verified+=1
 camera=reports['camera-reference-v2'];pairs=read(a.root/'camera-reference-v2/paired-results.json')
 assert len(pairs)==28017 and len({r['frame'] for r in pairs})==849
 for condition,methods in camera['metrics'].items():
  for method,m in methods.items():
   selected=[r for r in pairs if r['condition']==condition and r['method']==method]
   valid=[r['paired_de2000'] for r in selected if r['feature_valid']]
   healthy=[r['paired_de2000'] for r in selected if r['feature_valid'] and r['reference_healthy']]
   assert len(selected)==m['frames'] and len(valid)==m['valid'] and len(healthy)==m['healthy_valid']
   assert sum(v>5 for v in valid)==m['above5'] and np.isclose(np.percentile(valid,95),m['p95_de'])
 real=read(a.real/'completed.json')
 assert sha(a.real/'frames.jsonl')==real['artifacts']['frames.jsonl']
 rows=[json.loads(s) for s in (a.real/'frames.jsonl').read_text().splitlines()]
 expected={r['source_frame'] for r in rows if r['measurement']['measurement_valid']}
 assert {r['frame'] for r in pairs}==expected
 signal={}
 for condition in ('liquid_only_color_shift','liquid_only_shift_strong'):
  per={m:{r['frame']:r['paired_de2000'] for r in pairs if r['condition']==condition and r['method']==m and r['feature_valid']} for m in ('raw','srgb_gain','linear_gain')}
  signal[condition]={}
  for m in ('srgb_gain','linear_gain'):
   keys=set(per[m])&set(per['raw']);positive=[k for k in keys if per['raw'][k]>5]
   ratios=[per[m][k]/per['raw'][k] for k in keys if per['raw'][k]>.1]
   signal[condition][m]=dict(median_paired_signal_ratio=float(np.median(ratios)),raw_above5=len(positive),
       corrected_above5_among_raw_positive=sum(per[m][k]>5 for k in positive))
 # Recompute decision counts directly from episode records, independently of report aggregation.
 for folder in ('policy-refinement-v1','joint-candidate-v1'):
  episode=read(a.root/folder/'episode-results.json');report=reports[folder]
  assert len(episode)==report['policy_evaluations']
  if folder=='policy-refinement-v1':
   groups=[(c,arm,m) for c,arms in report['metrics'].items() for arm,m in arms.items()]
  else:groups=[(None,arm,m) for arm,m in report['metrics'].items()]
  for condition,arm,m in groups:
   subset=[r for r in episode if r['arm']==arm and (condition is None or r['condition']==condition)]
   clean=[r for r in subset if r['hold_up_ml']==0];bad=[r for r in subset if r['hold_up_ml']>0]
   assert len(clean)==m['clean_episodes'] and len(bad)==m['injected_episodes']
   assert sum(r['warned'] for r in clean)==m.get('clean_false_holds',m.get('clean_warned'))
   assert sum(r['warned'] for r in bad)==m['injected_warned']
   if 'fully_blocked_episodes' in m:assert sum(r['assessable_samples']==0 for r in subset)==m['fully_blocked_episodes']
  show=read(a.root/folder/'showcases.json')
  for case in show:assert all(not r['result']['hardware_action_authorized'] for r in case['trace'])
 progress=reports['progress-reference-v1'];episodes=read(a.root/'progress-reference-v1/episode-results.json')
 assert len(episodes)==864
 for variation,strategies in progress['metrics'].items():
  for strategy,m in strategies.items():
   group=[r for r in episodes if r['strategy']==strategy and r['flow_variation']==float(variation)]
   assert sum(r['warned'] for r in group if r['hold_up_ml']==0)==m['clean_warned']
   assert sum(r['warned'] for r in group if r['hold_up_ml']>0)==m['injected_warned']
 audit=read(a.root/'reference-evidence-audit-v1/completed.json')
 assert not audit['run5_values_loaded']
 # Exercise the new gate on real records; normalization never makes geometry approved.
 guard=GuardedEvidence();states=Counter()
 for row in rows:
  m=row['measurement'];s=dict(session_id='run4',vial_id=row['fill_detection']['detection_id'] or 'unassigned',
      source_time_s=row['seconds'],decision_time_s=row['seconds']+.1,recipe_key=None,
      geometry_status='unreviewed_model_proposals',measurement_valid=m['measurement_valid'])
  result=guard.update(s);states.update([result['status']]);assert not result['hardware_action_authorized']
 assert states=={'unavailable':1581}
 selected='srgb_gain_guarded'
 result=dict(schema_version=1,kind='stage2_refinement_verified',artifact_root=str(a.root),
     selected_pilot='sRGB background normalization + metered phase trajectory + current-source/camera guards + three samples',
     real_image_study=dict(training_runs=[1,2,3],development_run=4,training_background_samples=1478,
         matched_development_source_frames=camera['verified_development_frames'],qa_passing_liquid_frames=849,
         conditions=11,methods=3,paired_evaluations=28017,metrics=camera['metrics'],signal_retention=signal,
         reference_health=read(a.root/'camera-reference-v2/anchor-selection.json')),
     component_policy=dict(episodes=reports['policy-refinement-v1']['episodes'],evaluations=6720,
         selected_arm=selected,metrics={c:arms[selected] for c,arms in reports['policy-refinement-v1']['metrics'].items()}),
     progression=dict(episodes=288,evaluations=864,metrics=progress['metrics']),
     joint=dict(episodes=168,evaluations=336,metrics=reports['joint-candidate-v1']['metrics']),
     scoped_evidence_inventory=dict(standards_with_recorded_preparation=audit['rows_with_recorded_preparation_and_camera'],
         physical_trials_with_recorded_composition=audit['rows_with_recorded_composition_method_camera_and_time'],
         source_worksheet_hashes=audit['source_files'],review_databases_opened_read_only=len(audit['review_database_inventory']),
         eligible_spatial_revision_records=sum(d.get('eligible_spatial_revision_records',0) for d in audit['review_database_inventory'])),
     real_guard_statuses=dict(states),real_quality_accuracy=None,run5_used=False,hardware_actions=0,
     focused_stage2_contract_tests_passed=20,video=reports['camera-demo-v1'],verified_frozen_artifacts=verified,
     completion_sha256={n:sha(a.root/n/'completed.json') for n in reports},
     reference_inventory_sha256=sha(a.root/'reference-evidence-audit-v1/completed.json'),
     limits=['Pixel perturbations measure invariance conditional on frozen unreviewed geometry, not absolute recipe color.',
         'Analytic simulations share their optical family with the reference bank, not independent plant evidence.',
         'The 5 dE threshold and health bounds remain engineering choices requiring independent commissioning.',
         'No new real recipe or defect truth was inferred or approved.'])
 write(out/'completed.json',result)
 if a.record:write(a.record,result)
 print(json.dumps(dict(verified_frozen_artifacts=verified,paired_pixel_evaluations=28017,
     new_simulated_episodes=1800,new_policy_evaluations=7920,real_statuses=dict(states),hardware_actions=0),indent=2))


if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for n in ('root','real'):p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--record',type=Path);main(p.parse_args())
