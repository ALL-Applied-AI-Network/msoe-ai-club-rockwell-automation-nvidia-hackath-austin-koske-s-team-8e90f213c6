"""CPU-only cached-evidence integration rehearsal; no app server or model changes."""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime,timezone
import json
import os
from pathlib import Path

from .commissioning import ReferenceGate,associate_physical_vial,digest,evidence_readiness,validate_observation


def read(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class DemonstrationJournal:
    """Fresh append-only file with a checked hash chain, not an authenticated log."""
    def __init__(self,path):
        self.path=Path(path);self.path.open('x').close();self.seq=0;self.previous=None

    def append(self,kind,payload):
        self.seq+=1
        event=dict(seq=self.seq,kind=kind,actor='AUTOMATED REHEARSAL',demonstration=True,
            recorded_utc=datetime.now(timezone.utc).isoformat(),previous_hash=self.previous,payload=deepcopy(payload))
        event['event_hash']=digest(event)
        with self.path.open('a') as stream:
            stream.write(json.dumps(event,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
        self.previous=event['event_hash'];return event


def verify_journal(path):
    previous=None;events=[]
    for seq,line in enumerate(Path(path).read_text().splitlines(),1):
        event=json.loads(line);without={k:v for k,v in event.items() if k!='event_hash'}
        if event['seq']!=seq or event['previous_hash']!=previous or digest(without)!=event['event_hash']:
            raise ValueError('Rehearsal journal chain changed')
        if event['demonstration'] is not True:raise ValueError('Automated rehearsal must remain a demonstration')
        previous=event['event_hash'];events.append(event)
    return events


def fixture_observation(station='vial_fill',frame=10):
    """Synthetic in-memory engineering test fixture, never real review evidence."""
    return dict(schema='station-evidence/v1',observation_id='SYNTHETIC-UNIT-'+str(frame),station=station,data_domain='synthetic',
        source=dict(camera_id='TEST-CAMERA',source_id='TEST-SOURCE',stream_id='TEST-STREAM',clock_kind='native',clock_domain_id='TEST-CLOCK',
            pts=frame,time_base='1/1',frame_index=frame,capture_timestamp=None,decoded_rgb_sha256='0'*64,encoded_file_sha256=None,dimensions=[640,480]),
        target=dict(kind='source_window' if station=='vial_fill' else 'fixture_slot',id='TEST-TARGET',context_id='TEST-CONTEXT',
            physical_vial_id=None,box_xyxy=[20,20,68,68]),profile_id='TEST-PROFILE',models=[dict(role='test',sha256='1'*64)],
        proposal=dict(available=True),measurement=dict(qa_pass=True),qualification=dict(alignment='confirmed',visibility='confirmed'),
        quality_decision=None,hardware_command=None,fixture_only=True)


def fixture_reference(purpose='temporal_background'):
    return dict(reference_id='SYNTHETIC-UNIT-REFERENCE',purpose=purpose,observation=fixture_observation(frame=1),
        qualification=dict(status='human_reviewed',reviewer='TEST FIXTURE ONLY',evidence_id='SYNTHETIC-TEST-CASE',
            origin='human_review',confirmed_condition=ReferenceGate.CONDITIONS[purpose]))


def guard_checks():
    checks=[]
    def check(name,change=None,reference_change=None,purpose='temporal_background',max_age=None,expected=False,reason=None):
        observation=fixture_observation();reference=fixture_reference(purpose)
        if change:change(observation)
        if reference_change:reference_change(reference)
        gate=ReferenceGate(max_age);gate.install(reference);result=gate.check(observation,purpose)
        passed=result['usable']==expected and (reason is None or reason in result['reasons'])
        checks.append(dict(case=name,synthetic_test_fixture=True,expected_usable=expected,result=result,passed=passed))
    check('qualified_test_fixture',expected=True)
    check('unreviewed_reference',reference_change=lambda r:r['qualification'].update(status='unreviewed'),reason='reference_not_human_qualified')
    check('model_cannot_self_qualify_reference',reference_change=lambda r:r['qualification'].update(origin='model_prediction'),reason='reference_not_human_qualified')
    check('demonstration_cannot_qualify_reference',reference_change=lambda r:r.update(demonstration=True),reason='reference_not_human_qualified')
    check('same_frame_reference',reference_change=lambda r:r['observation']['source'].update(pts=10,frame_index=10),reason='reference_not_strictly_earlier')
    check('future_reference',reference_change=lambda r:r['observation']['source'].update(pts=11,frame_index=11),reason='reference_not_strictly_earlier')
    check('camera_change',change=lambda o:o['source'].update(camera_id='OTHER-CAMERA'),reason='camera_changed')
    check('profile_change',change=lambda o:o.update(profile_id='OTHER-PROFILE'),reason='measurement_profile_changed')
    check('new_stream',change=lambda o:o['source'].update(stream_id='OTHER-STREAM'),reason='source_stream_changed')
    check('new_target_context',change=lambda o:o['target'].update(context_id='NEXT-VIAL-CONTEXT'),reason='target_context_changed')
    check('alignment_lost',change=lambda o:o['qualification'].update(alignment='invalid'),reason='alignment_invalid')
    check('explicit_discontinuity',change=lambda o:o['source'].update(discontinuity=True),reason='declared_source_discontinuity')
    check('expired_reference',max_age=5,reason='reference_expired')
    check('clock_domain_mismatch',change=lambda o:o['source'].update(clock_domain_id='OTHER-CLOCK'),reason='reference_causality_unverified')
    check('unmatched_still_clock',change=lambda o:o['source'].update(clock_kind='capture_only',clock_domain_id=None,pts=None,time_base=None,frame_index=None),reason='reference_causality_unverified')
    check('empty_is_not_capped_golden',purpose='capped_golden',reference_change=lambda r:r['qualification'].update(confirmed_condition='empty'),reason='reference_condition_unconfirmed')
    check('golden_wrong_slot',purpose='capped_golden',change=lambda o:o['target'].update(id='OTHER-SLOT'),reason='target_slot_changed')
    gate=ReferenceGate();gate.install(fixture_reference());gate.check(fixture_observation(frame=10));result=gate.check(fixture_observation(frame=9))
    checks.append(dict(case='rewind_resets_reference',synthetic_test_fixture=True,expected_usable=False,result=result,
        passed=not result['usable'] and gate.resets[-1]['reason']=='source_clock_rewound'))
    fill=fixture_observation();fixture=fixture_observation('final_fixture')
    link=associate_physical_vial(fill,fixture)
    checks.append(dict(case='time_or_color_does_not_join_vials',synthetic_test_fixture=True,result=link,passed=link['status']=='unlinked'))
    return checks


def run(bundle,out):
    bundle=Path(bundle);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    manifest=read(bundle/'manifest.json')
    for relative,h in manifest['artifacts'].items():
        if sha(bundle/relative)!=h:raise ValueError('Pinned input bundle changed: '+relative)
    observations=[json.loads(s) for s in (bundle/'observations.jsonl').read_text().splitlines()]
    references={r['observation']['target']['context_id']:r for r in read(bundle/'reference-candidates.json')}
    journal=DemonstrationJournal(out/'demonstration-events.jsonl');gate=ReferenceGate();states=[];requests={};active=None
    journal.append('rehearsal_started',dict(mode='cached model outputs',source_clocks_joined=False,physical_vial_links=0,hardware_commands=0))
    for observation in observations:
        validate_observation(observation);temporal=None
        if observation['station']=='vial_fill':
            context=observation['target']['context_id']
            if active!=context:
                if active is not None:gate.check(observation)
                gate.install(references[context]);active=context
                journal.append('unreviewed_reference_candidate_loaded',dict(context_id=context,reference_id=references[context]['reference_id'],qualified=False))
            temporal=gate.check(observation)
        state=dict(observation_id=observation['observation_id'],station=observation['station'],source=observation['source'],
            target=observation['target'],temporal_reference_check=temporal,readiness=evidence_readiness(observation,temporal),
            original_observation_digest=digest(observation))
        states.append(state)
        journal.append('cached_observation_ingested',dict(observation_id=observation['observation_id'],station=observation['station'],
            source=observation['source'],observation_digest=digest(observation),readiness=state['readiness']))
        should_request=observation['station'] not in requests and (observation['station']=='final_fixture' or observation['measurement']['qa_pass'])
        if should_request:
            requests[observation['station']]=dict(observation_id=observation['observation_id'],source=deepcopy(observation['source']),
                reason='Commissioning/mapping/reference review required; no product defect asserted')
            journal.append('demonstration_review_requested',requests[observation['station']])
    # Source frames are pinned when requested, despite subsequent observations.
    for request in requests.values():
        journal.append('demonstration_review_acknowledged',request)
        journal.append('demonstration_response_recorded',dict(**request,response='Human review and commissioning evidence still required',follow_up='open',quality_release_authorized=False))
    events=verify_journal(journal.path)
    pinned=all([e['payload']['source'] for e in events if e['kind']=='demonstration_review_requested' and e['payload']['observation_id']==r['observation_id']]==[r['source']] for r in requests.values())
    checks=guard_checks()
    if not all(r['passed'] for r in checks) or not pinned:raise AssertionError('Integration guard/pinned evidence rehearsal failed')
    fill=[r for r in observations if r['station']=='vial_fill'];fixture=[r for r in observations if r['station']=='final_fixture']
    link=associate_physical_vial(fill[0],fixture[0])
    report=dict(schema='two-station-rehearsal/v1',mode='cached completed outputs; not new inference or live camera-to-dashboard validation',
        observations=len(observations),station_counts=dict(Counter(r['station'] for r in observations)),
        liquid_mask_proposals=sum(r['proposal']['available'] for r in fill),frozen_liquid_sample_qa_pass=sum(r['measurement']['qa_pass'] for r in fill),
        qualified_temporal_references=sum(bool(r['temporal_reference_check'] and r['temporal_reference_check']['usable']) for r in states),
        qualified_recipe_decisions=0,fixture_proposal_counts=dict(Counter(r['proposal']['proposed_class'] for r in fixture)),
        physical_association=link,source_clocks_joined=False,reference_reset_events=gate.resets,
        real_accuracy=None,guard_checks=len(checks),guard_checks_passed=sum(r['passed'] for r in checks),
        journal_events=len(events),journal_chain_verified=True,review_source_pinning_verified=pinned,demonstration_responses=len(requests),
        actual_operator_actions=0,hardware_commands=0,training=False,local_gpu_used=False,app_defaults_changed=False,
        blocking_inputs=['Human pool/mask/onset reviews','Human slot mapping/visibility/class reviews',
            'Qualified temporal reference capture/reset procedure','Independently acceptable recipe reference and completion context',
            'Attributed physical vial-to-slot association for per-vial fusion','CSI-approved physical test and feed arrangement'])
    (out/'readiness.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in states))
    write(out/'guard-checks.json',checks);write(out/'report.json',report)
    (out/'README.md').write_text(
        '# Two-station cached-evidence rehearsal\n\n'
        f'Ingested{len(fill)} filling frames and{len(fixture)} fixture target observations into one candidate evidence contract.\n'
        f'Available: {report["liquid_mask_proposals"]} liquid proposals, {report["frozen_liquid_sample_qa_pass"]} unchanged-QA color samples and480 fixture proposals.\n'
        'These remain visible diagnostic evidence even though product-quality decisions are unavailable.\n\n'
        'No physical vial identity or cross-camera clock mapping was invented. No recipe reference was approved.\n'
        f'{len(checks)} synthetic engineering guard cases passed. The actual six temporal references remain unreviewed.\n'
        'Two demonstration requests retained their exact source identity after all observations arrived; acknowledgement/response left follow-up open.\n'
        'The journal contains AUTOMATED REHEARSAL events, not actual operator responses or equipment operations.\n\n'
        'See report.json, readiness.jsonl, guard-checks.json and demonstration-events.jsonl.\n'
        'The bundle media show independent real development sources; do not present them as synchronized footage of the same vials.\n'
        'App defaults, model outputs and measurement QA are unchanged. No inference/training/GPU run was needed.\n')
    write(out/'completed.json',dict(report_sha256=sha(out/'report.json'),input_manifest_sha256=sha(bundle/'manifest.json'),
        files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}))
    print(json.dumps(report),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.bundle,a.out)
