"""Normalize completed real model outputs for an opt-in two-station rehearsal.

No inference, labels, camera connections or physical identity joins are created.
Different camera clocks remain separate; input file hashes and RGB hashes differ.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from replay.operator.commissioning import digest,validate_observation


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def identity(observation):
    observation['observation_id']='obs-'+digest({k:observation[k] for k in ('station','source','target','models','profile_id')})[:24]
    validate_observation(observation)
    return observation


def prepare(a):
    out=a.out;out.mkdir(parents=True,exist_ok=False)
    fill=a.results/'synthetic-iterations-20261022/gb10-results/sam-comparison-verified-v2'
    fixture=a.results/'dataset1-real-transfer-20261022/batch-v1'
    plan=read(fill/'plan.json');completion=read(fill/'completed.json')
    if plan['source_run']!=4 or plan['run5_used'] or plan['training']:
        raise ValueError('Expected completed unchanged Run4 comparison')
    for name,h in completion['artifacts'].items():
        if sha(fill/name)!=h:raise ValueError('Completed liquid evidence changed: '+name)
    fixturedone=read(fixture/'inference/completed.json');fixtureplan=read(fixture/'plan.json')
    if sha(fixture/'inference/predictions.json')!=fixturedone['predictions_sha256'] or fixtureplan['run5_used']:
        raise ValueError('Completed real fixture evidence changed')
    rows=[json.loads(s)['arms']['temporal'] for s in (fill/'pass-1.jsonl').read_text().splitlines()]
    fillmodels=[dict(role=role,sha256=plan['models'][key]['sha256'],domain_fitting='synthetic',
        generic_pretraining='COCO' if role=='vial' else None) for role,key in [('vial','vial'),('liquid','temporal')]]
    profile='profile-'+digest(dict(qa=plan['qa_config'],qa_code=plan['qa_code_sha256'],models=fillmodels,roi=[256,240,368,432]))[:24]
    observations=[];references=[];seen=set()
    for row in rows:
        case=row['case_id'];measurement=row['measurement']
        observation=identity(dict(schema='station-evidence/v1',station='vial_fill',data_domain='real',
            source=dict(camera_id='FillCam',source_id=plan['source_video_sha256'],stream_id=plan['source_video_sha256']+':'+case,
                clock_kind='native',clock_domain_id=plan['source_video_sha256'],pts=row['pts'],time_base=row['time_base'],
                frame_index=row['source_frame'],capture_timestamp=None,decoded_rgb_sha256=row['source_rgb_sha256'],encoded_file_sha256=None,dimensions=[640,480]),
            target=dict(kind='source_window',id=case,context_id=case,physical_vial_id=None,box_xyxy=measurement['geometry'].get('vial_box')),
            profile_id=profile,models=fillmodels,
            proposal=dict(available=row['proposal_available'],presence_score=row['presence_score'],raw_mask=row['raw_union'],status='unreviewed_model_proposal'),
            measurement=dict(qa_pass=measurement['measurement_valid'],qa_reasons=measurement['quality_flags'],rgb=measurement['rgb'],
                lab=measurement['lab'],sampling_mask=measurement['geometry']['mask'],source='original_RGB_unchanged_frozen_QA'),
            qualification=dict(alignment='unreviewed',visibility='unreviewed',physical_association='unlinked'),
            lineage=dict(file='fill-pass-1.jsonl',file_sha256=sha(fill/'pass-1.jsonl'),case_id=case,arm='temporal'),
            quality_decision=None,hardware_command=None))
        if case not in seen:
            references.append(dict(reference_id='unreviewed-'+case,purpose='temporal_background',observation=observation,
                qualification=dict(status='unreviewed',reviewer=None,evidence_id=None,origin='experiment_snapshot',confirmed_condition=None)))
            seen.add(case)
        observations.append(observation)
    samples={r['sample_id']:r for r in read(fixture/'samples.json')}
    registration=read(fixture/'registration.json')
    fixtureprofile='profile-'+digest(dict(registration_sha256=sha(fixture/'registration.json'),model=fixturedone['checkpoint_sha256'],crop=[48,48],resize=[64,64]))[:24]
    for row in read(fixture/'inference/predictions.json'):
        sample=samples[row['sample_id']]
        if sample['run_id'] not in (1,2,3,4) or sha(fixture/sample['image'])!=sample['source_sha256']:
            raise ValueError('Real fixture source/run mismatch')
        if row['source_sha256']!=sample['source_sha256'] or row['box_xyxy']!=sample['boxes'][row['slot']]:
            raise ValueError('Real target crop/source mismatch')
        observation=identity(dict(schema='station-evidence/v1',station='final_fixture',data_domain='real',
            source=dict(camera_id='CapSealCam',source_id=sample['source_sha256'],stream_id='cap-seal-run-'+str(sample['run_id']),
                clock_kind='capture_only',clock_domain_id=None,pts=None,time_base=None,frame_index=None,capture_timestamp=sample['capture_timestamp'],
                decoded_rgb_sha256=None,encoded_file_sha256=sample['source_sha256'],dimensions=[640,480]),
            target=dict(kind='fixture_slot',id=row['slot'],context_id=sample['sample_id'],physical_vial_id=None,box_xyxy=row['box_xyxy']),
            profile_id=fixtureprofile,models=[dict(role='slot_classifier',sha256=fixturedone['checkpoint_sha256'],domain_fitting='synthetic',generic_pretraining=None)],
            proposal=dict(available=True,proposed_class=row['proposed_class'],class_scores=row['class_scores'],scores_calibrated=False,status='unreviewed_model_proposal'),
            measurement=dict(qa_pass=None,qa_reasons=['slot_mapping_and_visibility_unreviewed'],rgb=None,lab=None,sampling_mask=None),
            qualification=dict(alignment='unreviewed',visibility='unreviewed',physical_association='unlinked'),
            lineage=dict(file='fixture-predictions.json',file_sha256=sha(fixture/'inference/predictions.json'),run_id=sample['run_id'],sample_id=sample['sample_id']),
            quality_decision=None,hardware_command=None))
        observations.append(observation)
    if len(observations)!=744 or len({r['observation_id'] for r in observations})!=744:
        raise ValueError('Expected264 filling plus480 unique fixture observations')
    (out/'observations.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in observations))
    write(out/'reference-candidates.json',references)
    sourcefiles={'fill-plan.json':fill/'plan.json','fill-pass-1.jsonl':fill/'pass-1.jsonl',
        'fixture-plan.json':fixture/'plan.json','fixture-predictions.json':fixture/'inference/predictions.json','fixture-registration.json':fixture/'registration.json'}
    (out/'lineage').mkdir()
    for name,path in sourcefiles.items():shutil.copy2(path,out/'lineage'/name)
    (out/'media').mkdir()
    media={
        'filling-real-development.mp4':a.results/'synthetic-iterations-20261022/gb10-results/temporal-preview-v1/temporal-pool-candidate.mp4',
        'fixture-real-still-proposals.mp4':a.results/'dataset1-real-transfer-20261022/summary-v1/real-fixture-predictions.mp4',
        'fixture-registration.png':fixture/'registration-review.png'}
    for name,path in media.items():shutil.copy2(path,out/'media'/name)
    manifest=dict(schema='two-station-evidence-bundle/v1',observations=744,station_counts=dict(Counter(r['station'] for r in observations)),
        source_clocks_joined=False,physical_vial_links=0,training=False,run5_used=False,real_accuracy=None,
        reference_candidates=6,qualified_references=0,media_relationship='Independent sources/development windows; not synchronized or asserted to be the same vials.',
        sources={name:sha(path) for name,path in sourcefiles.items()},
        artifacts={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()},helper_sha256=sha(__file__))
    write(out/'manifest.json',manifest)
    print(json.dumps(dict(bundle=str(out),counts=manifest['station_counts'],qualified_references=0)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    prepare(p.parse_args())
