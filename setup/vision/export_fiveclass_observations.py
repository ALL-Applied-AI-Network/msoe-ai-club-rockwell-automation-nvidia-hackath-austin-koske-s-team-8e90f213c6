"""Export frozen real fixture proposals to the opt-in shared evidence contract."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from replay.operator.commissioning import digest,validate_observation,evidence_readiness
from dataset1_fiveclass import CLASSES,read,sha,write


def export(batch,out):
    out.mkdir(exist_ok=False);completion=read(batch/'inference/completed.json');samples={s['sample_id']:s for s in read(batch/'samples.json')}
    source=batch/'inference/predictions.json'
    if sha(source)!=completion['predictions_sha256']:raise ValueError('Completed predictions differ')
    model=completion['checkpoint_sha256'];observations=[];states=[]
    profile='profile-'+digest(dict(registration_sha256=sha(batch/'registration.json'),model_sha256=model,crop=96,
        target_conditioning=completion['selected_review_model']))[:24]
    for prediction in read(source):
        sample=samples[prediction['sample_id']]
        if sample['run_id'] not in (1,2,3,4) or sha(batch/sample['image'])!=sample['source_sha256']:raise ValueError('Real source differs')
        if prediction['source_sha256']!=sample['source_sha256'] or prediction['box_xyxy']!=sample['boxes'][prediction['slot']]:raise ValueError('Source/target join differs')
        observation=dict(schema='station-evidence/v1',station='final_fixture',data_domain='real',
            source=dict(camera_id='CapSealCam',source_id=sample['source_sha256'],stream_id='cap-seal-run-'+str(sample['run_id']),
                clock_kind='capture_only',clock_domain_id=None,pts=None,time_base=None,frame_index=None,
                capture_timestamp=sample['capture_timestamp'],encoded_file_sha256=sample['source_sha256'],decoded_rgb_sha256=None,dimensions=[640,480]),
            target=dict(kind='fixture_slot',id=prediction['slot'],context_id=sample['sample_id'],physical_vial_id=None,box_xyxy=prediction['box_xyxy']),
            profile_id=profile,models=[dict(role='slot_classifier',sha256=model,domain_fitting='synthetic',generic_pretraining=None)],
            proposal=dict(available=True,proposed_class=prediction['proposed_class'],class_scores=prediction['class_scores'],
                supported_classes=CLASSES,defect_score=prediction['defect_score'],defect_threshold=prediction['defect_threshold'],
                occupied_defect_warning=prediction['occupied_defect_warning'],scores_calibrated=False,status='unreviewed_model_proposal'),
            measurement=dict(qa_pass=None,qa_reasons=['slot_mapping_and_visibility_unreviewed'],rgb=None,lab=None,sampling_mask=None),
            qualification=dict(alignment='unreviewed',visibility='unreviewed',physical_association='unlinked'),
            lineage=dict(file='predictions.json',file_sha256=sha(source),run_id=sample['run_id'],sample_id=sample['sample_id']),
            quality_decision=None,hardware_command=None)
        observation['observation_id']='obs-'+digest({k:observation[k] for k in ('station','source','target','models','profile_id')})[:24]
        validate_observation(observation);observations.append(observation)
        states.append(dict(observation_id=observation['observation_id'],readiness=evidence_readiness(observation)))
    (out/'station-observations.jsonl').write_text(''.join(json.dumps(o,allow_nan=False)+'\n' for o in observations))
    (out/'readiness.jsonl').write_text(''.join(json.dumps(o,allow_nan=False)+'\n' for o in states))
    write(out/'completed.json',dict(observations=len(observations),checkpoint_sha256=model,real_accuracy=None,
        qualified_product_decisions=0,physical_vials_linked=False,observations_sha256=sha(out/'station-observations.jsonl'),
        helper_sha256=sha(__file__),commissioning_helper_sha256=sha(Path(__file__).resolve().parents[2]/'replay/operator/commissioning.py')))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();export(a.batch,a.out)
