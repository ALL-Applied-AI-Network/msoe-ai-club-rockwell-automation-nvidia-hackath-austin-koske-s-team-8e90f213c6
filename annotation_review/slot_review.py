"""Offline real fixture review and source-matched scoring; evaluation labels only."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import zipfile

LABELS=['pass','missing_cap','crooked_cap','damaged_vial','no_vial','uncertain']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def build(batch,out):
    data=json.loads((batch/'samples.json').read_text());predictions=json.loads((batch/'inference/predictions.json').read_text())
    completed=json.loads((batch/'inference/completed.json').read_text())
    if sha(batch/'inference/predictions.json')!=completed['predictions_sha256']:
        raise ValueError('Predictions changed')
    drafts={(r['sample_id'],r['slot']):r for r in predictions}
    packet=dict(schema='dataset1-slot-review-packet/v1',training_eligible=False,approved_labels=False,
        model_sha256=completed['checkpoint_sha256'],model_classes=completed.get('classes',['pass','missing_cap','no_vial']),
        registration_sha256=sha(batch/'registration.json'),labels=LABELS,scenes=[],tasks=[])
    # Priority is positional and fixed independently of model predictions.
    priority={f'{fixture}{i:02d}' for fixture in ('G','S') for i in (1,3,5,16,26,30)}
    for sample in data:
        if sample['run_id'] not in (1,2,3,4) or sha(batch/sample['image'])!=sample['source_sha256']:
            raise ValueError('Source run/hash mismatch')
        scene=dict(sample_id=sample['sample_id'],source_sha256=sample['source_sha256'],filename=sample['source_filename'],
            run_id=sample['run_id'],timestamp=sample['capture_timestamp'],native_frame_index=None,
            image_data='data:image/jpeg;base64,'+base64.b64encode((batch/sample['image']).read_bytes()).decode())
        packet['scenes'].append(scene)
        for slot,box in sorted(sample['boxes'].items()):
            d=drafts[(sample['sample_id'],slot)]
            if d['source_sha256']!=scene['source_sha256'] or d['box_xyxy']!=box:
                raise ValueError('Draft/source/target mismatch')
            packet['tasks'].append(dict(task_id=sample['sample_id']+'/'+slot,sample_id=sample['sample_id'],run_id=sample['run_id'],
                slot=slot,box_xyxy=box,source_sha256=scene['source_sha256'],priority=slot in priority,draft=d,
                review_status='unreviewed',mapping_status='proposed-human-unreviewed'))
    packet['packet_id']=hashlib.sha256(json.dumps(packet,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    out.mkdir(exist_ok=False)
    write(out/'packet.json',packet)
    template=(Path(__file__).parent/'static/slot_review.html').read_text()
    script=(Path(__file__).parent/'static/slot_review.js').read_text()
    payload=json.dumps(packet,separators=(',',':')).replace('<','\\u003c')
    (out/'review.html').write_text(template.replace('__PACKET__',payload).replace('__SCRIPT__',script),encoding='utf-8')
    (out/'README.txt').write_text(
        'REAL FIXTURE SLOT REVIEW / EVALUATION ONLY\n\n'
        'Open review.html in Edge or Chrome. No server/install is needed. Enter your own name.\n'
        f'Start with Priority ({sum(t["priority"] for t in packet["tasks"])} targets), or All ({len(packet["tasks"])} targets across{len(data)} scenes).\n'
        'Positions use provisional authored fixture numbering. Confirm that the highlighted\n'
        'target corresponds to the intended aperture; use incorrect/uncertain mapping when needed.\n'
        'Inspect full-scene context and the exact target crop before deciding. Neighbors are not the target.\n'
        'pass: visible cap/body appear acceptable; no claim about hidden properties.\n'
        'missing_cap: vial is present with an open mouth; distinguish it from an empty hole.\n'
        'crooked_cap: cap is tilted/unseated; vial lean alone does not establish this.\n'
        'damaged_vial: visibly damaged body; inspect surrounding transparent objects.\n'
        'no_vial: target aperture is empty. It is not a missed placement without progress context.\n'
        'uncertain: cannot decide reliably. Robot/hand occlusion must not force a class.\n\n'
        'Model drafts are hidden by default and are shown only after you save an initial\n'
        'human decision. Raw-first review reduces anchoring but is not blinded forever.\n'
        'Download JSON regularly. Save appends history; imports reject divergent histories.\n'
        'No prefilled labels, training permission, team approval or physical vial mapping is supplied.\n'
        f'Model classes: {", ".join(packet["model_classes"])}. Review permits all five organizer classes.\n'
        'Use ?demonstration=1 only for testing; those exports cannot enter real scoring.\n',encoding='utf-8')
    write(out/'package-hashes.json',{p.name:sha(p) for p in out.iterdir() if p.is_file()})
    with zipfile.ZipFile(out.with_suffix('.zip'),'x',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.iterdir()):
            z.write(p,'real-fixture-review/'+p.name)
    print(json.dumps(dict(packet_id=packet['packet_id'],tasks=len(packet['tasks']),priority=sum(t['priority'] for t in packet['tasks']))))


def validate_history(packet,review):
    if review.get('schema')!='dataset1-slot-human-review/v1' or review.get('packet_id')!=packet['packet_id'] or review.get('training_eligible') is not False:
        raise ValueError('Review schema/packet/training boundary mismatch')
    tasks={t['task_id']:t for t in packet['tasks']};latest={}
    for seq,event in enumerate(review['history'],1):
        task=tasks.get(event.get('task_id'))
        if not task or event.get('seq')!=seq or event.get('source_sha256')!=task['source_sha256'] or event.get('slot')!=task['slot'] or event.get('box_xyxy')!=task['box_xyxy']:
            raise ValueError('History sequence/source/target mismatch')
        if event.get('label') not in LABELS or event.get('mapping') not in ('confirmed','incorrect','uncertain') or event.get('visibility') not in ('visible','occluded','uncertain'):
            raise ValueError('Invalid human decision')
        if not str(event.get('reviewer','')).strip() or not event.get('timestamp'):
            raise ValueError('Human attribution is required')
        if event['visibility']!='visible' and event['label']!='uncertain':
            raise ValueError('Occluded/uncertain visibility cannot supply a definite class')
        latest[event['task_id']]=event
    return latest


def validate_packet(packet):
    content={k:v for k,v in packet.items() if k!='packet_id'}
    expected=hashlib.sha256(json.dumps(content,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if packet.get('packet_id')!=expected or packet.get('training_eligible') is not False:
        raise ValueError('Packet content identity or training boundary differs')
    if len({t['task_id'] for t in packet['tasks']})!=len(packet['tasks']):
        raise ValueError('Duplicate target identity')


def score(packet,review):
    validate_packet(packet)
    if review.get('demonstration') is not False or review.get('eligible_for_team_review') is not True:
        raise ValueError('Demonstration or ineligible export cannot provide real truth')
    latest=validate_history(packet,review)
    classes=LABELS[:-1];matrix=[[0]*5 for _ in range(5)];eligible=[];reasons={'unreviewed':0,'mapping_unconfirmed':0,'visibility_unconfirmed':0,'uncertain_label':0}
    for task in packet['tasks']:
        event=latest.get(task['task_id'])
        if event is None:reasons['unreviewed']+=1;continue
        if event['mapping']!='confirmed':reasons['mapping_unconfirmed']+=1;continue
        if event['visibility']!='visible':reasons['visibility_unconfirmed']+=1;continue
        if event['label']=='uncertain':reasons['uncertain_label']+=1;continue
        truth=classes.index(event['label']);prediction=classes.index(task['draft']['proposed_class']);matrix[truth][prediction]+=1
        eligible.append(dict(task_id=task['task_id'],truth=event['label'],prediction=task['draft']['proposed_class'],reviewer=event['reviewer'],
            occupied_defect_warning=task['draft'].get('occupied_defect_warning'),defect_score=task['draft'].get('defect_score')))
    per_class={label:dict(reviewed_targets=sum(matrix[i]),correct=matrix[i][i],missed=sum(matrix[i])-matrix[i][i],
        false_positives=sum(row[i] for row in matrix)-matrix[i][i],
        recall=matrix[i][i]/sum(matrix[i]) if sum(matrix[i]) else None,
        precision=matrix[i][i]/sum(row[i] for row in matrix) if sum(row[i] for row in matrix) else None) for i,label in enumerate(classes)}
    supported=packet.get('model_classes',['pass','missing_cap','no_vial'])
    unsupported=sum(r['truth'] not in supported for r in eligible)
    scenes={};fixtures={}
    for row in eligible:
        scene,slot=row['task_id'].rsplit('/',1);scenes.setdefault(scene,[]).append(row)
        fixtures.setdefault((scene,slot[0]),[]).append(row)
    scored_warnings=[r for r in eligible if type(r['occupied_defect_warning']) is bool and r['truth']!='no_vial']
    acceptable=[r for r in scored_warnings if r['truth']=='pass']
    diagnostic=dict(status='available' if scored_warnings else 'unavailable',reviewed_occupied_targets=len(scored_warnings),
        acceptable_targets=len(acceptable),false_alarms=sum(r['occupied_defect_warning'] for r in acceptable),
        false_alarm_rate=sum(r['occupied_defect_warning'] for r in acceptable)/len(acceptable) if acceptable else None,
        per_class={c:dict(reviewed_targets=sum(r['truth']==c for r in scored_warnings),
            escaped_defects=sum(r['truth']==c and not r['occupied_defect_warning'] for r in scored_warnings))
            for c in ('missing_cap','crooked_cap','damaged_vial')},
        escaped_examples=[r for r in scored_warnings if r['truth']!='pass' and not r['occupied_defect_warning']],
        no_vial_excluded=True,real_thresholds_tuned=False,
        scope='Frozen occupied-target warning, separate from exact-class confusion and progress-dependent missed placement.')
    return dict(packet_id=packet['packet_id'],task_count=len(packet['tasks']),human_decisions=len(latest),scored_targets=len(eligible),
        excluded=reasons,classes=classes,confusion_rows_truth_columns_prediction=matrix,per_class=per_class,
        accuracy=sum(matrix[i][i] for i in range(5))/len(eligible) if eligible else None,
        reviewed_occupied_predicted_empty=sum(r['truth']!='no_vial' and r['prediction']=='no_vial' for r in eligible),
        fully_scored_30_slot_scenes=sum(len(rows)==30 for rows in fixtures.values()),
        all_30_correct_scenes=sum(len(rows)==30 and all(r['truth']==r['prediction'] for r in rows) for rows in fixtures.values()),
        fully_scored_60_slot_scenes=sum(len(rows)==60 for rows in scenes.values()),
        all_60_correct_scenes=sum(len(rows)==60 and all(r['truth']==r['prediction'] for r in rows) for rows in scenes.values()),
        unsupported_reviewed_defects=unsupported,examples=eligible,training_eligible=False,
        occupied_defect_diagnostic=diagnostic,
        scope='Source-matched reviewed visible targets with confirmed mapping; partial correlated development subset, not production accuracy.',
        coverage_warning='Unreviewed/uncertain/occluded/mismapped targets are excluded, never counted as correct. Rare-defect recall is unavailable without positive reviewed examples.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['build','score'])
    p.add_argument('--batch',type=Path);p.add_argument('--packet',type=Path);p.add_argument('--review',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.action=='build':build(a.batch,a.out)
    else:
        packet=json.loads((a.packet/'packet.json').read_text());review=json.loads(a.review.read_text())
        result=score(packet,review);a.out.mkdir(exist_ok=False);write(a.out/'reviewed-metrics.json',result)


if __name__=='__main__':main()
