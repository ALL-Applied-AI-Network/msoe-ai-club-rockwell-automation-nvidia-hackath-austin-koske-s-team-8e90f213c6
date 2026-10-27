"""Score matched SAM/temporal masks using the existing human review packet.

No second label copy is needed: immutable image/clock identities are identical.
Demonstration reviews are refused. Raw mask scoring includes missed positives;
QA availability remains separate from full visible-pool segmentation accuracy.
"""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path

from .score_mask_review import validated_decisions


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def score(a):
    import numpy as np
    from PIL import Image
    plan=read(a.comparison/'plan.json');complete=read(a.comparison/'completed.json')
    for name,digest in complete['artifacts'].items():
        if sha(a.comparison/name)!=digest:raise ValueError('Completed comparison changed: '+name)
    packet=read(a.packet/'manifest.json');review=read(a.review)
    if review.get('packet_id')!=packet['packet_id'] or review.get('source_video_sha256')!=plan['source_video_sha256']:
        raise ValueError('Review packet/source mismatch')
    tasks={t['sample_id']:t for t in read(a.comparison/'review-drafts.json')['tasks']}
    arms=list(plan['arms']);refined={}
    if getattr(a,'refinement',None):
        for name,digest in read(a.refinement/'completed.json')['artifacts'].items():
            if sha(a.refinement/name)!=digest:raise ValueError('Completed refinement changed: '+name)
        if read(a.refinement/'plan.json')['parent_plan_sha256']!=sha(a.comparison/'plan.json'):
            raise ValueError('Refinement ancestry mismatch')
        refined={r['source_frame']:r for r in [json.loads(s) for s in (a.refinement/'pass-1.jsonl').read_text().splitlines()]}
        arms.append('temporal_guided_sam')
    states,intervals=validated_decisions(review,tasks)
    definite={k:v for k,v in states.items() if v['status']!='uncertain'}
    if not definite:raise ValueError('No definite human reviews to score')
    models={}
    for arm in arms:
        tp=fp=fn=tn=0;ious=[];dices=[];rows=[]
        for sid,event in definite.items():
            task=tasks[sid]
            if sha(a.source_probe/task['image'])!=task['image_sha256']:
                raise ValueError('Frozen source/mask bytes differ')
            if arm=='temporal_guided_sam':
                row=refined[task['frame_index']]
                if row['source_rgb_sha256']!=task['source_rgb_sha256'] or row['pts']!=task['pts'] or row['time_base']!=task['time_base']:
                    raise ValueError('Refinement frame/source mismatch')
                mask=np.zeros((480,640),np.uint8);encoded=row['raw_union']
                if encoded:
                    x,y=encoded['origin'];w,h=encoded['size'];flat=np.zeros(w*h,np.uint8)
                    if not 0<=x<x+w<=640 or not 0<=y<y+h<=480:raise ValueError('Refined mask outside source')
                    for start,length in encoded['runs']:
                        if start<0 or length<=0 or start+length>len(flat):raise ValueError('Invalid refined mask run')
                        flat[start:start+length]=1
                    mask[y:y+h,x:x+w]=flat.reshape(h,w)
                mask=mask.astype(bool);available=row['proposal_available'] and bool(mask.any())
            else:
                draft=next(d for d in task['drafts'] if d['mode']==arm)
                if sha(a.comparison/draft['mask'])!=draft['mask_sha256']:raise ValueError('Frozen draft mask differs')
                mask=np.asarray(Image.open(a.comparison/draft['mask']))>=128
                available=bool(draft['proposal_available']) and bool(mask.any())
            if not available:mask=np.zeros_like(mask)
            positive=event['status']=='pool'
            tp+=positive and available;fp+=not positive and available
            fn+=positive and not available;tn+=not positive and not available
            truth=event['decoded_mask'];inter=int((mask&truth).sum());union=int((mask|truth).sum())
            iou=inter/union if union else 1.;dice=2*inter/max(1,int(mask.sum()+truth.sum()))
            if positive:ious.append(iou);dices.append(dice)
            rows.append(dict(sample_id=sid,review_status=event['status'],proposal_available=available,
                mask_iou=iou,dice=dice,reviewer=event['reviewer'],draft_assisted=event['draft_model_sha256'] is not None))
        models[arm]=dict(tp=tp,fp=fp,fn=fn,tn=tn,
            proposal_precision=tp/(tp+fp) if tp+fp else None,proposal_recall=tp/(tp+fn) if tp+fn else None,
            mean_positive_mask_iou=sum(ious)/len(ious) if ious else None,
            mean_positive_dice=sum(dices)/len(dices) if dices else None,frames=rows)
    predictions=[json.loads(s)['arms'] for s in (a.comparison/'pass-1.jsonl').read_text().splitlines()]
    if refined:
        for row in predictions:row['temporal_guided_sam']=refined[row['temporal']['source_frame']]
    onset={}
    for case,event in intervals.items():
        if states[event['last_empty']]['status']!='empty' or states[event['first_pool']]['status']!='pool':
            onset[case]=dict(status='stale_reviewed_endpoints');continue
        absent,present=(tasks[event[key]] for key in ['last_empty','first_pool'])
        low=float(absent['pts']*Fraction(absent['time_base']));high=float(present['pts']*Fraction(present['time_base']))
        onset[case]=dict(human_interval_frames=[absent['frame_index'],present['frame_index']],models={})
        for arm in arms:
            first=next((r[arm] for r in predictions if r[arm]['case_id']==case and r[arm]['proposal_available']),None)
            onset[case]['models'][arm]=None if first is None else dict(first_sampled_proposal_frame=first['source_frame'],
                delay_interval_s=[float(first['pts']*Fraction(first['time_base']))-high,float(first['pts']*Fraction(first['time_base']))-low])
    report=dict(schema='reviewed-sam-temporal-comparison/v1',review_sha256=sha(a.review),
        comparison_plan_sha256=sha(a.comparison/'plan.json'),packet_id=packet['packet_id'],reviewed=len(states),
        scored=len(definite),uncertain=len(states)-len(definite),unreviewed=len(tasks)-len(states),
        models=models,onset=onset,training=False,run5_used=False,thresholds_changed=False,
        team_approved=review.get('team_approved',False),
        limits=['SAM proposal absence is not a calibrated empty-vial decision; positives without a mask count as missed.',
            'No human-approved labels are generated by this scorer; it consumes explicit attributed reviews.',
            'Draft-assisted human reviews can favor the shown candidate. Include independent raw-first review.',
            'Six inspected development clips; differing training regimes; not a pure architecture or blind-test result.',
            'Onset is interval-valued, sampled and clip-bounded; early activation can be a false proposal.'])
    a.out.mkdir(parents=True,exist_ok=False)
    (a.out/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(reviewed=len(states),scored=len(definite),out=str(a.out))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['comparison','source-probe','packet','review','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--refinement',type=Path,help='Optional completed temporal-guided SAM refinement root')
    score(p.parse_args())
