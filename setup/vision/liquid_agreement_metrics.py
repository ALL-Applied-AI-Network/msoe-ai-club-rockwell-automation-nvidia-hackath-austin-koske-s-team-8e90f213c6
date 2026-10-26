"""NumPy-only provisional box-agreement metrics, extracted without algorithm changes."""
from collections import defaultdict
import numpy as np


def overlaps(box,boxes):
    if not len(boxes):
        return np.empty(0)
    boxes=np.asarray(boxes,dtype=np.float64).reshape(-1,4);box=np.asarray(box,dtype=np.float64)
    xy=np.maximum(0,np.minimum(box[2:],boxes[:,2:])-np.maximum(box[:2],boxes[:,:2]))
    intersection=xy[:,0]*xy[:,1]
    return intersection/np.maximum(1e-12,np.prod(box[2:]-box[:2])+np.prod(boxes[:,2:]-boxes[:,:2],axis=1)-intersection)


def score(rows,cls):
    targets={r['source_id']:[t['box'] for t in r['targets'] if t['class_id']==cls] for r in rows}
    count=sum(map(len,targets.values()))
    predictions=sorted([(r['source_id'],p) for r in rows for p in r['predictions'] if p['class_id']==cls],key=lambda pair:-pair[1]['confidence'])
    correct=np.zeros((len(predictions),10),bool)
    for j,threshold in enumerate(np.linspace(.5,.95,10)):
        claimed=defaultdict(set)
        for i,(sid,p) in enumerate(predictions):
            ious=overlaps(p['box'],targets[sid])
            eligible=[k for k in range(len(ious)) if ious[k]>=threshold and k not in claimed[sid]]
            if eligible:
                k=max(eligible,key=lambda k:ious[k]);claimed[sid].add(k);correct[i,j]=True
    aps=[]
    for j in range(10):
        tp=np.cumsum(correct[:,j]);fp=np.cumsum(~correct[:,j]);recall=tp/max(1,count);precision=tp/np.maximum(1,tp+fp)
        envelope=np.maximum.accumulate(precision[::-1])[::-1]
        aps.append(float(np.mean([max(envelope[recall>=r],default=0.) for r in np.linspace(0,1,101)])))
    operating=np.array([p['confidence']>=.35 for _,p in predictions],bool)
    tp=int(correct[operating,0].sum());number=int(operating.sum())
    strata={name:dict(targets=0,detected=0) for name in ['height_le32','height_32_to64','height_gt64']}
    for r in rows:
        if cls!=1:
            break
        ts=[t['box'] for t in r['targets'] if t['class_id']==cls]
        if len(ts)>1:
            raise ValueError('Expected at most one provisional liquid target')
        if ts:
            height=ts[0][3]-ts[0][1];key='height_le32' if height<=32 else 'height_32_to64' if height<=64 else 'height_gt64'
            strata[key]['targets']+=1
            ps=[p for p in r['predictions'] if p['class_id']==cls and p['confidence']>=.35]
            strata[key]['detected']+=int(any(overlaps(p['box'],ts)[0]>=.5 for p in ps))
    return dict(targets=count,ap50_101point=aps[0],ap50_95_101point=float(np.mean(aps)),
        confidence=.35,iou=.5,tp=tp,unmatched_boxes=number-tp,missed_targets=count-tp,
        precision=tp/number if number else 0.,recall=tp/count if count else None,
        provisional_negative_frames_with_predictions=sum(not any(t['class_id']==cls for t in r['targets']) and any(p['class_id']==cls and p['confidence']>=.35 for p in r['predictions']) for r in rows),
        target_height_strata=strata if cls==1 else None)


def metrics_self_check():
    target=dict(class_id=1,box=[0,0,10,10])
    pred=dict(**target,confidence=.9)
    r=score([dict(source_id='a',targets=[target],predictions=[pred,dict(**target,confidence=.8)]),dict(source_id='b',targets=[target],predictions=[])],1)
    assert (r['tp'],r['unmatched_boxes'],r['missed_targets'])==(1,1,1)
    assert r['precision']==r['recall']==.5
    assert .50<r['ap50_101point']<.51
