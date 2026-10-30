"""Experimental vial-relative accumulated-body association; no color/QA changes.

Calibrate only on new synthetic training families, freeze, then score real
development references. Returns original boxes and confidence, never fabricates
or expands a liquid box. A basal box can still be empty-glass confusion.
"""
import argparse
import json
from pathlib import Path

from target_synthetic_transfer import sha,read,write


def active_vial(vials):
    found=[p['box'] for p in vials if p['class_id']==0 and p['confidence']>=.35
           and p['box'][0]<306.5<p['box'][2] and p['box'][3]-p['box'][1]>180]
    return found[0] if len(found)==1 else None


def filter_predictions(liquids,vials,policy):
    vial=active_vial(vials)
    if vial is None:return []
    x,y,r,b=vial;w=r-x;h=b-y;mw=.20*w;mh=.03*h
    accepted=[]
    for p in liquids:
        if p['class_id']!=1:continue
        a,c,d,e=p['box'];area=max(0,d-a)*max(0,e-c)
        if area<=0:continue
        overlap=max(0,min(d,r+mw)-max(a,x-mw))*max(0,min(e,b+mh)-max(c,y-mh))/area
        allowed=(x<(a+d)/2<r and y<(c+e)/2<b and overlap>=.9
                 and (b-e)/h<=policy['maximum_bottom_gap_fraction'] and e<=b+.03*h)
        if allowed:accepted.append(p)
    # One accumulated body per active vial. Native score breaks alternatives;
    # no score calibration, box clipping, growth or color-based shape replacement.
    return sorted(accepted,key=lambda p:p['confidence'],reverse=True)[:1]


def main(a):
    import numpy as np
    from PIL import Image,ImageDraw,ImageFont
    from ultralytics import YOLO
    from benchmark_liquid_pipeline import VIAL_OPTIONS
    from experiment_sim_real_cpu import score
    a.out.mkdir(parents=True,exist_ok=False)
    completed=read(a.renders/'completed-render.json')
    if sha(a.renders/'manifest.jsonl')!=completed['manifest_sha256']:raise ValueError('Render inventory changed')
    rows=[json.loads(s) for s in (a.renders/'manifest.jsonl').read_text().splitlines()]
    checkpoint=a.tools/'results/real-synthetic-detector-20261019/models/sim-only/checkpoint.pt'
    metadata=read(checkpoint.with_name('model.json'))
    if sha(checkpoint)!=metadata['checkpoint_sha256'] or metadata['train_runs'] or metadata['development_runs']:
        raise ValueError('Vial model is not the frozen synthetic-only checkpoint')
    model=YOLO(str(checkpoint));vials={};gaps=[];heights=[]
    for row in rows:
        source=a.renders/row['image']
        if sha(source)!=row['image_sha256']:raise ValueError('Rendered RGB changed')
        result=model.predict([str(source)],**VIAL_OPTIONS)[0]
        ps=[dict(class_id=int(p[5]),box=p[:4].tolist(),confidence=float(p[4])) for p in result.boxes.data.cpu().numpy() if int(p[5])==0]
        vials[row['id']]=ps;v=active_vial(ps)
        if row['split']=='train' and row['box'] is not None and v is not None:
            gaps.append(max(0.,(v[3]-row['box'][3])/(v[3]-v[1])));heights.append(v[3]-v[1])
    if len(gaps)!=72:raise ValueError('Expected all72 synthetic training body cases with an unambiguous active vial')
    policy=dict(schema_version=1,kind='experimental_vial_basal_association',
        maximum_bottom_gap_fraction=max(gaps)+1./float(np.median(heights)),
        calibration='Maximum native body-bottom deficit vs frozen YOLO vial across72 synthetic training positives, plus1px at median synthetic vial height.',
        calibration_body_cases=len(gaps),synthetic_maximum_gap=max(gaps),
        vial_checkpoint_sha256=sha(checkpoint),source_manifest_sha256=sha(a.renders/'manifest.jsonl'),
        real_calibration_runs=[],run5_used=False,retained_bodies_per_vial=1,
        scope='Accumulated body only; may reject suspended droplets, stream and occluded pools. Does not establish that a retained box contains liquid.',
        limits=['Station-camera heuristic; a vial-relative basal filter, not improved learned detector accuracy.',
            'Adaptive hypothesis motivated by inspected real examples; freeze uses only synthetic training geometry.',
            'Real teacher agreement and provisional negatives do not establish physical absence or onset accuracy.'])
    write(a.out/'frozen-policy.json',policy)
    write(a.out/'synthetic-vial-predictions.json',vials)
    # All real files are deliberately opened only after the synthetic policy freezes.
    real_vial_rows=read(a.tools/'results/sim-real-robustness-20261022/cpu-transfer-v2/original_sim_only--raw.json')
    real_vials={r['source_id']:[p for p in r['predictions'] if p['class_id']==0] for r in real_vial_rows if r['domain']=='real_run4'}
    baseline=a.tools/'results/synthetic-liquid-20261022/fasterrcnn-v1'
    compare={};records={}
    for name,root in [('baseline',baseline),('paired',a.candidate)]:
        rows_real=read(root/'real-development-predictions.json')
        filtered=[dict(row,predictions=filter_predictions(row['predictions'],real_vials[row['source_id']],policy)) for row in rows_real]
        compare[name]=dict(raw=score(rows_real,1),basal=score(filtered,1))
        records[name]=filtered;write(a.out/(name+'-real-predictions.json'),filtered)
    syn_b=read(a.review/'baseline-new-synthetic-predictions.json')
    new_ids={r['source_id'] for r in syn_b}
    syn_c=[r for r in read(a.candidate/'synthetic-development-predictions.json') if r['source_id'] in new_ids]
    syn={name:dict(raw=score(data,1),basal=score([dict(row,predictions=filter_predictions(row['predictions'],vials[row['source_id']],policy)) for row in data],1)) for name,data in [('baseline',syn_b),('paired',syn_c)]}
    write(a.out/'comparison.json',dict(policy_sha256=sha(a.out/'frozen-policy.json'),real_run4_provisional_agreement=compare,
        new_grouped_synthetic_development=syn,training=False,app_changed=False,promoted=False))
    previous=a.tools/'results/real-vs-synthetic-20261022/review-v1'
    source_cases=read(previous/'plan.json')['source_cases']
    ids=['vial_fill_20260923_191017_576908','vial_fill_20260923_191029_656060']
    raw={r['source_id']:r for r in read(baseline/'real-development-predictions.json')}
    filtered={r['source_id']:r for r in records['baseline']}
    board=Image.new('RGB',(1350,1140),'#142638');draw=ImageDraw.Draw(board)
    font=lambda size:ImageFont.truetype('C:/Windows/Fonts/arial.ttf',size)
    draw.text((22,12),'BASELINE SYNTHETIC DETECTOR / OPTIONAL BASAL ASSOCIATION',font=font(26),fill='white')
    draw.text((22,49),'Rule frozen from synthetic training geometry. No liquid box is invented or enlarged.',font=font(20),fill='white')
    for row,sid in enumerate(ids):
        for col,data in enumerate([raw,filtered]):
            x=22+col*665;y=95+row*510
            draw.text((x,y),'RAW PROPOSALS' if col==0 else 'BASAL + ONE BODY PER VIAL',font=font(20),fill='white')
            im=Image.open(source_cases[sid]['source']).convert('RGB');d=ImageDraw.Draw(im)
            for p in data[sid]['predictions']:
                if p['confidence']>=.35:d.rectangle(p['box'],outline='#ffd05a',width=2)
            board.paste(im,(x,y+28))
    draw.text((22,1110),'Unreviewed real examples. Retaining fewer boxes does not prove the remaining box is correct.',font=font(18),fill='white')
    board.save(a.out/'basal-association-real-examples.png')
    write(a.out/'completed.json',dict(artifacts={p.name:sha(p) for p in a.out.iterdir() if p.is_file()},policy_sha256=sha(a.out/'frozen-policy.json')))
    print(json.dumps(dict(policy=policy,real=compare,synthetic=syn),indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['renders','candidate','review','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--tools',type=Path,default=Path('C:/Users/Austin/ai-club-tools'))
    main(p.parse_args())
