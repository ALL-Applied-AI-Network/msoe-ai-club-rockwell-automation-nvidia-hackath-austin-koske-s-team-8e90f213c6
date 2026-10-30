"""Compare frozen synthetic checkpoints, render evidence and human review cases."""
import argparse
import json
from pathlib import Path
import shutil
import sys

from target_synthetic_transfer import sha,read,write


def main(a):
    import numpy as np
    import torch
    from PIL import Image,ImageDraw,ImageFont
    from experiment_sim_real_cpu import score
    from train_synthetic_liquid_pilot import evaluate,validate
    from train_early_liquid import build_resnet

    a.out.mkdir(parents=True,exist_ok=False)
    baseline=a.tools/'results/synthetic-liquid-20261022/fasterrcnn-v1'
    previous=a.tools/'results/real-vs-synthetic-20261022/review-v1'
    cases=read(previous/'plan.json')['source_cases']
    base_model=read(baseline/'frozen-model.json');candidate_model=read(a.candidate/'frozen-model.json')
    for root,metadata in [(baseline,base_model),(a.candidate,candidate_model)]:
        if not (root/'completed.json').exists() or sha(root/'final.pt')!=metadata['checkpoint_sha256']:
            raise ValueError('Completed frozen checkpoint required')
        if metadata['training_real_runs'] or metadata['selection_real_runs']:raise ValueError('Real training boundary changed')
    plan=validate(a.candidate)
    candidate_syn=read(a.candidate/'synthetic-development-predictions.json')
    original_ids={r['source_id'] for r in read(baseline/'synthetic-development-predictions.json')}
    view=plan.get('input_view')
    view_model=None
    if view:
        from ultralytics import YOLO
        from vial_input_view import make_view,map_box,VIAL_OPTIONS
        if sha(view['vial_checkpoint'])!=view['vial_checkpoint_sha256']:raise ValueError('Input-view checkpoint differs')
        view_model=YOLO(view['vial_checkpoint'])
        reference_rows=read(baseline/'synthetic-development-predictions.json')+read(a.tools/'results/targeted-synthetic-20261022/review-v1/baseline-new-synthetic-predictions.json')
        references={r['source_id']:r for r in reference_rows}
        transforms={s['source_id']:s['input_view_affine'] for s in plan['samples']}
        candidate_syn=[dict(row,targets=references[row['source_id']]['targets'],
            predictions=[dict(p,box=box) for p in row['predictions'] if (box:=map_box(p['box'],transforms[row['source_id']],inverse=True)) is not None]) for row in candidate_syn]
    original=[r for r in candidate_syn if r['source_id'] in original_ids]
    new=[r for r in candidate_syn if r['source_id'] not in original_ids]
    synthetic_rows=[s for s in plan['samples'] if s['split']=='val' and s['source_id'] not in original_ids]
    torch.set_num_threads(4)
    if view:
        base_new_rows=[references[s['source_id']] for s in synthetic_rows]
        base_new=score(base_new_rows,1)
    else:
        model=build_resnet(baseline)
        model.load_state_dict(torch.load(baseline/'final.pt',map_location='cpu',weights_only=True)['model_state'])
        model.cuda().eval()
        base_new,base_new_rows=evaluate(model,synthetic_rows,a.candidate,'cuda')
        del model;torch.cuda.empty_cache()
    write(a.out/'baseline-new-synthetic-predictions.json',base_new_rows)
    model=build_resnet(a.candidate)
    model.load_state_dict(torch.load(a.candidate/'final.pt',map_location='cpu',weights_only=True)['model_state'])
    model.cuda().eval()
    illustrative_ids=['vial_fill_20260923_191017_576908','vial_fill_20260923_191029_656060',
                      'vial_fill_20260923_184720_822117','vial_fill_20260922_170600_743982']
    labels=['No supplied liquid target','Shallow blue','Pale low fill','Established blue']
    predictions={}
    with torch.inference_mode():
        for sid in illustrative_ids:
            source=Path(cases[sid]['source'])
            if sha(source)!=cases[sid]['source_sha256'] or cases[sid]['source_run']==5:
                raise ValueError('Diagnostic image identity changed')
            rgb=np.asarray(Image.open(source).convert('RGB')).copy()
            affine=None
            if view:
                v=view_model.predict([rgb[:,:,::-1].copy()],**VIAL_OPTIONS)[0]
                vials=[dict(class_id=int(p[5]),box=p[:4].tolist(),confidence=float(p[4])) for p in v.boxes.data.cpu().numpy() if int(p[5])==0]
                rgb,affine=make_view(rgb,vials,view)
            result=model([torch.from_numpy(rgb).permute(2,0,1).float().cuda()/255.])[0]
            predictions[sid]=[dict(box=b.tolist(),confidence=float(c)) for b,c in zip(result['boxes'].cpu(),result['scores'].cpu())]
            if view:predictions[sid]=[dict(p,box=box) for p in predictions[sid] if (box:=map_box(p['box'],affine,inverse=True)) is not None]
    write(a.out/'candidate-illustrative-predictions.json',predictions)
    old=read(previous/'predictions.json')['results']
    font=lambda size:ImageFont.truetype('C:/Windows/Fonts/arial.ttf',size)
    bg='#142638'; white='#eef3f7';yellow='#ffd05a'
    board=Image.new('RGB',(1360,2350),bg);draw=ImageDraw.Draw(board)
    draw.text((24,15),'TARGETED SYNTHETIC TRAINING / REAL CAMERA CHECK',font=font(29),fill=white)
    draw.text((24,56),'Same 3 epochs / 614 samples per epoch. Yellow: liquid proposals at 0.35.',font=font(21),fill=white)
    draw.text((24,88),'Development examples, not reviewed accuracy or a blind test.',font=font(20),fill=white)
    for i,sid in enumerate(illustrative_ids):
        for col,proposals in enumerate([old[sid],predictions[sid]]):
            x=24+col*672;y=140+i*540
            draw.text((x,y),('BASELINE' if col==0 else 'VIAL CROP CANDIDATE' if view else 'PAIRED RENDER CANDIDATE')+' / '+labels[i],font=font(19),fill=white)
            im=Image.open(cases[sid]['source']).convert('RGB');d=ImageDraw.Draw(im)
            for p in proposals:
                if p['confidence']>=.35:
                    b=p['box'];d.rectangle(b,outline=yellow,width=2)
                    d.text((max(0,b[0]),max(0,b[1]-18)),f'{p["confidence"]:.2f}',font=font(15),fill=yellow,stroke_width=1,stroke_fill=bg)
            board.paste(im,(x,y+30))
    draw.text((24,2310),'No real CSI weight fitting; generic COCO initialization. Original labels unchanged.',font=font(19),fill=white)
    board.save(a.out/'real-camera-before-after.png')

    rows=[json.loads(l) for l in (a.renders/'manifest.jsonl').read_text().splitlines()]
    contact=Image.new('RGB',(1440,1450),bg);d=ImageDraw.Draw(contact)
    d.text((24,12),'NEW SYNTHETIC PAIRS / SAME LIGHT, CAMERA AND GLASS',font=font(27),fill=white)
    d.text((24,52),'3x source crop; magenta = geometric body target, never a stream or vial target.',font=font(20),fill=white)
    byid={r['id']:r for r in rows}
    for i,group in enumerate([0,1,4,11]):
        for col,state in enumerate(['empty','stream-only','early','established']):
            row=byid[f'pair-{group:03d}-{state}'];x=24+col*354;y=96+i*330
            im=Image.open(a.renders/row['image']).convert('RGB')
            if row['box'] is not None:ImageDraw.Draw(im).rectangle(row['box'],outline='#f48bce',width=1)
            contact.paste(im.crop((265,310,360,435)).resize((228,300),Image.Resampling.NEAREST),(x,y+24))
            d.text((x,y),f'{group:02d} / {state}',font=font(17),fill=white)
    contact.save(a.out/'paired-synthetic-examples.png')
    real_b=read(baseline/'comparison.json')['real_run4_machine_agreement']
    real_c=read(a.candidate/'comparison.json')['real_run4_machine_agreement']
    result=dict(schema_version=1,baseline_model=base_model,candidate_model=candidate_model,
        matched_budget=plan['matched_budget'],real_run4_provisional_agreement=dict(baseline=real_b,candidate=real_c),
        original_synthetic_development=dict(baseline=read(baseline/'comparison.json')['synthetic_development'],candidate=score(original,1)),
        new_grouped_synthetic_development=dict(baseline=base_new,candidate=score(new,1)),
        new_synthetic_strata={state:{name:score([r for r in predictions_rows if byid[r['source_id']]['state']==state],1)
            for name,predictions_rows in [('baseline',base_new_rows),('candidate',new)]} for state in ['empty','stream-only','early','established']},
        input_view=view,synthetic_scoring='Mapped to full-frame original geometric targets; no reference clipping for ROI arm.',
        renders=read(a.renders/'completed-render.json'),run5_used=False,training_real_runs=[],
        app_changed=False,promoted=False,limits=plan['limitations'])
    write(a.out/'comparison.json',result)
    # Questions for human reviewers; do not convert model disagreement into
    # approved annotations or replace the existing machine-reference targets.
    queue_ids=illustrative_ids+['vial_fill_20260922_170544_697177','vial_fill_20260922_170548_711253','vial_fill_20260922_171633_379025']
    queue=[]
    for sid in queue_ids:
        row=cases[sid]
        queue.append(dict(source_id=sid,source_run=row['source_run'],source=row['source'],image_sha256=row['source_sha256'],
            existing_label_sha256=row['label_sha256'],unchanged_machine_targets=row['targets'],status='needs_human_review',
            question='Review visible accumulated pool/body, exclude stream/glass/label; mark uncertain if visibility or target scope is ambiguous. Do not treat a missing machine box as reviewed empty.'))
    write(a.out/'human-target-scope-review.json',dict(approved_labels_created=0,proposed_contract='Visible accumulated liquid body; stream separate. Human convention approval required.',cases=queue))
    write(a.out/'completed.json',dict(baseline_sha256=base_model['checkpoint_sha256'],candidate_sha256=candidate_model['checkpoint_sha256'],
        artifacts={p.name:sha(p) for p in a.out.iterdir() if p.is_file()},app_changed=False,promoted=False))
    print(json.dumps(dict(real_run4=result['real_run4_provisional_agreement'],new_synthetic=result['new_grouped_synthetic_development']),indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['candidate','renders','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--tools',type=Path,default=Path('C:/Users/Austin/ai-club-tools'))
    main(p.parse_args())
