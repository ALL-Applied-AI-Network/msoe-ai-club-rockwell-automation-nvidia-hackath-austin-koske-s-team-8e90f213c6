"""Source-exact real/render comparisons and frozen-model diagnostic overlays.

Illustrative, unpaired cases. No new labels, fitting, tuning or renderer edits.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[2]
BG='#152638';WHITE='#f2f5f8';MUTED='#bdcbd6';YELLOW='#ffd05a';PINK='#f48bce'
PAIRS=[
    ('No supplied real liquid target','vial_fill_20260923_191017_576908','00-empty'),
    ('Shallow blue example','vial_fill_20260923_191029_656060','02-tiny-blue'),
    ('Pale low-fill example','vial_fill_20260923_184720_822117','03-tiny-pale'),
    ('Established blue example','vial_fill_20260922_170600_743982','07-mid-blue'),
]


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def font(size):return ImageFont.truetype('C:/Windows/Fonts/arial.ttf',size)


def text(image,x,y,value,size=20,color=WHITE,max_width=None):
    draw=ImageDraw.Draw(image);f=font(size)
    if max_width is None:
        draw.text((x,y),value,font=f,fill=color);return
    lines=[];line=''
    for word in value.split():
        candidate=(line+' '+word).strip()
        if draw.textlength(candidate,font=f)>max_width and line:
            lines.append(line);line=word
        else:line=candidate
    if line:lines.append(line)
    for i,line in enumerate(lines):draw.text((x,y+i*(size+6)),line,font=f,fill=color)


def marked(image,boxes,color=YELLOW,scores=False):
    image=image.copy();draw=ImageDraw.Draw(image)
    for p in boxes:
        if p.get('confidence',1)<.35:continue
        box=p['box'];draw.rectangle(box,outline=color,width=2)
        if scores:
            x=max(0,min(575,box[0]));y=max(0,box[1]-18)
            draw.rectangle((x,y,x+60,y+18),fill=BG)
            draw.text((x+2,y),f'{p["confidence"]:.2f}',fill=color,font=font(15))
    return image


def labels(path):
    result=[]
    for line in Path(path).read_text().splitlines():
        c,x,y,w,h=map(float,line.split())
        result.append(dict(class_id=int(c),box=[(x-w/2)*640,(y-h/2)*480,(x+w/2)*640,(y+h/2)*480]))
    return result


def main(a):
    if a.out.exists():raise ValueError('Use a fresh comparison output')
    a.out.mkdir(parents=True);(a.out/'sources').mkdir()
    original=a.tools/'results/real-synthetic-detector-20261019';real_root=original/'data/real';render_root=original/'renders'
    real_rows=read(real_root/'provenance.json')['samples'];real={r['source_id']:r for r in real_rows if not r.get('parent_id')}
    renders={r['sample_id']:r for r in [json.loads(s) for s in (render_root/'manifest.jsonl').read_text().splitlines()]}
    fit=a.tools/'results/synthetic-liquid-20261022/fasterrcnn-v1';fit_plan=read(fit/'experiment-plan.json');model_meta=read(fit/'frozen-model.json')
    if model_meta['training_real_runs'] or sha(fit/'final.pt')!=model_meta['checkpoint_sha256']:raise ValueError('Synthetic-trained model identity differs')
    splits={s['source_id']:s['split'] for s in fit_plan['samples']}
    images={};cases={}

    def add_real(sid):
        row=real[sid]
        if row['source_run'] not in [1,2,3,4]:raise ValueError('Run 5 is excluded')
        path=real_root/row['image'];label=real_root/row['labels']
        if sha(path)!=row['image_sha256'] or sha(label)!=row['actual_label_sha256']:raise ValueError('Real source changed')
        images[sid]=Image.open(path).convert('RGB')
        cases[sid]=dict(origin='real',source_run=row['source_run'],source=str(path),source_sha256=sha(path),label_sha256=sha(label),targets=labels(label),status='Unreviewed machine targets; unchanged; no real CSI image used for this model fit')

    def add_render(sid):
        row=renders[sid];path=render_root/row['image_file']
        if sha(path)!=row['image_sha256']:raise ValueError('Render RGB changed')
        images[sid]=Image.open(path).convert('RGB')
        targets=[]
        for obj in row['annotations']:
            if obj['instance_id'] in [201,301] and obj['present']:
                targets.append(dict(class_id=1 if obj['instance_id']==201 else 2,box=obj['bbox_xyxy']))
        cases[sid]=dict(origin='synthetic',source=str(path),source_sha256=sha(path),parameters=row['parameters'],targets=targets,fit_split=splits[sid],status='Geometric body/stream proxies; stream is not a positive liquid-body training target')

    for _,rid,sid in PAIRS:add_real(rid);add_render(sid)
    for sid in ['04-tiny-clear-proxy']:add_render(sid)
    scope_ids=['vial_fill_20260922_170544_697177','vial_fill_20260922_170548_711253','vial_fill_20260922_171633_379025']
    for rid in scope_ids:add_real(rid)
    for sid,image in images.items():
        if image.size!=(640,480):raise ValueError('Source dimensions changed')
        image.save(a.out/'sources'/(sid+'.png'))
    write(a.out/'plan.json',dict(kind='illustrative_real_vs_synthetic_diagnostic',pairs=PAIRS,selection='Fixed stage examples and previously flagged target-scope cases; not paired volumes/recipes or an accuracy subset.',model=model_meta,source_cases=cases,run5_used=False,training=False,labels_changed=False,crop=[270,230,350,430]))

    def full_board(filename,predictions=None):
        image=Image.new('RGB',(1360,2440),BG)
        text(image,24,16,'REAL CAMERA vs ACTUAL SYNTHETIC CORPUS',32)
        text(image,24,58,'Original 640 x 480 pixels, same scale. Illustrative cases; not matched volumes or recipes.',22,MUTED)
        if predictions is not None:text(image,24,90,'Yellow = the SAME synthetic-trained Faster R-CNN liquid proposals at confidence >= 0.35.',20,YELLOW)
        for i,(name,rid,sid) in enumerate(PAIRS):
            y=140+i*550
            for col,key in enumerate([rid,sid]):
                x=24+col*672
                subtitle=f'REAL Run {cases[key]["source_run"]} / {name}' if col==0 else f'RENDER {sid} / synthetic {splits[sid]}'
                text(image,x,y,subtitle,20)
                im=images[key] if predictions is None else marked(images[key],predictions[key],scores=True)
                image.paste(im,(x,y+38))
                source=key.removeprefix('vial_fill_')
                text(image,x,y+520,source,16,MUTED)
        text(image,24,2350,'Rendered labels are geometric proxies. Real boxes remain unreviewed; no new optical truth is assigned.',20,MUTED)
        text(image,24,2384,'Qualitative diagnosis only. No model, threshold, image source or label was changed for these comparisons.',20,MUTED)
        image.save(a.out/filename)

    full_board('01-real-vs-synthetic.png')
    # Identical crop and nearest-neighbor enlargement expose the actual source pixels.
    crop=(270,230,350,430);crop_size=(240,600)
    close=Image.new('RGB',(1100,1460),BG)
    text(close,20,16,'SAME PIXEL CROP / lower vial detail',29)
    text(close,20,57,'3x nearest-neighbor enlargement; no enhancement or normalization.',20,MUTED)
    for i,(name,rid,sid) in enumerate(PAIRS):
        ox=20+(i%2)*550;oy=105+(i//2)*665
        for col,key in enumerate([rid,sid]):
            x=ox+col*265
            text(close,x,oy,('REAL' if col==0 else 'RENDER')+' / '+['no target','shallow blue','pale low','blue body'][i],18)
            close.paste(images[key].crop(crop).resize(crop_size,Image.Resampling.NEAREST),(x,oy+32))
    text(close,20,1430,'Different scenes and amounts; inspect appearance, not physical volume equivalence.',18,MUTED)
    close.save(a.out/'02-lower-vial-closeups.png')

    # Fresh inference only on the eight predeclared raw images.
    import torch
    sys.path.insert(0,str(fit/'helpers'));sys.path.insert(1,str(ROOT/'setup/vision'))
    from train_early_liquid import build_resnet
    torch.set_num_threads(4);model=build_resnet(fit)
    model.load_state_dict(torch.load(fit/'final.pt',map_location='cpu',weights_only=True)['model_state']);model.cuda().eval()
    predictions={}
    with torch.inference_mode():
        for _,rid,sid in PAIRS:
            for key in [rid,sid]:
                tensor=torch.from_numpy(np.asarray(images[key]).copy()).permute(2,0,1).float().cuda()/255.
                result=model([tensor])[0]
                predictions[key]=[dict(box=b.tolist(),confidence=float(score)) for b,score in zip(result['boxes'].cpu(),result['scores'].cpu())]
    write(a.out/'predictions.json',dict(model_sha256=model_meta['checkpoint_sha256'],results=predictions,confidence_display=.35,new_fits=0))
    full_board('03-same-model-predictions.png',predictions)

    # Unchanged supplied target scopes, kept separate from model proposals.
    scope=Image.new('RGB',(1350,1380),BG)
    text(scope,22,15,'LABEL CONTRACT: these targets do not always describe the same thing',29)
    text(scope,22,57,'Yellow = real provisional liquid or rendered body box. Pink = separate rendered stream, excluded from body training.',18,MUTED)
    captions=['Real: tall liquid target','Real: body target 4 seconds later','Real: 20px target near holder','Render: body + separate stream']
    keys=scope_ids+['02-tiny-blue']
    for i,key in enumerate(keys):
        x=22+i*330;text(scope,x,104,captions[i],18)
        im=marked(images[key],[p for p in cases[key]['targets'] if p['class_id']==1])
        im=marked(im,[p for p in cases[key]['targets'] if p['class_id']==2],PINK)
        scope.paste(im.crop((260,60,360,430)).resize((300,1110),Image.Resampling.NEAREST),(x,140))
    text(scope,22,1280,'The 20px target is a narrow band near the holder; small target height alone does not establish an early pool.',19,MUTED,max_width=1300)
    text(scope,22,1335,'Flag for human review. No replacement box, onset, liquid volume or approved annotation is asserted here.',19,MUTED)
    scope.save(a.out/'04-target-scope-mismatch.png')

    masks=Image.new('RGB',(1100,850),BG)
    text(masks,20,16,'GEOMETRY MASK vs VISIBLE RGB',30)
    text(masks,20,56,'All three are the same 2% proxy level; 2% is an uncalibrated renderer parameter.',19,MUTED)
    mask_data={}
    for i,key in enumerate(['02-tiny-blue','03-tiny-pale','04-tiny-clear-proxy']):
        x=20+i*360;row=renders[key];obj=next(o for o in row['annotations'] if o['instance_id']==201);path=render_root/obj['mask_file']
        if sha(path)!=obj['mask_sha256']:raise ValueError('Native mask changed')
        mask=Image.open(path).convert('L');pixels=int(np.count_nonzero(np.asarray(mask)))
        text(masks,x,100,key,20)
        masks.paste(images[key].crop((270,355,350,430)).resize((320,300),Image.Resampling.NEAREST),(x,135))
        text(masks,x,452,f'Native mask: {pixels} pixels',20,YELLOW)
        masks.paste(mask.convert('RGB').crop((270,355,350,430)).resize((320,300),Image.Resampling.NEAREST),(x,490))
        mask_data[key]=dict(mask_sha256=sha(path),pixels=pixels,box=obj['bbox_xyxy'])
    text(masks,20,809,'Native IDs verify geometry; they do not establish which pixels look like liquid through real glass.',19,MUTED)
    masks.save(a.out/'05-rgb-vs-geometric-mask.png');write(a.out/'mask-comparison.json',mask_data)
    shutil.copy2(__file__,a.out/Path(__file__).name)
    write(a.out/'completed.json',dict(illustrative_pairs=4,fresh_inference_frames=8,training=False,run5_used=False,labels_changed=False,artifacts={p.name:sha(p) for p in a.out.iterdir() if p.is_file()}))
    print(json.dumps(dict(out=str(a.out),pairs=4,fresh_inference_frames=8,geometric_masks=mask_data),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tools',type=Path,default=Path.home()/'ai-club-tools')
    p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
