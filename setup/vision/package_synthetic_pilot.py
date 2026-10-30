"""Package existing synthetic-only Dataset 2 RGB and geometry labels for the team."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(a):
    if a.out.exists() or a.out.with_suffix('.verification.json').exists():
        raise ValueError('Use a fresh archive path')
    plan=json.loads((a.model_root/'experiment-plan.json').read_text())
    if plan['training_real_runs'] or plan['selection_real_runs'] or plan['run5_used']:
        raise ValueError('Not a synthetic-only pilot')
    original=a.tools/'results/real-synthetic-detector-20261019/data/mixed'
    provenance=json.loads((a.tools/'results/real-synthetic-detector-20261019/data/sim-only/provenance.json').read_text())
    unique={}
    for row in provenance['samples']:
        unique.setdefault(row['native_sample_id'],row)
    metadata=[];coco={(split,cls):dict(images=[],annotations=[],categories=[dict(id=1,name='active_vial' if cls==0 else 'liquid_body')]) for split in ['train','val'] for cls in [0,1]}
    top='dataset2-synthetic-pilot-v1/'
    readme=(
        'DATASET 2 / SYNTHETIC TRAINING PILOT / October 22, 2026\n\n'
        '768 generated 640x480 images: 614 train, 154 development. Split by image\n'
        'hash groups with seed19. No real CSI images or labels are included.\n'
        'Use data.yaml for two-class YOLO: 0 active_vial, 1 liquid_body.\n'
        'COCO single-class vial/liquid files use category1; image paths are relative\n'
        'to this package root. Single-class liquid YOLO labels are also supplied\n'
        'under labels-liquid; an adapter must pair these with images explicitly.\n\n'
        'IMPORTANT: labels are provisional rendered geometric body boxes, not\n'
        'reviewed visible-liquid masks, first-visible onsets or real defect truth.\n'
        'The falling stream is not a positive liquid_body target. Empty body labels\n'
        'describe the synthetic scene, not confirmed absence in real footage.\n'
        'This older 768-scene fixture has one native vial pose and no liquid body\n'
        'box <=32px high. Pale/clear observability and optics are imperfect.\n'
        'This package enables synthetic-only training; it is not a new improved\n'
        'renderer or evidence that the sim-to-real gap is closed.\n\n'
        'Keep donor scenes, image hashes, and any augmented derivatives in one\n'
        'split. Do not convert these development images into training without\n'
        'creating a separately versioned experiment and independent evaluation.\n'
        'Real recordings are for inspection/evaluation; no new real-domain fitting.\n'
        'Disclose generic pretrained weights separately from CSI fine-tuning.\n'
        'No code, checkpoint, real image or approved optical mask is included.\n')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    files={}
    with zipfile.ZipFile(a.out,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        def add_bytes(name,content):
            if isinstance(content,str):content=content.encode('utf-8')
            z.writestr(top+name,content);files[name]=dict(bytes=len(content),sha256=hashlib.sha256(content).hexdigest())
        add_bytes('READ-ME.txt',readme)
        # Omitting path makes Ultralytics resolve paths relative to this YAML,
        # whereas an explicit '.' can resolve under its global datasets folder.
        add_bytes('data.yaml','train: images/train\nval: images/val\nnames:\n  0: active_vial\n  1: liquid_body\n')
        for i,s in enumerate(plan['samples']):
            row=unique[s['source_id']]
            if row['origin']!='synthetic' or row['source_run'] is not None:
                raise ValueError('A real row entered the package')
            image=a.model_root/s['image'];labels=original/row['labels']
            if sha(image)!=s['image_sha256'] or sha(labels)!=s['source_label_sha256']:
                raise ValueError('Frozen synthetic source changed')
            split=s['split'];image_rel=f'images/{split}/{s["source_id"]}.png';label_rel=f'labels/{split}/{s["source_id"]}.txt'
            add_bytes(image_rel,image.read_bytes());add_bytes(label_rel,labels.read_bytes())
            add_bytes(f'labels-liquid/{split}/{s["source_id"]}.txt',(a.model_root/s['label']).read_bytes())
            metadata.append(dict(source_id=s['source_id'],origin='synthetic',split=split,image=image_rel,image_sha256=s['image_sha256'],labels=label_rel,label_sha256=s['source_label_sha256'],annotation_status='provisional_geometric_body_boxes'))
            for cls in [0,1]:
                d=coco[(split,cls)];d['images'].append(dict(id=i+1,file_name=image_rel,width=640,height=480))
                for line in labels.read_text().splitlines():
                    c,x,y,w,h=map(float,line.split())
                    if c!=cls:continue
                    box=[(x-w/2)*640,(y-h/2)*480,w*640,h*480]
                    d['annotations'].append(dict(id=len(d['annotations'])+1,image_id=i+1,category_id=1,bbox=box,area=box[2]*box[3],iscrowd=0))
        add_bytes('metadata/samples.json',json.dumps(metadata,indent=2)+'\n')
        for (split,cls),d in coco.items():
            name=('vial' if cls==0 else 'liquid')+'-'+('development' if split=='val' else 'train')+'.json'
            add_bytes('coco/'+name,json.dumps(d,indent=2)+'\n')
        add_bytes('metadata/package-hashes.json',json.dumps(files,indent=2)+'\n')
    with zipfile.ZipFile(a.out) as z:
        if z.testzip() is not None:
            raise ValueError('Archive read-back failed')
        for name,entry in files.items():
            if hashlib.sha256(z.read(top+name)).hexdigest()!=entry['sha256']:
                raise ValueError('Archive bytes differ: '+name)
    proof=dict(schema='synthetic-pilot-package/v1',archive=str(a.out),archive_sha256=sha(a.out),archive_bytes=a.out.stat().st_size,
        image_count=768,train_images=614,development_images=154,real_images=0,code_files=0,model_files=0,
        annotation_status='Geometric proxy boxes; not approved visible-pool truth',source_plan_sha256=sha(a.model_root/'experiment-plan.json'),
        all_payload_hashes_verified=True,published=False)
    a.out.with_suffix('.verification.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps(proof,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tools',type=Path,default=Path.home()/'ai-club-tools')
    p.add_argument('--model-root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
