"""Expanded fixed-checkpoint real probe. Real labels/fit/threshold tuning absent."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
from dataset1_fiveclass import CLASSES,SLOTS,read,sha,write


def prepare(old,geometry,camera,out):
    import cv2
    import numpy as np
    out.mkdir(parents=True,exist_ok=False);(out/'images').mkdir()
    registration=read(old/'registration.json');world=read(geometry);spec=read(camera)
    f=spec['focal_px_final'];K=np.array([[f,0,320],[0,f,240],[0,0,1.]],np.float64);D=np.array([spec['lens_k1'],0,0,0,0.])
    points=np.array([world['apertures_world_m'][slot] for slot in SLOTS],np.float64)
    pixels,_=cv2.projectPoints(points,np.array(registration['rvec']),np.array(registration['tvec']),K,D);boxes={}
    for slot,(x,y) in zip(SLOTS,pixels.reshape(-1,2)):
        x,y=int(round(x)),int(round(y));box=[x-48,y-48,x+48,y+48]
        if not(0<=box[0]<box[2]<=640 and 0<=box[1]<box[3]<=480):raise ValueError('Crop outside source; do not clip silently')
        boxes[slot]=box
    samples=[]
    for sample in read(old/'samples.json'):
        if sample['run_id'] not in (1,2,3,4):raise ValueError('Unexpected run')
        path=old/sample['image']
        if sha(path)!=sample['source_sha256']:raise ValueError('Original JPEG differs')
        shutil.copy2(path,out/sample['image']);sample=dict(sample,boxes=boxes,mapping_status='proposed-G-anchor-to-both-fixtures-human-unreviewed')
        samples.append(sample)
    write(out/'samples.json',samples)
    write(out/'registration.json',dict(schema='proposed-two-fixture-registration/v1',parent_sha256=sha(old/'registration.json'),
        geometry_sha256=sha(geometry),boxes=boxes,points_xy=dict(zip(SLOTS,pixels.reshape(-1,2).tolist())),
        approved_mapping=False,S_independently_calibrated=False,anchor_image=registration['anchor_image'],anchor_sha256=registration['anchor_sha256']))
    write(out/'input-manifest.json',{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()})
    print(json.dumps(dict(samples=len(samples),targets=60*len(samples),out=str(out),real_accuracy=None)))


def infer(root,batch,out):
    import fcntl
    lock=(Path.home()/'ai-club-tools/spark-gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out.mkdir(exist_ok=False)
    for rel,digest in read(batch/'input-manifest.json').items():
        if sha(batch/rel)!=digest:raise ValueError('Real input differs')
    complete=read(root/'fitted-v1/completed.json');frozen=complete['models']
    # Pick the review arm only from frozen synthetic development, simpler RGB wins ties.
    selected=max(('rgb','target_rgb'),key=lambda mode:frozen[mode]['development']['macro_recall'])
    write(out/'plan.json',dict(checkpoints_frozen_sha256=sha(root/'fitted-v1/all-checkpoints-frozen.json'),
        batch_manifest_sha256=sha(batch/'input-manifest.json'),models=['rgb','target_rgb'],selected_review_model=selected,
        selection_basis='Synthetic development macro recall only; RGB wins exact ties',helper_sha256=sha(__file__),
        real_accuracy=None,real_fitting=False,real_threshold_tuning=False,paired_reference_status='unavailable: no qualified real golden',run5_used=False))
    from train_dataset1_fiveclass import build_model
    from PIL import Image
    import numpy as np
    import torch
    torch.set_num_threads(4);all_rows={};yy,xx=np.mgrid[:96,:96]
    heat=torch.from_numpy(np.exp(-((xx-48)**2+(yy-48)**2)/(2*9**2)).astype(np.float32))[None,None].cuda()
    for mode in ('rgb','target_rgb'):
        meta=frozen[mode];checkpoint=root/'fitted-v1'/mode/'best.pt'
        if sha(checkpoint)!=meta['checkpoint_sha256']:raise ValueError('Frozen model differs')
        model=build_model(meta['channels']).cuda();model.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=True)['state_dict']);model.eval();rows=[]
        with torch.inference_mode():
            for sample in read(batch/'samples.json'):
                if sample['run_id'] not in (1,2,3,4) or sha(batch/sample['image'])!=sample['source_sha256']:raise ValueError('Real source identity differs')
                with Image.open(batch/sample['image']) as im:
                    x=torch.stack([torch.from_numpy((np.array(im.convert('RGB').crop(sample['boxes'][slot]),dtype=np.float32)/255).transpose(2,0,1).copy()) for slot in SLOTS]).cuda()
                if mode=='target_rgb':x=torch.cat([x,heat.expand(60,-1,-1,-1)],1)
                probabilities=model(x).softmax(1).cpu().tolist()
                for slot,probs in zip(SLOTS,probabilities):rows.append(dict(sample_id=sample['sample_id'],slot=slot,fixture=slot[0],
                    source_sha256=sample['source_sha256'],run_id=sample['run_id'],box_xyxy=sample['boxes'][slot],
                    class_scores=dict(zip(CLASSES,probs)),proposed_class=CLASSES[max(range(5),key=lambda i:probs[i])],
                    mapping_status=sample['mapping_status'],visibility='unreviewed',truth=None,quality_decision=None,physical_vial_association='unlinked',
                    expected_occupied=None,scores_calibrated=False,model_sha256=meta['checkpoint_sha256']))
                for row in rows[-60:]:
                    row.update(defect_score=1-row['class_scores']['pass'],defect_threshold=meta['threshold'],
                        occupied_defect_warning=1-row['class_scores']['pass']>=meta['threshold'],
                        warning_scope='Frozen synthetic occupied-target diagnostic; empty-slot action requires progress context')
        write(out/(mode+'-predictions.json'),rows);all_rows[mode]=rows
    write(out/'predictions.json',all_rows[selected])
    write(out/'completed.json',dict(classes=CLASSES,checkpoint_sha256=frozen[selected]['checkpoint_sha256'],selected_review_model=selected,
        predictions_sha256=sha(out/'predictions.json'),helper_sha256=sha(__file__),plan_sha256=sha(out/'plan.json'),
        samples=16,targets_per_model=960,counts={mode:dict(Counter(r['proposed_class'] for r in rows)) for mode,rows in all_rows.items()},
        real_accuracy=None,approved_labels=False,training=False,run5_used=False,local_gpu_used=False))
    print(json.dumps(read(out/'completed.json')),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','infer']);p.add_argument('--old-batch',type=Path)
    p.add_argument('--geometry',type=Path);p.add_argument('--camera',type=Path);p.add_argument('--root',type=Path);p.add_argument('--batch',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    prepare(a.old_batch,a.geometry,a.camera,a.out) if a.action=='prepare' else infer(a.root,a.batch,a.out)
