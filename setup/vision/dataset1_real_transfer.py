"""Frozen RGB classifier transfer on real cap_seal; no fitting or approved labels.

Prepare runs on CPU. Infer runs only on the GB10 under the shared GPU lock.
Fixture registration is a geometric proposal; reviewed slot mapping is separate.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def sample_indices(length,count=4):
    if length<count:
        raise ValueError('Not enough distinct frames')
    return [round(i*(length-1)/(count-1)) for i in range(count)]


def validate_source_rows(rows):
    if not rows or any(int(row['run_id']) not in (1,2,3,4) for row in rows):
        raise ValueError('Only attributed Runs1-4 are eligible')
    if len({row['source_sha256'] for row in rows})!=len(rows):
        raise ValueError('Duplicate source image bytes')
    for row in rows:
        if set(row['boxes'])!={f'G{i:02d}' for i in range(1,31)}:
            raise ValueError('Missing target slot')
        for box in row['boxes'].values():
            if len(box)!=4 or not(0<=box[0]<box[2]<=640 and 0<=box[1]<box[3]<=480):
                raise ValueError('Invalid source crop')


def prepare(args):
    import cv2
    import numpy as np
    from PIL import Image,ImageDraw,ImageFont
    root=args.out;root.mkdir(exist_ok=False)
    assets=args.assets;data=args.data;pilot=args.pilot
    inventory=read(pilot/'inventory.json');camera=read(assets/'cameras/CapSealCam.json')
    source_plan=read(pilot/'gb10-results-v1/experiment-plan.json')
    if sha(assets/'cameras/CapSealCam.json')!=source_plan['source_camera_sha256']:
        raise ValueError('Camera source changed')
    index=list(csv.DictReader((data/'image_index.csv').open()))
    anchors=sorted([r for r in index if r['video_id']=='4'],key=lambda r:r['timestamp'])
    anchor=anchors[0];anchor_path=data/'images'/anchor['filename']
    rgb=np.array(Image.open(anchor_path).convert('RGB'));gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    # Predeclared image-space geometric search window, independent of class output.
    candidates=cv2.HoughCircles(gray[140:330,210:395],cv2.HOUGH_GRADIENT,1,22,
        param1=80,param2=15,minRadius=8,maxRadius=12)
    if candidates is None or len(candidates[0])!=30:
        raise ValueError('Registration requires exactly30 aperture candidates; inspect geometry, never silently invent a match')
    points=candidates[0,:,:2]+[210,140]
    rows=np.array(sorted(points.tolist(),key=lambda p:p[1])).reshape(5,6,2)
    if min(float(row[:,1].mean()-prev[:,1].mean()) for prev,row in zip(rows,rows[1:]))<24:
        raise ValueError('Aperture candidates do not form five distinct rows')
    targets={}
    for r,row in enumerate(rows):
        for c,xy in enumerate(sorted(row.tolist(),key=lambda p:p[0])):
            targets[f'G{(5-c)*5+r+1:02d}']=xy
    meshes={m['path']:m for m in inventory['meshes']}
    tray=meshes['/TrayInspection/tray_good_top/tray_good_top']
    z=tray['maximum'][2]
    aperture=[];caps=[]
    for i in range(1,31):
        slot=f'G{i:02d}'
        body=meshes[f'/TrayInspection/{slot}_tube/vial_tube']
        cap=meshes[f'/TrayInspection/{slot}_cap/vial_cap']
        center=(np.array(body['minimum'])+np.array(body['maximum']))/2
        aperture.append([center[0],center[1],z])
        caps.append((np.array(cap['minimum'])+np.array(cap['maximum']))/2)
    object_points=np.array(aperture,dtype=np.float64)
    observed=np.array([targets[f'G{i:02d}'] for i in range(1,31)],dtype=np.float64)
    f=camera['focal_px_final'];K=np.array([[f,0,320],[0,f,240],[0,0,1.]],dtype=np.float64)
    D=np.array([camera['lens_k1'],0,0,0,0.],dtype=np.float64)
    R=np.array(camera['R_world_to_cv']);rvec,_=cv2.Rodrigues(R)
    position=np.array(camera['position_m']);tvec=-R@position
    okay,rvec,tvec=cv2.solvePnP(object_points,observed,K,D,rvec,tvec,useExtrinsicGuess=True)
    if not okay:
        raise ValueError('Pose registration failed')
    reprojection,_=cv2.projectPoints(object_points,rvec,tvec,K,D)
    errors=np.linalg.norm(reprojection.reshape(-1,2)-observed,axis=1)
    estimated_R,_=cv2.Rodrigues(rvec);estimated_position=(-estimated_R.T@tvec).reshape(3)
    if float(np.sqrt(np.mean(errors**2)))>2 or float(np.linalg.norm(estimated_position-position))>.25:
        raise ValueError('Pose fit is inconsistent with supplied station camera')
    cap_pixels,_=cv2.projectPoints(np.array(caps),rvec,tvec,K,D)
    boxes={}
    for i,(x,y) in enumerate(cap_pixels.reshape(-1,2),1):
        x,y=int(round(x)),int(round(y));boxes[f'G{i:02d}']=[x-24,y-24,x+24,y+24]
    registration=dict(schema='proposed-fixture-registration/v1',anchor_image=anchor['filename'],anchor_run=4,
        anchor_sha256=sha(anchor_path),method='30 Hough aperture centers;5x6 grid; fixed intrinsics/distortion; solvePnP at CAD tray plane; project authored cap centers.',
        aperture_points=targets,projected_cap_centers=cap_pixels.reshape(-1,2).tolist(),boxes=boxes,
        rmse_pixels=float(np.sqrt(np.mean(errors**2))),max_error_pixels=float(max(errors)),
        source_camera_position=position.tolist(),estimated_camera_position=estimated_position.tolist(),rvec=rvec.reshape(3).tolist(),tvec=tvec.reshape(3).tolist(),
        approved_mapping=False,physical_slot_numbering_confirmed=False,training_eligible=False,
        boundary='Geometric proposal in authored G numbering. Human confirms target identity/visibility per review item. Other-run camera alignment is not independently calibrated.')
    write(root/'registration.json',registration)
    (root/'images').mkdir();samples=[]
    for run in (1,2,3,4):
        candidates=sorted([r for r in index if r['video_id']==str(run)],key=lambda r:r['timestamp'])
        for j in sample_indices(len(candidates)):
            source=candidates[j];path=data/'images'/source['filename'];destination=root/'images'/path.name
            shutil.copy2(path,destination)
            samples.append(dict(sample_id=f'run{run}-still-{j:03d}',run_id=run,image='images/'+path.name,
                source_sha256=sha(path),source_filename=path.name,capture_timestamp=source['timestamp'],
                native_pts=None,video_frame_index=None,video_mapping='unmatched; no wall-clock-to-frame conversion',
                boxes=boxes,source_index_row=j,source_dimensions=[640,480],mapping_status='geometry-proposed-human-unreviewed'))
    validate_source_rows(samples)
    model=read(pilot/'gb10-results-v1/fitted/rgb/frozen.json')
    shutil.copy2(pilot/'gb10-results-v1/fitted/rgb/best.pt',root/'rgb-best.pt')
    if sha(root/'rgb-best.pt')!=model['checkpoint_sha256']:
        raise ValueError('Frozen checkpoint differs')
    write(root/'frozen-rgb.json',model)
    # Retain the exact fitted architecture and its minimal dependency, never edit them.
    for name in ('train_dataset1_pilot.py','dataset1_pilot.py'):
        shutil.copy2(pilot/'gb10-results-v1'/name,root/name)
    write(root/'samples.json',samples)
    plan=dict(schema='dataset1-real-transfer/v1',samples=16,targets=480,runs=[1,2,3,4],run5_used=False,
        sampling='Four uniformly spaced image-index rows per run, independent of predictions/defect labels.',
        model='Frozen synthetic RGB classifier,selected epoch5',checkpoint_sha256=model['checkpoint_sha256'],
        registration_sha256=sha(root/'registration.json'),samples_sha256=sha(root/'samples.json'),helper_sha256=sha(__file__),
        trained=False,real_fitting=False,real_threshold_selection=False,approved_labels=False,
        paired_reference_status='unavailable: no human-qualified capped real golden; no substitute provided',
        truth_status='unreviewed; model outputs and reviewer instructions are not labels',
        real_accuracy=None,physical_vial_association='unlinked',
        limitations=['Run2/4 endpoint stills were previously visually inspected; this is development evidence, not a blind final test.',
            'Crops preserve the trained target-plane convention and can include neighboring objects. Mapping/occlusion must be reviewed.',
            'Five-class review includes unsupported crooked_cap/damaged_vial; the three-class model cannot clear them.'])
    write(root/'plan.json',plan)
    # The registration sheet uses no predictions or quality labels.
    font=ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf',12)
    sheet=Image.new('RGB',(1280,1060),'#142B40');draw=ImageDraw.Draw(sheet)
    for n,run in enumerate((1,2,3,4)):
        sample=next(r for r in samples if r['run_id']==run)
        im=Image.open(root/sample['image']).convert('RGB');d=ImageDraw.Draw(im)
        for slot,box in boxes.items():
            x,y=(box[0]+box[2])/2,(box[1]+box[3])/2
            d.ellipse((x-4,y-4,x+4,y+4),outline='#36FFFF',width=2)
            d.text((x-10,y-17),slot,fill='#36FFFF',font=font)
        x=(n%2)*640;y=(n//2)*530
        sheet.paste(im,(x,y+36));draw.text((x+10,y+9),f'RUN {run} / PROPOSED G SLOT CENTERS / no quality labels',font=font,fill='white')
    sheet.save(root/'registration-review.png')
    manifest={str(p.relative_to(root)).replace('\\','/'):sha(p) for p in root.rglob('*') if p.is_file()}
    write(root/'input-manifest.json',manifest)
    print(json.dumps(dict(out=str(root),samples=len(samples),targets=480,registration_rmse=registration['rmse_pixels'],max_error=registration['max_error_pixels'])),flush=True)


def infer(args):
    import fcntl
    import os
    import sys
    import time
    root=args.out.resolve();out=root/'inference';out.mkdir(exist_ok=False)
    for rel,digest in read(root/'input-manifest.json').items():
        if sha(root/rel)!=digest:
            raise ValueError('Transferred input changed: '+rel)
    lock=(Path.home()/'ai-club-tools/spark-gpu.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    sys.path.insert(0,str(root))
    from train_dataset1_pilot import build_model
    from PIL import Image
    import numpy as np
    import torch
    torch.set_num_threads(4)
    plan=read(root/'plan.json');samples=read(root/'samples.json');validate_source_rows(samples)
    if sha(root/'rgb-best.pt')!=plan['checkpoint_sha256']:
        raise ValueError('Model checkpoint differs')
    model=build_model(3).cuda();checkpoint=torch.load(root/'rgb-best.pt',map_location='cuda',weights_only=True)
    assert checkpoint['mode']=='rgb' and checkpoint['epoch']==5
    model.load_state_dict(checkpoint['state_dict']);model.eval()
    records=[]
    with torch.inference_mode():
        for sample in samples:
            source=Image.open(root/sample['image']).convert('RGB')
            slots=sorted(sample['boxes']);inputs=[]
            for slot in slots:
                crop=np.asarray(source.crop(sample['boxes'][slot]).resize((64,64)),dtype=np.float32)/255
                inputs.append(torch.from_numpy(crop.transpose(2,0,1).copy()))
            x=torch.stack(inputs).cuda();probabilities=model(x).softmax(1).cpu().tolist()
            for slot,p in zip(slots,probabilities):
                records.append(dict(sample_id=sample['sample_id'],source_sha256=sample['source_sha256'],run_id=sample['run_id'],
                    slot=slot,box_xyxy=sample['boxes'][slot],class_scores=dict(zip(checkpoint['classes'],p)),
                    proposed_class=checkpoint['classes'][int(np.argmax(p))],scores_calibrated=False,truth=None,
                    mapping_status=sample['mapping_status'],visible_and_unoccluded='unreviewed',
                    physical_vial_association='unlinked',expected_occupied=None,quality_decision=None))
    write(out/'predictions.json',records)
    write(out/'completed.json',dict(finished=time.time(),samples=16,target_predictions=len(records),gpu=torch.cuda.get_device_name(0),
        helper_sha256=sha(__file__),checkpoint_sha256=plan['checkpoint_sha256'],plan_sha256=sha(root/'plan.json'),
        predictions_sha256=sha(out/'predictions.json'),real_accuracy=None,training=False,run5_used=False,local_gpu_used=False))
    print(json.dumps(dict(completed=True,predictions=len(records),real_accuracy=None)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','infer']);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--data',type=Path);p.add_argument('--assets',type=Path);p.add_argument('--pilot',type=Path)
    a=p.parse_args();(prepare if a.action=='prepare' else infer)(a)
