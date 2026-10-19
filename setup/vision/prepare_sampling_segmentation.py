"""Prepare binary crop targets from Runs 1-3 candidates and Run 4 teacher proposals."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from replay.config import load_config
from replay.measurement import Extractor
from replay.geometry import decode_mask
from replay.segmentation import crop_input,INPUT_SIZE


def prepare(candidates,teacher,config,output):
    output.mkdir(parents=True,exist_ok=False)
    document=json.loads((candidates/'manifest.json').read_text())
    if any(s['source_run'] not in (1,2,3) for s in document['samples']):
        raise ValueError('Training candidates must contain only Runs 1-3')
    images={}; masks={}; rows=[]
    for split in ('train','val'): images[split]=[]; masks[split]=[]
    def append(split,image,mask,box,source):
        crop,region=crop_input(image,box); x0,y0,x1,y1=region
        target=cv2.resize(mask[y0:y1,x0:x1],INPUT_SIZE,interpolation=cv2.INTER_NEAREST)
        if not target.any(): raise ValueError('Empty training target')
        images[split].append(crop); masks[split].append(target)
        rows.append(dict(source,split=split,crop_box=region,crop_rgb_sha256=hashlib.sha256(crop.tobytes()).hexdigest(),
                         crop_mask_sha256=hashlib.sha256(target.tobytes()).hexdigest(),status='unreviewed_provisional_sampling_mask'))
    for s in document['samples']:
        image_path=candidates/s['image']; mask_path=candidates/s['mask']; box_path=candidates/s['boxes']
        image=cv2.cvtColor(cv2.imread(str(image_path)),cv2.COLOR_BGR2RGB)
        mask=(cv2.imread(str(mask_path),0)>0).astype(np.uint8)
        vial=[list(map(float,line.split()[1:])) for line in box_path.read_text().splitlines() if line.split()[0]=='0']
        if len(vial)!=1: raise ValueError('Expected one provisional vial box')
        x,y,w,h=vial[0]; height,width=image.shape[:2]
        box=[round((x-w/2)*width),round((y-h/2)*height),round((x+w/2)*width),round((y+h/2)*height)]
        append('train',image,mask,box,dict(id=s['id'],source_run=s['source_run'],source_frame=s['source_frame'],
               source_video_sha256=s['source_video_sha256'],parent_rgb_sha256=s['source_rgb_sha256'],transform=s['transform'],
               image_sha256=hashlib.sha256(image_path.read_bytes()).hexdigest(),mask_sha256=hashlib.sha256(mask_path.read_bytes()).hexdigest()))
    baseline=load_config(config); extractor=Extractor(baseline)
    manifest=json.loads((teacher/'manifest.json').read_text()); excluded=[]
    for s in manifest:
        if s['camera']!='vial_fill' or s['video_id']!='4': continue
        image_path=teacher/'review-pack/vial_fill/images'/s['source']
        if not image_path.is_file(): excluded.append(dict(id=s['id'],reason='no_source_image')); continue
        if hashlib.sha256(image_path.read_bytes()).hexdigest()!=s['source_sha256']: raise ValueError('Run 4 source changed')
        image=cv2.cvtColor(cv2.imread(str(image_path)),cv2.COLOR_BGR2RGB); feature=extractor.measure(image); g=feature['geometry']
        if g['mask_pixels']<60 or any(f!='vial_track_not_stable' for f in g['quality_flags']):
            excluded.append(dict(id=s['id'],reason=g['quality_flags'])); continue
        append('val',image,decode_mask(g['mask']),g['vial_box'],dict(id=s['id'],source_run=4,
               image_sha256=s['source_sha256'],teacher_model_sha256=baseline.vision_model_sha256,
               teacher_flags=feature['quality_flags'],note='Independent still; motion/timing QA is not a segmentation target.'))
    if not images['train'] or not images['val']: raise ValueError('Empty train/development split')
    if {r['crop_rgb_sha256'] for r in rows if r['split']=='train'} & {r['crop_rgb_sha256'] for r in rows if r['split']=='val'}:
        raise ValueError('Exact cropped image crosses train/development split')
    for split in ('train','val'):
        np.savez_compressed(output/f'{split}.npz',images=np.stack(images[split]),masks=np.stack(masks[split]))
    report=dict(schema_version=1,train_runs=[1,2,3],development_runs=[4],excluded_runs=[5],
        train_count=len(images['train']),val_count=len(images['val']),
        unique_training_source_frames=len({(r['source_run'],r.get('source_frame')) for r in rows if r['split']=='train'}),
        unique_training_crops=len({r['crop_rgb_sha256'] for r in rows if r['split']=='train'}),
        input_size=list(INPUT_SIZE),mask_target='provisional conservative sampled liquid interior; not full liquid outline',
        candidate_manifest_sha256=hashlib.sha256((candidates/'manifest.json').read_bytes()).hexdigest(),
        teacher_model_sha256=baseline.vision_model_sha256,
        train_sha256=hashlib.sha256((output/'train.npz').read_bytes()).hexdigest(),val_sha256=hashlib.sha256((output/'val.npz').read_bytes()).hexdigest(),
        samples=rows,excluded_development=excluded)
    (output/'provenance.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('samples','excluded_development')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--candidates',type=Path,required=True); p.add_argument('--teacher',type=Path,required=True); p.add_argument('--config',type=Path,required=True); p.add_argument('--out',type=Path,required=True); a=p.parse_args(); prepare(a.candidates,a.teacher,a.config,a.out)
