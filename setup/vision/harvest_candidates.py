"""Export bounded weak-label candidates and translations from training runs only.

This is a bootstrap augmentation experiment, not a fluid renderer. Neither
existing evaluation annotations nor Run 5 images are read by this tool.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from replay.config import load_config
from replay.dataset import Catalog
from replay.inputs import RecordingSource
from replay.measurement import Extractor
from replay.geometry import decode_mask


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--limit-per-run',type=int,default=20)
    args=p.parse_args()
    if not 1<=args.limit_per_run<=100:
        p.error('limit must be 1..100 per training run')
    args.out.mkdir(parents=True,exist_ok=False)
    for name in ('images','box-labels','liquid-masks'):
        (args.out/name).mkdir()
    c=load_config(args.config)
    catalog=Catalog(args.data,include_annotations=False)
    rows=[]
    for run_id in (1,2,3):
        run=catalog.runs[run_id]
        extractor=Extractor(c)
        count=0
        for packet in RecordingSource(run['cameras']['vial_fill'],c.sample_stride):
            row=extractor.measure(packet.rgb)
            if not row['measurement_valid'] or packet.frame_index%90 or count>=args.limit_per_run:
                continue
            count+=1
            g=row['geometry']; h,w=packet.rgb.shape[:2]
            mask=decode_mask(g['mask'])
            # Preserve parent recording/frame, transform, and model identities.
            # Output labels remain provisional even for high-confidence boxes.
            for dx in (0,12):
                name=f'run{run_id}-f{packet.frame_index:06d}-dx{dx}'
                transform=np.float32([[1,0,dx],[0,1,0]])
                image=cv2.warpAffine(packet.rgb,transform,(w,h),borderMode=cv2.BORDER_REFLECT_101)
                shifted=cv2.warpAffine(mask,transform,(w,h),flags=cv2.INTER_NEAREST)
                lines=[]
                for cls,key in ((0,'vial_box'),(1,'liquid_box')):
                    x0,y0,x1,y1=g[key]
                    if not 0<=x0+dx<x1+dx<=w:
                        raise ValueError('Augmentation would clip proposal')
                    lines.append(f'{cls} {((x0+x1)/2+dx)/w:.8f} {(y0+y1)/2/h:.8f} {(x1-x0)/w:.8f} {(y1-y0)/h:.8f}')
                cv2.imwrite(str(args.out/'images'/f'{name}.png'),cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(args.out/'liquid-masks'/f'{name}.png'),shifted*255)
                (args.out/'box-labels'/f'{name}.txt').write_text('\n'.join(lines)+'\n')
                rows.append(dict(id=name,source_run=run_id,source_video_sha256=run['cameras']['vial_fill']['sha256'],
                                 source_frame=packet.frame_index,mp4_s=packet.video_s,
                                 source_rgb_sha256=hashlib.sha256(packet.rgb.tobytes()).hexdigest(),
                                 model_sha256=c.vision_model_sha256,transform=dict(dx=dx,dy=0),
                                 status='unreviewed_model_proposal',mask_kind='heuristic_colored_liquid_sampling_mask',
                                 image=f'images/{name}.png',boxes=f'box-labels/{name}.txt',mask=f'liquid-masks/{name}.png'))
            if count>=args.limit_per_run:
                break
    (args.out/'manifest.json').write_text(json.dumps(dict(schema_version=1,artifact_kind='bootstrap_augmentation_candidates',
        train_runs=[1,2,3],excluded_runs=[4,5],note='These are candidate training labels, never evaluation ground truth. No 3D rendering or physical fluid simulation.',
        count=len(rows),samples=rows),indent=2))
    print(f'{len(rows)} provisional examples: {args.out}',flush=True)


if __name__=='__main__':
    main()
