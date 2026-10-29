"""Frozen student agreement with Run 5 teacher boxes; NOT mask accuracy."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--teacher',type=Path,required=True)
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',default='cpu')
    a=p.parse_args()
    if not a.weights.is_file():
        p.error('Weights must already exist; no implicit model download')
    a.out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((a.teacher/'manifest.json').read_text())
    gallery={r['id']:r for r in json.loads((a.teacher/'gallery.json').read_text()) if r['camera']=='vial_fill'}
    images=a.out/'data/images/val'; labels=a.out/'data/labels/val'
    images.mkdir(parents=True); labels.mkdir(parents=True)
    included=[]; excluded=[]
    for row in manifest:
        if row['camera']!='vial_fill' or row['video_id']!='5':
            continue
        g=gallery[row['id']]
        if g['status']!='review_candidates' or not g['boxes']:
            excluded.append(row['id'])
            continue
        image=a.teacher/'review-pack/vial_fill/images'/row['source']
        label=a.teacher/'review-pack/vial_fill/labels'/(row['id']+'.txt')
        if hashlib.sha256(image.read_bytes()).hexdigest()!=row['source_sha256']:
            raise ValueError('Teacher image hash mismatch')
        shutil.copyfile(image,images/image.name); shutil.copyfile(label,labels/label.name)
        included.append(dict(id=row['id'],image_sha256=row['source_sha256'],label_sha256=hashlib.sha256(label.read_bytes()).hexdigest()))
    yaml=a.out/'data/data.yaml'
    yaml.write_text(f'path: {(a.out/"data").resolve().as_posix()}\ntrain: images/val\nval: images/val\nnames:\n  0: vial\n  1: colored_liquid\n')
    # The train entry is a schema requirement only: this command NEVER trains.
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(4)
    metrics=YOLO(str(a.weights),task='detect').val(data=str(yaml),imgsz=640,batch=1,device=a.device,workers=0,
                                   project=str(a.out),name='evaluation',plots=True,save_json=False)
    result=dict(schema_version=1,kind='held_out_student_teacher_box_agreement',run_id=5,
                model_weights_sha256=hashlib.sha256(a.weights.read_bytes()).hexdigest(),
                included=included,excluded_unknown=excluded,metrics=metrics.results_dict,
                class_ap50=metrics.box.ap50.tolist(),class_ap50_95=metrics.box.ap.tolist(),
                limits=['Unreviewed Jev boxes are provisional scoring targets, not independent human truth.',
                        'Empty/manual cases excluded: this does not measure absent-vial false positives.',
                        'Run 5 excluded from student fitting/checkpoint selection; teacher previously saw Run 5.',
                        'No liquid mask IoU, color accuracy, or defect metric follows from these box scores.'])
    (a.out/'teacher-agreement.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(count=len(included),excluded_unknown=len(excluded),metrics=metrics.results_dict)),flush=True)


if __name__=='__main__':
    main()
