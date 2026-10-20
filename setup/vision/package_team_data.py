"""Immutable team starter: raw RGB, provisional boxes, source splits and formats.

No fitting, label adjudication, final prompt recipe or approved mask fabrication.
The source directories are read-only; every export goes to a fresh output root.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
from types import SimpleNamespace
import zipfile

CAMERAS = ('vial_fill', 'tank_level', 'cap_seal')
VERSION = 'liquid-team-data-v1-20261020'
ROOT_FOLDER = 'liquid-team-data-v1'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def csv_write(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)


def role(run):
    return 'train' if run in (1, 2, 3) else 'development' if run == 4 else 'diagnostic_only' if run == 5 else 'unassigned'


def parse_yolo(text, width=640, height=480):
    boxes = []
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = [float(x) for x in line.split()]
        if len(fields) != 5 or not all(math.isfinite(x) for x in fields):
            raise ValueError('Invalid YOLO row')
        cls, cx, cy, w, h = fields
        if cls not in (0., 1.) or w <= 0 or h <= 0:
            raise ValueError('Unexpected legacy class or size')
        x0, y0, x1, y1 = (cx-w/2)*width, (cy-h/2)*height, (cx+w/2)*width, (cy+h/2)*height
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError('YOLO target outside full source image')
        boxes.append(dict(legacy_class_id=int(cls), xyxy_pixels=[x0,y0,x1,y1]))
    return boxes


def validate_splits(samples, run5_hashes):
    by_id = {r['source_id']: r for r in samples}
    if len(by_id) != len(samples):
        raise ValueError('Duplicate training sample IDs')
    groups = defaultdict(set)
    for sample in samples:
        if sample['split'] not in ('train', 'val'):
            raise ValueError('Unexpected starter split')
        expected = 'train' if sample['source_run'] in (1,2,3) else 'val' if sample['source_run'] == 4 else None
        if sample['split'] != expected or sample['source_sha256'] in run5_hashes:
            raise ValueError('Run or diagnostic source entered wrong split')
        groups[sample['split']].add(sample['source_sha256'])
        if sample.get('parent_id'):
            parent = by_id.get(sample['parent_id'])
            if (not parent or parent['split'] != sample['split'] or parent['source_run'] != sample['source_run']
                    or parent['source_sha256'] != sample['source_sha256']):
                raise ValueError('Augmentation parent/split differs')
    if groups['train'] & groups['val']:
        raise ValueError('Source image leakage between train and development')


def coco(samples, boxes, classes):
    categories = [dict(id=i+1, name=('vial' if cls==0 else 'colored_liquid'),
                       supercategory='provisional_legacy_box') for i,cls in enumerate(classes)]
    class_map = {cls: i+1 for i,cls in enumerate(classes)}
    images = []; annotations = []
    for i, sample in enumerate(samples, 1):
        source_boxes = boxes[sample['source_id']]
        images.append(dict(id=i, file_name=sample['image'], width=640, height=480,
            source_id=sample['source_id'], source_run=sample['source_run'],
            source_sha256=sample['source_sha256'], image_sha256=sample['image_sha256'],
            parent_id=sample.get('parent_id'), annotation_status='unreviewed_provisional',
            no_liquid_target_is_confirmed_empty=False))
        for item in source_boxes:
            cls = item['legacy_class_id']
            if cls not in class_map:
                continue
            x,y,r,b = item['xyxy_pixels']
            annotations.append(dict(id=len(annotations)+1, image_id=i, category_id=class_map[cls],
                bbox=[x,y,r-x,b-y], area=(r-x)*(b-y), iscrowd=0,
                annotation_status='unreviewed_provisional'))
    return dict(info=dict(description='Provisional legacy box annotations; no segmentation ground truth.',
        dataset_version=VERSION, image_root='starter', coordinates='full-image pixel XYWH',
        legacy_colored_liquid_is_reviewed_pool=False, background_is_provisional=True),
        images=images, annotations=annotations, categories=categories)


def starter_record(sample, parsed, vial_label_sha256):
    positive=any(b['legacy_class_id']==1 for b in parsed)
    return dict(sample, label_status='derived_unreviewed_vial_only',
        annotation_status='unreviewed_provisional',label_sha256=vial_label_sha256,
        actual_label_sha256=vial_label_sha256,
        legacy_labels_path='source_legacy_labels/'+sample['labels'],
        legacy_label_sha256=sample['actual_label_sha256'],
        legacy_donor_label_sha256=sample['label_sha256'],
        legacy_liquid_target_present=positive,
        legacy_liquid_absence_status='not_applicable' if positive else 'provisional_not_confirmed_empty')


class Package:
    def __init__(self, args):
        self.a = args
        if args.out.exists():
            raise ValueError('Use a fresh package output; previous releases are immutable')
        self.root = args.out/'content'/ROOT_FOLDER
        self.components = defaultdict(set)
        self.files = {}

    @classmethod
    def restore(cls, out):
        state=read(out/'build-state.json')
        package=cls.__new__(cls);package.a=SimpleNamespace(out=out)
        package.root=out/'content'/ROOT_FOLDER
        package.components=defaultdict(set,{k:set(v) for k,v in state['components'].items()})
        package.files=state['files']
        return package,state

    def record(self, relative, component, origin=None):
        path = self.root/relative
        self.components[component].add(relative)
        self.files[relative] = dict(path=relative, component=component, bytes=path.stat().st_size,
                                    sha256=sha(path), source_sha256=sha(origin) if origin else None)

    def copy(self, source, relative, component, expected=None):
        target = self.root/relative; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if expected and sha(target) != expected:
            raise ValueError('Copied source differs: '+relative)
        self.record(relative, component, source)

    def json(self, relative, value, component='00-team-guide'):
        write(self.root/relative, value); self.record(relative, component)

    def text(self, relative, value, component='00-team-guide'):
        path = self.root/relative; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding='utf-8'); self.record(relative, component)


def inventory(package):
    a=package.a; teacher=read(a.teacher/'manifest.json')
    source_by_id={r['id']:r for r in teacher}
    if len(source_by_id)!=4647:
        raise ValueError('Unexpected source catalog identity')
    index=[]; proposals={}; exported_counts={}; classes={}; original_indices={}
    for camera in CAMERAS:
        original_indices[camera]=list(csv.DictReader((a.dataset/camera/'image_index.csv').open(encoding='utf-8-sig')))
        base=a.teacher/'review-pack'/camera
        proposals[camera]=read(base/'annotations.json')
        classes[camera]=(base/'classes.txt').read_text().splitlines()
        exported_counts[camera]=sum(len(p.read_text().splitlines()) for p in (base/'labels').glob('*.txt'))
        for row in original_indices[camera]:
            sample_id=Path(row['filename']).stem; source=source_by_id[sample_id]
            if source['camera']!=camera or source['source']!=row['filename']:
                raise ValueError('Teacher/source index mismatch')
            run=int(row['video_id']) if row['video_id'].isdigit() else None
            if str(run or 'unassigned')!=str(source['video_id']):
                raise ValueError('Run identity mismatch')
            relative=f'{camera}/images/{row["filename"]}'
            index.append(dict(sample_id=sample_id,camera=camera,run=run,split_role=role(run),
                capture_timestamp=row['timestamp'],source_relative_path=relative,
                sha256=source['source_sha256'],width=source['source_size'][0],height=source['source_size'][1],
                video_frame_index=None,video_pts_seconds=None,still_video_association_status='unmatched',
                annotation_status='unreviewed_machine_proposals'))
    if len(index)!=4647 or {r['sample_id'] for r in index}!=set(source_by_id):
        raise ValueError('Incomplete indexed source coverage')
    def check(row):
        import cv2
        import numpy as np
        path=a.dataset/row['source_relative_path'];data=path.read_bytes()
        if hashlib.sha256(data).hexdigest()!=row['sha256']:
            raise ValueError('Source image hash changed: '+row['sample_id'])
        image=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
        if image is None or image.shape[:2]!=(row['height'],row['width']):
            raise ValueError('Source image dimensions differ')
        row['bytes']=len(data)
        row['manual_review_required']=any(p['manual_only'] for p in proposals[row['camera']] if p['sample_id']==row['sample_id'])
        return row
    with ThreadPoolExecutor(max_workers=8) as executor:
        index=list(executor.map(check,index))
    unindexed=[]
    for camera in CAMERAS:
        expected={r['filename'] for r in original_indices[camera]}
        for path in sorted((a.dataset/camera/'images').glob('*.jpg')):
            if path.name in expected:
                continue
            alias=path.with_name(path.name.replace('(1)',''))
            unindexed.append(dict(camera=camera,filename=path.name,sha256=sha(path),included=False,
                reason='not_in_source_image_index',same_bytes_as_indexed_alias=alias.name in expected and sha(alias)==sha(path),
                alias_filename=alias.name if alias.name in expected else None))
    return index,proposals,classes,exported_counts,unindexed


def export_raw(package,index,proposals,classes):
    a=package.a;videos=[];by_camera=defaultdict(list)
    for row in index:
        by_camera[row['camera']].append(row)
        if row['run']==5:
            component='05-run5-diagnostic-only';prefix='diagnostic_run5/raw'
        elif row['camera']=='vial_fill':
            component='02-vial-source-runs1-4';prefix='raw'
        else:
            component='04-other-cameras-runs1-4';prefix='context/raw'
        row['package_path']=f'{prefix}/{row["source_relative_path"]}'
        row['component']=component
        package.copy(a.dataset/row['source_relative_path'],row['package_path'],component,row['sha256'])
    import av
    for camera in CAMERAS:
        for run in range(1,6):
            path=a.dataset/camera/'videos'/f'{camera}_video_{run}.mp4'
            if run==5:component='05-run5-diagnostic-only';prefix='diagnostic_run5/raw'
            elif camera=='vial_fill':component='02-vial-source-runs1-4';prefix='raw'
            else:component='04-other-cameras-runs1-4';prefix='context/raw'
            relative=f'{prefix}/{camera}/videos/{path.name}'
            with av.open(str(path)) as container:
                stream=container.streams.video[0]
                info=dict(camera=camera,run=run,split_role=role(run),package_path=relative,
                    sha256=sha(path),bytes=path.stat().st_size,width=stream.width,height=stream.height,
                    codec=stream.codec_context.name,average_rate=str(stream.average_rate),time_base=str(stream.time_base),
                    container_frame_count=stream.frames,full_decode_verified=False,
                    duration_s=float(stream.duration*stream.time_base) if stream.duration is not None else None,
                    cross_camera_physical_synchronization_verified=False)
            if (info['width'],info['height'])!=(640,480):
                raise ValueError('Unexpected video geometry')
            videos.append(info);package.copy(path,relative,component,info['sha256'])
        camera_rows=by_camera[camera]
        for diagnostic in (False,True):
            selected=[r for r in camera_rows if (r['run']==5)==diagnostic]
            if diagnostic:component='05-run5-diagnostic-only';prefix='diagnostic_run5'
            elif camera=='vial_fill':component='02-vial-source-runs1-4';prefix='raw_vial_metadata'
            else:component='04-other-cameras-runs1-4';prefix='context_metadata'
            ids={r['sample_id'] for r in selected}
            annotations=[r for r in proposals[camera] if r['sample_id'] in ids]
            package.json(f'{prefix}/provisional_labels/{camera}/all_proposal_records.json',annotations,component)
            package.text(f'{prefix}/provisional_labels/{camera}/classes.txt','\n'.join(classes[camera])+'\n',component)
            for row in selected:
                label=a.teacher/'review-pack'/camera/'labels'/(row['sample_id']+'.txt')
                if label.exists():
                    package.copy(label,f'{prefix}/provisional_labels/{camera}/yolo/{label.name}',component)
                row['existing_yolo_export_present']=label.exists()
    for path in sorted((a.dataset/'vial_fill/labels').iterdir()):
        if path.is_file():
            package.copy(path,'diagnostic_run5/organizer_reference/'+path.name,'05-run5-diagnostic-only')
    package.copy(a.dataset/'README.md','diagnostic_run5/source_context/original_dataset_readme.txt','05-run5-diagnostic-only')
    package.copy(a.dataset/'videos.csv','metadata/original_videos.csv','00-team-guide')
    for camera in CAMERAS:
        package.copy(a.dataset/camera/'image_index.csv',f'metadata/original_indices/{camera}.csv','00-team-guide')
    return videos


def export_starter(package,index):
    a=package.a;base=a.real;provenance=read(base/'provenance.json');samples=provenance['samples']
    if sha(base/'provenance.json')!='81851cba53c05a1811e81c32bc8328c4f4d5236e4f26bf6510505eae029b260c':
        raise ValueError('Frozen real source inventory differs')
    validate_splits(samples,{r['sha256'] for r in index if r['camera']=='vial_fill' and r['run']==5})
    if Counter(r['split'] for r in samples)!=dict(train=1442,val=157):
        raise ValueError('Starter counts differ')
    by_id={r['sample_id']:r for r in index};boxes={};output=[]
    for sample in samples:
        image=base/sample['image'];label=base/sample['labels']
        if sha(image)!=sample['image_sha256'] or sha(label)!=sample['actual_label_sha256']:
            raise ValueError('Frozen training bytes differ')
        parsed=parse_yolo(label.read_text())
        if sum(b['legacy_class_id']==0 for b in parsed)!=1:
            raise ValueError('Expected one provisional vial in this accepted starter subset')
        boxes[sample['source_id']]=parsed
        package.copy(image,'starter/'+sample['image'],'01-detector-starter',sample['image_sha256'])
        package.copy(label,'starter/source_legacy_labels/'+sample['labels'],'01-detector-starter',sample['actual_label_sha256'])
        vial='\n'.join(line for line in label.read_text().splitlines() if line.split()[0]=='0')+'\n'
        package.text('starter/'+sample['labels'],vial,'01-detector-starter')
        original_id=sample.get('parent_id',sample['source_id'])
        if by_id[original_id]['sha256']!=sample['source_sha256']:
            raise ValueError('Training donor identity differs from original RGB catalog')
        output.append(starter_record(sample,parsed,sha(package.root/'starter'/sample['labels'])))
    package.json('starter/frozen-source-provenance.json',provenance,'01-detector-starter')
    package.json('starter/samples.json',output,'01-detector-starter')
    originals=[r for r in samples if r['split']=='train' and not r.get('parent_id')]
    augmented=[r for r in samples if r['split']=='train'];development=[r for r in samples if r['split']=='val']
    package.text('starter/train-originals.txt',''.join('./'+r['image']+'\n' for r in originals),'01-detector-starter')
    package.text('starter/development.txt',''.join('./'+r['image']+'\n' for r in development),'01-detector-starter')
    for name,train in [('vial-originals','train-originals.txt'),('vial-with-translations','images/train')]:
        package.text(f'starter/{name}.yaml',f'train: {train}\nval: development.txt\nnames:\n  0: vial\n','01-detector-starter')
    for split,rows in [('train-originals',originals),('train-with-translations',augmented),('development',development)]:
        for name,classes in [('vial',[0]),('legacy-liquid',[1]),('legacy-two-class',[0,1])]:
            package.json(f'starter/coco/{name}-{split}.json',coco(rows,boxes,classes),'01-detector-starter')
    stats=dict(original_training_images=len(originals),translated_training_copies=len(augmented)-len(originals),
        historical_training_samples=len(augmented),development_images=len(development),
        original_train_liquid_positive=sum(any(b['legacy_class_id']==1 for b in boxes[r['source_id']]) for r in originals),
        historical_train_liquid_positive=sum(any(b['legacy_class_id']==1 for b in boxes[r['source_id']]) for r in augmented),
        development_liquid_positive=sum(any(b['legacy_class_id']==1 for b in boxes[r['source_id']]) for r in development),
        no_liquid_target_means_confirmed_empty=False,
        training_run_counts=dict(Counter(str(r['source_run']) for r in originals)),
        semantics='Legacy colored_liquid is provisional and may mix pool and stream.')
    return stats,output


def export_sam_frames(package,videos):
    a=package.a;original=read(a.pipeline/'render-v1/manifest.json')
    rows_path=a.pipeline/'render-v1/frames.jsonl'
    if sha(rows_path)!=original['artifacts']['frames.jsonl']:
        raise ValueError('Existing detector trace changed')
    rows=[json.loads(line) for line in rows_path.read_text().splitlines()]
    if [r['source_frame'] for r in rows]!=list(range(0,1800,6)):
        raise ValueError('Expected existing Run 4 diagnostic inventory')
    video=next(r for r in videos if r['camera']=='vial_fill' and r['run']==4)
    if video['sha256']!=original['source_sha256']:
        raise ValueError('Video/detector trace ancestry differs')
    import av
    from PIL import Image
    exported=[];count=0
    with av.open(str(a.dataset/'vial_fill/videos/vial_fill_video_4.mp4')) as container:
        for index,frame in enumerate(container.decode(video=0)):
            if index>=1800:break
            if index%6:continue
            row=rows[count];rgb=frame.to_ndarray(format='rgb24')
            if hashlib.sha256(rgb.tobytes()).hexdigest()!=row['source_rgb_sha256'] or frame.pts!=row['pts'] or str(frame.time_base)!=row['time_base']:
                raise ValueError('Fresh decoded frame differs from existing detections')
            relative=f'sam_starter/run4_frames/frame-{index:06d}.png'
            path=package.root/relative;path.parent.mkdir(parents=True,exist_ok=True)
            Image.fromarray(rgb).save(path)
            package.record(relative,'03-sam-frames-run4-development')
            exported.append(dict(sample_id=f'vial_fill_run4_frame_{index:06d}',camera='vial_fill',run=4,
                split_role='development',image_path=relative,image_sha256=sha(path),width=640,height=480,
                source_rgb_sha256=row['source_rgb_sha256'],source_video_sha256=video['sha256'],
                source_frame_index=index,pts=frame.pts,time_base=str(frame.time_base),pts_seconds=row['seconds'],
                detector_proposals=row['predictions'],detector_models={k:original['models'][k] for k in ('vial','liquid')},
                detector_label_status='machine_prediction_not_ground_truth',human_mask_status='unreviewed',
                first_visible_onset_status='unreviewed',training_eligible=False))
            count+=1
    if count!=300:raise ValueError('Incomplete SAM frame export')
    package.json('sam_starter/frame-manifest.json',dict(schema_version=1,frames=exported,
        source_run=4,ground_truth_masks_included=0,final_prompt_points_included=False,
        final_temporal_policy_included=False,final_sam_masks_included=False,
        note='Students design prompts and temporal handling. Current detector proposals supplied as unreviewed inputs.'),
        '03-sam-frames-run4-development')
    return dict(source_aligned_lossless_png_frames=count,source_run=4,ground_truth_masks_included=0,
                final_solution_traces_excluded=True)


def guides(package,stats,summary):
    package.text('START-HERE.txt',f'''LIQUID VISION TEAM DATA — {VERSION}

Extract selected archives into the SAME directory. Each ZIP contains the same
top-level folder: {ROOT_FOLDER}. Start with 00-team-guide.zip.

TEAM 1 / YOLO: 00 + 01; add 02 for source videos and transition work.
  starter/vial-originals.yaml: 721 original training + 157 development images.
  starter/vial-with-translations.yaml: historical 1,442 training samples.
  labels are provisional vial boxes, class 0. They are not reviewed truth.

TEAM 2 / FASTER R-CNN: 00 + 01 + 02.
  starter/coco/legacy-liquid-train-originals.json + legacy-liquid-development.json.
  COCO image root is the starter/ directory. Category 1 is legacy colored_liquid.
  This class is NOT approved liquid_body and can mix stream and accumulated pool.
  Background examples represent provisional label absence, not confirmed empty.

TEAM 3 / SAM: 00 + 01 + 02; add 03 for development frames and detector inputs.
  03 has 300 lossless Run 4 frames with exact decoded frame index/native PTS.
  It has machine detector proposals but no final-solution prompts, temporal
  policy, SAM masks, or reviewed masks. Design and evaluate your own pipeline.
  Run 4 is development only. Use Runs 1–3 for fitting; do not train on Run 4.

OPTIONAL 04: other cameras (tank level / cap seal), Runs 1–4 plus one unassigned
still. Context only: those camera-local classes are not vial/liquid classes.
05: Run 5 images/videos and organizer recipe/timing references, DIAGNOSTIC ONLY.
Keep 05 separate from training, checkpoint selection, tuning and sampling choices.

There are {summary['indexed_images']} indexed JPEG stills and {summary['videos']} MP4 recordings.
All existing machine labels remain unreviewed. No human-approved masks, onsets,
tracking identities or confirmed-empty decisions are supplied by this package.

Read DATA-FORMAT.txt and the two PDFs before loading labels. Original source
README/indices are preserved as provenance, not approval of clock alignment.
Capture timestamps must NOT be joined to video frames using assumed 30 fps.
All still/video frame associations remain unmatched unless separately reviewed.

No source code, model weights, private workspaces or final algorithm recipe is
included in these data archives. Source files and original annotations were
preserved; exports are new derived files in a versioned snapshot.
''')
    package.text('DATA-FORMAT.txt','''DATA FORMATS AND SEMANTIC BOUNDARIES

RGB SOURCE
  JPEG stills, original 640 x 480 pixels. MP4 recordings, nominal 30 fps.
  metadata/images.csv: unique sample ID, camera, run, capture timestamp, source
  byte hash, package path, split role, missing-label status. Capture time is not
  a verified native video PTS. Container frame counts are metadata, not a full
  decode verification. Individual exported SAM frames have exact verified PTS.

YOLO VIAL BOXES
  One TXT per accepted starter image, parallel images/{train,val} and
  labels/{train,val} directories. Each row is:
    class_id center_x center_y width height
  Coordinates are normalized to the FULL image. Class 0 = vial.
  x_left=(center_x-width/2)*640; y_top=(center_y-height/2)*480.
  No label file is fabricated for the excluded manual-review source images.
  Missing raw source labels mean unresolved, never automatically empty.
  Portable YAMLs omit absolute paths. Image-list entries starting ./ resolve
  relative to their list file. Do not randomly re-split neighboring stills.

COCO DETECTION BOXES
  JSON contains info, images, annotations, categories. Image file_name is relative
  to starter/. bbox is FULL-image PIXEL [x_left,y_top,width,height]. Category IDs
  start at 1; background is not an explicit annotation category.
  legacy-liquid: category 1 colored_liquid.
  vial: category 1 vial. legacy-two-class: category 1 vial, category 2 colored_liquid.
  No segmentation masks are implied by these detection JSONs. Images without a
  legacy liquid box remain explicitly provisional label-absence examples.

LEGACY MACHINE RECORDS
  all_proposal_records.json retains per-class hypotheses, even low-confidence
  and suppressed entries. Presence/confidence/draft/strict are machine metadata,
  not human decisions. Existing YOLO exports contain only the original exported
  subset; raw candidate records are not all positive boxes.
  Class IDs are LOCAL to each camera. The legacy two-class labels do not equal
  the separate human-review contract (vial, liquid_body, stream).

SAM FRAME STARTER
  Lossless RGB PNG, 640 x 480. Source frame indices are zero based. pts*time_base
  gives native MP4 seconds. Every exported frame's source RGB hash/native PTS was
  matched to the existing detector trace. Proposals are continuous full-image
  PIXEL [left,top,right,bottom], with right/bottom as exclusive image bounds.
  There are no approved masks. Review templates/workflow remain with the label
  owners; unreviewed is distinct from absent or uncertain.

SPLITS / ANCESTRY
  Runs 1–3 train; Run 4 development; Run 5 diagnostic only, already inspected.
  starter/samples.json preserves original source hash, run, split and translated
  parent linkage. The 721 translated training copies share their donor's split.
  The 721 originals are 551/76/94 from Runs 1/2/3. Development has 157 Run 4 images.
  Starter labels: train originals 354 positive liquid / 367 provisional negatives;
  historical train 708 / 734; development 89 / 68. These are target-presence counts.
  Use a new untouched recording for any blind final generalization test.

REVIEW AND TIMING
  contracts/active-station-visible-v1.txt defines human box review. It does NOT
  retroactively rename machine labels or approve them. Organizer Run 5 recipe,
  timing and still mappings are preserved separately and have not become pool
  masks, reviewed physical onsets, measured volumes or calibrated quality labels.

FORMAT REFERENCES
  YOLO: https://docs.ultralytics.com/datasets/detect/
  COCO: https://cocodataset.org/#format-data
''')
    package.copy(Path(__file__).with_name('LABEL-CONTRACT.md'),'contracts/active-station-visible-v1.txt','00-team-guide')
    for component in ('02-vial-source-runs1-4','04-other-cameras-runs1-4','05-run5-diagnostic-only'):
        prefix='diagnostic_run5' if component.startswith('05') else 'context_metadata' if component.startswith('04') else 'raw_vial_metadata'
        package.text(prefix+'/READ-ME.txt',
            ('DIAGNOSTIC ONLY — Run 5 must not enter fitting, model selection, tuning or sampling decisions.\n'
             if component.startswith('05') else 'Raw source/context assets; use the shared source-based split.\n')+
            'Existing labels are unreviewed. Missing labels do not confirm an empty image.\n'
            'Read the top-level START-HERE and DATA-FORMAT before use.\n',component)


def archive(package):
    output=package.a.out/'archives';output.mkdir(parents=True)
    reports=[]
    for component,paths in sorted(package.components.items()):
        path=output/(component+'.zip')
        with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as bundle:
            for relative in sorted(paths):
                kind=zipfile.ZIP_STORED if Path(relative).suffix.lower() in ('.jpg','.png','.mp4') else zipfile.ZIP_DEFLATED
                bundle.write(package.root/relative,ROOT_FOLDER+'/'+relative,compress_type=kind)
        with zipfile.ZipFile(path) as bundle:
            if bundle.testzip() is not None:
                raise ValueError('ZIP CRC verification failed')
            for relative in paths:
                info=package.files[relative]
                if hashlib.sha256(bundle.read(ROOT_FOLDER+'/'+relative)).hexdigest()!=info['sha256']:
                    raise ValueError('ZIP bytes differ from manifest')
        reports.append(dict(file=path.name,bytes=path.stat().st_size,sha256=sha(path),files=len(paths),verified=True))
        print(json.dumps(dict(archive=reports[-1])),flush=True)
    write(package.a.out/'archives.json',dict(dataset_version=VERSION,archives=reports))
    (output/'SHA256SUMS.txt').write_text(''.join(r['sha256']+'  '+r['file']+'\n' for r in reports))
    return reports


def archive_core(package):
    """Convenient single download; optional context and Run 5 remain separate."""
    paths=set().union(*(v for k,v in package.components.items() if k[:2] in ('00','01','02','03')))
    path=package.a.out/'archives/Team-Core-Data.zip'
    if path.exists():raise ValueError('Core release already exists; preserve its bytes')
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as bundle:
        for relative in sorted(paths):
            kind=zipfile.ZIP_STORED if Path(relative).suffix.lower() in ('.jpg','.png','.mp4') else zipfile.ZIP_DEFLATED
            bundle.write(package.root/relative,ROOT_FOLDER+'/'+relative,compress_type=kind)
    with zipfile.ZipFile(path) as bundle:
        if bundle.testzip() is not None:raise ValueError('Core ZIP CRC check failed')
        for relative in paths:
            if hashlib.sha256(bundle.read(ROOT_FOLDER+'/'+relative)).hexdigest()!=package.files[relative]['sha256']:
                raise ValueError('Core ZIP file differs')
    return dict(file=path.name,bytes=path.stat().st_size,sha256=sha(path),files=len(paths),verified=True,
                includes_components=['00','01','02','03'],run5_assets_included=False)


def finalize(out):
    package,state=Package.restore(out)
    for name in ('Team-Architecture-Implementation-Brief.pdf','Team-Data-Guide.pdf','team-architecture.png'):
        relative='guides/'+name
        if not (package.root/relative).is_file():
            raise ValueError('Required team-facing document is missing: '+name)
        package.record(relative,'00-team-guide')
    # The file manifest excludes its own bytes to avoid a recursive hash.
    relative='metadata/file-manifest.json'
    package.json(relative,dict(schema_version=1,files=[v for k,v in package.files.items() if k!=relative]))
    for relative,record in package.files.items():
        if sha(package.root/relative)!=record['sha256']:
            raise ValueError('Staged file changed before release: '+relative)
    reports=archive(package)
    reports.append(archive_core(package))
    write(out/'archives.json',dict(dataset_version=VERSION,archives=reports))
    (out/'archives/SHA256SUMS.txt').write_text(''.join(r['sha256']+'  '+r['file']+'\n' for r in reports))
    state.update(completion='complete',components={k:sorted(v) for k,v in package.components.items()},
                 files=package.files,archives=reports,finalizer_sha256=sha(__file__))
    write(out/'build-state.json',state)
    write(out/'completed-package.json',dict(schema_version=1,dataset_version=VERSION,
        completion='complete',archives=reports,source_images=4647,source_videos=15,
        final_algorithm_recipe_included=False,source_code_included=False))


def main(a):
    package=Package(a)
    print('Checking indexed JPEG hashes and dimensions',flush=True)
    index,proposals,classes,counts,unindexed=inventory(package)
    print('Copying indexed RGB/video sources and existing provisional label exports',flush=True)
    videos=export_raw(package,index,proposals,classes)
    print('Deriving detector starter formats from frozen original labels',flush=True)
    stats,samples=export_starter(package,index)
    accepted={r['source_id'] for r in samples if not r.get('parent_id')}
    for row in index:
        row['accepted_original_starter']=row['sample_id'] in accepted
        row['default_provisional_training_use']=row['camera']=='vial_fill' and row['run'] in (1,2,3) and row['sample_id'] in accepted
    print('Exporting source-aligned SAM development frames, without final solution traces',flush=True)
    sam=export_sam_frames(package,videos)
    summary=dict(schema_version=1,dataset_version=VERSION,indexed_images=len(index),videos=len(videos),
        by_camera={camera:dict(images=sum(r['camera']==camera for r in index),videos=5,
            by_run=dict(Counter(str(r['run']) if r['run'] else 'unassigned' for r in index if r['camera']==camera)),
            machine_proposal_records=len(proposals[camera]),existing_exported_boxes=counts[camera]) for camera in CAMERAS},
        vial_images_requiring_manual_review=sum(r['camera']=='vial_fill' and r['manual_review_required'] for r in index),
        machine_proposal_records=sum(map(len,proposals.values())),existing_exported_boxes=sum(counts.values()),
        source_jpeg_and_mp4_bytes=sum(r['bytes'] for r in index+videos),unindexed_files_excluded=unindexed,
        starter=stats,sam=sam,reviewed_pool_masks_included=0,reviewed_onsets_included=0,
        confirmed_empty_decisions_included=0,reviewed_tracking_identities_included=0,
        split_policy=dict(train=[1,2,3],development=[4],diagnostic_only=[5],blind_test_recordings=[]),
        provenance=dict(teacher_manifest_sha256=sha(a.teacher/'manifest.json'),
            frozen_real_provenance_sha256=sha(a.real/'provenance.json'),
            generator_sha256=sha(__file__),created_utc=datetime.now(timezone.utc).isoformat()),
        data_sources_modified=False,labels_adjudicated=False,training_started=False,
        final_algorithm_details_in_team_pdf=False)
    package.json('metadata/inventory.json',summary)
    package.json('metadata/images.json',index)
    package.json('metadata/videos.json',videos)
    package.json('metadata/excluded-unindexed-files.json',unindexed)
    for name,rows in [('images',index),('videos',videos)]:
        fields=list(dict.fromkeys(k for r in rows for k in r))
        csv_write(package.root/f'metadata/{name}.csv',rows,fields);package.record(f'metadata/{name}.csv','00-team-guide')
    guides(package,stats,summary)
    package.json('metadata/file-manifest.json',dict(schema_version=1,files=list(package.files.values())))
    write(a.out/'build-state.json',dict(schema_version=1,root=str(package.root),components={k:sorted(v) for k,v in package.components.items()},
        files=package.files,summary=summary,completion='data_staged_pending_documents_and_final_archive_verification'))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--finalize',action='store_true')
    for name in ('dataset','teacher','real','pipeline'):
        parser.add_argument('--'+name,type=Path)
    args=parser.parse_args()
    if args.finalize:finalize(args.out)
    else:
        if any(getattr(args,name) is None for name in ('dataset','teacher','real','pipeline')):
            parser.error('Preparing a fresh package needs dataset, teacher, real and pipeline roots')
        main(args)
