"""Frozen real-only control and real+synthetic two-class bbox experiment."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import shutil


BASELINE = '89ed11f7046171e5340dc45a9d528010de1cefbba21e163b23b7098151e0880e'
REAL_SOURCE = 'a7756db7360e4cbf142edc38d7584810e290fcb032401add5bd1bea8e35d3b9e'
BASELINE_CHECKPOINT = '10856c18a8a5c485e832adcc6b841b739e726a969dc043fe8db64c49163bb361'
PRETRAINED = '0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path, value): Path(path).write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')


def validate_split(rows, held):
    identities = {r['source_id']: r for r in rows}
    if not rows or len(identities) != len(rows):
        raise ValueError('Empty or duplicate dataset identities')
    sources = {'train': set(), 'val': set()}
    encoded = {'train': set(), 'val': set()}
    for row in rows:
        split = row['split']
        if split not in sources:
            raise ValueError('Unknown split')
        if row['source_sha256'] in held or row['image_sha256'] in held:
            raise ValueError('Run 5 exact source duplicate in fitting/development')
        if row['origin'] == 'real':
            if row['source_run'] not in ([1,2,3] if split == 'train' else [4]):
                raise ValueError('Real recording crosses fitting/development boundary')
        elif row['origin'] == 'synthetic':
            if split != 'train' or row['source_run'] is not None:
                raise ValueError('Synthetic data must not enter real development or claim a real run')
        else:
            raise ValueError('Unknown origin')
        if row.get('parent_id'):
            parent = identities.get(row['parent_id'])
            if not parent or any(parent[k] != row[k] for k in ('origin','split','source_run','source_sha256')):
                raise ValueError('Augmentation crosses source/split boundary')
        sources[split].add(row['source_sha256'])
        encoded[split].add(row['image_sha256'])
    if sources['train'] & sources['val'] or encoded['train'] & encoded['val']:
        raise ValueError('Exact source duplicate across fitting/development')


def yaml_for(data):
    (data/'data.yaml').write_text(f'path: {data.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: vial\n  1: colored_liquid\n', encoding='utf-8')


def synthetic_targets(row, renders):
    import sys
    import numpy as np
    from PIL import Image
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'isaac'))
    from box_annotations import mask_box
    selected = []; seen = set()
    for obj in row['annotations']:
        identity = obj['instance_id']
        if identity not in (101,201): continue
        if identity in seen: raise ValueError('Duplicate synthetic object identity')
        seen.add(identity)
        expected = 'vial' if identity==101 else 'liquid'
        if obj['class_name'] != expected: raise ValueError('Synthetic object class differs from identity contract')
        mask_file = renders/obj['mask_file']
        if sha(mask_file) != obj['mask_sha256']: raise ValueError('Synthetic mask changed')
        with Image.open(mask_file) as decoded: pixels = np.asarray(decoded)
        if pixels.shape!=(480,640) or not set(np.unique(pixels)).issubset({0,255}):
            raise ValueError('Synthetic mask is not native-size binary')
        mask = pixels>0
        if mask_box(mask) != obj['bbox_xyxy']: raise ValueError('Synthetic target is not mask-derived')
        if obj['present']:
            if not obj['bbox_xyxy']: raise ValueError('Expected synthetic target unavailable')
            c = 0 if identity==101 else 1
            x0,y0,x1,y1 = obj['bbox_xyxy']
            selected.append((c,f'{c} {(x0+x1)/1280:.9f} {(y0+y1)/960:.9f} {(x1-x0)/640:.9f} {(y1-y0)/480:.9f}'))
        elif obj['bbox_xyxy'] is not None:
            raise ValueError('Absent synthetic object has a mask')
    if seen!={101,201} or sum(c==0 for c,_ in selected)!=1:
        raise ValueError('Synthetic active vial/liquid record missing')
    return [line for _,line in sorted(selected)]


def prepare_real(source, teacher, fixture, out):
    from crop_contrast import read_labels
    source_provenance = source/'provenance.json'
    if sha(source_provenance) != REAL_SOURCE:
        raise ValueError('Use the immutable published real bootstrap dataset, not changing mentee labels')
    original = read(source_provenance)
    teacher_rows = read(teacher/'manifest.json')
    held = {r['source_sha256'] for r in teacher_rows if r['camera']=='vial_fill' and r['video_id']=='5'}
    if not held:
        raise ValueError('Missing Run 5 exclusion inventory')
    out.mkdir(parents=True, exist_ok=False)
    data = out/'data/real'; rows = []
    for old in original['samples']:
        image, label = source/old['image'], source/old['labels']
        read_labels(label.read_text(encoding='utf-8'))
        if not old.get('parent_id') and (sha(image)!=old['source_sha256'] or sha(label)!=old['label_sha256']):
            raise ValueError('Published real image/label changed')
        for relative in [old['image'], old['labels']]:
            target = data/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source/relative, target)
        rows.append(dict(old, origin='real', image_sha256=sha(image), actual_label_sha256=sha(label)))
    validate_split(rows, held)
    document = dict(schema_version=1, kind='real_only_frozen_training_copy', held_out_source_hashes=sorted(held),
        source_provenance_sha256=sha(source_provenance), source_teacher_manifest_sha256=sha(teacher/'manifest.json'),
        counts=dict(Counter(r['split'] for r in rows)), samples=rows)
    write(data/'provenance.json', document); yaml_for(data)
    render_fixture = read(fixture)
    plan = dict(schema_version=1, kind='frozen_real_plus_synthetic_bbox_pilot', seed=19, epochs=12,
        architecture='yolo11n_detect', pretrained='Same COCO yolo11n checkpoint as current detector',
        fitting_real_runs=[1,2,3], development_real_runs=[4], excluded_from_fit_and_selection=[5],
        expected_synthetic_images=len(render_fixture['cases']), synthetic_fixture_sha256=sha(fixture),
        baseline_onnx_sha256=BASELINE, baseline_checkpoint_sha256=BASELINE_CHECKPOINT, pretrained_sha256=PRETRAINED,
        real_provenance_sha256=sha(data/'provenance.json'),
        checkpoint_selection='Run 4 real box mAP50-95 only',
        preprocessing='Unenhanced native full-frame RGB; ordinary identical YOLO letterboxing in both arms',
        target_scope=dict(vial='Fill-position vial only: supplied synthetic object 101',
            colored_liquid='Accumulated liquid body only: supplied synthetic object 201',
            omitted_synthetic_targets='Neighbor vials 102/103 and stream 301; preserve in raw render export'),
        arms=['real-only','mixed'], video_confidence=.35, nms_iou=.45, ap_confidence=.001,
        control_policy='Repeat existing real training images to match mixed training size and optimizer-step budget; no new real targets',
        render_quality=dict(renderer='PathTracing',samples_per_pixel=32,subframes=1,scope='training pilot; no optical calibration upgrade'),
        synthetic_renderer_sha256=sha(Path(__file__).resolve().parents[1]/'isaac/render_box_batch.py'),
        synthetic_annotation_helper_sha256=sha(Path(__file__).resolve().parents[1]/'isaac/box_annotations.py'),
        authorization='Austin requested scale-up and provisional real+synthetic training on 2026-10-19',
        limits=['Real targets remain unreviewed teacher proposals.',
            'Synthetic targets are geometric proxy visibility, not calibrated refracted RGB or physical fluid truth.',
            'Run 5 was previously inspected and the teacher previously saw it; diagnostic comparison, not a new blind test.',
            'No detector promotion or measurement-QA change.'])
    write(out/'experiment-plan.json', plan)
    shutil.copyfile(fixture, out/'render-fixture.json')
    print(json.dumps(dict(real_counts=document['counts'], synthetic_planned=plan['expected_synthetic_images'])), flush=True)


def prepare_control(root):
    plan=read(root/'experiment-plan.json');base=root/'data/real';p=read(base/'provenance.json')
    if sha(base/'provenance.json')!=plan['real_provenance_sha256']:raise ValueError('Real inventory changed')
    out=root/'data/control';out.mkdir(parents=True,exist_ok=False)
    rows=[dict(row) for row in p['samples']]
    for row in rows:
        for key in ['image','labels']:
            target=out/row[key];target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(base/row[key],target)
    rng=random.Random(plan['seed']);train=[row for row in p['samples'] if row['split']=='train']
    for i in range(plan['expected_synthetic_images']):
        donor=rng.choice(train);name=f'control-repeat-{i:06d}'
        image=f"images/train/{name}{Path(donor['image']).suffix}";label=f'labels/train/{name}.txt'
        shutil.copyfile(base/donor['image'],out/image);shutil.copyfile(base/donor['labels'],out/label)
        rows.append(dict(donor,source_id=name,parent_id=donor['source_id'],image=image,labels=label,
            augmentation=dict(kind='repeat_for_step_matched_control',donor=donor['source_id'])))
    validate_split(rows,set(p['held_out_source_hashes']))
    write(out/'provenance.json',dict(schema_version=1,kind='step_matched_real_only_control',samples=rows,
        counts=dict(Counter(r['split'] for r in rows)),held_out_source_hashes=p['held_out_source_hashes'],
        real_provenance_sha256=sha(base/'provenance.json'),repeat_count=plan['expected_synthetic_images'],seed=plan['seed']))
    yaml_for(out)
    print(json.dumps(dict(control_counts=dict(Counter(r['split'] for r in rows)),repeated_real_images=plan['expected_synthetic_images'])),flush=True)


def prepare_mixed(root, renders, accept_provisional):
    from PIL import Image
    if not accept_provisional:
        raise ValueError('Explicit provisional-render admission required; raw eligibility flags stay false')
    plan = read(root/'experiment-plan.json'); native = read(renders/'run.json'); checked = read(renders/'verification.json')
    if not native['complete'] or not checked['passed'] or native['run5_used'] or not native['camera_parameters_unchanged'] or not native['source_assets_unchanged']:
        raise ValueError('Render export did not pass its integrity/camera/exclusion checks')
    if native['captured'] != plan['expected_synthetic_images'] or native['fixture_sha256'] != plan['synthetic_fixture_sha256']:
        raise ValueError('Render count/fixture differs from frozen plan')
    if native['host']!='spark8' or native['isaac_version']!='6.0.0.1' or not native['gpu'].startswith('NVIDIA GB10'):
        raise ValueError('Expected pinned GB10 render provenance; local pilots cannot replace it')
    if native.get('path_tracing_spp')!=32 or native.get('path_tracing_total_spp')!=32 or native.get('subframes')!=1:
        raise ValueError('Render sampling profile differs from frozen plan')
    if native['script_sha256']!=plan['synthetic_renderer_sha256'] or native['annotation_helper_sha256']!=plan['synthetic_annotation_helper_sha256']:
        raise ValueError('Executed renderer/helper differs from frozen plan')
    real = root/'data/real'; original = read(real/'provenance.json')
    if sha(real/'provenance.json') != plan['real_provenance_sha256']:
        raise ValueError('Real training inventory changed')
    data = root/'data/mixed'; data.mkdir(parents=True, exist_ok=False)
    rows = [dict(row) for row in original['samples']]
    for row in rows:
        for key in ['image','labels']:
            target = data/row[key]; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(real/row[key], target)
    native_rows = [json.loads(line) for line in (renders/'manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    manifest_sha = sha(renders/'manifest.jsonl')
    if len(native_rows) != plan['expected_synthetic_images']:
        raise ValueError('Render manifest count differs')
    for row in native_rows:
        image = renders/row['image_file']
        if sha(image) != row['image_sha256']:
            raise ValueError('Synthetic RGB changed')
        with Image.open(image) as decoded:
            if decoded.size != (640,480): raise ValueError('Synthetic camera shape changed')
        selected = synthetic_targets(row, renders)
        name = 'synthetic-'+row['sample_id']
        image_path = f'images/train/{name}.png'; labels_path = f'labels/train/{name}.txt'
        shutil.copyfile(image, data/image_path)
        (data/labels_path).write_text('\n'.join(selected)+'\n', encoding='utf-8')
        rows.append(dict(source_id=name, source_run=None, origin='synthetic', split='train',
            source_sha256=row['image_sha256'], image_sha256=sha(data/image_path),
            actual_label_sha256=sha(data/labels_path), image=image_path, labels=labels_path,
            native_sample_id=row['sample_id'], native_manifest_sha256=manifest_sha,
            label_status='provisional_scene_geometry; explicit user-authorized experiment; not certified optical truth'))
    validate_split(rows, set(original['held_out_source_hashes']))
    document = dict(schema_version=1, kind='real_plus_synthetic_training_copy', samples=rows,
        counts=dict(Counter((r['origin']+'_'+r['split']) for r in rows)), held_out_source_hashes=original['held_out_source_hashes'],
        real_provenance_sha256=sha(real/'provenance.json'), render_run_sha256=sha(renders/'run.json'),
        render_manifest_sha256=sha(renders/'manifest.jsonl'), render_verification_sha256=sha(renders/'verification.json'),
        provisional_admission_authorization=plan['authorization'], raw_render_eligibility_flags_modified=False,
        target_scope=plan['target_scope'])
    write(data/'provenance.json', document); yaml_for(data)
    print(json.dumps(document['counts']), flush=True)


def train(root, arm, pretrained):
    import torch
    import ultralytics
    from ultralytics import YOLO
    plan = read(root/'experiment-plan.json'); data = root/'data'/('control' if arm=='real-only' else 'mixed')
    provenance = read(data/'provenance.json')
    validate_split(provenance['samples'], set(provenance['held_out_source_hashes']))
    for row in provenance['samples']:
        if sha(data/row['image'])!=row['image_sha256'] or sha(data/row['labels'])!=row['actual_label_sha256']:
            raise ValueError('Prepared training source changed')
    if provenance['real_provenance_sha256']!=plan['real_provenance_sha256']:
        raise ValueError('Real ancestry differs from frozen plan')
    if sha(root/'data/real/provenance.json')!=plan['real_provenance_sha256']:
        raise ValueError('Frozen real inventory changed')
    base=read(root/'data/real/provenance.json');real_count=base['counts']['train']
    if arm=='real-only' and any(r['origin']!='real' for r in provenance['samples']):
        raise ValueError('Synthetic data entered the real-only control')
    if arm=='mixed' and sum(r['origin']=='synthetic' for r in provenance['samples'])!=plan['expected_synthetic_images']:
        raise ValueError('Mixed synthetic count differs from frozen plan')
    expected_val={r['source_id']:(r['image_sha256'],r['actual_label_sha256']) for r in base['samples'] if r['split']=='val'}
    actual_val={r['source_id']:(r['image_sha256'],r['actual_label_sha256']) for r in provenance['samples'] if r['split']=='val'}
    if actual_val!=expected_val:raise ValueError('Development sources/targets changed')
    if sum(row['split']=='train' for row in provenance['samples'])!=real_count+plan['expected_synthetic_images']:
        raise ValueError('Training-arm optimizer-step budgets differ')
    if sha(pretrained)!=plan['pretrained_sha256']:raise ValueError('Pretrained initialization differs from baseline')
    yaml_for(data)
    output = root/'models'/arm; output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    model = YOLO(str(pretrained))
    model.train(data=str(data/'data.yaml'), epochs=plan['epochs'], imgsz=640, batch=16, device=0, workers=0,
        seed=plan['seed'], deterministic=True, project=str(output), name='train', exist_ok=False, patience=plan['epochs'],
        pretrained=True, plots=True, amp=False, hsv_h=0, hsv_s=0, hsv_v=.1, degrees=2, translate=.04,
        scale=.1, fliplr=0, flipud=0, mosaic=0, mixup=0, copy_paste=0)
    best_path = output/'train/weights/best.pt'; best = YOLO(str(best_path))
    exported = Path(best.export(format='onnx', imgsz=640, opset=17, simplify=False, dynamic=False, nms=False))
    shutil.copyfile(exported, output/'detector.onnx'); shutil.copyfile(best_path, output/'best.pt')
    write(output/'model.json', dict(schema_version=1, arm=arm, architecture='yolo11n_detect',
        classes=['vial','colored_liquid'], model_sha256=sha(output/'detector.onnx'), checkpoint_sha256=sha(best_path),
        pretrained_sha256=sha(pretrained), experiment_plan_sha256=sha(root/'experiment-plan.json'),
        data_provenance_sha256=sha(data/'provenance.json'), trainer_sha256=sha(__file__),
        train_runs=[1,2,3], development_runs=[4], excluded_runs=[5], epochs=plan['epochs'], seed=plan['seed'],
        torch=torch.__version__, ultralytics=ultralytics.__version__, preprocessing=plan['preprocessing'],
        target_scope=plan['target_scope'], counts=provenance['counts'], automatic_promotion=False))
    print(json.dumps(read(output/'model.json')), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare-real','prepare-control','prepare-mixed','train'])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source', type=Path); parser.add_argument('--teacher', type=Path)
    parser.add_argument('--fixture', type=Path); parser.add_argument('--renders', type=Path)
    parser.add_argument('--accept-provisional-render-proxies', action='store_true')
    parser.add_argument('--arm', choices=['real-only','mixed']); parser.add_argument('--pretrained', type=Path)
    args = parser.parse_args()
    if args.command=='prepare-real': prepare_real(args.source,args.teacher,args.fixture,args.out)
    elif args.command=='prepare-control': prepare_control(args.out)
    elif args.command=='prepare-mixed': prepare_mixed(args.out,args.renders,args.accept_provisional_render_proxies)
    else: train(args.out,args.arm,args.pretrained)
