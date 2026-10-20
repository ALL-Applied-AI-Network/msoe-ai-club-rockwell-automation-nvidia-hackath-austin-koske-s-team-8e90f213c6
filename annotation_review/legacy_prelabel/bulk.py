"""All indexed stills, three camera-specific unreviewed YOLO prelabel packs.

Source data and human/reference labels are read-only. Successful exact prompt +
input matches are reused, and every source filename remains represented.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import getpass
import io
import json
import os
from pathlib import Path
import shutil
import statistics
import time
import zipfile

from PIL import Image, ImageDraw

from expand import frozen, infer_one, save
from run import to_yolo, valid_box
from tune import canonical, definition, digest, mapped, request

CAMERAS = ('vial_fill', 'tank_level', 'cap_seal')
TANK_ROI = [155, 280, 600, 480]
PRIOR_ROOTS = ('typesafe-box-expanded-20261017', 'typesafe-box-tuning-20261016')


def camera_profile(camera):
    if camera == 'vial_fill':
        return None
    questions, targets = {}, []
    if camera == 'tank_level':
        state = dict(scene='Four translucent reagent tanks, ordered from left to right. Identify positions visually; do not infer physical tank IDs, volumes, recipe or valve state.')
        for i, position in enumerate(('leftmost', 'second from left', 'third from left', 'rightmost')):
            body = f'The {position} of the four large reagent tank vessels in `photo`'
            targets.extend([
                (f'tank_{i+1}_body', f'{body}. Bound its vessel body from the top rim to the bottom of the tapered vessel. Exclude fittings above it, support legs, tubes and the white vertical gauge in front.',
                 f'Is the vessel body of the {position} reagent tank visibly identifiable in `photo`?'),
                (f'tank_{i+1}_visible_liquid', f'The visibly identifiable liquid region inside the {position} reagent tank in `photo`, from its horizontal liquid surface down to the vessel bottom. Use the region seen through the tank wall; exclude empty headspace, supports, gauge, hoses and background.',
                 f'Can the liquid region inside the {position} reagent tank in `photo` be visually distinguished from its empty headspace by a surface or coloration? Do not infer liquid merely because a vessel is present.')])
        crop = TANK_ROI
    else:
        state = dict(scene='Overhead cap/seal camera: yellow robot arm, blue/dark gripper at its tip, two perforated collection racks, and a narrow conveyor to the left. Only annotate the four named targets; other vials and caps are outside this partial label scope. Do not infer sorting decisions, sealing success or recipe.')
        targets = [
            ('good_collection_rack', 'The larger upper-left perforated metal vial collection rack in `photo`, above the pink Good note. Bound the perforated metal plate only, including all of that plate even if partly occluded; exclude its black base, clamps and loose vials.',
             'Is the larger upper-left perforated collection rack above the pink Good note visible in `photo`?'),
            ('scrap_collection_rack', 'The smaller lower-right perforated metal vial collection rack in `photo`, above the green Scrap note. Bound the perforated metal plate only, including all of that plate even if partly occluded; exclude its black base, clamps and loose vials.',
             'Is the smaller lower-right perforated collection rack above the green Scrap note visible in `photo`?'),
            ('robot_gripper', 'The blue or dark mechanical gripper/end-effector attached at the working tip of the yellow robot arm in `photo`. Bound the gripper and fingers only; exclude the large yellow arm, black cables, and any grasped vial.',
             'Is the robot gripper at the working tip of the yellow arm actually visible in `photo`? A visible yellow arm alone does not count.'),
            ('visibly_held_vial', 'The single distinct vial visibly grasped by the robot gripper in `photo`, if any. Bound only the visible vial body and its cap; exclude gripper fingers, yellow robot arm, conveyor fixtures and loose vials in the racks.',
             'Is a distinct vial visibly being grasped by the robot gripper in `photo`? Answer no when the gripper is empty, when no vial can actually be seen, or when a vial is only resting in a rack or on the conveyor.')]
        crop = None
    for name, box_text, present_text in targets:
        questions[name+'_present'] = dict(type='noul', instructions=present_text)
        questions[name+'_box'] = dict(type='box', instructions=box_text)
    return dict(name=camera+'_v1', model='jev-1.14.0', crop_xyxy=crop, state=state, questions=questions,
                targets=[t[0] for t in targets])


def specs():
    return dict(**frozen(), **{camera_profile(c)['name']:camera_profile(c) for c in CAMERAS if c!='vial_fill'})


def fingerprint(spec, sha):
    return digest(canonical(dict(spec=spec, input=sha, references={})))


def class_names(camera):
    return ['fill_position_vial','colored_liquid'] if camera=='vial_fill' else camera_profile(camera)['targets']


def input_row(row, out, spec):
    profile = spec['name']
    dest = out / profile / 'images' / (row['id']+'.jpg')
    dest.parent.mkdir(parents=True, exist_ok=True)
    (out / profile / 'responses').mkdir(exist_ok=True)
    if not dest.exists():
        if spec['crop_xyxy']:
            with Image.open(row['file']) as image:
                image = image.convert('RGB').crop(spec['crop_xyxy'])
                size = (280,800) if profile=='compact_crop' else (1335,600)
                image.resize(size, Image.Resampling.LANCZOS).save(dest, quality=95)
        else:
            shutil.copyfile(row['file'], dest)
    return dict(row, input_file=str(dest), input_sha256=digest(dest.read_bytes()), crop_xyxy=spec['crop_xyxy'])


def prepare(args):
    definitions = specs()
    existing = args.out / 'manifest.json'
    if existing.exists():
        if json.loads((args.out/'prompt-definitions.json').read_text()) != definitions:
            raise ValueError('Profile changed; preserve results and use a new output directory')
        return
    # Reuse only exact, successful responses to the frozen prompts, never older variants.
    cache = {}
    for root_name in PRIOR_ROOTS:
        root = args.out.parent / root_name
        for path in root.rglob('responses/*.json'):
            result = json.loads(path.read_text())
            name = result.get('profile')
            if (result.get('ok') and name in definitions and result.get('definition')==definitions[name]
                    and result['response'].get('model')=='jev-1.14.0'):
                cache.setdefault(result['fingerprint'], path)
    rows, reused = [], []
    for camera in CAMERAS:
        source = args.data / camera
        with (source/'image_index.csv').open(newline='') as handle:
            index = sorted(csv.DictReader(handle), key=lambda r:r['filename'])
        if len({r['filename'] for r in index})!=len(index):
            raise ValueError('Duplicate filenames in source index')
        actual = {p.name for p in (source/'images').iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png')}
        if actual != {r['filename'] for r in index}:
            raise ValueError('Source index does not cover every still image')
        for meta in index:
            path = source/'images'/meta['filename']
            with Image.open(path) as image:
                if image.size != (640,480):
                    raise ValueError('A new camera resolution requires recalibration')
            row = dict(id=path.stem, file=str(path.resolve()), source=meta['filename'], camera=camera,
                video_id=meta['video_id'], timestamp=meta['timestamp'], source_size=[640,480],
                source_sha256=digest(path.read_bytes()), split='all_existing_stills',
                validation_stage='previous_50_frame_spot_check' if camera=='vial_fill' else 'new_camera_smoke_check_only')
            profiles = ('compact_crop','compact_full') if camera=='vial_fill' else (camera_profile(camera)['name'],)
            for profile in profiles:
                current = input_row(row,args.out,definitions[profile])
                fp = fingerprint(definitions[profile],current['input_sha256'])
                if fp in cache:
                    result = json.loads(cache[fp].read_text())
                    result.update(id=row['id'],split=row['split'], reused_from=str(cache[fp]))
                    save(args.out/profile/'responses'/(row['id']+'.json'),result)
                    reused.append(dict(sample_id=row['id'],profile=profile,source=str(cache[fp])))
            rows.append(row)
    save(existing, rows)
    save(args.out/'prompt-definitions.json',definitions)
    save(args.out/'cache-reuse.json',reused)
    save(args.out/'protocol.json',dict(prepared_utc=datetime.now(timezone.utc).isoformat(),
        scope='All existing indexed JPEG/PNG stills from the three camera folders. No video frame extraction. All five runs included; cap/seal unassigned still retained.',
        source_root=str(args.data.resolve()),images=len(rows),by_camera=dict(Counter(r['camera'] for r in rows)),
        profile_sha256={k:digest(canonical(v)) for k,v in definitions.items()},
        classes={c:class_names(c) for c in CAMERAS},
        policy='Draft presence>=0.5; strict presence>=0.8 and box confidence>=0.5. All candidates unreviewed. Vial crop boundary cases excluded from YOLO candidates and sent to manual drawing.',
        caveats=['Camera-specific class IDs; do not merge class numbers across folders.',
            'Tank numbers mean left-to-right image position, not confirmed physical T100/T200/T300/T400 mapping.',
            'Partial task labels, not every object, cap or vial in the scene. Clear vial liquid is outside the colored_liquid class.',
            'Recipe, dispensing state, volume, cap seal success and sorting correctness are not inferred.',
            'New-camera profiles have only smoke checks, not accuracy validation. Same camera/recordings are correlated.',
            'Every source image requires human review, including missing and suppressed targets. No model confidence is approval.']))
    print(json.dumps(dict(prepared=len(rows),by_camera=dict(Counter(r['camera'] for r in rows)),cached_responses=len(reused))),flush=True)


def infer(row,out,key):
    if row['camera']=='vial_fill':
        sample_id, records, warning = infer_one(out,row,key)
        return dict(id=sample_id, records=records, warning=warning)
    spec = camera_profile(row['camera'])
    current = input_row(row,out,spec)
    result = request(current,spec['name'],out/spec['name'],{},key,spec=spec)
    if not result.get('ok'):
        return dict(id=row['id'],records=[],warning='primary_request_failed')
    answers, records = result['response']['answers'], []
    for class_id,target in enumerate(spec['targets']):
        p = answers[target+'_present']['noul']; a = answers[target+'_box']
        b = a['box']; c = a['confidence']
        cut = bool(spec['crop_xyxy'] and p>=.5 and (b['left']<=.01 or b['right']>=.99 or b['top']<=.01 or b['bottom']>=.99))
        records.append(dict(sample_id=row['id'],class_id=class_id,target=target,box=mapped(b,current),
            presence=p,confidence=c,source=spec['name'],strict=p>=.8 and c>=.5,draft=p>=.5,
            manual_only=cut,review_priority='high' if cut or p<.8 or c<.5 else 'normal',
            annotation_status='unreviewed_model_candidate'))
    return dict(id=row['id'],records=records,warning=None)


def smoke_rows(rows):
    selected=[]
    for camera in ('tank_level','cap_seal'):
        for video in ('1','2','3','4','5'):
            group=[r for r in rows if r['camera']==camera and r['video_id']==video]
            for fraction in (.2,.7):
                selected.append(group[int((len(group)-1)*fraction)])
    return selected


def run(args, pilot=False):
    frozen()
    if json.loads((args.out/'prompt-definitions.json').read_text()) != specs():
        raise ValueError('Profiles changed; preserve cached results in this output')
    rows=json.loads((args.out/'manifest.json').read_text())
    if pilot:
        rows=smoke_rows(rows)
    elif not (args.out/'smoke-review.json').exists():
        raise ValueError('Inspect new camera smoke results and record smoke-review.json before the full run')
    # Validate source snapshots even when resuming cached results.
    for row in rows:
        if digest(Path(row['file']).read_bytes())!=row['source_sha256']:
            raise ValueError('A source image changed after preparation')
    key=os.environ.get('TYPESAFE_API_KEY') or getpass.getpass('TypeSafe API key (hidden, memory only): ')
    path=args.out/('pilot-checkpoint.json' if pilot else 'checkpoint.json')
    completed=json.loads(path.read_text()) if path.exists() else {}
    pending=[r for r in rows if not completed.get(r['id'],{}).get('records') or completed[r['id']].get('warning')]
    previous_errors={r['id']:completed[r['id']] for r in pending if r['id'] in completed}
    if previous_errors:
        history=args.out/'retry-history'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        history.mkdir(parents=True)
        save(history/'prior-errors.json',previous_errors)
        for sample_id in previous_errors:
            for profile in specs():
                response=args.out/profile/'responses'/(sample_id+'.json')
                if response.exists() and not json.loads(response.read_text()).get('ok'):
                    shutil.copyfile(response,history/(profile+'__'+sample_id+'.json'))
    started=time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures={pool.submit(infer,row,args.out,key):row for row in pending}
            for n,future in enumerate(as_completed(futures),1):
                row=futures[future]
                try:
                    result=future.result()
                except Exception as exc:
                    result=dict(id=row['id'],records=[],warning=str(exc).replace(key,'[redacted]'))
                completed[row['id']]=result
                if n%20==0 or n==len(pending):
                    save(path,completed)
                    print(f'CHECKPOINT {n}/{len(pending)} pending workflows; {round(time.perf_counter()-started,1)}s',flush=True)
        # Recover transient connection/overload failures without redoing successes.
        for retry_round in range(2):
            failed=[row for row in rows if not completed.get(row['id'],{}).get('records') or completed[row['id']].get('warning')]
            if not failed:
                break
            print(f'RETRY ROUND {retry_round+1}: {len(failed)} image workflows',flush=True)
            time.sleep(1)
            for row in failed:
                try:
                    completed[row['id']]=infer(row,args.out,key)
                except Exception as exc:
                    completed[row['id']]=dict(id=row['id'],records=[],warning=str(exc).replace(key,'[redacted]'))
            save(path,completed)
    finally:
        del key
        save(path,completed)
    if pilot:
        pilot_report(args.out,rows,completed)
    else:
        export(args.out,rows,completed)


def overlay(row,records,dest):
    with Image.open(row['file']) as source:
        image=source.convert('RGB')
    d=ImageDraw.Draw(image)
    colors=('#00e4ff','#ffbd40','#e58dff','#61f58c','#ff7777','#faf58a','#87bbff','#ffffff')
    for r in records:
        if not r['draft']:
            continue
        b=r['box'];xy=[b[k]*(640 if k in ('left','right') else 480) for k in ('left','top','right','bottom')]
        color=colors[r['class_id']%len(colors)]
        d.rectangle(xy,outline=color,width=2)
        text=f"{r['class_id']} {r['presence']:.2f}/{r['confidence']:.2f}"
        y=max(0,xy[1]-13);x=max(0,min(xy[0],630-len(text)*6))
        d.rectangle((x,y,min(639,x+len(text)*6),y+13),fill='black');d.text((x,y),text,fill=color)
    dest.parent.mkdir(parents=True,exist_ok=True)
    image.save(dest,quality=88)


def pilot_report(out,rows,completed):
    for camera in ('tank_level','cap_seal'):
        group=[r for r in rows if r['camera']==camera]
        sheet=Image.new('RGB',(1280,2550),'#eeeeee');draw=ImageDraw.Draw(sheet)
        for n,row in enumerate(group):
            dest=out/'pilot-overlays'/(row['id']+'.jpg')
            overlay(row,completed[row['id']]['records'],dest)
            x,y=(n%2)*640,(n//2)*510
            sheet.paste(Image.open(dest),(x,y+26));draw.text((x+4,y+5),'run '+row['video_id']+' '+row['source'],fill='black')
        sheet.save(out/(camera+'-smoke.jpg'),quality=92)
    print(json.dumps(dict(smoke_images=len(rows),successful=sum(bool(v['records']) for v in completed.values()),failures={k:v['warning'] for k,v in completed.items() if v['warning']})),flush=True)


def export(out,rows,completed):
    by_camera={};gallery=[];all_queue=[]
    for camera in CAMERAS:
        pack=out/'review-pack'/camera
        for part in ('images','labels','manual_review_images','api_error_images'):
            (pack/part).mkdir(parents=True,exist_ok=True)
        names=class_names(camera)
        (pack/'classes.txt').write_text('\n'.join(names)+'\n')
        queue=[];annotations=[]
        for row in (r for r in rows if r['camera']==camera):
            result=completed.get(row['id'],dict(records=[],warning='not_processed'))
            group=result['records'];annotations.extend(group)
            # Service failures are never silently converted into negative labels.
            status='api_error' if not group else ('manual_full_image' if any(r['manual_only'] for r in group) else 'review_candidates')
            folder={'api_error':'api_error_images','manual_full_image':'manual_review_images','review_candidates':'images'}[status]
            # Remove only this generated image/label when a retry changes its route.
            for other in ('images','manual_review_images','api_error_images'):
                stale=pack/other/row['source']
                if other!=folder and stale.exists():
                    stale.unlink()
            if status!='review_candidates':
                (pack/'labels'/(row['id']+'.txt')).unlink(missing_ok=True)
            shutil.copyfile(row['file'],pack/folder/row['source'])
            lines=[]
            if status=='review_candidates':
                for r in group:
                    if r['draft']:
                        lines.append(str(r['class_id'])+' '+' '.join(f'{v:.8f}' for v in to_yolo(r['box'])))
                (pack/'labels'/(row['id']+'.txt')).write_text('\n'.join(lines)+ ('\n' if lines else ''))
            expected=({'good_collection_rack','scrap_collection_rack'} if camera=='cap_seal' else
                      {f'tank_{i}_body' for i in range(1,5)} if camera=='tank_level' else set())
            missing=sorted(expected-{r['target'] for r in group if r['draft']})
            high=status!='review_candidates' or bool(result.get('warning')) or row['video_id']=='unassigned' or bool(missing) or any(r['draft'] and r['review_priority']=='high' for r in group)
            q=dict(sample_id=row['id'],filename=row['source'],camera=camera,video_id=row['video_id'],timestamp=row['timestamp'],
                status=status,candidate_boxes=len(lines),high_priority=high,warning=result.get('warning') or '',review_reason='Missing expected static target: '+', '.join(missing) if missing else '',annotation_status='unreviewed',validation_stage=row['validation_stage'])
            queue.append(q);all_queue.append(q)
            dest=out/'overlays'/camera/(row['id']+'.jpg')
            overlay(row,group,dest)
            gallery.append(dict(id=row['id'],camera=camera,run=row['video_id'],timestamp=row['timestamp'],status=status,high=high,
                original=f'review-pack/{camera}/{folder}/{row["source"]}',overlay=f'overlays/{camera}/{row["id"]}.jpg',
                boxes=len(lines),review_reason=q['review_reason'], predictions=[dict(target=r['target'],presence=r['presence'],confidence=r['confidence'],draft=r['draft'],source=r['source']) for r in group]))
        save(pack/'annotations.json',annotations)
        write_csv(pack/'review-queue.csv',queue)
        (pack/'README.txt').write_text('UNREVIEWED CAMERA-SPECIFIC PRELABELS\n'
            'classes.txt defines the LOCAL class IDs for this camera. Do not merge numeric IDs across camera folders.\n'
            'images/ and labels/ have matching original filenames. YOLO coordinates refer to the full 640x480 frame.\n'
            'Every frame requires review for wrong/missing targets and inaccurate edges, even empty label files.\n'
            'manual_review_images/ requires full-frame drawing; no draft label file is exported for those images.\n'
            'api_error_images/ contains service failures, never verified negatives. Check warning fields in the queue.\n'
            'These are PARTIAL task-specific labels. Other objects are intentionally outside the class scope.\n'
            'No recipe, volume, dispensing state, capping success or sorting correctness is inferred.\n'
            'Save human corrections outside this generated pack. Report rebuilding replaces unreviewed exports.\n'
            'Split train/test by recording with care; do not treat correlated frames as independent evaluation.\n')
        eligible_ids={q['sample_id'] for q in queue if q['status']=='review_candidates'}
        by_camera[camera]=dict(images=len(queue),successful_images=sum(q['status']!='api_error' for q in queue),
            exportable_images=sum(q['status']=='review_candidates' for q in queue),manual_images=sum(q['status']=='manual_full_image' for q in queue),
            api_error_images=sum(q['status']=='api_error' for q in queue),candidate_boxes=sum(q['candidate_boxes'] for q in queue),high_priority=sum(q['high_priority'] for q in queue),
            candidate_classes=dict(Counter(names[r['class_id']] for r in annotations if r['draft'] and r['sample_id'] in eligible_ids)))
    # Preserve organizer metadata as reference material separate from predicted boxes.
    ref=out/'review-pack/vial_fill/reference_labels';ref.mkdir(exist_ok=True)
    source_root=Path(json.loads((out/'protocol.json').read_text())['source_root'])
    for path in (source_root/'vial_fill/labels').iterdir():
        if path.is_file():shutil.copyfile(path,ref/path.name)
    calls=[]
    for profile in specs():
        calls.extend(json.loads(p.read_text()) for p in (out/profile/'responses').glob('*.json'))
    fresh=[c for c in calls if not c.get('reused_from')];good=[c for c in fresh if c.get('ok')]
    tokens=sum(c['response']['usage']['input_tokens'] for c in good)
    summary=dict(images=len(rows),by_camera=by_camera,successful_images=sum(v['successful_images'] for v in by_camera.values()),
        candidate_boxes=sum(v['candidate_boxes'] for v in by_camera.values()),manual_images=sum(v['manual_images'] for v in by_camera.values()),
        api_errors=sum(v['api_error_images'] for v in by_camera.values()),new_request_records=len(fresh),new_successful_requests=len(good),
        reused_responses=sum(bool(c.get('reused_from')) for c in calls),new_input_tokens=tokens,estimated_new_usd=tokens*.042/1000000,
        median_new_request_seconds=statistics.median(c['seconds'] for c in good) if good else None,
        note='Candidate counts are inventory, not accuracy. New-camera profiles have only visual smoke checks. All labels require human review.')
    save(out/'summary.json',summary);save(out/'gallery.json',gallery)
    write_csv(out/'review-queue.csv',all_queue)
    save(out/'api-errors.json',{k:v['warning'] for k,v in completed.items() if v.get('warning')})
    (out/'review-pack/README.txt').write_text(
        'ALL THREE CAMERA VIEWS: UNREVIEWED BOUNDING-BOX CANDIDATES\n'
        'Use one camera folder at a time. Each has its own classes.txt; numeric class IDs do not have the same meaning across cameras.\n'
        'Original filenames are retained. Import that camera folder images/ and labels/ into your annotation tool.\n'
        'Review every image: correct edges, remove wrong candidates, and add missing targets within the documented class scope.\n'
        'Images in manual_review_images/ require drawing on the full image. API failures are kept separately.\n'
        'An empty label file is an unreviewed prediction, not a verified absence label.\n'
        'review-queue.csv lists high-priority cases and missing expected static targets.\n'
        'Video-5 organizer recipe/time metadata is copied under vial_fill/reference_labels/ and is not predicted box ground truth.\n'
        'Tank indices 1-4 mean image position left to right. Rack names describe locations, not sorting correctness.\n'
        'These are partial task labels, not every object in the scene. Save human corrections outside the generated export.\n')
    for name in ('summary.json','manifest.json','protocol.json','prompt-definitions.json','review-queue.csv','api-errors.json','smoke-review.json','cache-reuse.json','duplicate-source-check.json','duplicate-aliases.json'):
        if (out/name).exists():shutil.copyfile(out/name,out/'review-pack'/name)
    with zipfile.ZipFile(out/'review-pack.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted((out/'review-pack').rglob('*')):
            if path.is_file():archive.write(path,path.relative_to(out/'review-pack'))
    for camera in CAMERAS:
        with zipfile.ZipFile(out/(camera+'-review-pack.zip'),'w',zipfile.ZIP_DEFLATED) as archive:
            for path in sorted((out/'review-pack'/camera).rglob('*')):
                if path.is_file():archive.write(path,path.relative_to(out/'review-pack'/camera))
            for name in ('summary.json','protocol.json','prompt-definitions.json'):
                archive.write(out/name,name)
    labels_only(out)
    build_page(out)
    print(json.dumps(summary,indent=2),flush=True)


def labels_only(out):
    with zipfile.ZipFile(out/'labels-only.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted((out/'review-pack').rglob('*')):
            if path.is_file() and path.suffix.lower() in ('.json','.csv','.txt') and path!=out/'review-pack/README.txt':
                archive.write(path,path.relative_to(out/'review-pack'))
        archive.writestr('README.txt',
            'LABELS ONLY: IMAGES ARE ALREADY IN THE TEAM DATASET\n'
            'Pair each camera/labels/<original-stem>.txt with its matching source JPEG in that camera images/ folder.\n'
            'Class IDs are local to each camera; use that camera classes.txt. These are unreviewed partial prelabels.\n'
            'review-queue.csv records all images, including manual_full_image cases with no label file.\n'
            'Do not turn missing/manual label files into verified negatives. Correct wrong boxes and draw missing targets.\n'
            'Save reviewed corrections separately. Existing organizer recipe/time labels remain reference metadata.\n'
            'Use the full review pack if you also need image copies; this smaller archive contains labels and provenance only.\n')


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def build_page(out):
    page='''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>All cameras · Jev prelabels</title><style>
body{margin:clamp(12px,3vw,30px);font:16px system-ui;background:#101720;color:#e6eef8}a{color:#77d4ff}h1{font-size:clamp(24px,4vw,36px)}p{line-height:1.5;max-width:1100px}.note{color:#aebed0}button,select{max-width:100%;padding:9px;margin:4px;border:1px solid #64748b;background:#243347;color:white;border-radius:6px}#summary{background:#1f3043;padding:16px;border-radius:10px;line-height:1.6}#table{overflow:auto}table{border-collapse:collapse}td,th{padding:10px;border-bottom:1px solid #43556b;white-space:nowrap;text-align:left}#grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,430px),1fr));gap:16px;margin-top:16px}article{min-width:0;background:#1b2939;border:2px solid #34465a;padding:12px;border-radius:8px;overflow-wrap:anywhere}article.high{border-color:#d8a950}img{width:100%;cursor:pointer}pre{white-space:pre-wrap;font-size:12px}.stats{font-size:14px;color:#bdcddd}</style></head><body>
<h1>All three cameras · unreviewed prelabels</h1><div id="summary"></div>
<p><a href="labels-only.zip">Download labels only</a> · <a href="review-pack.zip">All cameras with images</a> · <a href="vial_fill-review-pack.zip">Vial fill</a> · <a href="tank_level-review-pack.zip">Tank level</a> · <a href="cap_seal-review-pack.zip">Cap/seal</a> · <a href="review-queue.csv">Full review queue</a> · <a href="protocol.json">Classes and scope</a> · <a href="smoke-review.json">New-camera smoke review</a></p>
<div id="table"></div><p class="note">All existing stills are included, including video 5 and the unassigned cap/seal still. Organizer recipe/time labels are preserved separately. Class numbers are LOCAL to each camera pack. Box captions show class ID then presence/confidence; click an image to toggle its original. Tank numbers mean image position left to right, not verified physical tank IDs. Partial task labels: other vials/caps/objects are outside scope. No volume, recipe, sorting correctness or capping success is inferred.</p>
<p class="note">Every image requires human review for missing targets and inaccurate edges. No confidence value approves a label. New tank and cap/seal prompts received only a small visual smoke check; counts here measure processing coverage, not accuracy. Save corrections separately from this generated export.</p>
<div><select id="camera"><option value="">All cameras</option><option>vial_fill</option><option>tank_level</option><option>cap_seal</option></select><select id="run"><option value="">All recordings</option><option>1</option><option>2</option><option>3</option><option>4</option><option>5</option><option>unassigned</option></select><select id="status"><option value="">All statuses</option><option value="high">High priority</option><option value="manual_full_image">Manual drawing</option><option value="api_error">API error</option><option value="zero">No draft boxes</option></select><button id="originals">Show originals</button><button id="prev">Previous</button><button id="next">Next</button><span id="count"></span></div><div id="grid"></div>
<script>let data=[],s=null,page=0,originals=false;const size=30;
async function init(){[data,s]=await Promise.all([fetch('gallery.json').then(r=>r.json()),fetch('summary.json').then(r=>r.json())]);document.querySelector('#summary').textContent=`${s.successful_images.toLocaleString()}/${s.images.toLocaleString()} images processed · ${s.candidate_boxes.toLocaleString()} exportable draft boxes · ${s.manual_images} manual-drawing images · ${s.api_errors} API-error images. ${s.new_successful_requests.toLocaleString()} new successful requests, ${s.reused_responses} cached responses reused. Estimated new API cost: $${s.estimated_new_usd.toFixed(3)} at published token price, not an invoice.`;
let t=document.createElement('table');t.innerHTML='<tr><th>Camera</th><th>Images</th><th>Draft boxes</th><th>Manual</th><th>API errors</th></tr>';for(const [camera,v] of Object.entries(s.by_camera)){let tr=document.createElement('tr');for(const value of [camera,v.images,v.candidate_boxes,v.manual_images,v.api_error_images]){let td=document.createElement('td');td.textContent=value;tr.append(td)}t.append(tr)}document.querySelector('#table').append(t);render()}
function filtered(){let camera=document.querySelector('#camera').value,run=document.querySelector('#run').value,status=document.querySelector('#status').value;return data.filter(r=>(!camera||r.camera===camera)&&(!run||r.run===run)&&(!status||status==='high'&&r.high||status==='zero'&&r.status==='review_candidates'&&r.boxes===0||r.status===status))}
function render(){let list=filtered(),pages=Math.max(1,Math.ceil(list.length/size));page=Math.max(0,Math.min(page,pages-1));document.querySelector('#count').textContent=`${list.length.toLocaleString()} images · page ${page+1}/${pages}`;document.querySelector('#prev').disabled=page===0;document.querySelector('#next').disabled=page===pages-1;let grid=document.querySelector('#grid');grid.replaceChildren();for(const r of list.slice(page*size,(page+1)*size)){let card=document.createElement('article');if(r.high)card.className='high';let title=document.createElement('b');title.textContent=r.id;let p=document.createElement('p');p.className='stats';p.textContent=`${r.camera} · run ${r.run} · ${r.status} · ${r.boxes} exportable boxes${r.review_reason ? " · "+r.review_reason : ""}`;let img=document.createElement('img');img.loading='lazy';img.alt=r.id;img.src=originals?r.original:r.overlay;img.onclick=()=>img.src=img.getAttribute('src')===r.original?r.overlay:r.original;let pre=document.createElement('pre');pre.textContent=r.predictions.map(q=>`${q.target}: p=${q.presence.toFixed(2)} c=${q.confidence.toFixed(2)} ${q.draft?'DRAFT':'suppressed'} (${q.source})`).join('\\n');card.append(title,p,img,pre);grid.append(card)}}
document.querySelectorAll('select').forEach(e=>e.onchange=()=>{page=0;render()});document.querySelector('#prev').onclick=()=>{page--;render()};document.querySelector('#next').onclick=()=>{page++;render()};document.querySelector('#originals').onclick=()=>{originals=!originals;document.querySelector('#originals').textContent=originals?'Show predictions':'Show originals';render()};init();</script></body></html>'''
    (out/'report.html').write_text(page,encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('prepare','pilot','run','report'))
    p.add_argument('--data',type=Path)
    p.add_argument('--out',required=True,type=Path)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    if args.mode=='prepare':
        if not args.data:p.error('--data is required to prepare')
        prepare(args)
    elif args.mode in ('pilot','run'):run(args,args.mode=='pilot')
    else:export(args.out,json.loads((args.out/'manifest.json').read_text()),json.loads((args.out/'checkpoint.json').read_text()))


if __name__=='__main__':main()
