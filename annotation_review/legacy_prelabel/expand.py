"""Bounded, reproducible expansion of the frozen crop-first Jev pilot.

Prepare stratified fresh frames and a blind audit before --run. Resumable API
responses, review-only exports, and no mutation of source data or labels.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import getpass
import json
import os
from pathlib import Path
import random
import shutil
import statistics
import zipfile

from PIL import Image, ImageDraw

from prelabel import combine, export_pack
from tune import ROI, canonical, definition, digest, needs_fallback, request

PROFILES = ('compact_crop', 'compact_full')
LOCKED_HASHES = {
    'compact_crop': '7f890e1b8cd0372699d21298218b8783d55f0efc5df9b84d6c9ed29233bb2863',
    'compact_full': 'ea082fa851b4e623de81f5bc084b683bd40d89c1f83facca4376cecc63e7820e',
}


def save(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def frozen():
    specs = {p: definition(p) for p in PROFILES}
    for name, spec in specs.items():
        if digest(canonical(spec)) != LOCKED_HASHES[name]:
            raise ValueError('Frozen prompt changed; use a separately named experiment')
    return specs


def prepare(args):
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError('Prepare requires a new output directory')
    specs = frozen()
    excluded_names, seen_pixels = set(), set()
    for manifest in args.exclude:
        for row in json.loads(manifest.read_text()):
            if row.get('source'):
                excluded_names.add(Path(row['source']).name)
            path = Path(row.get('file', ''))
            if path.is_file():
                with Image.open(path) as im:
                    seen_pixels.add(digest(im.convert('RGB').tobytes()))
    with (args.data / 'image_index.csv').open(newline='') as f:
        index = list(csv.DictReader(f))
    rng = random.Random(20261017)
    rows = []
    args.out.mkdir(parents=True)
    (args.out / 'originals').mkdir()
    for run in range(1, 6):
        sources = sorted((r for r in index if int(r['video_id']) == run), key=lambda r: r['timestamp'])
        for decile in range(10):
            candidates = list(range(len(sources)*decile//10, len(sources)*(decile+1)//10))
            rng.shuffle(candidates)
            chosen = []
            for i in candidates:
                source = sources[i]
                if source['filename'] in excluded_names:
                    continue
                path = args.data / 'images' / source['filename']
                with Image.open(path) as im:
                    if im.size != (640, 480):
                        raise ValueError('Recalibrate the camera crop for other resolutions')
                    pixel_hash = digest(im.convert('RGB').tobytes())
                if pixel_hash in seen_pixels:
                    continue
                seen_pixels.add(pixel_hash)
                chosen.append((i, source, pixel_hash))
                if len(chosen) == 4:
                    break
            if len(chosen) != 4:
                raise ValueError(f'Insufficient unique fresh images in run {run}, decile {decile}')
            for rank, (i, source, pixel_hash) in enumerate(chosen):
                sample_id = f'run{run}_d{decile:02d}_{rank}_i{i:04d}'
                original = args.out / 'originals' / (sample_id + '.jpg')
                shutil.copyfile(args.data / 'images' / source['filename'], original)
                rows.append(dict(id=sample_id, file=str(original.resolve()), source=source['filename'],
                    video_id=run, timestamp=source['timestamp'], source_index=i, decile=decile,
                    blind_audit=rank == 0, split='expanded_frozen', source_size=[640,480],
                    source_sha256=digest(original.read_bytes()), pixel_sha256=pixel_hash))
    save(args.out / 'manifest.json', rows)
    save(args.out / 'prompt-definitions.json', specs)
    save(args.out / 'protocol.json', dict(prepared_utc=datetime.now(timezone.utc).isoformat(),
        seed=20261017, selection='Four randomly selected unique stills per timeline decile per recording; first selected per decile is blind audit.',
        images=len(rows), audit_images=sum(r['blind_audit'] for r in rows), prompt_hashes=LOCKED_HASHES,
        excluded_manifests=[str(p) for p in args.exclude],
        caveats=['Same five recordings and camera as development; no new-recording generalization claim.',
                 'Timeline strata are not process-phase labels; neighboring frames remain correlated.',
                 'Audit is an assistant visual spot check, not independent human ground truth.',
                 'Strict presence>=0.8 and box confidence>=0.5; draft presence>=0.5; no policy retuning.',
                 'Every exported label requires human review, including missing targets.']))
    for profile in PROFILES:
        for part in ('images', 'responses'):
            (args.out / profile / part).mkdir(parents=True, exist_ok=True)
    for run in range(1, 6):
        group = [r for r in rows if r['video_id'] == run and r['blind_audit']]
        sheet = Image.new('RGB', (1040, 1660), '#eeeeee')
        draw = ImageDraw.Draw(sheet)
        for n, row in enumerate(group):
            x, y = (n%2)*520, (n//2)*332
            im = Image.open(row['file']).convert('RGB')
            sheet.paste(im.resize((400,300)), (x,y+25))
            sheet.paste(im.crop((230,40,390,440)).resize((120,300)), (x+400,y+25))
            draw.text((x+5,y+5), row['id'], fill='black')
        sheet.save(args.out / f'blind-audit-run{run}.jpg', quality=95)
    print(json.dumps(dict(prepared=len(rows), blind_audit=50, out=str(args.out))))


def infer_one(out, row, key):
    image = Image.open(row['file']).convert('RGB')
    path = out / 'compact_crop/images' / (row['id'] + '.jpg')
    if not path.exists():
        image.crop(ROI).resize((280,800), Image.Resampling.LANCZOS).save(path, quality=95)
    current = dict(row, crop_xyxy=ROI, input_file=str(path), input_sha256=digest(path.read_bytes()))
    primary = request(current, 'compact_crop', out / 'compact_crop', {}, key)
    if not primary.get('ok'):
        return row['id'], [], 'primary_request_failed'
    fallback, warning = None, None
    if needs_fallback(primary['response']['answers']):
        full_path = out / 'compact_full/images' / (row['id'] + '.jpg')
        if not full_path.exists():
            shutil.copyfile(row['file'], full_path)
        full = dict(row, crop_xyxy=None, input_file=str(full_path), input_sha256=digest(full_path.read_bytes()))
        fallback = request(full, 'compact_full', out / 'compact_full', {}, key)
        if not fallback.get('ok'):
            warning = 'fallback_request_failed'
    return row['id'], combine(current, primary, fallback), warning


def run(args):
    frozen()
    rows = json.loads((args.out / 'manifest.json').read_text())
    if len(rows) != 200:
        raise ValueError('This expansion is bounded to 200 prepared images')
    audit_path = args.out / 'blind-audit.json'
    if not audit_path.exists():
        raise ValueError('Record the 50 blind visual judgments before inference')
    audit = json.loads(audit_path.read_text())
    if {r['id'] for r in audit} != {r['id'] for r in rows if r['blind_audit']}:
        raise ValueError('Incomplete blind audit')
    # Freeze the judgments before any new responses exist.
    snapshot = args.out / 'blind-audit-before-api.json'
    if snapshot.exists():
        if snapshot.read_bytes() != audit_path.read_bytes():
            raise ValueError('Audit changed since inference began')
    else:
        shutil.copyfile(audit_path, snapshot)
    key = os.environ.get('TYPESAFE_API_KEY') or getpass.getpass('TypeSafe API key (hidden, memory only): ')
    completed = {}
    # Sequential smoke check, then at most three simultaneous image workflows.
    first = infer_one(args.out, rows[0], key)
    completed[first[0]] = dict(records=first[1], warning=first[2])
    if not first[1]:
        raise SystemExit('Smoke request failed; inspect saved response')
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(infer_one, args.out, row, key): row for row in rows[1:]}
        for future in as_completed(futures):
            sample_id, records, warning = future.result()
            completed[sample_id] = dict(records=records, warning=warning)
            save(args.out / 'checkpoint.json', completed)
    del key
    save(args.out / 'checkpoint.json', completed)
    report(args)


def report(args):
    rows = json.loads((args.out / 'manifest.json').read_text())
    completed = json.loads((args.out / 'checkpoint.json').read_text())
    # Failed requests are not empty/negative labels and cannot enter the YOLO pack.
    successful_rows = [r for r in rows if completed.get(r['id'], {}).get('records')]
    records = [r for row in successful_rows for r in completed[row['id']]['records']]
    queue = export_pack(args.out, successful_rows, records)
    audit = json.loads((args.out / 'blind-audit-before-api.json').read_text())
    by_records = {row['id']: {r['target']:r for r in records if r['sample_id']==row['id']} for row in successful_rows}
    audit_metrics = {}
    for target in ('vial','liquid'):
        known = [a for a in audit if a.get(target + '_visible') is not None and a['id'] in by_records]
        pos = [a for a in known if a[target + '_visible']]
        neg = [a for a in known if not a[target + '_visible']]
        audit_metrics[target] = dict(positives=len(pos), negatives=len(neg), uncertain=len(audit)-len(known))
        for policy in ('strict','draft'):
            audit_metrics[target][policy] = dict(
                visible_retained=sum(by_records[a['id']][target][policy] for a in pos),
                absent_retained=sum(by_records[a['id']][target][policy] for a in neg))
    unsupported = [a for a in audit if a.get('roi_supported') is False and a['id'] in by_records]
    audit_metrics['crop_guard'] = dict(visually_unsupported=len(unsupported),
        routed_manual=sum(by_records[a['id']]['vial']['manual_only'] for a in unsupported),
        note='Visual audit includes absent and moving targets. Guard success alone is not labeling accuracy.')
    supported = [a for a in audit if a.get('roi_supported') is True and a['id'] in by_records]
    audit_metrics['supported_liquid'] = dict(
        positives=sum(a['liquid_visible'] is True for a in supported),
        negatives=sum(a['liquid_visible'] is False for a in supported),
        strict_visible_retained=sum(a['liquid_visible'] is True and by_records[a['id']]['liquid']['strict'] for a in supported),
        draft_visible_retained=sum(a['liquid_visible'] is True and by_records[a['id']]['liquid']['draft'] for a in supported),
        strict_absent_retained=sum(a['liquid_visible'] is False and by_records[a['id']]['liquid']['strict'] for a in supported),
        draft_absent_retained=sum(a['liquid_visible'] is False and by_records[a['id']]['liquid']['draft'] for a in supported))
    calls = []
    for profile in PROFILES:
        calls.extend(json.loads(p.read_text()) for p in (args.out / profile / 'responses').glob('*.json'))
    good = [c for c in calls if c.get('ok')]
    tokens = sum(c['response']['usage']['input_tokens'] for c in good)
    metrics = dict(prepared_images=len(rows), successful_images=len(successful_rows),
        api_calls=len(calls), successful_calls=len(good), fallback_calls=sum(c['profile']=='compact_full' for c in calls),
        input_tokens=tokens, estimated_usd=tokens*.042/1000000,
        median_request_seconds=statistics.median(c['seconds'] for c in good) if good else None,
        candidate_boxes=sum(q['candidate_boxes'] for q in queue),
        manual_full_images=sum(q['status']=='manual_full_image' for q in queue),
        high_priority_images=sum(q['high_priority'] for q in queue),
        exported_draft_boxes=dict(Counter(r['target'] for r in records if r['draft'] and not r['manual_only'])),
        strict_boxes_before_manual_routing=dict(Counter(r['target'] for r in records if r['strict'])),
        draft_boxes_before_manual_routing=dict(Counter(r['target'] for r in records if r['draft'])),
        audit=audit_metrics, warnings={k:v['warning'] for k,v in completed.items() if v['warning']})
    save(args.out / 'metrics.json', metrics)
    save(args.out / 'annotations.json', records)
    save(args.out / 'api-errors.json', metrics['warnings'])
    (args.out / 'overlays').mkdir(exist_ok=True)
    gallery = []
    audit_lookup = {a['id']: a for a in audit}
    for row in rows:
        group = by_records.get(row['id'], {})
        q = next((q for q in queue if q['sample_id']==row['id']), None)
        im = Image.open(row['file']).convert('RGB')
        draw = ImageDraw.Draw(im)
        for target, r in group.items():
            box = r['box']; xy = [box[k]*(640 if k in ('left','right') else 480) for k in ('left','top','right','bottom')]
            color = '#00ffff' if target=='vial' else '#ffb700'
            draw.rectangle(xy, outline=color, width=2)
            text = f"{target} p={r['presence']:.2f} c={r['confidence']:.2f} {'DRAFT' if r['draft'] else 'SUPPRESSED'}"
            y = max(0,xy[1]-13)
            draw.rectangle((xy[0],y,min(639,xy[0]+len(text)*6),y+13), fill='black')
            draw.text((xy[0],y),text,fill=color)
        im.save(args.out / 'overlays' / (row['id']+'.jpg'), quality=92)
        gallery.append(dict(id=row['id'], run=row['video_id'], timestamp=row['timestamp'],
            original='originals/'+row['id']+'.jpg', overlay='overlays/'+row['id']+'.jpg',
            status=q['status'] if q else 'api_error', high=q['high_priority'] if q else True,
            audit=audit_lookup.get(row['id']), predictions=group))
    save(args.out / 'gallery.json', gallery)
    page = '''<!doctype html><html><head><meta charset="utf-8"><title>Jev expanded prelabel review</title>
<style>body{font:16px system-ui;background:#10151e;color:#e3e9f2;margin:clamp(12px,3vw,28px);overflow-wrap:anywhere}a{color:#6dd8ff}h1{margin-bottom:8px;font-size:clamp(24px,3vw,32px)}.muted{color:#a5b4c9;max-width:1100px;line-height:1.5}button,select{padding:9px;background:#233246;color:white;border:1px solid #54667f;border-radius:6px;margin:4px}#grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,420px),1fr));gap:18px;margin-top:20px}article{min-width:0;background:#1a2433;padding:14px;border-radius:9px}img{width:100%;cursor:pointer}pre{white-space:pre-wrap;font-size:13px}.high{border:2px solid #e5a632}.normal{border:2px solid #34445a}.tag{color:#ffd16b}#audit{overflow-x:auto}table{border-collapse:collapse;min-width:680px}td,th{padding:8px 16px;border-bottom:1px solid #45576c;text-align:left;white-space:nowrap}</style></head><body>
<h1>200 fresh frames · frozen prompts · 5 recordings</h1>
<p class="muted">Unreviewed prelabels for the fill-position vial and visibly colored liquid. Cyan=vial, gold=liquid. Suppressed predictions are shown so you can inspect misses. Click any image to toggle the original. These are the same camera/recordings as development; frames are correlated. Timeline sampling does not guarantee every process phase. No claim of mAP or measured human time savings.</p>
<p><a href="review-pack.zip">Download YOLO review pack</a> · <a href="metrics.json">Metrics</a> · <a href="review-pack/review-queue.csv">Review queue</a> · <a href="protocol.json">Sampling protocol</a> · <a href="blind-audit-before-api.json">Blind visual judgments</a> · <a href="visual-findings.json">Inspected failure examples</a></p>
<div id="summary"></div><h2>50-frame blind visual check</h2><div id="audit"></div>
<p id="routing" class="muted"></p><p class="muted">Assistant judgments were recorded before API inference. Counts above evaluate presence gates before manual routing, not box-boundary accuracy. Every image needs review for missing targets. No candidates are approved by this report. See the inspected failure examples for qualitative boundary problems.</p>
<div><select id="run"><option value="">All recordings</option></select><select id="filter"><option value="all">All images</option><option value="high">High priority</option><option value="manual">Manual full image</option><option value="audit">Blind audit only</option><option value="miss">Audited liquid misses (draft)</option><option value="negative">Audited liquid false positives (draft)</option></select><button id="toggle">Show originals</button><span id="count"></span></div><div id="grid"></div>
<script>const data=__DATA__,metrics=__METRICS__;let originals=false;
const supported=metrics.audit.supported_liquid,guard=metrics.audit.crop_guard;document.querySelector('#routing').textContent=`Among ${supported.positives} colored frames supported by the crop, strict retained ${supported.strict_visible_retained} and draft retained ${supported.draft_visible_retained}. ${guard.routed_manual}/${guard.visually_unsupported} visually unsupported audited cases were routed to manual review. Absent-vial audit examples: ${metrics.audit.vial.negatives}.`;
document.querySelector('#summary').textContent=`${metrics.successful_images}/${metrics.prepared_images} images processed; ${metrics.successful_calls}/${metrics.api_calls} successful API calls (${metrics.fallback_calls} fallbacks); ${metrics.candidate_boxes} exportable draft boxes; ${metrics.manual_full_images} manual full-image cases; ${metrics.high_priority_images} high-priority images. ${metrics.input_tokens.toLocaleString()} input tokens; estimated $${metrics.estimated_usd.toFixed(4)} at published token price (not an invoice).`;
const tbl=document.createElement('table');tbl.innerHTML='<tr><th>Target</th><th>Visible / absent / uncertain</th><th>Strict retained visible / false positive</th><th>Draft retained visible / false positive</th></tr>';
for(const t of ['vial','liquid']){const m=metrics.audit[t];const tr=document.createElement('tr');for(const v of [t,`${m.positives} / ${m.negatives} / ${m.uncertain}`,`${m.strict.visible_retained}/${m.positives} · ${m.strict.absent_retained}/${m.negatives}`,`${m.draft.visible_retained}/${m.positives} · ${m.draft.absent_retained}/${m.negatives}`]){const td=document.createElement('td');td.textContent=v;tr.append(td)}tbl.append(tr)}document.querySelector('#audit').append(tbl);
for(let i=1;i<=5;i++){let o=document.createElement('option');o.value=i;o.textContent=`Recording ${i}`;document.querySelector('#run').append(o)}
function render(){let run=document.querySelector('#run').value,f=document.querySelector('#filter').value;let items=data.filter(r=>(!run||r.run==run)&&(f==='all'||f==='high'&&r.high||f==='manual'&&r.status==='manual_full_image'||f==='audit'&&r.audit||f==='miss'&&r.audit?.liquid_visible===true&&!r.predictions.liquid?.draft||f==='negative'&&r.audit?.liquid_visible===false&&r.predictions.liquid?.draft));document.querySelector('#count').textContent=`${items.length} images`;const grid=document.querySelector('#grid');grid.replaceChildren();for(const r of items){let card=document.createElement('article');card.className=r.high?'high':'normal';let title=document.createElement('b');title.textContent=r.id;let p=document.createElement('p');p.textContent=`${r.timestamp} · ${r.status}`;let img=document.createElement('img');img.loading='lazy';img.src=originals?r.original:r.overlay;img.onclick=()=>{img.src=img.getAttribute('src')===r.original?r.overlay:r.original};card.append(title,p,img);if(r.audit){let a=document.createElement('p');a.className='tag';a.textContent=`Blind audit: vial=${r.audit.vial_visible}, liquid=${r.audit.liquid_visible}; ${r.audit.note||''}`;card.append(a)}let pre=document.createElement('pre');pre.textContent=Object.values(r.predictions).map(p=>`${p.target}: p=${p.presence.toFixed(3)}, confidence=${p.confidence.toFixed(3)} | strict=${p.strict} draft=${p.draft} | ${p.source}`).join('\\n');card.append(pre);grid.append(card)}}
document.querySelectorAll('select').forEach(s=>s.onchange=render);document.querySelector('#toggle').onclick=()=>{originals=!originals;document.querySelector('#toggle').textContent=originals?'Show predictions':'Show originals';render()};render();</script></body></html>'''
    page = page.replace('__DATA__',json.dumps(gallery).replace('<','\\u003c')).replace('__METRICS__',json.dumps(metrics).replace('<','\\u003c'))
    (args.out / 'report.html').write_text(page, encoding='utf-8')
    # Include immutable provenance and audit evidence in the portable review pack.
    for name in ('manifest.json','protocol.json','prompt-definitions.json','metrics.json','blind-audit-before-api.json','api-errors.json','visual-findings.json'):
        if not (args.out / name).exists():
            continue
        shutil.copyfile(args.out / name, args.out / 'review-pack' / name)
    with zipfile.ZipFile(args.out / 'review-pack.zip','w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted((args.out / 'review-pack').rglob('*')):
            if path.is_file():
                z.write(path,path.relative_to(args.out / 'review-pack'))
    print(json.dumps(metrics,indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','run','report'])
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--data',type=Path)
    parser.add_argument('--exclude',type=Path,nargs='*',default=[])
    args = parser.parse_args()
    if args.mode=='prepare' and not args.data:
        parser.error('--data is required to prepare')
    globals()[args.mode](args)


if __name__=='__main__':
    main()
