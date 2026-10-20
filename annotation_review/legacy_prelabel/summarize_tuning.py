"""Build the before/after review pack from cached, frozen-profile responses."""
import argparse
import csv
import html
import json
from pathlib import Path
import shutil
import statistics

from PIL import Image, ImageDraw, ImageFont

from prelabel import combine, export_pack
from tune import iou


def draw_records(image, records, font):
    draw = ImageDraw.Draw(image)
    for record in records:
        if not record['draft']:
            continue
        box = record['box']
        xy = [round(box[k] * (image.width if k in ('left', 'right') else image.height))
              for k in ('left', 'top', 'right', 'bottom')]
        color = '#20d9ff' if record['target'] == 'vial' else '#ffb52e'
        draw.rectangle(xy, outline=color, width=3)
        label = f"{record['target']} p={record['presence']:.2f} c={record['confidence']:.2f}"
        y = max(0, xy[1] - 18)
        x = min(xy[0], max(0, image.width - int(draw.textlength(label, font=font)) - 8))
        draw.rectangle(draw.textbbox((x, y), label, font=font), fill='#101820')
        draw.text((x, y), label, fill=color, font=font)
    return image


def measurements(records, rows):
    result = {}
    for name in ('vial', 'liquid'):
        group = [r for r in records if r['target'] == name]
        expected = lambda row: row['vial_visible'] if name == 'vial' else row['colored_liquid_visible']
        result[name] = {'stationary_visible': sum(expected(r) and r['roi_supported'] for r in rows)}
        for policy in ('strict', 'draft'):
            retained, wrong, useful, close, overlaps = 0, 0, 0, 0, []
            for record in group:
                row = next(r for r in rows if r['id'] == record['sample_id'])
                if not row['roi_supported']:
                    continue
                if not expected(row):
                    wrong += bool(record[policy])
                    continue
                reference = row.get('vial_reference_xyxy' if name == 'vial' else 'liquid_reference_xyxy')
                overlap = iou(record['box'], reference, [640, 480])
                if overlap is not None:
                    overlaps.append(overlap)
                if record[policy]:
                    retained += 1
                    useful += overlap is not None and overlap >= .5
                    close += overlap is not None and overlap >= .75
            result[name][policy] = dict(visible_retained=retained, absent_retained=wrong,
                                        approximate_iou_ge_05=useful, approximate_iou_ge_075=close)
            result[name]['mean_raw_reference_iou'] = statistics.mean(overlaps) if overlaps else None
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out
    rows = json.loads((out / 'compact_crop/holdout/manifest.json').read_text())
    old_annotations = json.loads((out / 'original_holdout/annotations.json').read_text())
    old, combined, metrics_rows = [], [], []
    for row in rows:
        name = row['id']
        primary = json.loads((out / 'compact_crop/holdout/responses' / (name + '.json')).read_text())
        fbpath = out / 'compact_full/holdout/responses' / (name + '.json')
        fallback = json.loads(fbpath.read_text()) if fbpath.exists() else None
        combined.extend(combine(row, primary, fallback))
        for target, baseline_name in (('vial', 'fill_position_vial'), ('liquid', 'colored_liquid')):
            source = next(r for r in old_annotations if r['sample_id'] == name and r['class_name'] == baseline_name)
            old.append(dict(sample_id=name, target=target, box=source['box'], presence=source['presence'],
                            confidence=source['confidence'], strict=source['exported'], draft=source['presence'] >= .5))
    queue = export_pack(out, rows, combined)
    previous, current = measurements(old, rows), measurements(combined, rows)
    responses = [json.loads(p.read_text()) for p in out.glob('*/dev/responses/*.json')]
    responses += [json.loads(p.read_text()) for p in out.glob('*/holdout/responses/*.json')]
    responses += [json.loads(p.read_text()) for p in out.glob('original_holdout/responses/*.json')]
    tokens = sum(r['response']['usage']['input_tokens'] for r in responses)
    summary = dict(model='jev-1.14.0', development_profiles=3, development_images=15,
                   fresh_images=12, stationary_images=11, moving_images=1,
                   unchanged_strict_thresholds={'presence': .8, 'box_confidence': .5},
                   draft_policy={'presence': .5, 'box_confidence': 'review priority only'},
                   original=previous, tuned=current,
                   review_pack=dict(candidate_boxes=sum(r['candidate_boxes'] for r in queue),
                                    images_with_prelabels=sum(r['status']=='review_candidates' for r in queue),
                                    manual_full_images=sum(r['status']=='manual_full_image' for r in queue),
                                    high_priority_images=sum(r['high_priority'] for r in queue)),
                   requests=len(responses), input_tokens=tokens,
                   estimated_cost_usd=tokens*.042/1_000_000,
                   median_api_request_seconds=statistics.median(r['seconds'] for r in responses),
                   limits='Small same-camera study. Assistant approximate references fixed before new calls, not independent ground truth. Human labeling time has not been measured.')
    (out / 'comparison.json').write_text(json.dumps(summary, indent=2) + '\n')
    (out / 'combined-annotations.json').write_text(json.dumps(combined, indent=2) + '\n')
    for name in ('overlays_old', 'overlays_v2', 'originals'):
        (out / name).mkdir(exist_ok=True)
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 14)
    cards, preview_tiles = [], []
    for row in rows:
        name = row['id']
        source = Image.open(row['file']).convert('RGB')
        shutil.copyfile(row['file'], out / 'originals' / (name + '.jpg'))
        before = [r for r in old if r['sample_id'] == name]
        after = [r for r in combined if r['sample_id'] == name]
        a = draw_records(source.copy(), before, font)
        b = draw_records(source.copy(), after, font)
        a.save(out / 'overlays_old' / (name + '.jpg'), quality=94)
        b.save(out / 'overlays_v2' / (name + '.jpg'), quality=94)
        manual = any(r['manual_only'] for r in after)
        badge = 'MANUAL FULL-IMAGE REVIEW: crop boundary' if manual else 'Draft candidates: review before accepting'
        details = []
        for record in after:
            details.append(f"<tr><td>{record['target']}</td><td>{record['presence']:.2f}</td><td>{record['confidence']:.2f}</td><td>{record['source']}</td><td>{'yes' if record['strict'] else 'review'}</td></tr>")
        cards.append(f'<article><h2>{name}</h2><p>{badge}</p><div class="pair"><figure><img src="overlays_old/{name}.jpg"><figcaption>Original prompt (same draft policy for a fair visual comparison)</figcaption></figure><figure><img src="overlays_v2/{name}.jpg"><figcaption>V2 crop + conditional full-frame liquid fallback</figcaption></figure></div><details><summary>Numbers, source and full-resolution original</summary><a href="originals/{name}.jpg">Original image</a><table><tr><th>Target</th><th>Presence</th><th>Box confidence</th><th>Source</th><th>Passes original strict gates?</th></tr>{"".join(details)}</table></details></article>')
        if name in ('hold_run5_vial09', 'hold_run5_vial14'):
            for label, group in (('Original prompt', before), ('V2 prompt', after)):
                clean = source.copy()
                draw = ImageDraw.Draw(clean)
                for record in group:
                    if not record['draft']: continue
                    box = record['box']
                    xy = [round(box[k] * (640 if k in ('left', 'right') else 480)) for k in ('left','top','right','bottom')]
                    draw.rectangle(xy, outline='#20d9ff' if record['target']=='vial' else '#ffb52e',width=3)
                zoom = clean.crop((240, 40, 380, 440)).resize((210, 600))
                tile = Image.new('RGB', (240, 700), '#101820')
                tile.paste(zoom, (15, 100))
                draw = ImageDraw.Draw(tile)
                draw.text((8, 7), name, font=font, fill='white')
                draw.text((8, 30), label, font=font, fill='#20d9ff')
                liquid = next(r for r in group if r['target']=='liquid')
                draw.text((8, 52), f"Liquid strict: {'KEEP' if liquid['strict'] else 'REJECT'}",font=font,fill='#ffb52e')
                draw.text((8, 74), f"p={liquid['presence']:.2f} c={liquid['confidence']:.2f}",font=font,fill='#ffb52e')
                preview_tiles.append(tile)
    preview = Image.new('RGB', (960, 700), '#101820')
    for i, tile in enumerate(preview_tiles): preview.paste(tile, (i * 240, 0))
    preview.save(out / 'preview.jpg', quality=94)
    metrics_table = ''
    for target in ('liquid', 'vial'):
        denominator = current[target]['stationary_visible']
        for policy in ('strict', 'draft'):
            old_metric, new_metric = previous[target][policy], current[target][policy]
            metrics_table += f'<tr><td>{target}</td><td>{policy}</td><td>{old_metric["visible_retained"]}/{denominator}</td><td>{new_metric["visible_retained"]}/{denominator}</td><td>{new_metric["absent_retained"]}</td></tr>'
    spec = json.loads((out / 'locked-profile.json').read_text())
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Jev prompt tuning: review-ready prelabels</title><style>
body{font:16px system-ui;background:#101820;color:#e8f0f4;max-width:1400px;margin:32px auto;padding:0 22px}h1{color:#20d9ff}a{color:#20d9ff}article,section{background:#172735;padding:20px;border-radius:12px;margin:20px 0}.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}figure{margin:0}img{width:100%}figcaption{color:#b8cbd7}table{border-collapse:collapse;width:100%}td,th{padding:10px;text-align:left;border-bottom:1px solid #365065}pre{white-space:pre-wrap;overflow-wrap:anywhere}li{margin:8px 0}.callout{border-left:4px solid #ffb52e;padding-left:15px}</style>
<h1>Prompt v2: better liquid recall, review-first exports</h1><p>October 16, 2026 · Jev 1.14 · 15 development images, three prompt variants, then 12 fresh frames with the chosen rules frozen.</p>
<section><h2>Same images, same strict gates</h2><p>On the 11 stationary fresh frames, the original strict filters kept liquid boxes on 5 of 8 colored fills. The new pipeline kept 8 of 8. Neither kept a liquid box on the three clear frames. The moving/blurred vial was caught by the crop-boundary guard and sent to manual full-image review.</p>
<table><tr><th>Target</th><th>Policy</th><th>Old retained</th><th>V2 retained</th><th>V2 absent retained</th></tr>'''
    page += metrics_table + '</table><p>Strict is unchanged: presence ≥ 0.8 and Box confidence ≥ 0.5. Draft is presence ≥ 0.5, with Box confidence used to prioritize review. The draft-policy gain is shown separately from the prompt-only strict comparison.</p></section>'
    page += '<section class="callout"><h2>Recall improved; boundary precision did not</h2><p>On the eight colored stationary frames, mean overlap against the approximate liquid reference boxes decreased from 0.80 with the original raw boxes to 0.71 with V2. None of the V2 liquid boxes reached 0.75 IoU against those references. More objects are recovered, but fluid edges still need human adjustment. The whole-vial mean remained about 0.88.</p><p><a href="liquid_v3/report.html">Final follow-up: short presence + original localization wording on another 12 fresh frames</a>. That check also improved presence without establishing better boundary accuracy. No further tuning was done after it.</p></section>'
    page += '<section><h2>Ready for a mentee trial</h2><p>The review pack contains 19 candidate boxes on 11 stationary images; one moving-vial image is queued separately. All 19 overlap the approximate visual references by at least 0.5 IoU. Some edges still need adjustment. This is not a trained-detector accuracy result or approval to use the labels without review.</p><p class="callout">A time saving is plausible but unmeasured. Trial ten frames in your annotation tool: accept, adjust or redraw each prelabel and record active labeling time against drawing from scratch.</p><a href="comparison.json">Metrics and limitations</a> · <a href="review-pack/review-queue.csv">Review queue</a> · <a href="combined-annotations.json">Candidate annotations</a> · <a href="locked-profile.json">Frozen selection rules</a></section>'
    page += '<section><h2>What changed</h2><ul><li>Moved target identity into shared state; shortened the independent presence and box questions.</li><li>Defined colored-liquid yes/no criteria, including dark and nearly black fluid.</li><li>Used an enlarged crop around the vial for its full extent, including below the holder.</li><li>Retried plausible but uncertain liquid on the full image with the shorter prompt.</li><li>Used a deterministic box-at-crop-edge guard alongside the cut questions.</li><li>Reference-image criteria were tested and rejected: more tokens, worse strict recall.</li></ul><details><summary>Exact selected prompt</summary><pre>'
    page += html.escape(json.dumps(json.loads((out / 'prompt-definitions.json').read_text())['compact_crop'], indent=2))
    page += '</pre></details></section><section><h2>Recovered blue and dark-green fills</h2><img src="preview.jpg"><p>Cyan: vial. Orange: colored-liquid body. All pictures display the same draft policy on both sides; the strict-policy results above isolate the prompt change.</p></section>'
    page += '<section><h2>All fresh-frame comparisons</h2><p>Every frame is shown, including clears and motion. Source recipes and timing labels were never supplied as answers. References are approximate assistant visual judgments fixed before the new API calls; they have not been independently reviewed by the team. New frames come from the same recordings/camera, so this is not a new-camera or recording-independent benchmark.</p></section>'
    page += ''.join(cards) + '</html>'
    (out / 'report.html').write_text(page, encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
