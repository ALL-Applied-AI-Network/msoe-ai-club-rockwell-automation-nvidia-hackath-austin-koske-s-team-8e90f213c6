"""One final liquid-only profile: short presence + original localization prompt."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import getpass
import json
import os
from pathlib import Path
import shutil
import statistics

from run import TARGETS
from tune import definition, digest, canonical, request, iou


def liquid_definition():
    spec = definition('compact_full')
    spec['name'] = 'hybrid_liquid_v3'
    spec['questions'] = {
        'liquid_present': spec['questions']['liquid_present'],
        'liquid_box': {'type': 'box', 'instructions': TARGETS['colored_liquid']['box']},
    }
    return spec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--phase', choices=('dev', 'validation'), required=True)
    args = parser.parse_args()
    spec = liquid_definition()
    if args.phase == 'validation':
        lock = json.loads((args.out / 'locked.json').read_text())
        assert digest(canonical(spec)) == lock['definition_sha256']
        assert digest(args.manifest.read_bytes()) == lock['manifest_sha256']
    folder = args.out / args.phase
    for name in ('images', 'responses'):
        (folder / name).mkdir(parents=True, exist_ok=True)
    rows = json.loads(args.manifest.read_text())
    for row in rows:
        dest = folder / 'images' / (row['id'] + '.jpg')
        shutil.copyfile(row['file'], dest)
        row.update(input_file=str(dest), input_sha256=digest(dest.read_bytes()), crop_xyxy=None)
    (folder / 'manifest.json').write_text(json.dumps(rows, indent=2) + '\n')
    (args.out / 'definition.json').write_text(json.dumps(spec, indent=2) + '\n')
    key = os.environ.get('TYPESAFE_API_KEY') or getpass.getpass('TypeSafe API key (hidden, memory only): ')
    results = {}
    first = request(rows[0], 'hybrid_liquid_v3', folder, {}, key, spec)
    if not first.get('ok'): raise SystemExit('First request failed; inspect the saved response')
    results[rows[0]['id']] = first
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(request, row, 'hybrid_liquid_v3', folder, {}, key, spec): row['id'] for row in rows[1:]}
        for future in as_completed(futures): results[futures[future]] = future.result()
    del key
    records = []
    for row in rows:
        result = results[row['id']]
        if not result.get('ok'): continue
        a = result['response']['answers'];p=a['liquid_present']['noul'];c=a['liquid_box']['confidence'];box=a['liquid_box']['box']
        records.append(dict(sample_id=row['id'], box=box, presence=p, confidence=c,
                            visible=row['colored_liquid_visible'], strict=p>=.8 and c>=.5,draft=p>=.5,
                            reference_iou=iou(box,row.get('liquid_reference_xyxy'),[640,480])))
    positive = [r for r in records if r['visible']]
    overlaps = [r['reference_iou'] for r in positive if r['reference_iou'] is not None]
    metrics = dict(requests=len(results), visible=len(positive), strict_retained=sum(r['strict'] for r in positive),
                   draft_retained=sum(r['draft'] for r in positive),
                   false_strict=sum(r['strict'] for r in records if not r['visible']),
                   false_draft=sum(r['draft'] for r in records if not r['visible']),
                   mean_raw_reference_iou=statistics.mean(overlaps) if overlaps else None,
                   draft_overlap_ge_05=sum(r['draft'] and (r['reference_iou'] or 0)>=.5 for r in positive),
                   draft_overlap_ge_075=sum(r['draft'] and (r['reference_iou'] or 0)>=.75 for r in positive))
    (folder / 'annotations.json').write_text(json.dumps(records,indent=2)+'\n')
    (folder / 'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print(json.dumps(metrics,indent=2))


if __name__ == '__main__': main()
