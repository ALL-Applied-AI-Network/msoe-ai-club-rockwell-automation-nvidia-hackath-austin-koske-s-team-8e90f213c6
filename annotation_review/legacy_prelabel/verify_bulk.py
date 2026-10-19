"""Check bulk coverage, source integrity, local class IDs and review exports."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import zipfile

from bulk import CAMERAS, class_names, specs
from tune import canonical, digest
from run import valid_box


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();out=args.out
    rows=json.loads((out/'manifest.json').read_text())
    protocol=json.loads((out/'protocol.json').read_text())
    summary=json.loads((out/'summary.json').read_text())
    queue=list(csv.DictReader((out/'review-queue.csv').open(newline='')))
    definitions=json.loads((out/'prompt-definitions.json').read_text())
    assert definitions==specs(), 'Prompts differ from generated results'
    assert {k:digest(canonical(v)) for k,v in definitions.items()}==protocol['profile_sha256']
    assert len(rows)==len(queue)==summary['images']
    assert len({r['id'] for r in rows})==len(rows)
    assert {q['sample_id'] for q in queue}=={r['id'] for r in rows}
    source=Path(protocol['source_root']);checks=Counter()
    lookup={q['sample_id']:q for q in queue}
    for camera in CAMERAS:
        indexed=list(csv.DictReader((source/camera/'image_index.csv').open(newline='')))
        group=[r for r in rows if r['camera']==camera]
        assert {r['source'] for r in group}=={r['filename'] for r in indexed}
        pack=out/'review-pack'/camera
        names=(pack/'classes.txt').read_text().splitlines()
        assert names==class_names(camera)
        annotations=json.loads((pack/'annotations.json').read_text())
        for a in annotations:
            assert valid_box(a['box'])
            assert 0<=a['class_id']<len(names)
            assert a['annotation_status']=='unreviewed_model_candidate'
        for row in group:
            assert digest(Path(row['file']).read_bytes())==row['source_sha256'], 'Source changed'
            q=lookup[row['id']]
            folder={'api_error':'api_error_images','manual_full_image':'manual_review_images','review_candidates':'images'}[q['status']]
            assert digest((pack/folder/row['source']).read_bytes())==row['source_sha256']
            label=pack/'labels'/(row['id']+'.txt')
            assert label.exists()==(q['status']=='review_candidates')
            for other in ('images','manual_review_images','api_error_images'):
                assert (pack/other/row['source']).exists()==(other==folder)
            count=0
            if label.exists():
                for line in label.read_text().splitlines():
                    cls,*values=line.split();values=list(map(float,values))
                    assert 0<=int(cls)<len(names) and len(values)==4
                    cx,cy,w,h=values
                    assert w>0 and h>0 and cx-w/2>=-1e-7 and cx+w/2<=1+1e-7 and cy-h/2>=-1e-7 and cy+h/2<=1+1e-7
                    count+=1
            assert count==int(q['candidate_boxes'])
            checks['images']+=1;checks['boxes']+=count
        assert len(group)==summary['by_camera'][camera]['images']
        assert sum(int(q['candidate_boxes']) for q in queue if q['camera']==camera)==summary['by_camera'][camera]['candidate_boxes']
    assert checks['boxes']==summary['candidate_boxes']
    for reference in (source/'vial_fill/labels').iterdir():
        if reference.is_file():
            assert digest(reference.read_bytes())==digest((out/'review-pack/vial_fill/reference_labels'/reference.name).read_bytes())
    for name in ('review-pack.zip','labels-only.zip',*(c+'-review-pack.zip' for c in CAMERAS)):
        with zipfile.ZipFile(out/name) as archive:
            assert archive.testzip() is None
    for path in out.rglob('*'):
        if path.is_file() and path.suffix in ('.json','.csv','.html','.txt'):
            assert b'apikey_' not in path.read_bytes(), 'Credential-like token found in output'
    assert summary['api_errors']==0, 'API-error images remain; inspect and retry before calling coverage complete'
    result=dict(status='passed',images=checks['images'],candidate_boxes=checks['boxes'],
        source_images_unchanged=True,reference_labels_preserved=True,prompt_hashes_match=True,
        original_filenames=True,camera_local_class_ids=True,all_archives_crc_valid=True)
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
