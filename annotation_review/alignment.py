"""Reviewed associations are external inputs. There is deliberately no clock fit."""
import csv
from pathlib import Path
from .core import sha


def load_mappings(catalog,path):
    with Path(path).open(newline='',encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    seen=set()
    for row in rows:
        sample=catalog.samples[row['sample_id']]
        if sample['source']['kind']!='still' or row['sample_id'] in seen:
            raise ValueError('Mappings require unique still sample IDs')
        seen.add(row['sample_id'])
        if row['still_sha256']!=sample['source']['sha256']:
            raise ValueError('Reviewed mapping refers to different still bytes')
        if row['status'] not in ('reviewed','proposed','unmatched'):
            raise ValueError('Unknown mapping status')
        if row['status']=='reviewed':
            if not row.get('reviewer','').strip() or not row.get('evidence','').strip():
                raise ValueError('Reviewed mappings require reviewer and evidence')
            if not row.get('frame_index','').isdigit() or int(row['frame_index'])<0:
                raise ValueError('Reviewed mappings need an explicit decoded frame index')
            relative=f'{sample["source"]["camera"]}/videos/{sample["source"]["camera"]}_video_{sample["source"]["run"]}.mp4'
            if row['video_sha256']!=sha(catalog.dataset/relative):
                raise ValueError('Reviewed mapping source video hash mismatch')
        # Proposed/unmatched rows remain visibly uncertain; no promotion by score.
        sample['association']=dict(status=row['status'],reviewer=row.get('reviewer'),evidence=row.get('evidence'),
            frame_index=int(row['frame_index']) if row.get('frame_index','').isdigit() else None,
            video_sha256=row.get('video_sha256'),mapping_file_sha256=sha(path),
            note=row.get('notes','') or 'Imported separate association; spatial review does not approve timing.')
