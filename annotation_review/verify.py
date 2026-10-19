"""Read-only integrity verification against the saved prelabel inputs."""
import argparse
import json
from pathlib import Path
from .core import Catalog, sha


def verify(dataset,prelabels,cases=None):
    catalog=Catalog(dataset,prelabels,cases)
    counts={'still_images':0,'case_frames':0,'organizer_files':0,'source_videos':0}
    for sample in catalog.samples.values():
        catalog.path(sample)
        counts['still_images' if sample['source']['kind']=='still' else 'case_frames']+=1
    root=Path(prelabels)
    reference=root/'review-pack/vial_fill/reference_labels'
    if not reference.exists():reference=root/'vial_fill/reference_labels'
    if not reference.exists():raise ValueError('Provide the original bulk output or full labels archive containing reference_labels')
    for path in reference.iterdir():
        if path.is_file():
            current=Path(dataset)/'vial_fill/labels'/path.name
            if sha(current)!=sha(path):raise ValueError(f'Organizer reference changed: {path.name}')
            counts['organizer_files']+=1
    videos={s['source']['video_relative_path']:s['source']['source_video_sha256'] for s in catalog.samples.values() if s['source']['kind']=='video_frame'}
    for relative,expected in videos.items():
        if sha(Path(dataset)/relative)!=expected:raise ValueError('Source video differs from extracted frame provenance')
        counts['source_videos']+=1
    return dict(status='passed',**counts,all_machine_suggestions_remain_unreviewed=True,
        claim='Coverage/integrity only; this does not measure annotation accuracy or approve timing.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True,type=Path);p.add_argument('--prelabels',required=True,type=Path)
    p.add_argument('--cases',type=Path);p.add_argument('--out',required=True,type=Path)
    args=p.parse_args();result=verify(args.dataset,args.prelabels,args.cases)
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
