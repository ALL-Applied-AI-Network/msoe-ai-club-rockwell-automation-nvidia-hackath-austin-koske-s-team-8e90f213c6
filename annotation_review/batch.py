"""Prioritized case packaging and exact decoded-frame extraction, never clock joins."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import zipfile
import hashlib

from .core import Catalog, sha

CASE_SCHEMA='first-vial-review-cases/v1'
QA_SCHEMA='first-vial-replay-diagnostics/v1'
PEOPLE=('Aleyna','Zoe','Reagan')


def append_replay_qa(manifest_path,dataset,batch):
    """Adapter for the replay owner's published measurement_qa_review v1."""
    manifest_path=Path(manifest_path).resolve();batch=Path(batch).resolve()
    doc=json.loads(manifest_path.read_text())
    if doc.get('artifact_kind')!='measurement_qa_review' or doc.get('schema_version')!=1:
        raise ValueError('Expected published measurement_qa_review schema_version 1')
    source=doc['source']
    if source['camera']!='vial_fill' or source['run_id']!=5:
        raise ValueError('Initial diagnostic adapter requires Run 5 vial_fill')
    video=Path(dataset)/'vial_fill/videos/vial_fill_video_5.mp4'
    if sha(video)!=source['video_sha256']:
        raise ValueError('Replay diagnostic video hash mismatch')
    result=[];copies=[]
    rejected=[c for c in doc['cases'] if c['measurement_available'] is False]
    for n,case in enumerate(rejected):
        if case.get('review_status')!='unreviewed' or case.get('reviewed_label') is not None:
            raise ValueError('Expected unreviewed generated diagnostics, not adjudicated labels')
        detection=case['detection_id'];key=hashlib.sha256(detection.encode()).hexdigest()[:12]
        for rep in case['representative_frames']:
            frame=rep['frame_index']
            if type(frame)!=int or not 0<=frame<source['frame_count']:
                raise ValueError('Diagnostic frame outside declared source recording')
            if rep['role'] not in ('window_sample','departure'):
                raise ValueError('Unknown diagnostic frame role')
            original=(manifest_path.parent/rep['path']).resolve()
            if not original.is_relative_to(manifest_path.parent) or sha(original)!=rep['sha256']:
                raise ValueError('Unsafe or changed diagnostic representative image')
            name=f'diagnostics/{key}_frame{frame:06d}_{rep["role"]}.jpg'
            dest=batch/name;copies.append((original,dest))
            result.append(dict(sample_id=f'qa_run5_{key}_frame{frame}_{rep["role"]}',priority=50,
                cohort='run5_rejected_qa',assignee=PEOPLE[n%3],qa_triage_owner='Aleyna',
                reason='Rejected replay window '+detection+' / '+rep['role']+': '+', '.join(case['rejection_reasons']),
                source=dict(kind='video_frame',camera='vial_fill',run='5',filename=dest.name,
                    image_relative_path=name,sha256=rep['sha256'],width=None,height=None,
                    dimension_status='Not included in diagnostic manifest; browser reads original JPEG dimensions',
                    video_relative_path='vial_fill/videos/vial_fill_video_5.mp4',source_video_sha256=source['video_sha256'],
                    frame_index=frame,video_pts_seconds=None,video_pts=None,time_base=None,capture_timestamp=None,
                    export_reported_mp4_seconds=rep['mp4_s'],timestamp_provenance='replay diagnostic MP4 clock; native PTS not provided',
                    clock_status='No still association or physical-time conversion'),
                context_references=dict(kind='measurement_qa_review/v1',manifest_sha256=sha(manifest_path),
                    detection_id=detection,config_id=doc['config_id'],implementation_id=doc['implementation_id'],
                    replay_url=rep['replay_url'],representative=rep,diagnostic_case_snapshot=case,
                    label_status='unreviewed; QA rejection is not a product-quality label or timing approval')))
    document=json.loads((batch/'cases.json').read_text())
    existing={c['sample_id'] for c in document['cases']}
    prior={c['sample_id']:c for c in document['cases']}
    for case in result:
        if case['sample_id'] in prior and prior[case['sample_id']]!=case:
            raise ValueError('Diagnostic case changed: use a fresh batch to preserve old review provenance')
    for original,dest in copies:
        dest.parent.mkdir(exist_ok=True);shutil.copyfile(original,dest)
    target=batch/'qa-source';target.mkdir(exist_ok=True)
    shutil.copyfile(manifest_path,target/'manifest.json')
    document['cases'].extend(c for c in result if c['sample_id'] not in existing)
    document['qa_manifest_status']='imported measurement_qa_review/v1'
    document['qa_manifest']=dict(relative_path='qa-source/manifest.json',sha256=sha(manifest_path),
        config_id=doc['config_id'],implementation_id=doc['implementation_id'],rejected_windows=len(rejected),representative_frames=len(result))
    (batch/'cases.json').write_text(json.dumps(document,indent=2)+'\n')
    return document['qa_manifest']


def diagnostic_cases(path,dataset):
    """Explicit external adapter: input fields are documented in CONTRACTS.md."""
    doc=json.loads(Path(path).read_text())
    if doc.get('schema')!=QA_SCHEMA:
        raise ValueError('Unsupported diagnostics schema; adapt the replay export explicitly before import')
    cases=[]
    for row in doc['cases']:
        if row['camera']!='vial_fill' or str(row['run'])!='5':
            raise ValueError('Initial QA adapter accepts Run 5 vial_fill cases only')
        video=Path(dataset)/'vial_fill/videos/vial_fill_video_5.mp4'
        if row['source_video_sha256']!=sha(video):
            raise ValueError('Diagnostic source video differs from configured dataset')
        indices=row.get('frame_indices')
        if not isinstance(indices,list) or not indices or any(type(i)!=int or i<0 for i in indices):
            raise ValueError('Diagnostic cases need explicit nonnegative decoded frame_indices; no time-to-frame conversion')
        for n,frame in enumerate(indices):
            cases.append(dict(sample_id=f'qa_{row["case_id"]}_f{frame}',camera='vial_fill',run='5',frame_index=frame,
                priority=50,assignee=PEOPLE[n%3],reason='Replay diagnostic: '+str(row['reason']),
                context_references=dict(kind='replay_diagnostic_export',manifest_sha256=sha(path),
                    diagnostic_case_id=row['case_id'],measurement_decision_owner='replay agent',
                    snapshot=row,review_status='unreviewed_navigation_context')))
    return cases


def extract_frames(dataset,out,cases):
    try:import av
    except ImportError as exc:raise ValueError('Install annotation_review[video] to extract exact frames') from exc
    for run in sorted({str(c['run']) for c in cases if 'frame_index' in c}):
        group=[c for c in cases if str(c.get('run'))==run and 'frame_index' in c]
        wanted={c['frame_index'] for c in group}
        video=Path(dataset)/'vial_fill/videos'/f'vial_fill_video_{run}.mp4'
        video_hash=sha(video)
        found={}
        with av.open(str(video)) as container:
            stream=container.streams.video[0]
            for index,frame in enumerate(container.decode(stream)):
                if index not in wanted:continue
                name=f'frames/run{run}_frame{index:06d}.jpg';dest=out/name;dest.parent.mkdir(exist_ok=True)
                frame.to_image().save(dest,quality=95)
                found[index]=dict(kind='video_frame',camera='vial_fill',run=run,filename=dest.name,
                    image_relative_path=name,sha256=sha(dest),width=frame.width,height=frame.height,
                    video_relative_path=f'vial_fill/videos/vial_fill_video_{run}.mp4',source_video_sha256=video_hash,
                    frame_index=index,video_pts=frame.pts,time_base=str(frame.time_base),
                    video_pts_seconds=float(frame.pts*frame.time_base) if frame.pts is not None else None,
                    capture_timestamp=None,decoder='PyAV '+av.__version__,
                    clock_status='Decoded MP4 PTS; physical/wall time and still association not inferred')
                if len(found)==len(wanted):break
        missing=wanted-found.keys()
        if missing:raise ValueError(f'Requested frames beyond decoded source: {sorted(missing)}')
        if sha(video)!=video_hash:raise ValueError('Source video changed during extraction')
        for case in group:case['source']=found[case['frame_index']]


def build(dataset,prelabels,out,diagnostics=None):
    dataset,prelabels,out=Path(dataset).resolve(),Path(prelabels).resolve(),Path(out).resolve()
    if out.exists() and any(out.iterdir()):raise ValueError('Use a fresh batch output directory')
    out.mkdir(parents=True,exist_ok=True)
    catalog=Catalog(dataset,prelabels)
    stills=sorted([s for s in catalog.samples.values() if s['source']['camera']=='vial_fill' and s['source']['run']=='3'],key=lambda s:s['source']['filename'])
    cases=[]
    for n,sample in enumerate(stills):
        liquid=next((c for c in sample['candidates'] if c['class_id']==1),None)
        low=liquid and not liquid['machine_record'].get('draft')
        case=dict(sample_id=sample['id'],priority=10 if low else 20,assignee=PEOPLE[min(2,n*3//len(stills))],
            reason='Run 3 complete still sweep: inspect faint/partial/missed colored fills, including suppressed candidates.',
            cohort='run3_stills',association_status='unmatched')
        cases.append(case)
        dest=out/'data'/sample['source']['relative_path'];dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(catalog.path(sample),dest)
    # Supplied decoded frame bounds are navigation references, not reviewed timing.
    annotations=dataset/'vial_fill/labels/video5_vial_recipes.csv'
    with annotations.open(newline='',encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    groups={}
    for row in rows:groups.setdefault(row['recipe_id'],[]).append(row)
    for group_index,(recipe,group) in enumerate(groups.items()):
        for position,row in (('first',group[0]),('later',group[-1])):
            for location,frame in (('middle',(int(row['start_frame'])+int(row['end_frame']))//2),('provided_end',int(row['end_frame']))):
                cases.append(dict(sample_id=f'run5_provided_vial{int(row["vial"]):02d}_{location}',camera='vial_fill',run='5',frame_index=frame,
                    cohort='run5_transition_navigation',priority=100+group_index,assignee=PEOPLE[group_index%3],
                    reason=f'Provided recipe {recipe}, proposed {position} vial navigation anchor ({location}). Not a measurement window or reviewed timing decision.',
                    context_references=dict(kind='organizer_annotation_navigation_only',relative_path='vial_fill/labels/video5_vial_recipes.csv',
                        sha256=sha(annotations),provided_vial=row['vial'],provided_recipe_id=recipe,
                        provided_start_frame=int(row['start_frame']),provided_end_frame=int(row['end_frame']),
                        review_status='unreviewed_navigation_context')))
    if diagnostics:cases.extend(diagnostic_cases(diagnostics,dataset))
    extract_frames(dataset,out,cases)
    manifest=dict(schema=CASE_SCHEMA,cases=cases,priorities=['Run 3 stills first','Run 5 video navigation anchors second','Run 5 diagnostic cases when explicitly exported'],
        qa_manifest_status='imported' if diagnostics else 'pending; initial batch ships independently',
        limitations=['No still-to-video/vial conversion. All stills unmatched unless a separate reviewed mapping is loaded.',
            'Organizer annotations are navigation context, not mentee-approved timing or measurement decisions.',
            'No synthetic generation, detector training, color-threshold or replay behavior changes.'])
    (out/'cases.json').write_text(json.dumps(manifest,indent=2)+'\n')
    # Full saved prelabels stay available without copying full imagery.
    pack=prelabels/'review-pack' if (prelabels/'review-pack').is_dir() else prelabels
    target=out/'prelabels';target.mkdir()
    shutil.copyfile(prelabels/'manifest.json',target/'manifest.json')
    for camera in ('vial_fill','tank_level','cap_seal'):
        folder=target/camera;folder.mkdir()
        for filename in ('classes.txt','annotations.json'):shutil.copyfile(pack/camera/filename,folder/filename)
    references=out/'data/vial_fill/labels';references.mkdir(exist_ok=True)
    for path in (dataset/'vial_fill/labels').iterdir():
        if path.is_file():shutil.copyfile(path,references/path.name)
    snapshot={str(path.relative_to(dataset)):sha(path) for path in (dataset/'vial_fill/labels').iterdir() if path.is_file()}
    (out/'source-reference-hashes.json').write_text(json.dumps(snapshot,indent=2)+'\n')
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--prelabels',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--diagnostics',type=Path)
    args=p.parse_args();result=build(args.dataset,args.prelabels,args.out,args.diagnostics)
    print(json.dumps(dict(cases=len(result['cases']),qa_manifest_status=result['qa_manifest_status'])))


if __name__=='__main__':main()
