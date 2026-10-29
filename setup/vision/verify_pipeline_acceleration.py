"""Verify the measured pipeline, media and unreviewed human-review package."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

import av
import cv2
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from liquid_specialists_data import read,write,sha
from compare_pipeline_backends import rows,distribution,frame_difference,summarize_differences


def artifacts(folder,marker):
    n=0
    for name,digest in marker['artifacts'].items():
        if sha(folder/name)!=digest:raise ValueError('Artifact changed: '+str(folder/name))
        n+=1
    return n


def main(a):
    if a.out.exists():raise ValueError('Use a fresh verification path')
    root=a.root;experiment=root/'full-run4-v1';analysis=root/'comparison-v1'
    package=root/'early-fill-review-v1';media=root/'media-v1'
    plan=read(experiment/'plan.json');complete=read(experiment/'completed.json')
    report=read(analysis/'comparison.json');media_done=read(media/'completed.json')
    count=artifacts(root/'engine-640-v1',read(root/'engine-640-v1/completed.json'))
    count+=artifacts(analysis,read(analysis/'completed.json'))
    count+=artifacts(media,media_done)
    for name,digest in complete['helper_sha256'].items():
        if sha(experiment/name)!=digest:raise ValueError('Executed pipeline helper changed')
        count+=1
    if sha(experiment/'plan.json')!=complete['plan_sha256'] or sha(experiment/'completed.json')!=report['input_completion_sha256']:
        raise ValueError('Pipeline identity differs')
    prior_done=read(a.previous/'completed.json')
    if sha(a.previous/'frames.jsonl')!=prior_done['artifacts']['frames.jsonl']:
        raise ValueError('Original evidence changed')
    prior=rows(a.previous/'frames.jsonl');data={}
    for key,digest in complete['runs'].items():
        if sha(experiment/key/'completed.json')!=digest:raise ValueError('Run marker differs')
        count+=artifacts(experiment/key,read(experiment/key/'completed.json'))
        rr=rows(experiment/key/'frames.jsonl');data[key]=rr
        if len(rr)!=1581:raise ValueError('Incomplete pipeline sample set')
        marker=read(experiment/key/'completed.json')
        if sum(r['measurement']['measurement_valid'] for r in rr)!=marker['counts']['qa_pass']:
            raise ValueError('Sample QA count differs')
        for r,old in zip(rr,prior):
            if any(r[k]!=old[k] for k in ('source_frame','source_rgb_sha256','pts','time_base')):
                raise ValueError('Source identity differs')
            if r['hardware_command']!='none' or r['recipe_decision']['status']!='unavailable' or r['recipe_decision']['hardware_action_authorized']:
                raise ValueError('Unexpected operational decision')
    # Independently decode the native recording once and check all 1,581 RGBs.
    expected={r['source_frame']:r for r in prior};source_samples=0;native_frames=0
    if sha(a.source)!=plan['source_sha256']:raise ValueError('Wrong source video')
    with av.open(str(a.source)) as c:
        for index,f in enumerate(c.decode(video=0)):
            native_frames+=1
            if index not in expected:continue
            r=expected[index];rgb=f.to_ndarray(format='rgb24')
            if hashlib.sha256(rgb.tobytes()).hexdigest()!=r['source_rgb_sha256'] or f.pts!=r['pts'] or str(f.time_base)!=r['time_base']:
                raise ValueError('Native source pixels or PTS differ')
            source_samples+=1
    control=data['pytorch-pass0']
    control_summary=dict(compared_frames=len(control),
        measurement_exact_matches=sum(l['measurement']==r['measurement'] for l,r in zip(control,prior)),
        raw_masks_exact_matches=sum(l['raw_masks']==r['raw_masks'] for l,r in zip(control,prior)),
        fill_state_exact_matches=sum(l['fill_detection']==r['fill_detection'] for l,r in zip(control,prior)))
    if control_summary!=report['original_pytorch_control']:raise ValueError('Original control comparison differs')
    for backend in ('pytorch','tensorrt'):
        values=[r for key,rr in data.items() if key.startswith(backend+'-') for r in rr]
        for metric in ('processing','decode_plus_processing','vial'):
            found=distribution([r['timing_ms'][metric] for r in values])
            if found!=report['timing'][backend][metric]:raise ValueError('Latency aggregate differs')
    lower_tail={}
    for repeat in range(2):
        differences=[frame_difference(l,r) for l,r in zip(data[f'pytorch-pass{repeat}'],data[f'tensorrt-pass{repeat}'])]
        if summarize_differences(differences)!=report['backend_agreement'][f'pass{repeat}']:
            raise ValueError('Backend output comparison differs')
        lower_tail[f'pass{repeat}']={}
        for kind in ('raw_mask','final_mask'):
            values=[d[kind]['iou_both_nonempty'] for d in differences if d[kind]['both_nonempty']]
            lower_tail[f'pass{repeat}'][kind]=dict(both_nonempty_frames=len(values),
                iou_min=float(min(values)) if values else None,
                iou_p05=float(np.percentile(values,5)) if values else None,
                iou_median=float(np.median(values)) if values else None)
    for backend in ('pytorch','tensorrt'):
        differences=[frame_difference(l,r) for l,r in zip(data[f'{backend}-pass0'],data[f'{backend}-pass1'])]
        if summarize_differences(differences)!=report['repeat_agreement'][backend]:raise ValueError('Repeat comparison differs')
    cases=read(package/'cases.json')['cases'];mask_review=read(package/'mask-review.json')
    onsets=read(package/'onset-review.json')['cycles'];package_manifest=read(package/'package-hashes.json')
    for name,digest in package_manifest.items():
        if sha(package/name)!=digest:raise ValueError('Review package changed')
        count+=1
    if len(cases)!=72 or len(onsets)!=6:raise ValueError('Unexpected review batch inventory')
    for case in cases:
        if case['review_status']!='unreviewed' or case['reviewed_label'] is not None:raise ValueError('Unexpected approved review')
        source=case['source'];im=cv2.imread(str(package/source['image_relative_path']))
        rgb=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
        if hashlib.sha256(rgb.tobytes()).hexdigest()!=source['source_rgb_sha256']:raise ValueError('Review PNG changed source pixels')
    if any(t['reviewed_mask_path'] is not None or t['reviewer'] is not None for t in mask_review['tasks']):raise ValueError('Unexpected mask adjudication')
    if any(o['reviewed_first_visible_frame'] is not None for o in onsets):raise ValueError('Unexpected onset adjudication')
    archive=root/'early-fill-review-v1.zip'
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise ValueError('Review ZIP failed CRC')
        if json.loads(z.read('package-hashes.json'))!=package_manifest:raise ValueError('ZIP manifest differs')
        for name,digest in package_manifest.items():
            if hashlib.sha256(z.read(name)).hexdigest()!=digest:raise ValueError('Review ZIP file differs')
    clips=0;clip_frames=0
    for p in (package/'clips').glob('*.mp4'):
        with av.open(str(p)) as c:decoded=sum(1 for _ in c.decode(video=0))
        if decoded!=240:raise ValueError('Incomplete raw review clip')
        clips+=1;clip_frames+=decoded
    decoded_media=[]
    for video in media_done['videos']:
        with av.open(str(media/video['path'])) as c:decoded=sum(1 for _ in c.decode(video=0))
        if decoded!=video['decoded_frames']:raise ValueError('Incomplete pipeline application video')
        trace=read(media/(Path(video['path']).stem+'-source-trace.json'))
        if len(trace)!=decoded:raise ValueError('Incomplete media source trace')
        for frame in trace:
            actual=expected[frame['source_frame']]
            if any(frame[k]!=actual[k] for k in ('source_rgb_sha256','pts','time_base')):raise ValueError('Media trace differs')
        decoded_media.append(dict(path=video['path'],frames=decoded))
    published=root/'published-media-v1';encoding=read(published/'encoding.json');published_checks=[]
    for name,digest in encoding['files'].items():
        if sha(published/name)!=digest:raise ValueError('Published media changed')
        count+=1
    for video in encoding['videos']:
        if sha(media/video['path'])!=video['source_sha256']:raise ValueError('Encoded media source differs')
        with av.open(str(published/video['path'])) as c:
            if c.streams.video[0].codec_context.name!='h264':raise ValueError('Expected H.264 media')
            decoded=sum(1 for f in c.decode(video=0) if (f.width,f.height)==(1280,720))
        if decoded!=video['decoded_frames']:raise ValueError('Incomplete published video decode')
        published_checks.append(dict(path=video['path'],frames=decoded,sha256=video['sha256']))
    result=dict(schema_version=1,kind='full_pipeline_acceleration_verification',
        artifacts_hash_verified=count,source_native_frames=native_frames,source_samples_rgb_pts_verified=source_samples,
        measured_pipeline_samples=sum(len(rr) for rr in data.values()),
        comparison_sha256=sha(analysis/'comparison.json'),engine_sha256=plan['engine_sha256'],
        median_latency_speedup=report['decode_plus_processing_median_speedup'],
        timing=report['timing'],backend_agreement=report['backend_agreement'],
        mask_agreement_lower_tail=lower_tail,
        repeat_agreement=report['repeat_agreement'],original_pytorch_control=report['original_pytorch_control'],
        early_fill_review=dict(cycles=len(onsets),exact_pngs=len(cases),decoded_clips=clips,decoded_clip_frames=clip_frames,
            zip_sha256=sha(archive),zip_bytes=archive.stat().st_size,approved_onsets=0,approved_masks=0,
            existing_workspaces_modified=False),pipeline_videos=decoded_media,published_videos=published_checks,
        csi_plan_sha256=sha(root/'CSI-VALIDATION-PLAN.md'),csi_request_sent=False,
        reviewed_onset_accuracy=None,reviewed_mask_accuracy=None,real_recipe_accuracy=None,
        promoted=False,run5_used_for_fitting_or_decisions=False,hardware_actions=0,
        verifier_sha256=sha(Path(__file__)))
    shutil.copy2(__file__,a.out.with_name(a.out.stem+'-helper.py'))
    write(a.out,result)
    print(json.dumps({k:result[k] for k in ('artifacts_hash_verified','source_native_frames','source_samples_rgb_pts_verified','measured_pipeline_samples','median_latency_speedup','early_fill_review')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','source','previous','out'):p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
