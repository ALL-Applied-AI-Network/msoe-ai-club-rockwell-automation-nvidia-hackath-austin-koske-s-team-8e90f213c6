"""Independently verify retrieved generation and saved paired model evidence."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import av
import cv2

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from liquid_specialists_data import read,write,sha
from render_liquid_pipeline import pixel_hash
from replay.geometry import decode_mask
from replay.measurement import delta_e


def main(a):
    if a.out.exists():raise ValueError('Use a fresh verification file')
    bundle=a.root/'neutral-v4';manifest=read(bundle/'completed-neutral-artifacts-v1.manifest.json')
    for rel,item in manifest['files'].items():
        p=bundle/rel
        if not p.is_file() or p.stat().st_size!=item['size'] or sha(p)!=item['sha256']:raise ValueError('Retrieved evidence changed: '+rel)
    launch=read(bundle/'launch-neutral-v4.json')
    if launch['status']!='completed' or launch['exit_code'] or not launch['guardrail_success_logged_without_error']:raise ValueError('Generation did not succeed')
    log=bundle/'inference-neutral-v4.log'
    if sha(log)!=launch['log_sha256']:raise ValueError('Inference log changed')
    for directory in [a.review,a.evaluation]:
        for rel,digest in read(directory/'completed.json')['artifacts'].items():
            if sha(directory/rel)!=digest:raise ValueError('Analysis changed: '+rel)
    plan=read(a.evaluation/'plan.json');comparison=read(a.evaluation/'comparison.json')
    if plan['models']!=comparison['model_pins']:raise ValueError('Model identity differs')
    original=read(a.root/'inputs-v1/preparation.json')
    for rel,digest in original['artifacts'].items():
        if sha(a.root/'inputs-v1'/rel)!=digest or sha(bundle/'inputs-v1'/rel)!=digest:raise ValueError('Prepared input changed')
    paths={'source':a.root/'inputs-v1/source-nominal.mp4','generated':bundle/'outputs-neutral-v4/vial-neutral.mp4'}
    if sha(paths['source'])!=plan['input_sha256'] or sha(paths['generated'])!=plan['generated_sha256']:raise ValueError('Evaluated media changed')
    mapping=read(a.root/'inputs-v1/frame-map.json');wanted=plan['samples'];decoded={};saved={};summary={}
    expected_indices=[i for i,m in enumerate(mapping) if i==0 or m['source_frame']!=mapping[i-1]['source_frame']]
    if wanted!=expected_indices:raise ValueError('Sampling differs from frozen source mapping')
    for name,path in paths.items():
        rows=[json.loads(s) for s in (a.evaluation/(name+'-frames.jsonl')).read_text().splitlines()]
        counts=Counter();count=0;at=0
        with av.open(str(path)) as c:
            if c.streams.video[0].average_rate!=16:raise ValueError('Wrong video cadence')
            for i,f in enumerate(c.decode(video=0)):
                rgb=f.to_ndarray(format='rgb24')
                if rgb.shape!=(720,960,3):raise ValueError('Wrong image geometry')
                if i in wanted:
                    row=rows[at];at+=1
                    resized=cv2.resize(rgb,(640,480),interpolation=cv2.INTER_AREA)
                    if pixel_hash(resized)!=row['source_rgb_sha256'] or row['source_frame']!=i:raise ValueError('Model pixel identity differs')
                    if row['pts']!=f.pts or row['time_base']!=str(f.time_base) or abs(row['seconds']-float(f.pts*f.time_base))>1e-9:raise ValueError('Model clock differs')
                    if row['source_mapping']!=mapping[i]:raise ValueError('Synthetic frame lineage differs')
                    m=row['measurement'];mask=decode_mask(m['geometry']['mask'],(480,640))
                    if int(mask.sum())!=m['geometry']['mask_pixels']:raise ValueError('Saved mask pixel count differs')
                    counts.update(samples=1,qa_pass=int(m['measurement_valid']),vial_present=int(any(p['class_id']==0 for p in row['predictions'])),
                        liquid_present=int(any(p['class_id']==1 for p in row['predictions'])),nonempty_final_mask=int(m['geometry']['mask_pixels']>0))
                    if row['operational_recipe_decision']['status']!='unavailable':raise ValueError('Operational boundary changed')
                count+=1
        if count!=93 or at!=29 or len(rows)!=29:raise ValueError('Incomplete decoded sequence or model evidence')
        actual=dict(counts)
        actual['first_model_liquid_clip_s']=next((r['seconds'] for r in rows if any(p['class_id']==1 for p in r['predictions'])),None)
        actual['first_qa_valid_clip_s']=next((r['seconds'] for r in rows if r['measurement']['measurement_valid']),None)
        if actual!=comparison['summary'][name]:raise ValueError('Summary differs from retained model output')
        decoded[name]=count;saved[name]=rows;summary[name]=actual
    for x,y,pair in zip(saved['source'],saved['generated'],read(a.evaluation/'paired-model-disagreement.json')['samples']):
        mx=x['measurement'];my=y['measurement']
        aa=decode_mask(mx['geometry']['mask'],(480,640)).astype(bool);bb=decode_mask(my['geometry']['mask'],(480,640)).astype(bool)
        union=(aa|bb).sum();iou=float((aa&bb).sum()/union) if union else None
        score=float(delta_e(mx['lab'],my['lab'])) if mx['measurement_valid'] and my['measurement_valid'] else None
        for actual,expected in [(iou,pair['final_sampling_mask_iou']),(score,pair['raw_lab_de2000'])]:
            if actual is None and expected is None:continue
            if actual is None or expected is None or abs(actual-expected)>1e-8:raise ValueError('Paired diagnostic differs')
    media=[]
    for path in [paths['generated'],a.evaluation/'generated-inspection.mp4']:
        with av.open(str(path)) as c:
            stream=c.streams.video[0];frames=sum(1 for _ in c.decode(video=0))
            if frames!=93 or stream.average_rate!=16 or stream.codec_context.name!='h264':raise ValueError('Final H264 export incomplete')
        media.append(dict(path=str(path),sha256=sha(path),decoded_frames=frames,fps=16))
    report=dict(kind='verified_completed_cosmos_transfer_neutral_trial',generated_frames_verified=93,
        retrieved_files_verified=len(manifest['files']),paired_model_samples_pixel_and_clock_verified=58,
        source_and_generated_model_summary=summary,generation_elapsed_s=launch['elapsed_s'],
        guardrail_success_logged_without_error=True,final_media=media,
        model_pins=plan['models'],training_eligible=False,real_accuracy=None,
        disposition='Hold from training and automatic label inheritance; generated empty-frame liquid and hue changes flagged during visual inspection.',
        verification_scope='Recomputed source pixels, clocks, counts, mask sizes and paired diagnostics from retained actual inference; neural inference was not repeated.',
        evidence_sha256={str(p.relative_to(a.root)):sha(p) for p in [
            bundle/'completed-neutral-artifacts-v1.manifest.json',bundle/'local-retrieval-verification.json',
            a.review/'review.json',a.review/'completed.json',a.evaluation/'completed.json',
            a.evaluation/'comparison.json',a.evaluation/'paired-model-disagreement.json']})
    write(a.out,report);print(json.dumps(dict(decoded=decoded,saved_model_frames_verified=58,summary=summary)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['root','review','evaluation','out']:p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
