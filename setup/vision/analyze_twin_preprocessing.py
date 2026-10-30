"""Verify the adaptive saved-RGB preprocessing replay and publish its results."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import av
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from liquid_specialists_data import read, write, sha
from analyze_twin_experiments import close_values
from render_liquid_pipeline import pixel_hash
from replay.geometry import decode_mask
from twin_camera_preprocess import anchors, normalize
from twin_diagnostics import DiagnosticSuite


def main(a):
    if a.out.exists(): raise ValueError('Use a fresh analysis output')
    done=read(a.replay/'completed.json'); pinned=read(a.replay/'plan.json')
    original=read(a.analysis/'comparison.json')
    verified=read(a.verification/'verification.json')
    for rel,digest in verified['evidence_sha256'].items():
        if sha(a.source/rel)!=digest: raise ValueError('Original live evidence changed: '+rel)
    for name,digest in pinned['helpers_sha256'].items():
        if sha(a.replay/'helpers'/name)!=digest: raise ValueError('Replay helper changed')
    if sha(a.source/'observer/frames.jsonl')!=pinned['source_frames_sha256']: raise ValueError('Source changed')
    if sha(a.source/'completed.json')!=pinned['source_completion_sha256']: raise ValueError('Source completion changed')
    if sha(a.replay/'frames.jsonl')!=done['frames_sha256']: raise ValueError('Replay changed')
    if pinned['model_pins']!=read(a.source/'observer/models.json')['models']: raise ValueError('Model identity differs')
    # The exact correction and diagnostic implementation used in the worker is retained.
    for name in ['twin_camera_preprocess.py','twin_diagnostics.py']:
        if sha(Path(__file__).parent/name)!=pinned['helpers_sha256'][name]: raise ValueError('Verification helper differs')
    raw=[json.loads(s) for s in (a.source/'observer/frames.jsonl').read_text().splitlines()]
    rows=[json.loads(s) for s in (a.replay/'frames.jsonl').read_text().splitlines()]
    if len(raw)!=len(rows) or len(rows)!=done['frames']: raise ValueError('Replay is incomplete')
    source_plan=read(a.source/'plan.json'); reference=None; suite=DiagnosticSuite(); last=None
    cases=[]; group=[]; counts=Counter(); corrected_images={}; trace=[]
    def finish():
        if not group:return
        idx=int(group[0]['public_source']['episode_id'].split('-')[-1]); ep=source_plan['episodes'][idx]
        samples=[r['diagnostic_suite']['methods']['metered_raw'] for r in group if r['diagnostic_suite'] is not None]
        scores=[m['delta_e2000'] for m in samples if m['delta_e2000'] is not None]
        old=original['results'][idx]
        cases.append(dict(episode_id=last,case=ep['id'],group=ep['group'],frames=len(group),
            processed=sum(r['model_output'] is not None for r in group),
            blocked=sum(r['model_output'] is None for r in group),
            qa_pass=sum(bool(r['model_output'] and r['model_output']['measurement']['measurement_valid']) for r in group),
            comparisons=len(scores),warning=any(m['warning'] for m in samples),
            maximum_de2000=max(scores) if scores else None,
            first_warning_sim_s=next((r['public_source']['sim_time_s'] for r in group if r['diagnostic_suite'] and r['diagnostic_suite']['methods']['metered_raw']['warning']),None),
            original_qa_pass=old['qa_pass'],original=old['methods']['metered_normalized_guarded']))
    for before,r in zip(raw,rows):
        h=r['public_source'];eid=h['episode_id'];idx=int(eid.split('-')[-1]);ep=source_plan['episodes'][idx]
        if h!=before['public_source'] or r['raw_source_rgb_sha256']!=h['rgb_sha256']: raise ValueError('Source association differs')
        if eid!=last:
            finish();group=[];suite.reset();last=eid
        rgb=np.array(Image.open(a.source/'simulation/rgb'/f'{eid}-{h["frame_index"]:05d}.png').convert('RGB'))
        if pixel_hash(rgb)!=h['rgb_sha256']:raise ValueError('Raw pixel hash differs')
        if reference is None:reference=anchors(rgb)
        corrected,camera=normalize(rgb,reference)
        if not close_values(camera,r['camera_preprocessing']):raise ValueError('Correction metadata differs')
        if corrected is None:
            if any(r[k] is not None for k in ['processed_rgb_sha256','model_output','diagnostic_suite']):raise ValueError('Blocked camera has inference evidence')
            for state in suite.states.values():state['count']=0
            counts['blocked']+=1
        else:
            digest=pixel_hash(corrected);m=r['model_output']
            if digest!=r['processed_rgb_sha256'] or digest!=m['source_rgb_sha256']:raise ValueError('Processed model pixel identity differs')
            if m['source_frame']!=h['frame_index']*6 or m['pts']!=h['frame_index'] or m['time_base']!='1/5':raise ValueError('Processed clock differs')
            actual=suite.update(h,m,corrected,ep['reference'])
            if not close_values(actual,r['diagnostic_suite']):raise ValueError('Corrected diagnostic differs')
            counts['processed']+=1
        if r['operational_recipe_decision']['status']!='unavailable' or r['hardware_action']!='none':raise ValueError('Operational boundary changed')
        group.append(r)
        if idx in (8,19,13):corrected_images[(eid,h['frame_index'])]=corrected if corrected is not None else rgb
    finish()
    if not close_values(suite.export(),read(a.replay/'reference.json')['diagnostics']):raise ValueError('Reference differs')
    if not close_values(reference.tolist(),read(a.replay/'reference.json')['reference_background']):raise ValueError('Background reference differs')
    summary={}
    for group_name in ['nominal','residue','purge','appearance_probe']:
        cc=[c for c in cases if c['group']==group_name]
        summary[group_name]=dict(cases=len(cc),cases_with_comparisons=sum(c['comparisons']>0 for c in cc),
            warned=sum(c['warning'] for c in cc),comparisons=sum(c['comparisons'] for c in cc))
    a.out.mkdir(parents=True)
    report=dict(kind='adaptive_recorded_twin_preprocessing_followup',frames=len(rows),counts=dict(counts),
        independently_recomputed_camera_and_diagnostic_frames=len(rows),summary=summary,results=cases,
        source_verification_sha256=sha(a.verification/'verification.json'),
        replay_frames_sha256=sha(a.replay/'frames.jsonl'),model_pins=pinned['model_pins'],
        new_live_render=False,held_out=False,training=False,hardware_actions=0,
        limits=['Adaptive development replay on the same 21 cases, not an independent new live study.',
            'Model predictions are retained actual replay inference; this verifier recomputes pixels and diagnostics, not neural inference.',
            'No real defect/mask/onset accuracy or commissioned recipe references; all operational decisions unavailable.'])
    write(a.out/'comparison.json',report)
    lines=['# Correction before inference: adaptive follow-up','',
        '| Case | Original QA | Corrected QA | Blocked | Eligible comparisons | Warning |',
        '| --- | ---: | ---: | ---: | ---: | --- |']
    for c in cases:lines.append(f'| {c["case"]} | {c["original_qa_pass"]}/{c["frames"]} | {c["qa_pass"]}/{c["frames"]} | {c["blocked"]} | {c["comparisons"]} | {"Yes" if c["warning"] else "No"} |')
    (a.out/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    fig,ax=plt.subplots(figsize=(12,9));yy=np.arange(len(cases));width=.38
    ax.barh(yy-width/2,[c['original_qa_pass'] for c in cases],width,label='Original RGB',color='#8296a6')
    ax.barh(yy+width/2,[c['qa_pass'] for c in cases],width,label='Corrected RGB before frozen models',color='#3b9f82')
    ax.set_yticks(yy,labels=[c['case'] for c in cases],fontsize=8);ax.invert_yaxis();ax.legend(fontsize=9)
    ax.set(xlabel='QA-valid frame samples (32-46 attempted per case)',title='Background correction before inference recovers missing geometry')
    fig.text(.5,.015,'Adaptive replay of saved simulation frames; not held out. 16 unhealthy-anchor frames blocked. No real accuracy claim.',ha='center',fontsize=9)
    fig.subplots_adjust(left=.36,right=.98,bottom=.12,top=.94);fig.savefig(a.out/'preprocessing-scorecard.png',dpi=150);plt.close(fig)
    writer=cv2.VideoWriter(str(a.out/'corrected-pipeline-examples-raw.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),5.,(960,720))
    if not writer.isOpened():raise RuntimeError('Video writer failed')
    poster=False
    for r in rows:
        h=r['public_source'];key=(h['episode_id'],h['frame_index'])
        if key not in corrected_images:continue
        rgb=corrected_images[key];bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR);m=r['model_output']
        if m:
            mask=decode_mask(m['measurement']['geometry']['mask'],rgb.shape[:2]).astype(bool)
            bgr[mask]=(.5*bgr[mask]+.5*np.array([80,230,240])).astype(np.uint8)
            for p in m['predictions']:
                x,y,right,bottom=map(int,p['box']);cv2.rectangle(bgr,(x,y),(right,bottom),(240,170,55) if p['class_id']==0 else (70,230,80),1)
        image=np.full((720,960,3),(20,25,34),np.uint8);image[110:590,160:800]=bgr
        def line(s,y,size=.55,color=(230,235,240)):cv2.putText(image,s,(20,y),cv2.FONT_HERSHEY_SIMPLEX,size,color,1,cv2.LINE_AA)
        case=cases[int(h['episode_id'].split('-')[-1])]['case']
        line('RECORDED TWIN / CAMERA CORRECTION BEFORE FROZEN MODELS',30,.65)
        line(f'{case} | simulation {h["sim_time_s"]:.1f}s',65,.6)
        if m:
            d=r['diagnostic_suite']['methods']['metered_raw'];val='unavailable' if d['delta_e2000'] is None else f'{d["delta_e2000"]:.2f}'
            line(f'Sample QA: {"passes" if m["measurement"]["measurement_valid"] else "unavailable"} | color dE: {val}',620)
            line(f'Provisional diagnostic: {d["status"]}',648,color=(90,170,250) if d['warning'] else (230,235,240))
        else:
            line('CAMERA REFERENCE UNHEALTHY / inference and comparison blocked',620,color=(90,170,250))
            line('Raw RGB shown while blocked; no inferred liquid mask',648)
        line('Adaptive simulation replay; no real defect accuracy. Operational decision unavailable.',682,.51)
        line('No hardware actions. Source and corrected pixel hashes recorded separately.',708,.51)
        writer.write(image);trace.append(dict(output_frame=len(trace),episode_id=h['episode_id'],source_frame=h['frame_index'],
            raw_rgb_sha256=r['raw_source_rgb_sha256'],processed_rgb_sha256=r['processed_rgb_sha256']))
        if not poster and m and r['diagnostic_suite']['methods']['metered_raw']['warning']:
            cv2.imwrite(str(a.out/'corrected-pipeline-warning.png'),image);poster=True
    writer.release();decoded=0
    with av.open(str(a.out/'corrected-pipeline-examples-raw.mp4')) as container:
        for _ in container.decode(video=0):decoded+=1
    if decoded!=len(trace):raise ValueError('Incomplete video')
    write(a.out/'video-source-trace.json',trace)
    write(a.out/'completed.json',dict(frames_verified=len(rows),decoded_video_frames=decoded,
        artifacts={p.name:sha(p) for p in sorted(a.out.iterdir()) if p.is_file()}))
    print(json.dumps(dict(counts=dict(counts),summary=summary,video_frames=decoded)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['source','replay','analysis','verification','out']:p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
