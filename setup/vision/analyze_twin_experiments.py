"""Verify and summarize the predeclared live-twin ablation study."""
import argparse
from collections import Counter,defaultdict
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

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from liquid_specialists_data import read,write,sha
from twin_diagnostics import DiagnosticSuite,METHODS,SETTINGS


def close_values(a,b):
    if isinstance(a,dict):return set(a)==set(b) and all(close_values(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return isinstance(b,(list,tuple)) and len(a)==len(b) and all(close_values(x,y) for x,y in zip(a,b))
    if isinstance(a,float):return b is not None and abs(a-b)<=1e-8
    return a==b


def main(a):
    if a.out.exists():raise ValueError('Use a fresh study-analysis directory')
    verified=read(a.verification/'verification.json');plan=read(a.run/'plan.json')
    for rel,digest in verified['evidence_sha256'].items():
        if sha(a.run/rel)!=digest:raise ValueError('Verified experiment changed: '+rel)
    rows=[json.loads(s) for s in (a.run/'observer/frames.jsonl').read_text().splitlines()]
    truth=[json.loads(s) for s in (a.run/'simulation/private-truth.jsonl').read_text().splitlines()]
    observed=read(a.run/'observer/completed.json')['episodes']
    groups=defaultdict(list);private=defaultdict(list)
    for r,t in zip(rows,truth):groups[r['public_source']['episode_id']].append(r);private[t['episode_id']].append(t)
    suite=DiagnosticSuite();results=[];recomputed=0
    for number,(ep,old) in enumerate(zip(plan['episodes'],observed)):
        eid=f'episode-{number:03d}';rr=groups[eid];tt=private[eid]
        suite.reset()
        for r,t in zip(rr,tt):
            rgb=np.array(Image.open(a.run/'simulation'/t['rgb_file']).convert('RGB'))
            actual=suite.update(r['public_source'],r,rgb,ep['reference'])
            if not close_values(actual,r['diagnostic_suite']):raise ValueError('Diagnostic recomputation differs: '+eid)
            if t['episode_parameters']!=ep:raise ValueError('Private case configuration differs from frozen plan')
            recomputed+=1
        methods={}
        for method in METHODS:
            samples=[r['diagnostic_suite']['methods'][method] for r in rr]
            scores=[s['delta_e2000'] for s in samples if s['delta_e2000'] is not None]
            first=next((r['seconds'] for r,s in zip(rr,samples) if s['warning']),None)
            methods[method]=dict(warning=first is not None,first_warning_sim_s=first,
                comparisons=len(scores),maximum_de2000=max(scores) if scores else None,
                median_de2000=float(np.median(scores)) if scores else None,
                unavailable_reasons=dict(Counter(s['reason'] for s in samples if s['reason'] and s['reason']!='building_reference')))
        signal=np.array([r['measurement']['stream_contrast'] for r in rr])
        active=np.array([t['state']['active_tank'] is not None for t in tt])
        starts=[e for e in old['image_events'] if e['kind']=='fill_started']
        first_active=next((r['seconds'] for r,is_active in zip(rr,active) if is_active),None)
        results.append(dict(episode_id=eid,case=ep['id'],group=ep['group'],parameters=ep,frames=len(rr),
            qa_pass=sum(r['measurement']['measurement_valid'] for r in rr),
            background_healthy=sum(r['diagnostic_suite']['background_healthy'] for r in rr),
            methods=methods,stream=dict(active_frames=int(active.sum()),
                active_above_start_threshold=int(((signal>=3.)&active).sum()),
                inactive_above_start_threshold=int(((signal>=3.)&~active).sum()),
                maximum_contrast=float(signal.max()),active_median_contrast=float(np.median(signal[active])) if active.any() else None,
                image_fill_starts=len(starts),image_fill_completed=len(old['image_completed_fills']),
                first_image_start_available_sim_s=starts[0]['video_s'] if starts else None,
                declared_first_flow_sim_s=first_active,
                start_availability_delay_from_flow_s=starts[0]['video_s']-first_active if starts and first_active is not None else None),
            final_components_ml=tt[-1]['state']['actual_components_ml']))
    if recomputed!=verified['live_frames_rgb_and_ack_verified']:raise ValueError('Incomplete diagnostic verification')
    summaries={}
    for method in METHODS:
        summaries[method]={}
        for group in ['nominal','residue','purge','appearance_probe']:
            cases=[r for r in results if r['group']==group]
            summaries[method][group]=dict(cases=len(cases),warned=sum(r['methods'][method]['warning'] for r in cases),
                cases_with_comparisons=sum(r['methods'][method]['comparisons']>0 for r in cases),
                comparisons=sum(r['methods'][method]['comparisons'] for r in cases))
    a.out.mkdir(parents=True)
    report=dict(schema_version=1,kind='live_twin_reference_camera_stream_ablation',cases=len(results),
        live_frames=recomputed,settings=SETTINGS,results=results,summary=summaries,
        source_verification_sha256=sha(a.verification/'verification.json'),
        independently_recomputed_feature_and_decision_frames=recomputed,
        operational_recipe_decisions=verified['operational_recipe_decisions'],hardware_actions=0,
        real_defect_accuracy=None,real_mask_accuracy=None,human_visibility_labels=None,
        limits=['Predeclared deterministic development cases, one trajectory per condition, not independent production trials.',
            'Initial residue is a known model perturbation, not a product acceptance criterion.',
            'Optical profiles are separate perception probes, not nominal quality controls or certified liquid properties.',
            'Three-frame warnings are provisional; deployed models, QA and stream thresholds are unchanged.',
            'No carrier departure is simulated, so completed-vial lifecycle cannot be validated by EOF.'])
    write(a.out/'comparison.json',report)
    # A complete, readable metric table accompanies the machine evidence.
    lines=['# Live twin experiment results','',f'{len(results)} cases / {recomputed} live model frames; one trajectory per condition.','',
        '| Case | QA passes | Exact raw | Metered raw | Normalized | Normalized + guard | Stream peak / image starts |',
        '| --- | ---: | --- | --- | --- | --- | ---: |']
    for r in results:
        cells=[]
        for method in METHODS:
            m=r['methods'][method]
            cells.append('Reference' if r['group']=='reference' else f'{"WARN" if m["warning"] else "No warning"}; {m["comparisons"]} comparisons')
        lines.append(f'| {r["case"]} | {r["qa_pass"]}/{r["frames"]} | '+ ' | '.join(cells)+f' | {r["stream"]["maximum_contrast"]:.2f} / {r["stream"]["image_fill_starts"]} |')
    (a.out/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    labels=[r['case'] for r in results if r['group']!='reference']
    grid=[]
    for r in results[1:]:
        grid.append([2 if r['methods'][m]['warning'] else 1 if r['methods'][m]['comparisons'] else 0 for m in METHODS])
    from matplotlib.colors import ListedColormap
    fig,(ax,stream)=plt.subplots(1,2,figsize=(14,10),gridspec_kw={'width_ratios':[1.5,1]})
    ax.imshow(grid,cmap=ListedColormap(['#b9bec7','#4aab86','#dd795a']),vmin=0,vmax=2,aspect='auto')
    ax.set_yticks(range(len(labels)),labels=labels,fontsize=8)
    ax.set_xticks(range(4),labels=['Exact raw','Metered raw','Normalized','Normalized\n+ guard'],fontsize=9)
    ax.set_title('Gray: no comparison | green: no warning | orange: warning',fontsize=10)
    for i,r in enumerate(results[1:]):
        for j,m in enumerate(METHODS):ax.text(j,i,str(r['methods'][m]['comparisons']),ha='center',va='center',fontsize=8)
    stream.barh(range(len(labels)),[r['stream']['maximum_contrast'] for r in results[1:]],color='#617fa0')
    stream.set_yticks(range(len(labels)),labels=['']*len(labels));stream.invert_yaxis()
    stream.axvline(3.,color='#a3732d',linestyle='--',label='Start threshold 3')
    stream.axvline(6.,color='#a83e3a',linestyle=':',label='Peak threshold 6')
    stream.set(xlabel='Maximum image stream contrast',title='Same stream detector / thresholds');stream.legend(fontsize=8)
    fig.suptitle('Live Isaac + frozen vision models / predeclared development study')
    fig.text(.5,.015,'Numbers in cells are eligible comparisons. Warnings are provisional; no real defect accuracy. Optical probes are a separate group.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.96));fig.savefig(a.out/'experiment-scorecard.png',dpi=160);plt.close(fig)
    # Complete four illustrative clips, selected by predeclared case identity.
    selected=['nominal-flow-slow','nominal-sensor-green-low','residue-slow-sensor-green-low','stream-wider-shorter-absorption']
    writer=cv2.VideoWriter(str(a.out/'twin-experiment-examples-raw.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),5.,(1280,720))
    if not writer.isOpened():raise RuntimeError('Study video writer failed')
    count=0;trace=[]
    for r in results:
        if r['case'] not in selected:continue
        rr=groups[r['episode_id']]
        with av.open(str(a.run/'observer'/(r['episode_id']+'.mp4'))) as container:
            for i,f in enumerate(container.decode(video=0)):
                image=f.to_ndarray(format='bgr24');row=rr[i]
                cv2.rectangle(image,(0,44),(1279,95),(20,25,34),-1)
                cv2.putText(image,f'{r["case"]} | simulation {row["seconds"]:.1f}s',(22,76),cv2.FONT_HERSHEY_SIMPLEX,.68,(230,235,240),1,cv2.LINE_AA)
                cv2.rectangle(image,(0,626),(1279,719),(20,25,34),-1)
                for j,(method,title) in enumerate([('exact_raw','Exact meter / raw'),('metered_normalized_guarded','Meter interpolation / normalized + guard')]):
                    m=row['diagnostic_suite']['methods'][method]
                    value='unavailable' if m['delta_e2000'] is None else f'dE {m["delta_e2000"]:.2f}'
                    text=f'{title}: {"REVIEW" if m["warning"] else m["status"]} / {value}'
                    cv2.putText(image,text,(22,650+j*23),cv2.FONT_HERSHEY_SIMPLEX,.55,(90,170,250) if m['warning'] else (225,235,240),1,cv2.LINE_AA)
                cv2.putText(image,'Uncalibrated development simulation. Operational quality unavailable; no model/hardware actions.',(22,706),cv2.FONT_HERSHEY_SIMPLEX,.53,(200,210,220),1,cv2.LINE_AA)
                writer.write(image);trace.append(dict(output_frame=count,episode_id=r['episode_id'],source_frame=i,source_rgb_sha256=row['source_rgb_sha256']));count+=1
    writer.release()
    with av.open(str(a.out/'twin-experiment-examples-raw.mp4')) as c:
        if sum(1 for _ in c.decode(video=0))!=count:raise ValueError('Study video incomplete')
    write(a.out/'video-source-trace.json',trace)
    write(a.out/'completed.json',dict(cases=len(results),frames_recomputed=recomputed,video_frames=count,
        artifacts={p.name:sha(p) for p in a.out.iterdir() if p.is_file()}))
    print(json.dumps(dict(cases=len(results),frames=recomputed,summary=summaries),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['run','verification','out']:p.add_argument('--'+n,type=Path,required=True)
    main(p.parse_args())
