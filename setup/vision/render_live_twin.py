"""Publish a verified live run's actual inference outputs with evaluator titles."""
import argparse
import json
from pathlib import Path

import av
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from liquid_specialists_data import read,write,sha


NAMES={'reference':'Nominal reference capture','normal-repeat':'Nominal repeat',
    'carryover':'Different shared-path residue','carryover-after-scripted-flush':'Same start + scripted 8 mL flush'}


def main(a):
    if a.out.exists():raise ValueError('Preserve existing media; use a new directory')
    report=read(a.verification/'verification.json')
    if report['live_frames_rgb_and_ack_verified']!=148:raise ValueError('This publishing layout expects the complete four-case pilot')
    for rel,expected in report['evidence_sha256'].items():
        if sha(a.run/rel)!=expected:raise ValueError('Evidence changed since verification: '+rel)
    rows=[json.loads(s) for s in (a.run/'observer/frames.jsonl').read_text().splitlines()]
    a.out.mkdir(parents=True)
    video=a.out/'live-digital-twin-pipeline-raw.mp4'
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),5.,(1280,720))
    if not writer.isOpened():raise RuntimeError('Video writer failed')
    count=0;trace=[];poster=None
    for ep in report['episodes']:
        expected=[r for r in rows if r['public_source']['episode_id']==ep['episode_id']]
        with av.open(str(a.run/'observer'/(ep['episode_id']+'.mp4'))) as container:
            for i,f in enumerate(container.decode(video=0)):
                image=f.to_ndarray(format='bgr24')
                cv2.rectangle(image,(0,44),(1279,95),(20,25,34),-1)
                name=NAMES[ep['scenario']]
                t=expected[i]['public_source']['sim_time_s']
                cv2.putText(image,f'{name} | simulation {t:.1f}s | scenario {len(trace)//37+1}/4',
                    (22,76),cv2.FONT_HERSHEY_SIMPLEX,.75,(225,235,245),1,cv2.LINE_AA)
                writer.write(image)
                trace.append(dict(output_frame=count,episode_id=ep['episode_id'],
                    source_frame=i,source_rgb_sha256=expected[i]['source_rgb_sha256']))
                count+=1
                if ep['scenario']=='carryover' and expected[i]['synthetic_color_diagnostic']['warning'] and poster is None:
                    poster=image.copy()
    writer.release()
    decoded=0
    with av.open(str(video)) as container:
        for f in container.decode(video=0):decoded+=1
    if count!=148 or decoded!=count:raise ValueError('Published sequence incomplete')
    if poster is not None:cv2.imwrite(str(a.out/'live-twin-provisional-warning.png'),poster)
    fig,(ax,qa)=plt.subplots(2,1,figsize=(11,7),gridspec_kw={'height_ratios':[2,1]})
    for ep in report['episodes'][1:]:
        rr=[r for r in rows if r['public_source']['episode_id']==ep['episode_id']]
        points=[r for r in rr if r['synthetic_color_diagnostic'].get('delta_e2000') is not None]
        ax.plot([r['seconds'] for r in points],[r['synthetic_color_diagnostic']['delta_e2000'] for r in points],
            marker='.',label=NAMES[ep['scenario']])
    ax.axhline(5,color='gray',linestyle='--',label='Experimental threshold (5 dE)')
    ax.set(xlabel='Simulation time (s)',ylabel='Measured color difference (dE2000)',
        title='Actual model measurements vs nominal rendered reference')
    ax.legend(fontsize=8);ax.grid(alpha=.2)
    qa.barh([NAMES[e['scenario']] for e in report['episodes']],
        [e['sample_qa_pass'] for e in report['episodes']],color=['#6b8eab','#479d80','#ce795b','#7a9b43'])
    qa.set(xlabel='QA-valid frame samples (37 attempted per scenario)',xlim=(0,37));qa.invert_yaxis()
    fig.suptitle('Live Isaac -> YOLO -> Faster R-CNN -> SAM -> unchanged sample QA',fontsize=13)
    fig.text(.5,.015,'One trajectory per condition; uncalibrated process/optics. No real defect accuracy or model-triggered flush.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.95));fig.savefig(a.out/'live-twin-scorecard.png',dpi=160);plt.close(fig)
    write(a.out/'source-trace.json',trace)
    write(a.out/'completed.json',dict(frames=count,decoded_frames=decoded,fps=5,
        verification_sha256=sha(a.verification/'verification.json'),
        artifacts={p.name:sha(p) for p in sorted(a.out.iterdir()) if p.is_file()}))
    print(json.dumps(dict(frames=count,video=str(video))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['run','verification','out']:p.add_argument('--'+n,type=Path,required=True)
    main(p.parse_args())
