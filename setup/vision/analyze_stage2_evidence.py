"""Causal phase checkpoints on frozen Run 4 evidence; no recipe inference."""
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

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from liquid_specialists_data import read,write,sha
from replay.config import load_config
from replay.geometry import decode_mask
from replay.measurement import delta_e
from render_liquid_pipeline import pixel_hash


def checkpoints(rows,config):
    """All inputs are current/past detector states. No last-episode lookup here."""
    pending=[];completed=[];identity=None;last_time=None;last_state=None;stop=None;emitted=False
    for r in rows:
        state=r['fill_detection'];m=r['measurement'];frame=r['source_frame']
        key=(state['detection_id'],state['episode'])
        if key!=identity:
            pending=[];stop=None;emitted=False;identity=key
        if state['phase']=='ingredient_pause' and last_state!='ingredient_pause':
            # Stream off is only known now; do not backdate availability to visual stop.
            stop=frame;pending=[];emitted=False
        if state['phase']!='ingredient_pause' or stop is None:
            pending=[]
        elif frame-stop>=round(config.window_start_after_end_s*30):
            contiguous=last_time is None or frame-last_time==6
            if not m['measurement_valid'] or not contiguous:pending=[]
            if m['measurement_valid']:
                pending.append(r);pending=pending[-config.minimum_window_samples:]
            if len(pending)==config.minimum_window_samples and not emitted:
                tracks={s['measurement']['geometry']['track_id'] for s in pending}
                labs=[s['measurement']['lab'] for s in pending]
                median=np.median(labs,axis=0)
                spread=max(delta_e(v,median) for v in labs)
                if len(tracks)==1 and spread<=config.max_temporal_de2000:
                    completed.append(dict(detection_id=state['detection_id'],episode=state['episode'],
                        available_frame=frame,available_s=r['seconds'],stream_off_available_frame=stop,
                        sample_frames=[s['source_frame'] for s in pending],lab=median.tolist(),
                        temporal_spread_de2000=spread,geometry_status='unreviewed_model_proposals',
                        scope='phase_checkpoint_only',recipe_assessment='unavailable',
                        final_dispense_complete_known=False,hardware_action_authorized=False))
                    emitted=True
        last_time=frame;last_state=state['phase']
    return completed


def main(a):
    if a.out.exists():raise ValueError('Use a fresh output directory')
    done=read(a.input/'completed.json')
    for n in ('frames.jsonl','observations.json','plan.json'):
        if sha(a.input/n)!=done['artifacts'][n]:raise ValueError('Frozen evidence changed')
    config=load_config(Path(__file__).resolve().parents[2]/'replay/examples/bootstrap-vision.json')
    rows=[json.loads(s) for s in (a.input/'frames.jsonl').read_text().splitlines()]
    observations=read(a.input/'observations.json');points=checkpoints(rows,config)
    a.out.mkdir(parents=True);summary=[]
    for o in observations:
        candidates=[c for c in points if c['detection_id']==o['detection_id']]
        # Retrospective diagnostic join only. The checkpoint generator never sees future episodes.
        final=[c for c in candidates if c['episode']==len(o['episodes'])]
        within=[r for r in rows if o['start_frame']<=r['source_frame']<o['departure_frame']]
        valid=[r for r in within if r['measurement']['measurement_valid']]
        summary.append(dict(detection_id=o['detection_id'],start_s=o['start_frame']/30,
            departure_s=o['departure_frame']/30,final_visual_stream_stop_s=o['end_frame']/30,
            episodes=len(o['episodes']),sample_qa_pass=len(valid),fill_samples=len(within),
            accepted_fraction=len(valid)/len(within),current_departure_window_valid=o['measurement_valid'],
            current_window_reasons=o['quality_flags'],causal_phase_checkpoints=len(candidates),
            final_episode_checkpoint=final[0] if final else None,
            lead_before_departure_s=o['departure_frame']/30-final[0]['available_s'] if final else None,
            first_accepted_after_image_start_s=valid[0]['seconds']-o['start_frame']/30 if valid else None))
    report=dict(schema_version=1,source_run=4,fill_candidates=len(summary),source_decoded_frames=done['source_decoded_frames'],
        sampled_frames=len(rows),whole_recording_sample_qa_pass=done['counts']['sample_qa_pass'],
        within_fill_sample_qa_pass=sum(s['sample_qa_pass'] for s in summary),within_fill_samples=sum(s['fill_samples'] for s in summary),
        current_completed_window_valid=sum(s['current_departure_window_valid'] for s in summary),
        causal_phase_checkpoints=len(points),final_episode_checkpoint_opportunities=sum(s['final_episode_checkpoint'] is not None for s in summary),
        phase_checkpoint_is_not_a_final_completion_signal=True,unknown_real_recipes=len(summary),
        actionable_real_recipe_decisions=0,hardware_actions=0,run5_used=False,per_fill=summary,
        limits=['Development-only, six unreviewed image fill candidates, no real defect labels.',
                'Final-episode identity is retrospective analysis, never an online policy input.',
                'Commissioned recipe/phase references and an authoritative dispense-complete signal remain necessary.',
                'QA gates unchanged; earlier phase checkpoint is a new experimental temporal policy.',
                'First sample delay is relative to image stream start, not reviewed liquid visibility onset.'])
    write(a.out/'phase-checkpoints.json',points);write(a.out/'metrics.json',report)
    times=np.array([r['seconds'] for r in rows]);labs=np.array([r['measurement']['lab'] or [np.nan]*3 for r in rows])
    valid=np.array([r['measurement']['measurement_valid'] for r in rows])
    fig,axes=plt.subplots(3,1,figsize=(13,8),sharex=True,gridspec_kw={'height_ratios':[2,1,1]})
    for k,c,label in [(0,'#333333','L*'),(1,'#a44765','a*'),(2,'#345a9e','b*')]:
        axes[0].plot(times,np.where(valid,labs[:,k],np.nan),color=c,label=label,lw=1.1)
    axes[0].legend(ncol=3);axes[0].set_ylabel('Camera Lab features')
    axes[0].set_title('Run 4: measured evidence availability; recipe assessments remain unavailable')
    axes[1].step(times,valid.astype(int),where='post',color='#167d9a');axes[1].set_ylabel('Frame QA');axes[1].set_yticks([0,1])
    for i,s in enumerate(summary):
        for ax in axes:ax.axvspan(s['start_s'],s['departure_s'],color='#63a699',alpha=.09)
        axes[0].text(s['start_s'],39,f"Fill {i+1}",fontsize=9)
        axes[2].plot([s['start_s'],s['departure_s']],[i+1]*2,lw=5,color='#bed6dd')
        for p in [c for c in points if c['detection_id']==s['detection_id']]:
            axes[2].scatter(p['available_s'],i+1,marker='D',color='#d18b19',s=32)
        axes[2].scatter(s['departure_s'],i+1,color='#13865a' if s['current_departure_window_valid'] else '#c34747',s=40)
    axes[2].set_ylabel('Fill candidate');axes[2].set_xlabel('Native source PTS (seconds)')
    axes[2].set_title('Gold: causal phase checkpoint. End dot: current departure-window QA (green/pass, red/fail).',fontsize=10)
    for ax in axes:ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(a.out/'run4-evidence-timeline.png',dpi=160);plt.close(fig)
    write(a.out/'verification.json',dict(metrics_sha256=sha(a.out/'metrics.json'),
        all_checkpoint_samples_are_past_or_current=all(max(c['sample_frames'])<=c['available_frame'] for c in points),
        no_checkpoint_claims_recipe_or_final_completion=all(not c['final_dispense_complete_known'] and c['recipe_assessment']=='unavailable' for c in points),
        input_completion_sha256=sha(a.input/'completed.json'),script_sha256=sha(Path(__file__))))
    print(json.dumps({k:v for k,v in report.items() if k not in ('per_fill','limits')},indent=2))
    print(json.dumps([dict(fill=s['detection_id'],lead_s=s['lead_before_departure_s']) for s in summary]))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
