"""Independently join private twin truth to saved model outputs, after inference."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import sys

import av
import cv2
import numpy as np
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from replay.geometry import decode_mask
from liquid_specialists_data import read,write,sha
from twin_contract import PUBLIC_FIELDS,digest,episode_states


def stats(values):
    values=np.asarray(values,dtype=float)
    return dict(count=len(values),median=float(np.median(values)) if len(values) else None,
        minimum=float(values.min()) if len(values) else None,p95=float(np.percentile(values,95)) if len(values) else None,
        maximum=float(values.max()) if len(values) else None)


def main(a):
    if a.out.exists():raise ValueError('Use a new verification output directory')
    root=a.run
    done=read(root/'completed.json');simulation=read(root/'simulation/completed.json');observer=read(root/'observer/completed.json')
    if done['status']!='completed' or not simulation['source_assets_unchanged']:raise ValueError('Incomplete experiment')
    truth=[json.loads(s) for s in (root/'simulation/private-truth.jsonl').read_text().splitlines()]
    rows=[json.loads(s) for s in (root/'observer/frames.jsonl').read_text().splitlines()]
    if len(truth)!=len(rows):raise ValueError('Dropped model outputs or private truth')
    launch=read(root/'launch.json');plan=read(root/'plan.json')
    for name,expected in launch['helpers_sha256'].items():
        if sha(root/'helpers'/name)!=expected:raise ValueError('Helper snapshot changed')
    for name,expected in launch.get('dependency_sha256',{}).items():
        if sha(root/'helpers/dependencies'/name)!=expected:raise ValueError('Dependency snapshot changed')
    by=defaultdict(list);collage=[]
    for private,row in zip(truth,rows):
        h=row['public_source'];index=h['frame_index'];eid=h['episode_id']
        if set(h)!=PUBLIC_FIELDS or private['episode_id']!=eid or private['state']['frame_index']!=index:
            raise ValueError('Source association or public/private boundary failed')
        image=np.array(Image.open(root/'simulation'/private['rgb_file']).convert('RGB'))
        if digest(image.tobytes())!=h['rgb_sha256'] or h['rgb_sha256']!=row['source_rgb_sha256'] or h['rgb_sha256']!=private['rgb_sha256']:
            raise ValueError('RGB bytes differ between renderer and model evidence')
        if any(private['ack'][k]!=h[k] for k in ['episode_id','frame_index','rgb_sha256']):
            raise ValueError('Frame not acknowledged before next simulation step')
        if abs(h['sim_time_s']-row['seconds'])>1e-8:raise ValueError('Simulation clock mismatch')
        if row['recipe_decision']['status']!='unavailable' or row['simulated_control_command']!='none':
            raise ValueError('Operational evidence gate or no-control boundary changed')
        if 'declared_recipe_context' in row:
            context=row['declared_recipe_context']
            if any(context[k]!=h[k] for k in ['session_id','recipe_key','phase_key']):
                raise ValueError('Declared process context differs from live source')
            if row['recipe_decision']['session_id']!=h['session_id'] or context['geometry_status']!='unreviewed_model_proposals':
                raise ValueError('Live source or geometry approval boundary changed')
        state=private['state']
        if abs(sum(state['actual_components_ml'])-sum(state['metered_ml']))>1e-7:raise ValueError('Mass balance failed')
        geometric=np.array(Image.open(root/'simulation'/private['mask_file']))>0
        if int(geometric.sum())!=private['geometric_mask_pixels']:raise ValueError('Geometric mask differs')
        if sum(state['actual_components_ml'])>1e-8 and not geometric.any():
            raise ValueError('Invalid private geometric mask for a present body; preserve this run as diagnostic')
        final=decode_mask(row['measurement']['geometry']['mask'],image.shape[:2]).astype(bool)
        union=np.logical_or(geometric,final).sum();overlap=np.logical_and(geometric,final).sum()
        predicted_present=bool(final.any());geometric_present=bool(geometric.any())
        record=dict(frame=index,simulation_s=h['sim_time_s'],actual_body_nonzero=sum(state['actual_components_ml'])>1e-8,
            geometric_present=geometric_present,predicted_present=predicted_present,
            qa_pass=bool(row['measurement']['measurement_valid']),
            sampling_mask_geometric_proxy_iou=float(overlap/union) if union else None,
            provisional=row['synthetic_color_diagnostic'],render_save_s=private['render_save_s'],
            inference_roundtrip_s=private['inference_roundtrip_s'],frame_wall_s=private['frame_wall_s'])
        by[eid].append(record)
    report=[]
    for i,(episode,summary) in enumerate(zip(plan['episodes'],simulation['episodes'])):
        eid=summary['episode_id'];records=by[eid]
        expected=list(episode_states(plan,episode,launch['limit_frames']))
        actual=[t for t in truth if t['episode_id']==eid]
        if [t['state'] for t in actual]!=expected:raise ValueError('Process trace does not reproduce the frozen plan')
        if len(records)!=len(expected):raise ValueError('Incomplete scenario')
        first_body=next((r['simulation_s'] for r in records if r['actual_body_nonzero']),None)
        first_pass=next((r['simulation_s'] for r in records if r['qa_pass']),None)
        comparisons=[r['provisional']['delta_e2000'] for r in records if r['provisional'].get('delta_e2000') is not None]
        video=root/'observer'/(eid+'.mp4');decoded=0
        with av.open(str(video)) as container:
            if container.streams.video[0].average_rate!=5:raise ValueError('Changed video cadence')
            for frame in container.decode(video=0):
                if (frame.width,frame.height)!=(1280,720):raise ValueError('Wrong demo dimensions')
                decoded+=1
                if decoded==len(records):collage.append(frame.to_ndarray(format='bgr24'))
        if decoded!=len(records):raise ValueError('Incomplete demo video')
        report.append(dict(episode_id=eid,scenario=episode['id'],reference=episode['reference'],frames=len(records),
            complete_dispense=expected[-1]['dispense_complete'],
            frames_with_geometric_body=sum(r['geometric_present'] for r in records),
            frames_with_sampling_mask=sum(r['predicted_present'] for r in records),
            sample_qa_pass=sum(r['qa_pass'] for r in records),
            first_nonzero_body_sim_time_s=first_body,first_valid_sample_sim_time_s=first_pass,
            delay_from_geometric_body_s=first_pass-first_body if first_body is not None and first_pass is not None else None,
            reviewed_first_visible_delay_s=None,
            mask_iou_against_geometric_proxy=stats([r['sampling_mask_geometric_proxy_iou'] for r in records if r['geometric_present'] and r['predicted_present']]),
            provisional_color_comparisons=len(comparisons),provisional_delta_e2000=stats(comparisons),
            provisional_warning=any(r['provisional']['warning'] for r in records),
            first_provisional_warning_sim_time_s=next((r['simulation_s'] for r in records if r['provisional']['warning']),None),
            final_components_ml=expected[-1]['actual_components_ml'],scripted_flush_ml=episode['flush_ml'],
            native_video_frames_decoded=decoded,video_sha256=sha(video)))
    a.out.mkdir(parents=True)
    wall=stats([t['frame_wall_s'] for t in truth])
    result=dict(schema_version=1,kind='verified_live_isaac_camera_to_frozen_models',
        source_run=str(root),live_frames_rgb_and_ack_verified=len(rows),
        private_composition_fields_received_by_observer=0,episodes=report,
        per_frame_wall_s=wall,inference_roundtrip_s=stats([t['inference_roundtrip_s'] for t in truth]),
        render_and_save_s=stats([t['render_save_s'] for t in truth]),
        operational_recipe_decisions=dict(Counter(r['recipe_decision']['status'] for r in rows)),
        operational_unavailable_reasons=dict(Counter(r['recipe_decision']['reason'] for r in rows)),
        image_only_fill_states=dict(Counter(r['fill_detection']['phase'] for r in rows)),
        frames_with_image_fill_identity=sum(r['fill_detection']['detection_id'] is not None for r in rows),
        reference_knots=observer['reference_knots'],source_assets_unchanged=True,
        real_onset_accuracy=None,real_mask_accuracy=None,real_recipe_accuracy=None,hardware_actions=0,
        model_triggered_interventions=0,physical_parameters_calibrated=False,run5_used=False,
        evidence_sha256={str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*'))
            if p.is_file() and p.suffix in ('.json','.jsonl','.png','.mp4','.py')},
        limits=['Software-in-loop with backpressure, not a wall-clock real-time guarantee.',
                'Procedural mesh/materials and hypothesized mixed-compartment model, not calibrated CFD.',
                'Model sampling masks score against geometric proxy masks, not optically reviewed human masks.',
                'One rendered nominal reference and paired development scenarios, not a statistical performance study.',
                'Color comparison is provisional; operational geometry gate stays unavailable.',
                'Flush is a predeclared simulated operator intervention, not a model-triggered action.'])
    write(a.out/'verification.json',result)
    if collage:cv2.imwrite(str(a.out/'scenario-final-frames.png'),np.concatenate([cv2.resize(im,(640,360)) for im in collage],axis=0))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
