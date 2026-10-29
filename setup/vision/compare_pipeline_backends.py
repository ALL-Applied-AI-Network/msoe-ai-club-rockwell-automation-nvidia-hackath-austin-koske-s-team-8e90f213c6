"""Compare full-pipeline timings, state, masks and colors without new labels."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from liquid_specialists_data import read, write, sha
from replay.geometry import decode_mask
from replay.measurement import delta_e
from benchmark_vial_tensorrt import compare
from analyze_stage2_evidence import checkpoints
from replay.config import load_config


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines()]


def distribution(values):
    return {'count':len(values), 'p50':float(np.median(values)) if values else None,
            'p95':float(np.percentile(values,95)) if values else None,
            'max':float(max(values)) if values else None}


def masks(left, right):
    a, b = decode_mask(left).astype(bool), decode_mask(right).astype(bool)
    an, bn = int(a.sum()), int(b.sum())
    both = an > 0 and bn > 0
    union = int(np.logical_or(a,b).sum())
    return dict(left_pixels=an, right_pixels=bn, identical=bool(np.array_equal(a,b)),
        both_nonempty=both, emptiness_equal=(an > 0) == (bn > 0),
        iou_both_nonempty=float(np.logical_and(a,b).sum()/union) if both else None,
        symmetric_difference_pixels=int(np.logical_xor(a,b).sum()))


def frame_difference(left, right):
    if any(left[k] != right[k] for k in ('source_frame','source_rgb_sha256','pts','time_base')):
        raise ValueError('Backend sources differ')
    lm, rm = left['measurement'], right['measurement']
    lg, rg = lm['geometry'], rm['geometry']
    vial = compare([p['box']+[p['confidence'],0] for p in left['predictions'] if p['class_id']==0],
                   [p['box']+[p['confidence'],0] for p in right['predictions'] if p['class_id']==0])
    raw = masks(left['raw_masks'][0] if left['raw_masks'] else None,
                right['raw_masks'][0] if right['raw_masks'] else None)
    final = masks(lg['mask'],rg['mask'])
    both_valid = lm['measurement_valid'] and rm['measurement_valid']
    color_delta = delta_e(lm['lab'],rm['lab']) if both_valid else None
    lc, rc = left['camera_feature'].get('lab'),right['camera_feature'].get('lab')
    return dict(source_frame=left['source_frame'], seconds=left['seconds'], vial=vial,
        liquid_predictions_identical=[p for p in left['predictions'] if p['class_id']==1]==[p for p in right['predictions'] if p['class_id']==1],
        raw_mask=raw, final_mask=final, both_sample_qa_pass=both_valid,
        sample_qa_equal=lm['measurement_valid']==rm['measurement_valid'],
        quality_flags_equal=lm['quality_flags']==rm['quality_flags'],
        track_state_equal=(lg['track_id'],lg['track_age'])==(rg['track_id'],rg['track_age']),
        fill_state_equal=left['fill_detection']==right['fill_detection'],
        bridge_mode_equal=left['bridge']['mode']==right['bridge']['mode'],
        camera_health_equal=left['camera_feature'].get('healthy')==right['camera_feature'].get('healthy'),
        eligible_raw_color_de2000=color_delta,
        eligible_normalized_color_de2000=delta_e(lc,rc) if both_valid and lc is not None and rc is not None else None,
        recipe_status_equal=left['recipe_decision']['status']==right['recipe_decision']['status'],
        left_sample_valid=lm['measurement_valid'],right_sample_valid=rm['measurement_valid'])


def summarize_differences(rr):
    summary = {'frames':len(rr)}
    for key in ('liquid_predictions_identical','sample_qa_equal','quality_flags_equal','track_state_equal',
                'fill_state_equal','bridge_mode_equal','camera_health_equal','recipe_status_equal'):
        summary[key] = sum(r[key] for r in rr)
    for name in ('raw_mask','final_mask'):
        values=[r[name] for r in rr]
        summary[name]=dict(identical_frames=sum(v['identical'] for v in values),
            emptiness_equal_frames=sum(v['emptiness_equal'] for v in values),
            both_nonempty_frames=sum(v['both_nonempty'] for v in values),
            iou_both_nonempty=distribution([v['iou_both_nonempty'] for v in values if v['both_nonempty']]),
            changed_pixels=distribution([v['symmetric_difference_pixels'] for v in values]))
    for key in ('eligible_raw_color_de2000','eligible_normalized_color_de2000'):
        summary[key]=distribution([r[key] for r in rr if r[key] is not None])
    pairs=[p for r in rr for p in r['vial']['pairs']]
    summary['vial']=dict(unmatched_torch=sum(r['vial']['unmatched_torch'] for r in rr),
        unmatched_tensorrt=sum(r['vial']['unmatched_tensorrt'] for r in rr),
        iou=distribution([p['iou'] for p in pairs]),
        confidence_abs_difference=distribution([p['confidence_abs_difference'] for p in pairs]))
    summary['qa_disagreements']=[{k:r[k] for k in ('source_frame','seconds','left_sample_valid','right_sample_valid')} for r in rr if not r['sample_qa_equal']]
    return summary


def main(a):
    if a.out.exists():
        raise ValueError('Use a fresh comparison directory')
    complete, plan = read(a.input/'completed.json'),read(a.input/'plan.json')
    if sha(a.input/'plan.json') != complete['plan_sha256']:
        raise ValueError('Pipeline plan changed')
    prior_done=read(a.previous/'completed.json')
    if sha(a.previous/'frames.jsonl') != prior_done['artifacts']['frames.jsonl']:
        raise ValueError('Original pipeline evidence changed')
    a.out.mkdir(parents=True)
    shutil.copy2(__file__,a.out/Path(__file__).name)
    data={};markers={}
    for key,digest in complete['runs'].items():
        folder=a.input/key
        if sha(folder/'completed.json') != digest:
            raise ValueError('Run completion changed')
        marker=read(folder/'completed.json');markers[key]=marker
        for name,expected in marker['artifacts'].items():
            if sha(folder/name) != expected:raise ValueError('Run evidence changed')
        data[key]=rows(folder/'frames.jsonl')
        if len(data[key]) != plan['samples_per_arm']:
            raise ValueError('Incomplete sample set')
    config=load_config(Path(__file__).resolve().parents[2]/'replay/examples/bootstrap-vision.json')
    timing={};parity={};per_pass={}
    for backend in ('pytorch','tensorrt'):
        collected=[r for key,rr in data.items() if key.startswith(backend+'-') for r in rr]
        timing[backend]={k:distribution([r['timing_ms'][k] for r in collected]) for k in
            ('vial','liquid','prompting','sam','measurement_and_state','result_encoding','processing','decode_and_rgb','decode_plus_processing')}
        timing[backend]['prompted_processing']=distribution([r['timing_ms']['processing'] for r in collected if r['timing_ms']['sam_prompted']])
        timing[backend]['unprompted_processing']=distribution([r['timing_ms']['processing'] for r in collected if not r['timing_ms']['sam_prompted']])
        per_pass[backend]=[]
        for repeat in range(plan['passes']):
            key=f'{backend}-pass{repeat}';rr=data[key]
            points=checkpoints(rr,config)
            write(a.out/f'{key}-checkpoints.json',points)
            per_pass[backend].append(dict(repeat=repeat,counts=markers[key]['counts'],
                sample_latency_ms=distribution([r['timing_ms']['decode_plus_processing'] for r in rr]),
                window_qa_pass=markers[key]['window_qa_pass'],completed_fill_candidates=markers[key]['completed_fill_candidates'],
                causal_phase_checkpoints=len(points)))
    for repeat in range(plan['passes']):
        left,right=data[f'pytorch-pass{repeat}'],data[f'tensorrt-pass{repeat}']
        diff=[frame_difference(l,r) for l,r in zip(left,right)]
        write(a.out/f'pass{repeat}-frame-differences.json',diff)
        parity[f'pass{repeat}']=summarize_differences(diff)
    original=rows(a.previous/'frames.jsonl')[:plan['samples_per_arm']]
    control=data['pytorch-pass0']
    reference=dict(compared_frames=len(control),
        measurement_exact_matches=sum(l['measurement']==r['measurement'] for l,r in zip(control,original)),
        raw_masks_exact_matches=sum(l['raw_masks']==r['raw_masks'] for l,r in zip(control,original)),
        fill_state_exact_matches=sum(l['fill_detection']==r['fill_detection'] for l,r in zip(control,original)))
    repeats={}
    if plan['passes']==2:
        for backend in ('pytorch','tensorrt'):
            rr=[frame_difference(l,r) for l,r in zip(data[f'{backend}-pass0'],data[f'{backend}-pass1'])]
            repeats[backend]=summarize_differences(rr)
    # These are system-behavior proxies, never physical first-visible timing.
    proxies=[]
    events=read(a.previous/'events.json')
    for e in events:
        if e['kind']!='fill_started':continue
        anchor=e['start_frame']; key=e['detection_id']
        entry=dict(image_detection_id=key, image_stream_anchor_frame=anchor,
            reviewed_first_visible_frame=None, physical_onset_delay_s=None, mask_accuracy=None)
        for backend in ('pytorch','tensorrt'):
            neighborhood=[r for r in data[f'{backend}-pass0'] if anchor-60<=r['source_frame']<=anchor+180]
            entry[backend]={name:next((r['source_frame'] for r in neighborhood if condition(r)),None) for name,condition in [
                ('first_liquid_proposal',lambda r:any(p['class_id']==1 for p in r['predictions'])),
                ('first_raw_mask',lambda r:bool(r['raw_masks'])),
                ('first_nonempty_sample',lambda r:r['measurement']['geometry']['mask_pixels']>0),
                ('first_sample_qa_pass',lambda r:r['measurement']['measurement_valid'])]}
        proxies.append(entry)
    ratio=timing['pytorch']['decode_plus_processing']['p50']/timing['tensorrt']['decode_plus_processing']['p50']
    report=dict(schema_version=1,kind='full_liquid_pipeline_comparison',source_run=4,
        samples_per_pass=plan['samples_per_arm'],passes=plan['passes'],timing=timing,
        decode_plus_processing_median_speedup=ratio,
        median_latency_reduction_percent=100*(1-1/ratio),
        per_pass=per_pass,backend_agreement=parity,repeat_agreement=repeats,
        original_pytorch_control=reference,early_fill_navigation_proxies=proxies,
        real_quality_accuracy=None,reviewed_onset_accuracy=None,reviewed_mask_iou=None,
        recipe_unavailable_samples=sum(r['recipe_decision']['status']=='unavailable' for rr in data.values() for r in rr),
        source_labels_used=False,run5_used=False,hardware_actions=0,deployed_defaults_changed=False,
        full_pipeline_live_models=True,timing_scope=plan['timing_scope'],
        input_completion_sha256=sha(a.input/'completed.json'),
        limits=['Inspected Run 4 development recording, not an independent accuracy test.',
            'Backend and repeated-run agreement are not agreement with human masks or physical onset truth.',
            'Sampled 5Hz local processing; capture, network, encode/disk and final aggregate reports excluded.',
            'Missing recipe/meter/reference/approved geometry keeps real quality decisions unavailable.'])
    write(a.out/'comparison.json',report)
    # Plot after GPU runs, outside the measured workload.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,stage,title in [(axes[0],'decode_plus_processing','Full sampled processing, including decode'),
                           (axes[1],'vial','Vial component only')]:
        values=[timing[b][stage]['p50'] for b in ('pytorch','tensorrt')]
        ax.bar(['PyTorch vial','TensorRT vial'],values,color=['#73859b','#159a8d'])
        ax.set_ylabel('Median milliseconds per sampled frame');ax.set_title(title)
        for i,v in enumerate(values):ax.text(i,v,f'{v:.2f} ms',ha='center',va='bottom')
        ax.set_ylim(0,max(values)*1.18)
    fig.suptitle('Same live liquid detector, SAM, prompts and measurement gates in both arms')
    fig.tight_layout();fig.savefig(a.out/'pipeline-scorecard.png',dpi=160);plt.close(fig)
    write(a.out/'completed.json',dict(input_completion_sha256=sha(a.input/'completed.json'),
        artifacts={p.name:sha(p) for p in a.out.iterdir() if p.is_file()}))
    print(json.dumps({k:report[k] for k in ('samples_per_pass','passes','decode_plus_processing_median_speedup','median_latency_reduction_percent','original_pytorch_control','recipe_unavailable_samples')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('input','previous','out'):p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
