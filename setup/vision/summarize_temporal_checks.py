"""CPU report/plots from completed and hash-verified GB10 candidate checks."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())


def main(a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    manifest=read(a.reports/'manifest.json')
    for name,digest in manifest['files'].items():
        if sha(a.reports/name)!=digest:raise ValueError('Collected report differs: '+name)
    synthetic=read(a.reports/'synthetic-stability/comparison.json')
    real=read(a.reports/'real-stability/comparison.json')
    bench=read(a.reports/'pipeline-benchmark/comparison.json')
    duration=read(a.reports/'duration-controls/comparison.json')
    a.out.mkdir(parents=True,exist_ok=False)
    names=['raw','current_exposure085','current_exposure070','current_exposure130','current_green085','reference_shift2','reference_shift5','other_empty_reference']
    labels=['Clean','Exposure .85','Exposure .70','Exposure 1.30','Green .85','Ref shift 2px','Ref shift 5px','Different ref']
    fig,axes=plt.subplots(2,1,figsize=(13,8),sharex=True,constrained_layout=True)
    colors={'single_rgb':'#8b8f96','paired_diff':'#ed9a30','paired_diff_gain':'#1686ae'}
    for mode in colors:
        rows=[synthetic[mode]['conditions'][k] for k in names]
        axes[0].plot(range(8),[100*r['positive_presence_detected']/r['positive_images'] for r in rows],marker='o',label=mode,color=colors[mode])
        axes[1].plot(range(8),[100*r['negative_presence_alarms']/r['negative_images'] for r in rows],marker='o',label=mode,color=colors[mode])
    axes[0].set_title('Frozen synthetic stress tests — 24 geometric positives / 24 negatives per condition')
    axes[0].set_ylabel('Presence detections (%)');axes[1].set_ylabel('Presence alarms on negatives (%)')
    for ax in axes:ax.set_ylim(-3,103);ax.grid(alpha=.2);ax.legend(loc='best')
    axes[1].set_xticks(range(8),labels,rotation=25,ha='right')
    fig.savefig(a.out/'synthetic-stability.png',dpi=160);plt.close(fig)
    real_names=names[:-1]+['late_reference'];real_labels=labels[:-1]+['Late ref']
    fig,axes=plt.subplots(2,1,figsize=(13,8),sharex=True,constrained_layout=True)
    for mode in ['paired_diff','paired_diff_gain']:
        rows=[real['models'][mode][k] for k in real_names]
        axes[0].plot(range(8),[r['clean_positive_mean_prediction_iou'] for r in rows],marker='o',label=mode,color=colors[mode])
        axes[1].plot(range(8),[r['presence_decision_flips'] for r in rows],marker='o',label=mode,color=colors[mode])
    axes[0].set_title('Digitally perturbed real clips — prediction stability, not human accuracy')
    axes[0].set_ylabel('Mask IoU with clean output\non clean accepted frames');axes[0].set_ylim(-.03,1.03)
    axes[1].set_ylabel('Proposal acceptance flips / 264')
    for ax in axes:ax.grid(alpha=.2);ax.legend(loc='best')
    axes[1].set_xticks(range(8),real_labels,rotation=25,ha='right')
    fig.savefig(a.out/'real-prediction-stability.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4),constrained_layout=True)
    stages=['vial_detector','temporal_mask','unchanged_QA','total'];x=np.arange(len(stages));width=.35
    ax.bar(x-width/2,[bench['timings_ms'][s]['median'] for s in stages],width,label='Median')
    ax.bar(x+width/2,[bench['timings_ms'][s]['p95'] for s in stages],width,label='p95')
    ax.set_xticks(x,['Vial detector','Temporal mask','Existing QA','Complete CV path'])
    ax.set_ylabel('Milliseconds');ax.legend();ax.grid(axis='y',alpha=.2)
    ax.set_title('GB10 cached-input processing — 528 samples; decode/network/browser excluded')
    fig.savefig(a.out/'pipeline-latency.png',dpi=160);plt.close(fig)
    duration_rows=[]
    for arm,entry in duration.items():
        for label,report in [('epoch3',entry['controls']['epoch-03.pt']),('synthetic_selected',entry['primary']),('last15',entry['controls']['last.pt'])]:
            metrics=report['real_run4_machine_agreement']
            duration_rows.append(dict(arm=arm,checkpoint=label,
                epoch=entry['metadata']['epoch'] if label=='synthetic_selected' else report['epoch'],
                synthetic_ap=report['synthetic_development']['ap50_95_101point'],
                real_provisional_ap=metrics['ap50_95_101point'],matched=metrics['tp'],
                unmatched=metrics['unmatched_boxes'],missed=metrics['missed_targets']))
    report=dict(primary_mode='paired_diff_gain',frozen_model_sha256='6714fe32191d8eba230aa87246f38a25c2906c619d60d7ba30c9e60205352eef',
        synthetic_stress=synthetic,real_prediction_stability=real,full_cv_benchmark=bench,
        detector_duration_controls=duration_rows,human_mask_accuracy=None,human_presence_accuracy=None,human_onset_delay=None,
        collection_manifest_sha256=sha(a.reports/'manifest.json'),helper_sha256=sha(__file__),
        local_gpu_used=False,app_changed=False,run5_used=False,
        limits=['Real perturbation agreement is stability, not accuracy.',
            'Real detector AP uses provisional machine references, not reviewed first-visible pools.',
            'Six correlated inspected development clips; no independent physical test.',
            'Cached-input processing time is not camera-to-dashboard latency.'])
    (a.out/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    lines=['# GB10 temporal candidate checks','',
        'Frozen synthetic-trained candidate. Real human accuracy remains pending; app defaults are unchanged.','',
        '## Cached-input processing','',
        '| Stage | Median ms | p95 ms |','| --- | ---: | ---: |']
    for stage in stages:
        v=bench['timings_ms'][stage];lines.append(f'| {stage} | {v["median"]:.2f} | {v["p95"]:.2f} |')
    lines+=['','528 samples over two passes. Excludes decode, transport, rendering and browser.',
        '', '## Fixed duration detector controls','',
        '| Arm | Checkpoint | Epoch | Synthetic AP | Real provisional AP | Matched / unmatched / missed |',
        '| --- | --- | ---: | ---: | ---: | --- |']
    for r in duration_rows:
        lines.append(f'| {r["arm"]} | {r["checkpoint"]} | {r["epoch"]} | {r["synthetic_ap"]:.2%} | {r["real_provisional_ap"]:.2%} | {r["matched"]} / {r["unmatched"]} / {r["missed"]} |')
    lines+=['','Checkpoint selection used synthetic data only. Real tables are diagnostic, not a selection rule.',
        '', '## Digital stress tests','',
        '| Condition | Synthetic detections / positives | Synthetic alarms / negatives | Real proposal flips / 264 | Real mask consistency IoU |',
        '| --- | ---: | ---: | ---: | ---: |']
    for sk,rk,label in zip(names,real_names,real_labels):
        s=synthetic['paired_diff_gain']['conditions'][sk];r=real['models']['paired_diff_gain'][rk]
        lines.append(f'| {label} | {s["positive_presence_detected"]} / {s["positive_images"]} | {s["negative_presence_alarms"]} / {s["negative_images"]} | {r["presence_decision_flips"]} | {r["clean_positive_mean_prediction_iou"]:.3f} |')
    lines+=['','The last row uses a different empty-family reference for synthetic inputs and a later clip reference for real inputs. These are separate invalid-reference controls.',
        'Real consistency compares predictions to clean predictions, never to approved labels.',
        'See comparison.json for both temporal models, the single-RGB synthetic control, all denominators and QA/mask reproducibility counts.']
    (a.out/'README.md').write_text('\n'.join(lines)+'\n')
    (a.out/'completed.json').write_text(json.dumps(dict(artifacts={p.name:sha(p) for p in a.out.iterdir() if p.is_file()}),indent=2)+'\n')
    print(json.dumps(dict(out=str(a.out),pipeline_ms=bench['timings_ms']['total'],passes=bench['passes'],duration=duration_rows),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reports',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
