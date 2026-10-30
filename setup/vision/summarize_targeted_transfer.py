"""Verify and summarize completed targeted synthetic-transfer experiments."""
import argparse
import csv
from pathlib import Path

from target_synthetic_transfer import read,write,sha


def main(a):
    from train_synthetic_liquid_pilot import validate
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    a.out.mkdir(parents=True,exist_ok=False)
    base=a.tools/'results/synthetic-liquid-20261022/fasterrcnn-v1'
    paired=a.root/'fasterrcnn-paired-v1';roi=a.root/'fasterrcnn-roi-v1'
    roots=[base,paired,roi,a.root/'review-v1',a.root/'roi-review-v1',a.root/'basal-v1',
           a.root/'paired-pipeline-v1',a.root/'roi-pipeline-v1',a.root/'basal-pipeline-v1']
    for root in roots:
        done=read(root/'completed.json')
        for name,expected in done.get('artifacts',{}).items():
            if sha(root/name)!=expected:raise ValueError('Completed artifact changed: '+str(root/name))
    for root in [base,paired,roi]:
        plan=validate(root);meta=read(root/'frozen-model.json');curve=read(root/'training-curve.json')
        if sha(root/'final.pt')!=meta['checkpoint_sha256']:raise ValueError('Checkpoint changed')
        assert len(curve)==3 and all(r['steps']==154 for r in curve)
        assert meta['epoch']==3 and not meta['training_real_runs'] and not meta['selection_real_runs']
        assert (root/'frozen-model.json').stat().st_mtime < (root/'real-development-predictions.json').stat().st_mtime
    render_done=read(a.root/'paired-render-v1/completed-render.json')
    render_plan=read(a.root/'paired-render-v1/render-plan.json')
    assets=Path(render_plan['assets'])
    assert render_done['source_assets_unchanged']
    for path,expected in render_done['source_assets_sha256'].items():
        if sha(assets/path)!=expected:raise ValueError('Original USD/camera asset changed')
    baseline_pipeline=a.tools/'results/sim-real-robustness-20261022/synthetic-fasterrcnn-pipeline-v1'
    pipeline_roots=[baseline_pipeline,a.root/'paired-pipeline-v1',a.root/'roi-pipeline-v1',a.root/'basal-pipeline-v1']
    reference=read(baseline_pipeline/'plan.json')
    keys=['domain','group','source_frame','pts','time_base','source_rgb_sha256']
    identity=lambda plan:[[r[k] for k in keys] for r in plan['ordered_samples']]
    for root in pipeline_roots:
        plan=read(root/'plan.json')
        assert identity(plan)==identity(reference) and plan['qa_config']==reference['qa_config']
        assert all(plan['models'][key]==reference['models'][key] for key in ['vial','sam'])
        pins={k.replace('\\','/'):v for k,v in plan['helper_sha256'].items()}
        reference_pins={k.replace('\\','/'):v for k,v in reference['helper_sha256'].items()}
        for path in ['replay/config.py','replay/measurement.py','replay/geometry.py']:
            assert pins[path]==reference_pins[path]
    raw_qa=[read(p/'comparison.json')['summary']['real_run4']['raw']['qa_pass'] for p in pipeline_roots]
    r=read(a.root/'basal-v1/comparison.json')
    metrics=[read(base/'comparison.json')['real_run4_machine_agreement'],
             read(paired/'comparison.json')['real_run4_machine_agreement'],
             read(roi/'comparison.json')['real_run4_machine_agreement'],
             r['real_run4_provisional_agreement']['baseline']['basal']]
    labels=['Baseline','Paired renders','Vial crop','Baseline + basal']
    rows=[dict(arm=label,real_provisional_ap50_95=m['ap50_95_101point'],matches=m['tp'],
        unmatched_boxes=m['unmatched_boxes'],missed_targets=m['missed_targets'],precision=m['precision'],recall=m['recall'],
        provisional_negative_frames_with_predictions=m['provisional_negative_frames_with_predictions'],
        real_raw_pipeline_qa_passes=qa,real_raw_pipeline_attempts=111) for label,m,qa in zip(labels,metrics,raw_qa)]
    with (a.out/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    review=read(a.root/'review-v1/comparison.json')
    result=dict(schema_version=1,rows=rows,new_render_geometric_development=review['new_grouped_synthetic_development'],
        new_render_small_boxes=render_done['small_body_boxes_le32'],new_render_images=render_done['images'],
        source_assets_unchanged=True,qa_config_and_implementation_unchanged=True,pipeline_sources_and_clocks_identical=True,
        new_fits=2,fixed_epochs=3,training_images_per_epoch=614,optimizer_steps_per_fit=462,
        training_real_runs=[],selection_real_runs=[],run5_used=False,promoted=False,app_changed=False,
        pipeline_attempts_this_followup=222+222+666,
        conclusion='Paired renders improve their own held-out synthetic coverage, including5/7 small geometric targets vs0/7. Neither new fit improves real development agreement or unchanged-QA availability. Basal association trades unmatched boxes for missed targets and loses QA passes. Retain baseline; no successful real-transfer improvement claimed.',
        limits=['Real targets and missing-target frames are provisional; unmatched boxes are not reviewed false positives.',
            'QA passes count availability, not mask/measurement/onset/defect accuracy.',
            'Development scenes and hypotheses were previously inspected; no blind physical test or real label approval.',
            'Generic COCO/SAM pretraining and existing real-informed camera/prompt design remain disclosed.',
            'Multiple renderer/data-mix changes are coupled; lower-vial crop also truncates some synthetic targets.'])
    write(a.out/'comparison.json',result)
    fig,axes=plt.subplots(1,3,figsize=(15,4.6))
    values=[[100*m['ap50_95_101point'] for m in metrics],[m['unmatched_boxes'] for m in metrics],raw_qa]
    titles=['Run 4 provisional liquid AP50–95 (%)','Unmatched boxes at confidence .35','Real raw QA passes / 111']
    for ax,vals,title in zip(axes,values,titles):
        bars=ax.barh(labels,vals,color=['#4b8db9','#c26969','#bd9460','#78a291'])
        ax.invert_yaxis();ax.set_title(title,fontsize=11);ax.spines[['top','right']].set_visible(False)
        ax.set_xlim(0,max(vals)*1.25)
        for bar,v in zip(bars,vals):ax.text(v+.01*max(vals),bar.get_y()+bar.get_height()/2,f'{v:.1f}' if title.endswith('(%)') else str(v),va='center',fontsize=10)
    fig.suptitle('Targeted synthetic-transfer pilots | no candidate promoted',fontsize=14)
    fig.text(.5,.02,'Provisional teacher agreement and correlated development frames. QA availability is not reviewed accuracy.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.05,1,.92));fig.savefig(a.out/'targeted-transfer-scorecard.png',dpi=170);plt.close(fig)
    write(a.out/'completed.json',dict(artifacts={p.name:sha(p) for p in a.out.iterdir() if p.is_file()},verified_roots=[str(p) for p in roots]))
    print(result['conclusion'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['root','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--tools',type=Path,default=Path('C:/Users/Austin/ai-club-tools'))
    main(p.parse_args())
