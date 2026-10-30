"""Verify and summarize completed synthetic-only Dataset 2 development pilots."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main(a):
    if a.out.exists():
        raise ValueError('Use a fresh summary output')
    root=a.tools/'results';probe=root/'sim-real-robustness-20261022';fits=root/'synthetic-liquid-20261022'
    paths=[probe/'cpu-transfer-v2',probe/'synthetic-pipeline-v1',probe/'synthetic-fasterrcnn-pipeline-v1',probe/'synthetic-fasterrcnn-aug-pipeline-v1',fits/'fasterrcnn-v1',fits/'fasterrcnn-aug-v1']
    verified={}
    for path in paths:
        completed=read(path/'completed.json')
        for name,expected in completed['artifacts'].items():
            if sha(path/name)!=expected:
                raise ValueError('Completed evidence changed: '+str(path/name))
        verified[str(path)]=dict(completed_sha256=sha(path/'completed.json'),artifacts=len(completed['artifacts']))
    cpu=read(probe/'cpu-transfer-v2/comparison.json')
    pipelines={name:read(probe/path/'comparison.json') for name,path in [('joint_synthetic_yolo','synthetic-pipeline-v1'),('synthetic_fasterrcnn','synthetic-fasterrcnn-pipeline-v1'),('augmented_synthetic_fasterrcnn','synthetic-fasterrcnn-aug-pipeline-v1')]}
    training={}
    for name in ['fasterrcnn-v1','fasterrcnn-aug-v1']:
        path=fits/name;plan=read(path/'experiment-plan.json');model=read(path/'frozen-model.json');comparison=read(path/'comparison.json')
        assert not plan['training_real_runs'] and not plan['selection_real_runs'] and not plan['run5_used']
        assert all(s['origin']=='synthetic' and s['source_run'] is None for s in plan['samples'])
        train={s['image_sha256'] for s in plan['samples'] if s['split']=='train'}
        val={s['image_sha256'] for s in plan['samples'] if s['split']=='val'}
        assert not train&val and len(train)==614 and len(val)==154
        assert sha(path/'final.pt')==model['checkpoint_sha256']==comparison['checkpoint_sha256']
        for name2,expected in plan['helpers'].items():
            assert sha(path/'helpers'/name2)==expected
        training[name]=dict(checkpoint_sha256=model['checkpoint_sha256'],epochs=3,train_images=614,synthetic_development_images=154,
            synthetic_geometric_ap50_95=comparison['synthetic_development']['ap50_95_101point'],
            real_machine_agreement=comparison['real_run4_machine_agreement'],
            plan_sha256=sha(path/'experiment-plan.json'),augmentation=plan.get('augmentation','none'))
    assert {s['image_sha256']:s['split'] for s in read(fits/'fasterrcnn-v1/experiment-plan.json')['samples']}=={s['image_sha256']:s['split'] for s in read(fits/'fasterrcnn-aug-v1/experiment-plan.json')['samples']}
    # The legacy fixed-point comparator reproduces the pre-existing Run 4 audit.
    old=read(root/'early-liquid-20261020/development-resnet/comparison.json')['models']['Deployed']
    prior=next(p for p in old['operating_points'] if p['confidence']==.35)
    current=cpu['summary']['legacy_real_baseline']['raw']['real_liquid']
    assert all(prior[k]==current[k] for k in ['tp','unmatched_boxes','missed_targets'])
    # All full-pipeline raw runs use identical original RGB, frame and native PTS.
    identities=[]
    for name,path in [('joint','synthetic-pipeline-v1'),('rcnn','synthetic-fasterrcnn-pipeline-v1'),('aug','synthetic-fasterrcnn-aug-pipeline-v1')]:
        plan=read(probe/path/'plan.json')
        identities.append([(s['domain'],s['group'],s['source_frame'],s['pts'],s['time_base'],s['source_rgb_sha256']) for s in plan['ordered_samples']])
    assert identities[0]==identities[1]==identities[2]
    a.out.mkdir(parents=True)
    report=dict(schema_version=1,kind='synthetic_training_and_transfer_development_summary',prepared='2026-10-22',
        training=training,cpu_transfer=cpu['summary'],pipelines={name:v['summary'] for name,v in pipelines.items()},
        completed_evidence=verified,scored_inference_inputs=cpu['attempts']+sum(v['attempts'] for v in pipelines.values())+2*(154+157),
        verified_training_split_disjoint=True,verified_pipeline_source_join=True,verified_legacy_operating_point_parity=True,
        generic_pretraining='COCO initialization and generic SAM weights; CSI task fine-tuning for candidate arms uses synthetic data only.',
        prior_real_development='Camera geometry/prompt/QA policies were already developed with real footage; this is not zero-shot commissioning.',
        original_corpus_audit=read(probe/'synthetic-corpus-audit-v2.json'),
        real_accuracy=None,sim_to_real_gap_closed=False,defaults_promoted=False,hardware_actions=0,
        conclusion='Synthetic-only detector candidates now exist for the original Faster R-CNN stage, but high geometric synthetic AP does not transfer cleanly to provisional real targets. The tested RGB augmentation did not improve real agreement. QA availability is not mask correctness.',
        limits=['Real labels are provisional and can include stream; synthetic body boxes are geometric proxies. Cross-domain AP values have different reference semantics.',
            'Fixed three-epoch, one-seed family pilots; no optimized architecture or statistically independent physical accuracy claim.',
            'Repeated correlated frames contribute to inference counts; no reviewed first-visible-pool onset or mask accuracy is available.',
            'Dataset 1 final-fixture classification and cross-camera physical-vial linking are not implemented or evaluated here.'])
    (a.out/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(1,2,figsize=(14,5.8))
    names=list(training);x=np.arange(2)
    axes[0].bar(x-.18,[training[n]['synthetic_geometric_ap50_95']*100 for n in names],width=.36,label='Held-out synthetic geometry')
    axes[0].bar(x+.18,[training[n]['real_machine_agreement']['ap50_95_101point']*100 for n in names],width=.36,label='Real Run 4 machine references')
    axes[0].set_xticks(x,['Synthetic-only R-CNN','Synthetic-only R-CNN + RGB variation']);axes[0].set_ylim(0,100);axes[0].set_ylabel('AP50–95 / 101-point interpolation (%)');axes[0].set_title('Training success does not establish real transfer');axes[0].legend(fontsize=8)
    labels=['Joint synthetic YOLO + SAM','Synthetic YOLO + R-CNN + SAM','Synthetic YOLO + augmented R-CNN + SAM'];x=np.arange(3)
    for i,condition in enumerate(['raw','green_085','jpeg_q40']):
        axes[1].bar(x+(i-1)*.24,[pipelines[n]['summary']['real_run4'][condition]['qa_pass'] for n in pipelines],width=.24,label=condition)
    axes[1].set_xticks(x,labels,fontsize=8,rotation=12);axes[1].set_ylim(0,111);axes[1].set_ylabel('QA passes / 111 selected real samples');axes[1].set_title('Full inference: availability, not mask correctness');axes[1].legend(fontsize=8)
    fig.suptitle('Dataset 2: synthetic-only CSI fine-tuning and real-development transfer',fontsize=14)
    fig.text(.5,.025,'One seed, fixed three epochs; provisional/geometric references differ. No real CSI training, Run 5 tuning or default promotion.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,.93));fig.savefig(a.out/'synthetic-transfer-scorecard.png',dpi=180);plt.close(fig)
    completion=dict(source_files_verified=verified,artifacts={p.name:sha(p) for p in a.out.iterdir() if p.is_file()})
    (a.out/'completed.json').write_text(json.dumps(completion,indent=2)+'\n')
    print(json.dumps(dict(scored_inference_inputs=report['scored_inference_inputs'],training=training,verified_sources=len(verified)),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tools',type=Path,default=Path.home()/'ai-club-tools')
    p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
