"""Explicit provisional admission of the frozen 18-render revision.

Reuses the previous fitting functions without altering their split/hash guards.
Repeats are optimizer exposures, never additional unique rendered observations.
"""
import argparse
from collections import Counter
from pathlib import Path
import random
import shutil

from train_mixed_detector import read, write, sha, synthetic_targets, validate_split
from train_sim_only import validate_rows


def prepare(previous, renders, out, accept):
    if not accept: raise ValueError('Explicit provisional-render admission required')
    evidence=read(Path(__file__).resolve().parents[1]/'verification/pale-early-gb10.json')
    for name in ('run.json','verification.json','manifest.jsonl'):
        expected=next(x['sha256'] for x in evidence['outputs'] if Path(x['path']).name==name)
        if sha(renders/name)!=expected:raise ValueError('Published final batch changed: '+name)
    native=read(renders/'run.json');verified=read(renders/'verification.json')
    if (native['captured']!=18 or not native['complete'] or not verified['passed'] or
            native['renderer_profile']!='meniscus-transmission-v2' or native['host']!='spark8' or
            native['isaac_version']!='6.0.0.1' or native['run5_used'] or native['training_eligible'] or
            not native['camera_parameters_unchanged'] or not native['source_assets_unchanged'] or
            any(x['source_run'] not in (1,2,3) for x in native['appearance_references'])):
        raise ValueError('Native runtime/reference/eligibility boundary changed')
    old=read(previous/'experiment-plan.json');base=read(previous/'data/real/provenance.json')
    if sha(previous/'data/real/provenance.json')!=old['real_provenance_sha256']:
        raise ValueError('Frozen real inventory changed')
    out.mkdir(parents=True,exist_ok=False)
    (out/'data/real').mkdir(parents=True)
    shutil.copyfile(previous/'data/real/provenance.json',out/'data/real/provenance.json')
    plan=dict(old,kind='revised_18_native_render_training_probe',arms=['mixed','sim-only'],
        unique_synthetic_images=18,expected_synthetic_images=768,
        synthetic_fixture_sha256=native['fixture_sha256'],
        synthetic_renderer_sha256=native['script_sha256'],
        synthetic_annotation_helper_sha256=native['annotation_helper_sha256'],
        geometry_helper_sha256=native['geometry_helper_sha256'],
        renderer_profile=native['renderer_profile'],render_quality=dict(renderer='PathTracing',samples_per_pixel=32,subframes=4),
        render_manifest_sha256=sha(renders/'manifest.jsonl'),render_run_sha256=sha(renders/'run.json'),
        render_verification_sha256=sha(renders/'verification.json'),
        appearance_reference_runs=[1,2,3],raw_render_eligibility_flags_modified=False,
        authorization='Austin: "we should train it and try", 2026-10-19; provisional research use, no appearance approval inferred',
        source_experiment_plan_sha256=sha(previous/'experiment-plan.json'),
        repetition_policy='Include all 18 once, then seeded random draws from those 18 to reach 768 synthetic mixed exposures; sim-only budget 2210',
        cached_real_only_control=dict(checkpoint_sha256=sha(previous/'models/real-only/best.pt'),
            model_sha256=sha(previous/'models/real-only/detector.onnx'),
            source_plan_sha256=sha(previous/'experiment-plan.json'),
            policy='Reuse previously completed identical real-only budget/init/settings; do not relabel it as fitted under this plan'),
        limits=old['limits']+['Only 18 unique revised renders, repeated heavily; not 768 new scenes.',
            'Renderer was qualitatively guided by real Runs 1-3 RGB; sim-only describes domain fine-tuning pixels, not an absence of real design guidance.',
            'Revised-vs-old comparisons change render diversity as well as appearance/geometry; no causal renderer-only ablation.',
            'Body/vial classes unchanged. Streams retained raw, excluded as positive targets in this two-class probe.'])
    write(out/'experiment-plan.json',plan)
    import json
    originals=[]
    for native_row in map(json.loads,(renders/'manifest.jsonl').read_text().splitlines()):
        image=renders/native_row['image_file']
        if sha(image)!=native_row['image_sha256']:raise ValueError('Native RGB changed')
        target=synthetic_targets(native_row,renders);name='synthetic-'+native_row['sample_id']
        image_rel=f'images/train/{name}.png'; label_rel=f'labels/train/{name}.txt'
        donor_image='synthetic/'+image_rel;donor_labels='synthetic/'+label_rel
        for relative in (donor_image,donor_labels):(out/relative).parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(image,out/donor_image)
        (out/donor_labels).write_text('\n'.join(target)+'\n')
        originals.append(dict(source_id=name,source_run=None,origin='synthetic',split='train',
            source_sha256=sha(image),image_sha256=sha(image),actual_label_sha256=sha(out/donor_labels),
            image=image_rel,labels=label_rel,package_image=donor_image,package_labels=donor_labels,
            native_sample_id=native_row['sample_id'],native_manifest_sha256=plan['render_manifest_sha256'],
            label_status='unreviewed geometric proxy; explicit user-authorized experiment'))
    if len(originals)!=18:raise ValueError('Expected exactly 18 unique revised renders')
    def exposures(budget,prefix):
        rows=[dict(r) for r in originals];rng=random.Random(plan['seed'])
        for i in range(budget-len(rows)):
            donor=rng.choice(originals);name=f'{prefix}-{i:06d}'
            rows.append(dict(donor,source_id=name,parent_id=donor['source_id'],
                image=f'images/train/{name}.png',labels=f'labels/train/{name}.txt'))
        return rows
    rows=[dict(r) for r in base['samples']]+exposures(768,'revised-repeat')
    validate_split(rows,set(base['held_out_source_hashes']))
    (out/'data/mixed').mkdir(parents=True)
    write(out/'data/mixed/provenance.json',dict(schema_version=1,kind='real_plus_revised_render_repeated_probe',
        samples=rows,counts=dict(Counter(r['origin']+'_'+r['split'] for r in rows)),
        held_out_source_hashes=base['held_out_source_hashes'],real_provenance_sha256=plan['real_provenance_sha256'],
        render_manifest_sha256=plan['render_manifest_sha256'],render_verification_sha256=plan['render_verification_sha256'],
        unique_synthetic_images=18,provisional_admission_authorization=plan['authorization'],
        raw_render_eligibility_flags_modified=False,target_scope=plan['target_scope']))
    sim=dict(schema_version=1,arm='sim-only',base_plan_sha256=sha(out/'experiment-plan.json'),
        architecture=plan['architecture'],seed=plan['seed'],epochs=plan['epochs'],
        pretrained_sha256=plan['pretrained_sha256'],expected_unique_images=18,training_images=2210,
        checkpoint_selection='Fixed final epoch 12; no real checkpoint selection',
        training_real_runs=[],checkpoint_selection_real_runs=[],real_data_used_for_fit_or_selection=False,
        validation='Training-set synthetic resubstitution only; not held-out accuracy',
        source_scope=plan['target_scope'],preprocessing=plan['preprocessing'],
        appearance_reference_runs=[1,2,3],
        generic_real_pretraining='COCO initialization; renderer also qualitatively guided by Runs 1-3 RGB; fine-tuning/selection use only generated pixels')
    write(out/'sim-only-plan.json',sim)
    sim_rows=exposures(2210,'revised-sim-repeat')
    for row in sim_rows:
        donor=next(r for r in originals if r['native_sample_id']==row['native_sample_id'])
        row.update(donor_image=donor['image'],donor_labels=donor['labels'])
    validate_rows(sim_rows,18,2210)
    (out/'data/sim-only').mkdir(parents=True)
    write(out/'data/sim-only/provenance.json',dict(schema_version=1,arm='sim-only',samples=sim_rows,
        plan_sha256=sha(out/'sim-only-plan.json'),source_provenance_sha256=sha(out/'data/mixed/provenance.json'),
        render_manifest_sha256=plan['render_manifest_sha256'],render_verification_sha256=plan['render_verification_sha256'],
        unique_synthetic_images=18,training_images=2210,real_pixels_or_labels_copied=False))
    write(out/'preparation.json',dict(plan_sha256=sha(out/'experiment-plan.json'),
        sim_plan_sha256=sha(out/'sim-only-plan.json'),
        provenance_sha256={arm:sha(out/f'data/{arm}/provenance.json') for arm in ('real','mixed','sim-only')},
        source_scripts={name:sha(Path(__file__).with_name(name)) for name in
            ('train_mixed_detector.py','train_sim_only.py','train_revised_render.py')}))
    print('Frozen 18 unique renders; mixed 1442 real + 768 synthetic exposures; sim-only 2210 synthetic exposures')


def materialize(root,previous):
    frozen=read(root/'preparation.json');plan=read(root/'experiment-plan.json')
    if sha(root/'experiment-plan.json')!=frozen['plan_sha256'] or sha(root/'sim-only-plan.json')!=frozen['sim_plan_sha256']:
        raise ValueError('Frozen plan changed')
    if sha(previous/'data/real/provenance.json')!=plan['real_provenance_sha256']:
        raise ValueError('Remote real source differs')
    for arm in ('real','mixed','sim-only'):
        if sha(root/f'data/{arm}/provenance.json')!=frozen['provenance_sha256'][arm]:raise ValueError('Inventory changed')
    for arm in ('real','mixed'):
        data=root/'data'/arm;p=read(data/'provenance.json')
        validate_split(p['samples'],set(p['held_out_source_hashes']))
        for row in p['samples']:
            for key,hash_key in [('image','image_sha256'),('labels','actual_label_sha256')]:
                source=previous/'data/real'/row[key] if row['origin']=='real' else root/row['package_'+key]
                if sha(source)!=row[hash_key]:raise ValueError('Materialization source changed')
                target=data/row[key];target.parent.mkdir(parents=True,exist_ok=True)
                if not target.exists():shutil.copyfile(source,target)
                if sha(target)!=row[hash_key]:raise ValueError('Materialized file changed')
    print('Materialized frozen real/mixed inputs; sim-only trainer materializes synthetic donors only')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','materialize'])
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--previous',type=Path,required=True)
    parser.add_argument('--renders',type=Path);parser.add_argument('--accept-provisional-render-proxies',action='store_true')
    args=parser.parse_args()
    if args.command=='prepare':prepare(args.previous,args.renders,args.out,args.accept_provisional_render_proxies)
    else:materialize(args.out,args.previous)
