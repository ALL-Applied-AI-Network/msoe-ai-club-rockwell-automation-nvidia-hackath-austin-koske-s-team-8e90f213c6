"""Assemble completed Run 4 pilot evaluations without repeating GPU inference."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from audit_liquid_specialists import development_rows, targets_for, summarize, gallery
from liquid_specialists_data import read, sha, write
from train_early_liquid import ARMS, validate


def main(a):
    if a.out.exists():
        raise ValueError('Use a fresh collection directory')
    plan = validate(a.experiment)
    data, sources = development_rows(a.source_experiment)
    previous = read(a.previous_audit/'comparison.json')
    predictions = {name: read(a.previous_audit/(name+'-predictions.json')) for name in previous['models']}
    expected = [(r['source_id'], targets_for(data, r)) for r in sources]
    if previous['experiment_plan_sha256'] != plan['source_experiment_plan_sha256']:
        raise ValueError('Earlier audit ancestry differs')
    models = {}; hashes = {}; report_hashes = {}
    for folder in a.inputs:
        report = read(folder/'comparison.json')
        if (report['kind'] != 'early_liquid_run4_architecture_audit' or report['development_run'] != 4
                or report['run5_images_used'] or report['detector_promoted'] or report['measurement_qa_changed']
                or report['experiment_plan_sha256'] != sha(a.experiment/'experiment-plan.json')
                or report['previous_audit_sha256'] != sha(a.previous_audit/'comparison.json')):
            raise ValueError('Evaluation provenance differs')
        for name, digest in report['checkpoint_hashes'].items():
            if name not in ARMS or name in models:
                raise ValueError('Duplicate or unexpected arm')
            model = read(a.experiment/'models'/name/'model.json')
            if digest != model['checkpoint_sha256'] or sha(a.experiment/'models'/name/'best.pt') != digest:
                raise ValueError('Model checkpoint changed')
            rows = read(folder/(name+'-predictions.json'))
            if [(r['source_id'], r['targets']) for r in rows] != expected:
                raise ValueError('Scoring sources or targets differ')
            if summarize(rows) != report['models'][name]:
                raise ValueError('Stored metrics differ from prediction bytes')
            models[name] = model; hashes[name] = digest
            report_hashes[name] = sha(folder/'comparison.json'); predictions[name] = rows
    if set(models) != set(ARMS):
        raise ValueError('Missing completed arm')
    for name, rows in predictions.items():
        if [(r['source_id'], r['targets']) for r in rows] != expected:
            raise ValueError('Previous scoring inventory changed')
    a.out.mkdir(parents=True)
    for name, rows in predictions.items():
        write(a.out/(name+'-predictions.json'), rows)
    gallery(data, sources, predictions, a.out, short_only=True)
    scores = {name: summarize(rows) for name, rows in predictions.items()}
    names = list(scores)
    points = [next(p for p in scores[n]['operating_points'] if p['confidence'] == .35) for n in names]
    ap = [100*scores[n]['liquid_ap50_95'] for n in names]
    short = [p['target_height_strata']['height_32_to_64']['detected'] for p in points]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [1.2, 1]})
    axes[0].barh(names, ap); axes[0].invert_yaxis(); axes[0].set_xlim(0, 65)
    axes[0].set_title('Overall liquid AP50-95 (%)')
    axes[1].barh(names, short); axes[1].invert_yaxis(); axes[1].set_xlim(0, 20)
    axes[1].set_title('Matched short targets / 18 at confidence .35')
    for axis, values, pattern in [(axes[0], ap, '{:.2f}'), (axes[1], short, '{:d}/18')]:
        for y, value in enumerate(values):
            axis.text(value+.3, y, pattern.format(value), va='center', fontsize=9)
    fig.suptitle('Run 4 development: provisional teacher boxes; not first-visible-liquid accuracy')
    fig.text(.5, .01, '157 images / 89 targets. Short group: 32-64px high. No targets <=32px. Same diagnostic cutoff, uncalibrated scores.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .95)); fig.savefig(a.out/'model-scorecard.png', dpi=160); plt.close(fig)
    write(a.out/'comparison.json', dict(schema_version=1, kind='completed_early_liquid_development_comparison',
        source_experiment_plan_sha256=plan['source_experiment_plan_sha256'],
        experiment_plan_sha256=sha(a.experiment/'experiment-plan.json'),
        previous_audit_sha256=sha(a.previous_audit/'comparison.json'), input_report_sha256=report_hashes,
        checkpoint_hashes=hashes, model_metadata=models, models=scores,
        source_count=len(sources), source_runs=[4], run5_images_used=False,
        detector_promoted=False, measurement_qa_changed=False, masks_scored=False,
        collector_script_sha256=sha(Path(__file__)), scorecard_sha256=sha(a.out/'model-scorecard.png'),
        limits=plan['limits']))
    for name in names:
        point = next(p for p in scores[name]['operating_points'] if p['confidence'] == .35)
        print(name, round(100*scores[name]['liquid_ap50_95'], 2), point['target_height_strata']['height_32_to_64'], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('experiment', 'source-experiment', 'previous-audit', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--inputs', nargs='+', type=Path, required=True)
    main(p.parse_args())
