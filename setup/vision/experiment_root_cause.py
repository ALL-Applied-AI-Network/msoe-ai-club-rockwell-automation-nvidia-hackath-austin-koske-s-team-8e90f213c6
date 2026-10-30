"""CPU diagnosis scenario study, with private cause labels outside inference.

Reuses the hypothesized SharedPath and analytic appearance family. No Isaac
rendering, camera feed, fitted physics, training, real accuracy or actuator I/O.
"""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict
from fractions import Fraction
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from replay.diagnostics import DiagnosticPolicy, diagnose
from replay.measurement import delta_e, rgb_to_lab
from twin_contract import SharedPath
from experiment_stage2_policy import appearance


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def dispense(path, volumes, order):
    delivered = np.zeros(3)
    for tank in order:
        delivered += path.dispense(tank, float(volumes[tank]))
    return delivered


def conditioned(volumes, order, capacity):
    path = SharedPath(capacity, tuple(np.asarray(volumes) / sum(volumes)))
    for _ in range(8):
        endpoint = dispense(path, volumes, order)
    return path, endpoint


def color(parts):
    return rgb_to_lab(appearance(np.asarray(parts, float))).tolist()


def document(labs, previous, expected, tag):
    doc = dict(schema='vial-diagnosis-input/v1', data_domain='synthetic', session_id=tag,
               camera_id='sim-camera', profile_id='sim-Lab-endpoint', clock_domain_id='sim-clock',
               recipe_id='B', previous_recipe_id='A', transition_evidence_id='sim-transition',
               context_evidence_id='sim-plan', context_available_s=0., as_of_s=90., references=[], vials=[])
    for recipe, values in [('A', previous), ('B', expected)]:
        doc['references'].append(dict(reference_id='sim-reference-' + recipe, recipe_id=recipe, lab=values,
            data_domain='synthetic', camera_id=doc['camera_id'], profile_id=doc['profile_id'],
            clock_domain_id=doc['clock_domain_id'], available_s=0., source_vial_id='sim-standard-' + recipe,
            independent_of_evaluation=True, qualification=dict(status='simulated',
                condition='accepted_recipe_endpoint', evidence_id='sim-reference-preparation')))
    for i, values in enumerate(labs, 1):
        doc['vials'].append(dict(vial_id='v' + str(i), observation_id='endpoint-' + str(i),
            source_evidence_id='simulated-feature-' + str(i), recipe_id='B', camera_id=doc['camera_id'],
            profile_id=doc['profile_id'], clock_domain_id=doc['clock_domain_id'], position_after_change=i,
            start_s=float(20 * i - 19), source_time_s=float(20 * i - 2), available_s=float(20 * i - 1),
            completed=True, completion_evidence_id='sim-complete-' + str(i), qa_pass=True,
            camera_health_valid=True, alignment_confirmed=True, identity_confirmed=True,
            temporal_reference_status='simulated', lab=values))
    return doc


def process_record(doc, kind):
    return dict(kind=kind, matches_plan=False, available_s=80., data_domain='synthetic',
                evidence_id='simulated-process-export', recorded_by='synthetic-generator',
                session_id=doc['session_id'], recipe_id='B', clock_domain_id=doc['clock_domain_id'],
                vial_ids=[v['vial_id'] for v in doc['vials']], origin='simulation')


def real_audit(probe, out):
    source = probe / 'frames.jsonl'
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    plan = json.loads((probe / 'plan.json').read_text())
    if plan.get('source_run') != 4 or plan.get('run5_used'):
        raise ValueError('Expected the frozen Run4 development probe only')
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['case_id']].append(row)
    examples = []
    for case, frames in sorted(grouped.items()):
        # Audit the existing record structure, not a claim that the clip end is a completed vial.
        row = frames[-1]
        doc = dict(schema='vial-diagnosis-input/v1', data_domain='real', session_id=case,
                   camera_id='vial-fill', profile_id='paired_diff_gain', clock_domain_id='run4-native',
                   recipe_id=None, previous_recipe_id=None, context_evidence_id=None,
                   context_available_s=None, as_of_s=float(row['pts'] * Fraction(row['time_base'])),
                   references=[], vials=[])
        examples.append(dict(case_id=case, outcome=diagnose(doc),
                             note='No completed-vial identity/context/reference record supplied; raw frames are not relabeled as vials.'))
    report = dict(source_sha256=sha(source), source_video_sha256=plan['source_video_sha256'],
                  frames=len(rows), correlated_clip_groups=len(grouped),
                  QA_eligible_frames=sum(r['modes']['paired_diff_gain']['measurement']['measurement_valid'] for r in rows),
                  reference_status_counts=dict(Counter(r['reference_status'] for r in rows)),
                  completed_vial_RCA_inputs_available=0, real_cause_accuracy=None,
                  required_inputs=['Independently accepted current/previous recipe references.',
                                   'Attributed intended recipe, transition and physical-vial order/identity.',
                                   'Confirmed completion, source timing, camera health and aligned qualified temporal reference.',
                                   'Staff-provided process records or matched physical comparison for cause attribution.'],
                  adapter_examples=examples)
    write(out / 'real-input-audit.json', report)
    return report


def main(args):
    started = time.monotonic()
    args.out.mkdir(parents=True, exist_ok=False)
    policy = DiagnosticPolicy()
    families = ['nominal', 'carryover', 'transient_wrong_dose_same_features',
                'local_optical_same_features', 'persistent_wrong_recipe', 'persistent_wrong_dose',
                'camera_unhealthy', 'unqualified_reference', 'wrong_recipe_with_record', 'wrong_dose_with_record']
    profiles = [([16., 9., 5.], [5., 9., 16.]), ([5., 20., 5.], [15., 5., 10.]),
                ([10., 5., 15.], [5., 20., 5.])]
    plan = dict(schema='root-cause-scenario-plan/v1', policy=asdict(policy),
                profiles=profiles, orders=list(itertools.permutations(range(3))),
                capacities_ml=[2., 8., 16.], seeds=[311, 499, 733], vials_per_sequence=4,
                feature_noise_sd_Lab=.2, families=families, new_training=False, GPU_used=False,
                comparators=dict(current_dashboard_first_vial_de2000=2.3,
                                 matched_threshold_first_vial_de2000=policy.deviation_de2000),
                real_accuracy=None, run5_used=False,
                limits=['Feature-level analytic scenarios, not rendered optical validation.',
                        'SharedPath capacity/mixing and absorbance are hypotheses, not fitted CSI physics.',
                        'References and observations share an analytic family.',
                        'Cause labels and path truth never enter diagnosis input.',
                        'Matched feature traces can have different causes; suggestions are not causal proof.'])
    helpers = [Path(__file__), ROOT / 'replay/diagnostics.py', ROOT / 'replay/measurement.py',
               ROOT / 'setup/vision/twin_contract.py', ROOT / 'setup/vision/experiment_stage2_policy.py']
    plan['helper_sha256'] = {str(p.relative_to(ROOT)): sha(p) for p in helpers}
    write(args.out / 'plan.json', plan)
    frozen = args.out / 'helpers'; frozen.mkdir()
    for p in helpers:
        (frozen / p.name).write_bytes(p.read_bytes())
    examples = args.out / 'examples'; examples.mkdir()
    results, ambiguity, showcases = [], [], {}
    for recipe, (a, b) in enumerate(profiles):
        wrong = [b[0] * 1.4, b[1] * .7, b[2]]
        for order in itertools.permutations(range(3)):
            for capacity in plan['capacities_ml']:
                previous_path, previous_parts = conditioned(a, order, capacity)
                nominal_path, expected_parts = conditioned(b, order, capacity)
                previous, expected = color(previous_parts), color(expected_parts)
                transient_parts = [dispense(previous_path, b, order).tolist() for _ in range(4)]
                _, wrong_parts = conditioned(wrong, order, capacity)
                for seed in plan['seeds']:
                    tag = f'sim-r{recipe}-o{"".join(map(str, order))}-h{capacity:g}-s{seed}'
                    rng = np.random.default_rng(seed + recipe * 10000)
                    noise = rng.normal(0, plan['feature_noise_sd_Lab'], (4, 3))
                    labs = lambda parts: (np.asarray([color(v) for v in parts]) + noise).tolist()
                    transient = document(labs(transient_parts), previous, expected, tag)
                    nominal = document(labs([expected_parts] * 4), previous, expected, tag)
                    recipe_bad = document(labs([previous_parts] * 4), previous, expected, tag)
                    dose_bad = document(labs([wrong_parts] * 4), previous, expected, tag)
                    cases = dict(nominal=nominal, carryover=transient,
                                 transient_wrong_dose_same_features=deepcopy(transient),
                                 local_optical_same_features=deepcopy(transient),
                                 persistent_wrong_recipe=recipe_bad, persistent_wrong_dose=dose_bad,
                                 camera_unhealthy=deepcopy(recipe_bad), unqualified_reference=deepcopy(transient),
                                 wrong_recipe_with_record=deepcopy(recipe_bad), wrong_dose_with_record=deepcopy(dose_bad))
                    for row in cases['camera_unhealthy']['vials']:
                        row['camera_health_valid'] = False
                    cases['unqualified_reference']['references'][1]['qualification']['status'] = 'unreviewed'
                    cases['wrong_recipe_with_record']['process_checks'] = [process_record(recipe_bad, 'commanded_recipe')]
                    cases['wrong_dose_with_record']['process_checks'] = [process_record(dose_bad, 'metered_delivery')]
                    group = []
                    for family in families:
                        doc = cases[family]
                        result = diagnose(doc, policy)
                        if any(result[k] is not None for k in ['confirmed_root_cause', 'quality_decision', 'hardware_command']):
                            raise AssertionError('Causal/product/hardware authority was fabricated')
                        first_delta = delta_e(doc['vials'][0]['lab'], expected)
                        baseline_cause = first_delta > 2.3  # Exact first-vial/context behavior of the current dashboard.
                        row = dict(case_id=tag + '/' + family, private_family=family,
                                   input_sha256=digest(doc), status=result['status'], pattern=result['pattern'],
                                   primary_hypothesis=result['primary_hypothesis'],
                                   dashboard_rule_suggests_carryover=baseline_cause,
                                   matched_threshold_rule_suggests_carryover=first_delta > policy.deviation_de2000,
                                   first_delta_e2000=first_delta, confirmed_cause=None)
                        results.append(row)
                        if family in ('carryover', 'transient_wrong_dose_same_features', 'local_optical_same_features'):
                            group.append(dict(family=family, input_sha256=digest(doc), output_sha256=digest(result)))
                        if family not in showcases and (result['primary_hypothesis'] or family in (
                                'nominal', 'persistent_wrong_recipe', 'persistent_wrong_dose',
                                'camera_unhealthy', 'unqualified_reference')):
                            showcases[family] = dict(input=doc, output=result)
                            write(examples / (family + '.input.json'), doc)
                            write(examples / (family + '.diagnosis.json'), result)
                    assert len({r['input_sha256'] for r in group}) == 1
                    assert len({r['output_sha256'] for r in group}) == 1
                    ambiguity.append(dict(case_id=tag, indistinguishable_causes=group,
                                          meaning='Identical observed features cannot identify these different private causes.'))
    metrics = {}
    for family in families:
        selected = [r for r in results if r['private_family'] == family]
        metrics[family] = dict(sequences=len(selected),
                              statuses=dict(Counter(r['status'] for r in selected)),
                              patterns=dict(Counter(r['pattern'] for r in selected)),
                              primary_hypotheses=dict(Counter(r['primary_hypothesis'] or 'undetermined' for r in selected)),
                              dashboard_rule_carryover_suggestions=sum(r['dashboard_rule_suggests_carryover'] for r in selected),
                              matched_threshold_rule_carryover_suggestions=sum(r['matched_threshold_rule_suggests_carryover'] for r in selected),
                              confirmed_cause_claims=0)
    write(args.out / 'scenario-results.json', results)
    write(args.out / 'identifiability-pairs.json', ambiguity)
    audit = real_audit(args.probe, args.out) if args.probe else None
    summary = dict(sequences=len(results), families=metrics, identical_input_cause_triplets=len(ambiguity),
                   real_cause_accuracy=None, warning='Counts describe scenario behavior, not cause accuracy or physical validation.',
                   elapsed_s=time.monotonic() - started, real_audit=audit and dict(frames=audit['frames'],
                       QA_eligible_frames=audit['QA_eligible_frames'], RCA_inputs_available=0))
    write(args.out / 'summary.json', summary)
    write(args.out / 'completed.json', dict(plan_sha256=sha(args.out / 'plan.json'),
          artifacts={p.relative_to(args.out).as_posix(): sha(p) for p in args.out.rglob('*') if p.is_file()},
          GPU_used=False, new_training=False, hardware_actions=0, confirmed_causes=0))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--probe', type=Path)
    main(parser.parse_args())
