"""Verify diagnosis artifacts and construct alternative synthetic mechanisms.

The certificates establish equivalence inside the analytic feature model, not
physical feasibility/calibration on the CSI line or real cause accuracy.
"""
import argparse
from collections import Counter
import itertools
import json
from pathlib import Path

import numpy as np
from experiment_root_cause import conditioned, dispense, sha, write
from experiment_stage2_policy import appearance
from twin_contract import SharedPath


def main(args):
    args.out.mkdir(parents=True, exist_ok=False)
    complete = json.loads((args.run / 'completed.json').read_text())
    for relative, expected in complete['artifacts'].items():
        if sha(args.run / relative) != expected:
            raise ValueError('Frozen result changed: ' + relative)
    plan = json.loads((args.run / 'plan.json').read_text())
    rows = json.loads((args.run / 'scenario-results.json').read_text())
    summary = json.loads((args.run / 'summary.json').read_text())
    for family, metric in summary['families'].items():
        selected = [r for r in rows if r['private_family'] == family]
        if len(selected) != metric['sequences'] or dict(Counter(r['status'] for r in selected)) != metric['statuses']:
            raise ValueError('Summary count mismatch')
        if any(r['confirmed_cause'] is not None for r in selected):
            raise ValueError('Unsupported confirmed-cause claim')
    groups = json.loads((args.run / 'identifiability-pairs.json').read_text())
    for group in groups:
        if (len({r['input_sha256'] for r in group['indistinguishable_causes']}) != 1
                or len({r['output_sha256'] for r in group['indistinguishable_causes']}) != 1):
            raise ValueError('Input-equivalence assertion failed')
    certificates = []
    for recipe, (a, b) in enumerate(plan['profiles']):
        for order in itertools.permutations(range(3)):
            for capacity in plan['capacities_ml']:
                path, _ = conditioned(a, order, capacity)
                _, expected_parts = conditioned(b, order, capacity)
                before = list(path.inventory)
                expected_rgb = appearance(expected_parts)
                endpoints = []
                for vial in range(1, 5):
                    carried = dispense(path, b, order)
                    # Different mechanism: zero hold-up, with actual dosing changed
                    # to the carried endpoint's constituent amounts.
                    wrong_dose = dispense(SharedPath(0.), carried, order)
                    carried_rgb = appearance(carried)
                    wrong_dose_rgb = appearance(wrong_dose)
                    # Different feature mechanism: nominal composition with a
                    # per-vial local RGB gain; background anchors are unchanged.
                    local_gain = carried_rgb / expected_rgb
                    optical_rgb = expected_rgb * local_gain
                    if not (np.allclose(carried, wrong_dose, atol=1e-10, rtol=0)
                            and np.allclose(carried_rgb, wrong_dose_rgb, atol=1e-10, rtol=0)
                            and np.allclose(carried_rgb, optical_rgb, atol=1e-10, rtol=0)):
                        raise ValueError('Alternative-mechanism equivalence failed')
                    endpoints.append(dict(vial=vial, carryover_components_ml=carried.tolist(),
                        alternative_zero_holdup_doses_ml=wrong_dose.tolist(),
                        alternative_local_RGB_gain=local_gain.tolist(),
                        ideal_RGB_max_difference=float(max(np.max(abs(carried_rgb - wrong_dose_rgb)),
                                                          np.max(abs(carried_rgb - optical_rgb))))))
                certificates.append(dict(profile=recipe, dispense_order=list(order), capacity_ml=capacity,
                    initial_path_components_ml=before, expected_recipe_components_ml=list(b), endpoints=endpoints))
    write(args.out / 'alternative-mechanisms.json', certificates)
    result = dict(frozen_artifacts_verified=len(complete['artifacts']), scenario_counts_recomputed=True,
                  scenario_sequences=len(rows), identical_input_cause_triplets=len(groups),
                  distinct_analytic_process_settings=len(certificates),
                  equivalent_endpoints=sum(len(c['endpoints']) for c in certificates),
                  max_ideal_RGB_difference=max(v['ideal_RGB_max_difference'] for c in certificates for v in c['endpoints']),
                  mechanism_claim='Equivalence under the stated analytic assumptions; no real optical/physical validation.',
                  references_and_observations_share_optical_family=True,
                  real_cause_accuracy=None, physical_cause_confirmed=False, GPU_used=False,
                  helper_sha256=sha(Path(__file__)), plan_sha256=sha(args.run / 'plan.json'))
    write(args.out / 'verification.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    main(parser.parse_args())
