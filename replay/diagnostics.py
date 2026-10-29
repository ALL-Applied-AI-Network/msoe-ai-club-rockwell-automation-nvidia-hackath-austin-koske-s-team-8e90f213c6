"""Opt-in completed-vial diagnosis. Patterns support hypotheses, not proof.

Consumes endpoint evidence after measurement QA; never runs a detector, changes
QA, authorizes product release or sends equipment commands. Lab direction is an
appearance descriptor, not a physical mixing fraction or residue-volume fit.
"""
from dataclasses import asdict, dataclass
from math import isfinite, sqrt
from statistics import mean

from .measurement import delta_e


@dataclass(frozen=True)
class DiagnosticPolicy:
    deviation_de2000: float = 5.0
    minimum_vials: int = 3
    recovery_ratio: float = 0.5
    previous_direction_cosine: float = 0.8

    def __post_init__(self):
        if (not isfinite(self.deviation_de2000) or self.deviation_de2000 <= 0
                or type(self.minimum_vials) is not int or self.minimum_vials < 3
                or not isfinite(self.recovery_ratio) or not 0 < self.recovery_ratio < 1
                or not isfinite(self.previous_direction_cosine)
                or not 0 < self.previous_direction_cosine <= 1):
            raise ValueError('Invalid research diagnosis policy')


def number(value):
    return type(value) in (int, float) and isfinite(value)


def lab(value):
    return (isinstance(value, list) and len(value) == 3
            and all(number(v) for v in value) and 0 <= value[0] <= 100)


def text(value):
    return isinstance(value, str) and bool(value.strip())


def qualified_reference(reference, document, first_start, evaluation_vials):
    if not isinstance(reference, dict):
        return False
    if any(reference.get(k) != document.get(k)
           for k in ('data_domain', 'camera_id', 'profile_id', 'clock_domain_id')):
        return False
    if (not text(reference.get('reference_id')) or not text(reference.get('source_vial_id'))
            or not lab(reference.get('lab'))
            or not number(reference.get('available_s'))
            or reference['available_s'] > first_start
            or reference.get('independent_of_evaluation') is not True
            or reference.get('source_vial_id') in evaluation_vials):
        return False
    q = reference.get('qualification', {})
    if not isinstance(q, dict):
        return False
    if q.get('demonstration') is True or q.get('condition') != 'accepted_recipe_endpoint':
        return False
    if document['data_domain'] == 'synthetic':
        return q.get('status') == 'simulated' and bool(q.get('evidence_id'))
    return (q.get('status') == 'human_reviewed' and bool(q.get('reviewer'))
            and bool(q.get('evidence_id')) and q.get('origin') == 'human_review')


def process_evidence(document, seen):
    checks = document.get('process_checks', [])
    if not isinstance(checks, list):
        raise ValueError('Process checks must be a list')
    supported = []
    for check in checks:
        if (not isinstance(check, dict) or check.get('kind') not in ('commanded_recipe', 'metered_delivery')
                or not number(check.get('available_s')) or check['available_s'] > document['as_of_s']
                or check.get('data_domain') != document['data_domain'] or not text(check.get('evidence_id'))
                or not text(check.get('recorded_by'))
                or check.get('session_id') != document['session_id']
                or check.get('recipe_id') != document['recipe_id']
                or check.get('clock_domain_id') != document['clock_domain_id']
                or not isinstance(check.get('vial_ids'), list) or not check['vial_ids']
                or not all(text(v) and v in seen for v in check['vial_ids'])
                or check.get('origin') not in (('simulation',) if document['data_domain'] == 'synthetic' else ('staff_export', 'operator_record'))):
            continue
        if check.get('matches_plan') is False:
            supported.append(dict(cause=check['kind'] + '_record_mismatch',
                                  support='attributed_process_record', evidence_id=check['evidence_id']))
    return supported


def diagnose(document, policy=DiagnosticPolicy()):
    """Analyze only evidence available by as_of_s in one declared clock domain.

    All vial positions must be supplied from process/identity evidence, never
    inferred from color. Qualification fields are attestations requiring external
    verification; this function cannot authenticate them.
    """
    if document.get('schema') != 'vial-diagnosis-input/v1':
        raise ValueError('Unknown diagnosis input schema')
    if document.get('data_domain') not in ('real', 'synthetic'):
        raise ValueError('Explicit real/synthetic domain required')
    if not number(document.get('as_of_s')):
        raise ValueError('Finite as_of_s in the declared clock domain required')
    if not isinstance(document.get('vials'), list):
        raise ValueError('Vial evidence must be a list')
    out = dict(schema='vial-diagnosis/v1', session_id=document.get('session_id'),
               data_domain=document['data_domain'], as_of_s=document['as_of_s'],
               policy=asdict(policy), status='unavailable', pattern=None,
               primary_hypothesis=None, hypotheses=[], evidence=[], blocked_reasons=[],
               cause_status='not_identified', next_checks=[],
               confirmed_root_cause=None, quality_decision=None, hardware_command=None,
               validation='software scenario only' if document['data_domain'] == 'synthetic'
               else 'requires independent physical validation',
               limitation='Pattern consistency and attributed process records do not establish a physical root cause.')
    reasons = []
    for key in ('session_id', 'camera_id', 'profile_id', 'clock_domain_id', 'recipe_id', 'context_evidence_id'):
        if not text(document.get(key)):
            reasons.append('missing_' + key)
    rows = []
    for row in document['vials']:
        if not isinstance(row, dict) or not number(row.get('available_s')):
            reasons.append('invalid_vial_availability')
        elif row['available_s'] <= document['as_of_s']:
            rows.append(row)
    if not rows:
        reasons.append('no_available_completed_vials')
    seen = set()
    previous_time = None
    previous_position = None
    first_start = min((r['start_s'] for r in rows if number(r.get('start_s'))), default=None)
    if (first_start is None or not number(document.get('context_available_s'))
            or document['context_available_s'] > first_start):
        reasons.append('recipe_context_not_available_before_evaluation')
    for i, row in enumerate(rows):
        prefix = 'vial_' + str(i + 1) + ':'
        if not text(row.get('vial_id')) or row.get('vial_id') in seen:
            reasons.append(prefix + 'missing_or_duplicate_vial_identity')
        if text(row.get('vial_id')):
            seen.add(row['vial_id'])
        for key in ('camera_id', 'profile_id', 'clock_domain_id', 'recipe_id'):
            if row.get(key) != document.get(key):
                reasons.append(prefix + key + '_mismatch')
        for key in ('observation_id', 'source_evidence_id', 'completion_evidence_id'):
            if not text(row.get(key)):
                reasons.append(prefix + 'missing_' + key)
        for key in ('qa_pass', 'camera_health_valid', 'alignment_confirmed', 'identity_confirmed', 'completed'):
            if row.get(key) is not True:
                reasons.append(prefix + key + '_not_confirmed')
        expected_temporal = 'simulated' if document['data_domain'] == 'synthetic' else 'human_reviewed'
        if row.get('temporal_reference_status') != expected_temporal:
            reasons.append(prefix + 'temporal_reference_unqualified')
        t, start, available = row.get('source_time_s'), row.get('start_s'), row.get('available_s')
        if (not number(t) or not number(start) or not number(available)
                or not 0 <= start <= t <= available):
            reasons.append(prefix + 'invalid_source_time')
        elif previous_time is not None and t <= previous_time:
            reasons.append(prefix + 'source_time_not_increasing')
        if number(t):
            previous_time = t
        position = row.get('position_after_change')
        if type(position) is not int or position < 1:
            reasons.append(prefix + 'position_after_change_unknown')
        elif previous_position is not None and position != previous_position + 1:
            reasons.append(prefix + 'sequence_gap_or_duplicate')
        if type(position) is int:
            previous_position = position
        if not lab(row.get('lab')):
            reasons.append(prefix + 'invalid_measured_lab')
    refs = document.get('references', [])
    if not isinstance(refs, list):
        raise ValueError('References must be a list')
    current_refs = [r for r in refs if isinstance(r, dict) and r.get('recipe_id') == document.get('recipe_id')]
    if len(current_refs) != 1 or first_start is None or not qualified_reference(current_refs[0], document, first_start, seen):
        reasons.append('current_recipe_reference_unqualified_or_ambiguous')
    if reasons:
        out['blocked_reasons'] = sorted(set(reasons))
        out['operator_message'] = 'Diagnosis unavailable: review source, reference, completion and sampling evidence.'
        return out

    expected = current_refs[0]['lab']
    distances = [delta_e(r['lab'], expected) for r in rows]
    out['evidence'] = [dict(vial_id=r['vial_id'], observation_id=r['observation_id'],
                            source_evidence_id=r['source_evidence_id'], position_after_change=r['position_after_change'],
                            source_time_s=r['source_time_s'], delta_e2000=d)
                       for r, d in zip(rows, distances)]
    out['current_reference_id'] = current_refs[0]['reference_id']
    process_support = process_evidence(document, seen)
    if not any(d > policy.deviation_de2000 for d in distances):
        if process_support:
            out.update(status='hypotheses', pattern='process_discrepancy_without_visual_deviation',
                       primary_hypothesis=process_support[0]['cause'], hypotheses=process_support,
                       next_checks=['Verify the attributed process record and its vial/clock association.'],
                       operator_message='Appearance is within the research band, but an attributed process record disagrees with the plan. Review the record; color does not clear the discrepancy.')
        else:
            out.update(status='no_deviation_observed', pattern='within_research_band',
                       operator_message='No deviation observed in eligible endpoints; this does not approve the product.')
        return out

    previous_id = document.get('previous_recipe_id')
    previous_refs = [r for r in refs if isinstance(r, dict) and previous_id
                     and previous_id != document['recipe_id'] and r.get('recipe_id') == previous_id]
    previous = previous_refs[0] if len(previous_refs) == 1 and qualified_reference(previous_refs[0], document, first_start, seen) else None
    out['previous_reference_id'] = previous['reference_id'] if previous else None
    cosine = None
    if previous is not None and delta_e(previous['lab'], expected) > 2 * policy.deviation_de2000:
        direction = [a - b for a, b in zip(previous['lab'], expected)]
        observed = [a - b for a, b in zip(rows[0]['lab'], expected)]
        scale = sqrt(sum(v * v for v in direction) * sum(v * v for v in observed))
        if scale > 0:
            cosine = sum(a * b for a, b in zip(direction, observed)) / scale
    enough = len(rows) >= policy.minimum_vials
    declining = (enough and distances[0] > policy.deviation_de2000
                 and mean(distances[-2:]) <= policy.recovery_ratio * distances[0])
    begins_at_change = bool(document.get('transition_evidence_id')) and previous_id != document['recipe_id'] and rows[0]['position_after_change'] == 1
    supported = declining and begins_at_change and cosine is not None and cosine >= policy.previous_direction_cosine
    out['appearance_descriptors'] = dict(previous_recipe_direction_cosine=cosine,
                                        late_to_first_deviation_ratio=mean(distances[-2:]) / distances[0] if enough and distances[0] else None,
                                        physical_residue_fraction=None, physical_residue_volume_ml=None)
    if supported:
        pattern = 'transient_toward_previous_recipe'
        primary = 'previous_recipe_carryover'
        message = 'Suspected carryover: the first endpoint shifted toward the previous recipe and later endpoints converged. Local optical and other process causes remain possible.'
    elif enough and all(d > policy.deviation_de2000 for d in distances[-policy.minimum_vials:]):
        pattern, primary = 'persistent_deviation', None
        message = 'Persistent recipe-appearance deviation: check actual recipe and dispensing records; cause remains undetermined.'
    elif declining:
        pattern, primary = 'transient_deviation_without_donor_attribution', None
        message = 'A transient deviation was observed, but the evidence does not identify the previous recipe as its source.'
    else:
        pattern, primary = 'insufficient_or_irregular_sequence', None
        message = 'Deviation observed; retain more completed-vial evidence before assigning a cause.'
    hypotheses = [dict(cause='previous_recipe_carryover', support='consistent_pattern' if supported else 'not_established'),
                  dict(cause='vial_local_optical_or_sampling_change', support='not_excluded_by_healthy_background'),
                  dict(cause='recipe_or_dispensing_mismatch', support='not_excluded_without_process_records')]
    if process_support:
        hypotheses = process_support + hypotheses
        primary = process_support[0]['cause']
        message = 'An attributed process record disagrees with the plan. Verify that record and the observed deviation before assigning a physical root cause.'
    out.update(status='hypotheses', pattern=pattern, primary_hypothesis=primary,
               hypotheses=hypotheses, operator_message=message,
               next_checks=['Review source-aligned masks and independent recipe references.',
                            'Obtain staff-provided recipe command and metered-delivery records.',
                            'Request a staff-approved matched changeover/conditioning comparison.'])
    return out


def main():
    import argparse
    import hashlib
    import json
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('Use a fresh output directory')
    data = json.loads(args.input.read_text())
    result = diagnose(data)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'input.snapshot.json').write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    (args.out / 'diagnosis.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    complete = dict(schema='vial-diagnosis-completion/v1',
                    helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    artifacts={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.out.iterdir()},
                    quality_decision=None, hardware_command=None)
    (args.out / 'completed.json').write_text(json.dumps(complete, indent=2) + '\n')
    print(json.dumps(dict(status=result['status'], pattern=result['pattern'],
                          primary_hypothesis=result['primary_hypothesis'], confirmed_root_cause=None)))


if __name__ == '__main__':
    main()
