"""Diagnosis boundaries; all test references/identities are software fixtures."""
from copy import deepcopy
import unittest

from replay.diagnostics import diagnose


def fixture():
    doc = dict(schema='vial-diagnosis-input/v1', data_domain='synthetic', session_id='test',
               camera_id='camera', profile_id='profile', clock_domain_id='clock',
               recipe_id='B', previous_recipe_id='A', transition_evidence_id='sim-transition',
               context_evidence_id='sim-plan', context_available_s=0., as_of_s=30., references=[], vials=[])
    for recipe, color in [('A', [55., -35., 45.]), ('B', [55., 35., -35.])]:
        doc['references'].append(dict(reference_id='sim-ref-' + recipe, recipe_id=recipe, lab=color,
            data_domain='synthetic', camera_id='camera', profile_id='profile', clock_domain_id='clock',
            available_s=0., source_vial_id='sim-reference-' + recipe, independent_of_evaluation=True,
            qualification=dict(status='simulated', condition='accepted_recipe_endpoint', evidence_id='sim-preparation')))
    for i, color in enumerate([[55., -10., 15.], [55., 30., -30.], [55., 35., -35.]], 1):
        doc['vials'].append(dict(vial_id='v' + str(i), observation_id='o' + str(i),
            source_evidence_id='sim-feature-' + str(i), recipe_id='B', camera_id='camera',
            profile_id='profile', clock_domain_id='clock', position_after_change=i,
            source_time_s=float(i * 8), start_s=float(i * 8 - 6), available_s=float(i * 8 + .1),
            completed=True, completion_evidence_id='sim-complete', qa_pass=True,
            camera_health_valid=True, alignment_confirmed=True, identity_confirmed=True,
            temporal_reference_status='simulated', lab=color))
    return doc


class DiagnosisContracts(unittest.TestCase):
    def test_sequence_supports_carryover_without_confirming_it(self):
        result = diagnose(fixture())
        self.assertEqual(result['pattern'], 'transient_toward_previous_recipe')
        self.assertEqual(result['primary_hypothesis'], 'previous_recipe_carryover')
        self.assertIn('vial_local_optical_or_sampling_change', [h['cause'] for h in result['hypotheses']])
        for key in ['confirmed_root_cause', 'quality_decision', 'hardware_command']:
            self.assertIsNone(result[key])

    def test_one_first_vial_mismatch_is_insufficient(self):
        doc = fixture(); doc['vials'] = doc['vials'][:1]
        self.assertIsNone(diagnose(doc)['primary_hypothesis'])

    def test_failed_missing_or_untrusted_measurements_abstain(self):
        for change in [dict(qa_pass=False), dict(camera_health_valid=False),
                       dict(alignment_confirmed=False), dict(identity_confirmed=False),
                       dict(completed=False), dict(temporal_reference_status='unreviewed'),
                       dict(recipe_id='other'), dict(profile_id='other'),
                       dict(clock_domain_id='other'), dict(lab=[float('nan'), 0, 0])]:
            with self.subTest(change=change):
                doc = fixture(); doc['vials'][1].update(change)
                self.assertEqual(diagnose(doc)['status'], 'unavailable')

    def test_duplicate_or_missing_vials_cannot_count_as_recovery(self):
        doc = fixture(); doc['vials'][2]['vial_id'] = doc['vials'][1]['vial_id']
        self.assertEqual(diagnose(doc)['status'], 'unavailable')
        doc = fixture(); doc['vials'].pop(1)
        self.assertEqual(diagnose(doc)['status'], 'unavailable')

    def test_unqualified_future_or_self_reference_is_blocked(self):
        for change in [dict(available_s=29), dict(independent_of_evaluation=False),
                       dict(source_vial_id='v1'), dict(profile_id='other'), dict(lab=None)]:
            doc = fixture(); doc['references'][1].update(change)
            self.assertEqual(diagnose(doc)['status'], 'unavailable')

    def test_simulated_or_demonstration_reference_cannot_qualify_real_evidence(self):
        doc = fixture(); doc['data_domain'] = 'real'
        for r in doc['references']:
            r['data_domain'] = 'real'
        for r in doc['vials']:
            r['temporal_reference_status'] = 'human_reviewed'
        self.assertEqual(diagnose(doc)['status'], 'unavailable')
        for ref in doc['references']:
            ref['qualification'].update(status='human_reviewed', origin='human_review',
                                        reviewer='Unit-test fixture', demonstration=True)
        self.assertEqual(diagnose(doc)['status'], 'unavailable')

    def test_future_vials_do_not_change_a_prefix_diagnosis(self):
        doc = fixture(); doc['as_of_s'] = 9.
        prefix = deepcopy(doc); prefix['vials'] = prefix['vials'][:1]
        self.assertEqual(diagnose(doc), diagnose(prefix))

    def test_missing_previous_reference_prevents_donor_attribution(self):
        doc = fixture(); doc['references'] = doc['references'][1:]
        self.assertIsNone(diagnose(doc)['primary_hypothesis'])

    def test_persistent_deviation_does_not_identify_wrong_recipe_or_pump(self):
        doc = fixture()
        for r in doc['vials']:
            r['lab'] = [55., -35., 45.]
        result = diagnose(doc)
        self.assertEqual(result['pattern'], 'persistent_deviation')
        self.assertIsNone(result['primary_hypothesis'])

    def test_process_records_require_matching_identity_clock_and_availability(self):
        doc = fixture()
        record = dict(kind='commanded_recipe', matches_plan=False, available_s=25.,
                      data_domain='synthetic', evidence_id='sim-controller-record', recorded_by='simulation',
                      session_id='test', recipe_id='B', clock_domain_id='clock',
                      vial_ids=['v1'], origin='simulation')
        for change in [dict(session_id='other'), dict(recipe_id='other'), dict(clock_domain_id='other'),
                       dict(available_s=40.), dict(vial_ids=['unseen']), dict(origin='model')]:
            invalid = dict(record, **change); doc['process_checks'] = [invalid]
            self.assertEqual(diagnose(doc)['primary_hypothesis'], 'previous_recipe_carryover')
        doc['process_checks'] = [record]
        result = diagnose(doc)
        self.assertEqual(result['primary_hypothesis'], 'commanded_recipe_record_mismatch')
        self.assertIsNone(result['confirmed_root_cause'])

    def test_input_is_not_modified(self):
        doc = fixture(); original = deepcopy(doc); diagnose(doc)
        self.assertEqual(doc, original)

    def test_normal_appearance_cannot_clear_a_process_record_discrepancy(self):
        doc = fixture()
        for row in doc['vials']:
            row['lab'] = [55., 35., -35.]
        doc['process_checks'] = [dict(kind='metered_delivery', matches_plan=False, available_s=25.,
            data_domain='synthetic', evidence_id='sim-meter', recorded_by='simulation',
            session_id='test', recipe_id='B', clock_domain_id='clock', vial_ids=['v1'], origin='simulation')]
        result = diagnose(doc)
        self.assertEqual(result['status'], 'hypotheses')
        self.assertEqual(result['pattern'], 'process_discrepancy_without_visual_deviation')
        self.assertEqual(result['primary_hypothesis'], 'metered_delivery_record_mismatch')
        self.assertIsNone(result['quality_decision'])


if __name__ == '__main__':
    unittest.main()
