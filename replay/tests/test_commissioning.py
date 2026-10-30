"""Synthetic engineering fixtures, never physical reference approvals or labels."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from replay.operator.commissioning import ReferenceGate,associate_physical_vial,digest,evidence_readiness,validate_observation
from replay.operator.rehearsal import DemonstrationJournal,fixture_observation,fixture_reference,guard_checks,verify_journal


class CommissioningTests(unittest.TestCase):
    def test_declared_failure_matrix(self):
        checks=guard_checks();self.assertGreaterEqual(len(checks),18)
        self.assertTrue(all(r['passed'] for r in checks),checks)

    def test_unreviewed_reference_does_not_discard_valid_raw_sample(self):
        observation=fixture_observation();before=deepcopy(observation)
        reference=fixture_reference();reference['qualification']['status']='unreviewed'
        gate=ReferenceGate();gate.install(reference);check=gate.check(observation)
        state=evidence_readiness(observation,check)
        self.assertTrue(state['proposal_display_available']);self.assertTrue(state['frozen_measurement_qa_pass'])
        self.assertFalse(state['qualified_product_decision_available']);self.assertFalse(state['recipe_comparison_ready'])
        self.assertEqual(observation,before)

    def test_reference_copy_cannot_be_qualified_by_mutating_callers_input(self):
        reference=fixture_reference();reference['qualification']['status']='unreviewed'
        gate=ReferenceGate();gate.install(reference);reference['qualification']['status']='human_reviewed'
        self.assertFalse(gate.check(fixture_observation())['usable'])

    def test_new_context_requires_recapture(self):
        gate=ReferenceGate();gate.install(fixture_reference())
        current=fixture_observation();current['target']['context_id']='NEXT-CONTEXT'
        self.assertFalse(gate.check(current)['usable']);self.assertNotIn('temporal_background',gate.references)
        self.assertFalse(gate.check(fixture_observation())['usable'])

    def test_unknown_recipe_and_completion_cannot_be_inferred(self):
        gate=ReferenceGate();reference=fixture_reference('recipe_endpoint');reference['recipe_id']='TEST-RECIPE';gate.install(reference)
        result=gate.check(fixture_observation(),'recipe_endpoint')
        self.assertIn('intended_recipe_or_reference_missing',result['reasons'])
        self.assertIn('fill_completion_unconfirmed',result['reasons'])
        self.assertIn('physical_vial_identity_unconfirmed',result['reasons'])

    def test_capture_still_does_not_gain_assumed_video_frame(self):
        observation=fixture_observation();observation['source']['clock_kind']='capture_only'
        with self.assertRaises(ValueError):validate_observation(observation)

    def test_camera_replay_identity_is_not_physical_identity(self):
        fill=fixture_observation();fixture=fixture_observation('final_fixture')
        self.assertEqual(associate_physical_vial(fill,fixture)['status'],'unlinked')
        attestation=dict(reviewer='TEST FIXTURE',physical_vial_id='TEST-VIAL',evidence_id='TEST-EVIDENCE',basis='test',demonstration=True,
            observation_ids=[fill['observation_id'],fixture['observation_id']])
        self.assertEqual(associate_physical_vial(fill,fixture,attestation)['status'],'unlinked')

    def test_no_cross_domain_physical_join(self):
        fill=fixture_observation();fixture=fixture_observation('final_fixture');fixture['data_domain']='real'
        self.assertEqual(associate_physical_vial(fill,fixture)['status'],'unlinked')

    def test_quality_or_hardware_command_is_not_an_observation(self):
        observation=fixture_observation();observation['hardware_command']='stop'
        with self.assertRaises(ValueError):validate_observation(observation)

    def test_five_class_capability_does_not_approve_mapping_or_product(self):
        observation=fixture_observation('final_fixture');observation['qualification']['alignment']='unreviewed'
        old=evidence_readiness(observation)
        self.assertIn('crooked_cap_and_damaged_body_unsupported',old['reasons'])
        observation['proposal']['supported_classes']=['pass','missing_cap','crooked_cap','damaged_vial','no_vial']
        new=evidence_readiness(observation)
        self.assertNotIn('crooked_cap_and_damaged_body_unsupported',new['reasons'])
        self.assertIn('slot_mapping_unreviewed',new['reasons'])
        self.assertFalse(new['qualified_product_decision_available'])

    def test_acknowledgement_journal_does_not_authorize_release(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'events.jsonl';journal=DemonstrationJournal(path)
            source=dict(source_id='TEST',pts=1,time_base='1/1')
            journal.append('review_requested',dict(source=source));journal.append('acknowledged',dict(source=source,quality_release_authorized=False))
            events=verify_journal(path);self.assertEqual(events[0]['payload']['source'],source)
            self.assertFalse(events[-1]['payload']['quality_release_authorized'])
            rows=path.read_text().splitlines();event=json.loads(rows[0]);event['payload']['source']['pts']=2;rows[0]=json.dumps(event)
            path.write_text('\n'.join(rows)+'\n')
            with self.assertRaises(ValueError):verify_journal(path)


if __name__=='__main__':unittest.main()
