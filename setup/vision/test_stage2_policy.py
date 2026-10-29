"""Contract tests for time, context and operator acknowledgement boundaries."""
import unittest
import numpy as np
from stage2_policy import Policy, EvidencePolicy, ActionCoordinator


def sample(t=1.,**changes):
    s=dict(session_id='s1',vial_id='v1',source_time_s=t,decision_time_s=t+.05,
        recipe_key='recipe-a',reference_recipe_key='recipe-a',phase_key='T100',expected_phase_key='T100',
        measurement_profile='camera-v1',reference_measurement_profile='camera-v1',reference_id='r1',
        reference_available_s=-1.,context_available_s=0.,fill_started_s=0.,
        geometry_status='simulated',vial_identity_unambiguous=True,measurement_valid=True,
        expected_lab=[40.,0.,0.],observed_lab=[40.,40.,0.])
    return dict(s,**changes)


class EvidenceContracts(unittest.TestCase):
    def test_warning_requires_current_consecutive_samples(self):
        p=EvidencePolicy()
        self.assertEqual(p.update(sample(1))['status'],'collecting_evidence')
        p.update(sample(1.2,measurement_valid=False))
        self.assertIsNone(p.update(sample(1.4))['warning_time_s'])
        self.assertIsNone(p.update(sample(1.6))['warning_time_s'])
        self.assertAlmostEqual(p.update(sample(1.8))['warning_time_s'],1.85)

    def test_evidence_cannot_cross_phase_gap_or_vial(self):
        for change in [dict(phase_key='T200',expected_phase_key='T200'),dict(vial_id='v2')]:
            p=EvidencePolicy();p.update(sample(1));p.update(sample(1.2))
            self.assertIsNone(p.update(sample(1.4,**change))['warning_time_s'])
        p=EvidencePolicy();p.update(sample(1));p.update(sample(1.2))
        self.assertIsNone(p.update(sample(2))['warning_time_s'])

    def test_guarded_inputs_never_issue_a_warning(self):
        cases=[dict(recipe_key=None),dict(geometry_status='unreviewed_model_proposals'),
            dict(vial_identity_unambiguous=False),dict(reference_available_s=2.),
            dict(context_available_s=99.),dict(reference_recipe_key='other'),
            dict(reference_measurement_profile='other'),dict(expected_phase_key='other'),
            dict(reference_id=None),dict(phase_key=None),dict(observed_lab=[np.nan,0,0]),
            dict(observed_lab=[1,2]),dict(decision_time_s=99.),dict(measurement_valid=False)]
        for change in cases:
            with self.subTest(change=change):
                p=EvidencePolicy()
                for t in [1.,1.2,1.4,1.6]:
                    result=p.update(sample(t,**change))
                    self.assertEqual(result['status'],'unavailable');self.assertFalse(result['hardware_action_authorized'])

    def test_midfill_context_change_abstains_and_does_not_clear_latch(self):
        p=EvidencePolicy()
        for t in [1,1.2,1.4]:p.update(sample(t))
        r=p.update(sample(1.6,recipe_key='b',reference_recipe_key='b'))
        self.assertEqual(r['status'],'review_required')
        self.assertEqual(r['reason'],'recipe_or_reference_changed_during_fill')
        self.assertEqual(p.update(sample(1.8,measurement_valid=False))['status'],'review_required')

    def test_duplicate_time_does_not_accumulate_evidence(self):
        p=EvidencePolicy();p.update(sample(1))
        self.assertEqual(p.update(sample(1))['reason'],'nonincreasing_sample_time')
        self.assertIsNone(p.update(sample(1.2))['warning_time_s'])
        self.assertIsNone(p.update(sample(1.4))['warning_time_s'])


class ActionContracts(unittest.TestCase):
    def test_complete_operator_loop_needs_two_postflush_good_vials(self):
        a=ActionCoordinator('s1')
        def done(v,status,t):return a.complete(v,'r',status,t,session_id='s1')
        self.assertEqual(done('v1','review_required',10)['action'],'hold_review_requested')
        self.assertEqual(done('v2','review_required',20)['action'],'flush_recommended')
        # A good observation without an acknowledged intervention cannot release a hold.
        self.assertEqual(done('v3','within_expected_band',30)['action'],'verify_next_fill')
        a.acknowledge_flush(31,'operator','receipt',session_id='s1')
        self.assertTrue(a.hold)
        self.assertEqual(done('v4','within_expected_band',40)['action'],'verify_next_fill')
        self.assertEqual(done('v5','within_expected_band',50)['action'],'resume_recommended')
        self.assertTrue(all(not e['hardware_action_authorized'] for e in a.events))

    def test_unknown_evidence_breaks_recovery_sequence(self):
        a=ActionCoordinator('s1');a.complete('v1','r','review_required',1,session_id='s1')
        a.acknowledge_flush(2,'operator','receipt',session_id='s1')
        a.complete('v2','r','within_expected_band',3,session_id='s1')
        a.complete('v3','r','unavailable',4,session_id='s1')
        self.assertEqual(a.complete('v4','r','within_expected_band',5,session_id='s1')['action'],'verify_next_fill')
        self.assertEqual(a.complete('v5','r','within_expected_band',6,session_id='s1')['action'],'resume_recommended')

    def test_duplicate_identity_source_and_ack_are_rejected(self):
        a=ActionCoordinator('s1');a.complete('v1','r','review_required',1,session_id='s1')
        with self.assertRaises(ValueError):a.complete('v1','r','review_required',2,session_id='s1')
        with self.assertRaises(ValueError):a.complete('v2','r','within_expected_band',2,session_id='s2')
        with self.assertRaises(ValueError):a.acknowledge_flush(2,'','',session_id='s1')


class PhaseCheckpointContracts(unittest.TestCase):
    def test_checkpoint_uses_only_qualified_past_samples_and_is_not_completion(self):
        from analyze_stage2_evidence import checkpoints
        from replay.config import Config
        rows=[]
        for i in range(11):
            rows.append(dict(source_frame=i*6,seconds=i*.2,
                fill_detection=dict(detection_id='f1',episode=1,phase='ingredient_pause'),
                measurement=dict(measurement_valid=True,lab=[40.,0.,0.],geometry=dict(track_id=1))))
        prefix=checkpoints(rows[:5],Config())
        self.assertEqual(prefix,[])
        full=checkpoints(rows,Config())
        self.assertEqual(len(full),1)
        self.assertEqual(full[0]['available_frame'],30)
        self.assertEqual(full[0]['sample_frames'],[18,24,30])
        self.assertFalse(full[0]['final_dispense_complete_known'])
        self.assertEqual(full[0]['recipe_assessment'],'unavailable')
        self.assertEqual(checkpoints(rows[:6],Config()),full)

    def test_motion_bad_qa_and_track_changes_prevent_checkpoint(self):
        from analyze_stage2_evidence import checkpoints
        from replay.config import Config
        rows=[dict(source_frame=i*6,seconds=i*.2,
            fill_detection=dict(detection_id='f1',episode=1,phase='ingredient_pause'),
            measurement=dict(measurement_valid=i%2==0,lab=[40.,0.,0.],geometry=dict(track_id=i))) for i in range(12)]
        self.assertEqual(checkpoints(rows,Config()),[])
        for r in rows:r['measurement']['measurement_valid']=True
        self.assertEqual(checkpoints(rows,Config()),[])


if __name__=='__main__':unittest.main()
