"""Camera anchoring, signal preservation and process-metadata boundary checks."""
import json,unittest
import numpy as np
from stage2_camera_reference import correction,perturb,feature,anchor_health
from stage2_guard import GuardedEvidence
from test_stage2_policy import sample


def guarded(t=1.,**kw):
    s=sample(t,process_state_source_s=t,process_phase_key='T100',camera_reference_key='c1',
             reference_camera_key='c1',camera_health_valid=True)
    return dict(s,**kw)


class CameraReferenceTests(unittest.TestCase):
    def setUp(self):
        self.anchors=np.array([[130,135,140],[155,160,165],[185,190,195]],float)
        self.liquid=np.array([[35,65,85],[36,66,86],[34,64,84]],float)

    def test_gain_correction_uses_background_and_inverts_common_gain(self):
        gain=[1.1,.9,1.2]
        noisy=perturb(self.liquid,gain);anchors=perturb(self.anchors,gain)
        corrected,_,_=correction(noisy,anchors,self.anchors,'linear_gain')
        np.testing.assert_allclose(corrected,self.liquid,atol=1e-9)
        self.assertTrue(anchor_health(anchors,self.anchors,.03)[0])

    def test_liquid_only_color_change_is_not_normalized_away(self):
        altered=perturb(self.liquid,[.6,1.4,.9])
        corrected,_,_=correction(altered,self.anchors,self.anchors,'linear_gain')
        np.testing.assert_allclose(corrected,altered,atol=1e-9)
        self.assertFalse(np.allclose(corrected,self.liquid))

    def test_disagreeing_or_clipped_anchors_are_rejected(self):
        changed=self.anchors.copy();changed[0]*=.8
        self.assertFalse(anchor_health(changed,self.anchors,.03)[0])
        changed=self.anchors.copy();changed[1]=255
        self.assertFalse(anchor_health(changed,self.anchors,.03)[0])


class GuardTests(unittest.TestCase):
    def test_bad_metadata_abstains_without_crashing_or_nonfinite_json(self):
        for key in ('source_time_s','decision_time_s','fill_started_s','reference_available_s','context_available_s','process_state_source_s'):
            for value in (None,float('nan'),'bad'):
                with self.subTest(key=key,value=value):
                    r=GuardedEvidence().update(guarded(**{key:value}))
                    self.assertEqual(r['status'],'unavailable');json.dumps(r,allow_nan=False)

    def test_stale_process_and_camera_faults_block_evidence(self):
        for changes in [dict(process_state_source_s=0.),dict(process_state_source_s=2.),
            dict(process_phase_key='T200'),dict(camera_health_valid=False),
            dict(camera_health_valid='true'),dict(reference_camera_key='other')]:
            with self.subTest(changes=changes):
                g=GuardedEvidence()
                for t in (1.,1.2,1.4):
                    r=g.update(guarded(t,**changes));self.assertIsNone(r['warning_time_s'])

    def test_block_breaks_consecutive_count_but_preserves_existing_hold(self):
        g=GuardedEvidence();g.update(guarded(1));g.update(guarded(1.2))
        g.update(guarded(1.4,camera_health_valid=False))
        self.assertIsNone(g.update(guarded(1.6))['warning_time_s'])
        self.assertIsNone(g.update(guarded(1.8))['warning_time_s'])
        self.assertIsNotNone(g.update(guarded(2.))['warning_time_s'])
        r=g.update(guarded(2.2,process_state_source_s=0.))
        self.assertEqual(r['status'],'review_required');self.assertIsNotNone(r['warning_time_s'])

    def test_normalized_camera_evidence_does_not_approve_real_geometry(self):
        r=GuardedEvidence().update(guarded(geometry_status='unreviewed_model_proposals'))
        self.assertEqual(r['status'],'unavailable');self.assertEqual(r['reason'],'unreviewed_measurement_geometry')


class TrajectoryTests(unittest.TestCase):
    def profile(self):
        return dict(recipe_key='a',phase_key='ingredient-1',camera_profile='c',cumulative_ml=[0.,10.],
                    lab=[[20.,0.,0.],[40.,10.,0.]],reference_id='r',available_s=-1.,reference_status='simulated')

    def test_reference_depends_on_phase_and_meter_progress(self):
        from stage2_trajectory import TrajectoryBank
        bank=TrajectoryBank([self.profile()])
        r=bank.lookup('a','ingredient-1','c',5.,0.)
        np.testing.assert_allclose(r['expected_lab'],[30.,5.,0.])
        self.assertEqual(bank.lookup('a','ingredient-2','c',5.,0.)['status'],'unavailable')
        self.assertEqual(bank.lookup('b','ingredient-1','c',5.,0.)['status'],'unavailable')

    def test_missing_future_or_extrapolated_reference_is_unavailable(self):
        from stage2_trajectory import TrajectoryBank
        p=self.profile();bank=TrajectoryBank([p])
        for value in (-1.,11.,None,float('nan')):
            self.assertEqual(bank.lookup('a','ingredient-1','c',value,0.)['status'],'unavailable')
        self.assertEqual(bank.lookup('a','ingredient-1','other',5.,0.)['status'],'unavailable')
        p['available_s']=10.;bank=TrajectoryBank([p])
        self.assertEqual(bank.lookup('a','ingredient-1','c',5.,0.)['reason'],'future_reference')

    def test_duplicate_or_nonmonotonic_bank_is_rejected(self):
        from stage2_trajectory import TrajectoryBank
        p=self.profile()
        with self.assertRaises(ValueError):TrajectoryBank([p,p])
        p['cumulative_ml']=[10.,0.]
        with self.assertRaises(ValueError):TrajectoryBank([p])


if __name__=='__main__':unittest.main()
