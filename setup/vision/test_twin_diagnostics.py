"""Reference range/phase safeguards and custom-plan conservation."""
import copy
import unittest
import numpy as np
from twin_contract import default_plan,validate_plan,episode_states
from twin_diagnostics import lookup,DiagnosticSuite
from prepare_twin_experiments import plan
from replay.geometry import encode_mask
from replay.measurement import rgb_to_lab


class DiagnosticTests(unittest.TestCase):
    def test_normalization_removes_camera_gain_but_keeps_liquid_change(self):
        rgb=np.full((480,640,3),180,dtype=np.uint8);mask=np.zeros((480,640),bool);mask[300:330,310:340]=True
        rgb[mask]=[60,100,150]
        def row(image):
            return dict(measurement=dict(measurement_valid=True,lab=rgb_to_lab(np.median(image[mask],axis=0)).tolist(),
                geometry=dict(mask=encode_mask(mask))))
        header=dict(phase_key='1',metered_ml=[2.,0.,0.]);suite=DiagnosticSuite()
        suite.update(header,row(rgb),rgb,True);suite.reset()
        gained=np.rint(rgb.astype(float)*[1.,.85,1.]).astype(np.uint8)
        for _ in range(3):result=suite.update(header,row(gained),gained,False)
        self.assertTrue(result['background_healthy'])
        self.assertLess(result['methods']['metered_normalized_guarded']['delta_e2000'],.01)
        self.assertFalse(result['methods']['metered_normalized_guarded']['warning'])
        changed=rgb.copy();changed[mask]=[140,20,40]
        changed=np.rint(changed.astype(float)*[1.,.85,1.]).astype(np.uint8)
        for _ in range(3):result=suite.update(header,row(changed),changed,False)
        self.assertTrue(result['methods']['metered_normalized_guarded']['warning'])

    def test_interpolation_is_bounded_and_phase_specific(self):
        reference={('1',1.):[10.,0.,0.],('1',3.):[30.,0.,0.],('2',9.):[90.,0.,0.]}
        self.assertIsNone(lookup(reference,'1',2.,exact=True)[0])
        self.assertEqual(lookup(reference,'1',2.)[0],[20.,0.,0.])
        self.assertIsNone(lookup(reference,'1',4.)[0])
        self.assertIsNone(lookup(reference,'2',2.)[0])
        reference[('1',7.)]=[70.,0.,0.]
        self.assertEqual(lookup(reference,'1',5.)[1],'reference_gap_too_large')

    def test_custom_flow_keeps_quantity_but_changes_time(self):
        p=plan();validate_plan(p)
        nominal=list(episode_states(p,p['episodes'][1]));slow=list(episode_states(p,p['episodes'][4]))
        self.assertGreater(len(slow),len(nominal))
        np.testing.assert_allclose(slow[-1]['actual_components_ml'],nominal[-1]['actual_components_ml'],atol=1e-8)
        np.testing.assert_allclose(slow[-1]['metered_ml'],p['volumes_ml'])
        for s in slow:self.assertAlmostEqual(sum(s['actual_components_ml']),sum(s['metered_ml']))
        p['volumes_ml']=[5.,10.,10.]
        self.assertEqual(next(episode_states(p,p['episodes'][0]))['recipe_key'],'synthetic-5-10-10-order-0-1-2')

    def test_missing_reference_or_bad_plan_cannot_be_approved(self):
        self.assertEqual(lookup({},'1',2.)[1],'outside_reference_range')
        p=default_plan();p['cadence_s']=.1
        with self.assertRaises(ValueError):validate_plan(p)
        p=default_plan();p['episodes'][1]['sensor_gain']=[1.,float('nan'),1.]
        with self.assertRaises(ValueError):validate_plan(p)


if __name__=='__main__':unittest.main()
