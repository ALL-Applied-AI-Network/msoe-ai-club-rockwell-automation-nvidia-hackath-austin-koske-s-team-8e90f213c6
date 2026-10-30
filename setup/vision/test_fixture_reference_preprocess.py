import unittest
import numpy as np

from fixture_reference_preprocess import estimate_scene_gain


class ReferenceContextTests(unittest.TestCase):
    def test_known_reference_gain_recovered_without_center_target_pixels(self):
        golden=np.random.default_rng(91).uniform(.15,.7,(60,3,96,96)).astype('float32');current=golden.copy()
        current[:,:,32:64,32:64]=.02  # Distinct target changes cannot drive gain.
        result=estimate_scene_gain(current,golden*.9)
        self.assertTrue(result['available']);self.assertAlmostEqual(result['gain'],1/.9,places=5)
        self.assertFalse(result['physical_reference_qualified'])

    def test_uninformative_scene_cannot_be_qualified(self):
        rgb=np.zeros((60,3,96,96),dtype='float32')
        result=estimate_scene_gain(rgb,rgb)
        self.assertFalse(result['available']);self.assertIsNone(result['gain'])

    def test_gain_does_not_approve_registration(self):
        rgb=np.full((60,3,96,96),.5,dtype='float32')
        result=estimate_scene_gain(rgb,rgb)
        self.assertTrue(result['available']);self.assertFalse(result['reference_alignment_qualified'])


if __name__=='__main__':unittest.main()
