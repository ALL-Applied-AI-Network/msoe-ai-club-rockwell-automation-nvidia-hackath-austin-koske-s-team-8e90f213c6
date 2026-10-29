import unittest
import numpy as np
from compare_liquid_visibility import clahe_rgb,views


class VisibilityChecks(unittest.TestCase):
    def test_source_preserved_and_identical_reference_has_zero_difference(self):
        rgb=np.random.default_rng(19).integers(0,256,(48,64,3),dtype=np.uint8);original=rgb.copy()
        result=views(rgb,rgb.copy(),30,0,[10,5,50,40])
        np.testing.assert_array_equal(rgb,original)
        np.testing.assert_array_equal(result['raw'],original[5:40,10:50])
        self.assertFalse(result['past_difference'].any())
        self.assertEqual(clahe_rgb(rgb).shape,rgb.shape)

    def test_only_earlier_reference_and_valid_crop_are_accepted(self):
        rgb=np.full((48,64,3),100,np.uint8)
        for earlier in (30,36):
            with self.assertRaises(ValueError):views(rgb,rgb,30,earlier,[0,0,20,40])
        with self.assertRaises(ValueError):views(rgb,rgb,30,0,[-1,0,20,40])


if __name__=='__main__':unittest.main()
