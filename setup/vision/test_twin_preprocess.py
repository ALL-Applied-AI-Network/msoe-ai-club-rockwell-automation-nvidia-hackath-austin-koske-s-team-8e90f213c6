import unittest
import numpy as np
from twin_camera_preprocess import anchors,normalize


class PreprocessTests(unittest.TestCase):
    def test_camera_gain_is_estimated_without_liquid_pixels(self):
        image=np.full((480,640,3),180,np.uint8);image[300:330,300:330]=[60,100,140]
        reference=anchors(image)
        altered=np.rint(image.astype(float)*[1.,.85,1.]).astype(np.uint8)
        result,metadata=normalize(altered,reference)
        self.assertTrue(metadata['healthy']);np.testing.assert_array_equal(result,image)

    def test_one_bad_anchor_blocks_inference_input(self):
        image=np.full((480,640,3),180,np.uint8);reference=anchors(image)
        image[160:208,112:160]=120
        result,metadata=normalize(image,reference)
        self.assertIsNone(result);self.assertFalse(metadata['healthy'])


if __name__=='__main__':unittest.main()
