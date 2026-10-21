import copy
import unittest
from compare_pipeline_backends import masks,frame_difference
from replay.geometry import encode_mask
import numpy as np


class PipelineAgreementTests(unittest.TestCase):
    def test_empty_masks_are_not_reported_as_segmentation_iou(self):
        result=masks(None,None)
        self.assertTrue(result['identical'])
        self.assertIsNone(result['iou_both_nonempty'])

    def test_disappearing_mask_is_reported(self):
        mask=np.zeros((480,640),np.uint8);mask[390:395,300:310]=1
        result=masks(encode_mask(mask),None)
        self.assertFalse(result['emptiness_equal'])
        self.assertEqual(result['symmetric_difference_pixels'],50)

    def row(self):
        return dict(source_frame=0,source_rgb_sha256='x',pts=0,time_base='1/30',seconds=0.,
            predictions=[],raw_masks=[],measurement=dict(measurement_valid=False,lab=None,
                quality_flags=['missing'],geometry=dict(mask=None,track_id=None,track_age=0)),
            fill_detection={'phase':'waiting'},bridge={'mode':'none'},camera_feature={'lab':None},
            recipe_decision={'status':'unavailable'})

    def test_wrong_source_frame_cannot_be_compared(self):
        left=self.row();right=copy.deepcopy(left);right['pts']=1
        with self.assertRaises(ValueError):frame_difference(left,right)

    def test_missing_color_is_not_zero_error_or_quality_accuracy(self):
        r=self.row();result=frame_difference(r,r)
        self.assertIsNone(result['eligible_raw_color_de2000'])
        self.assertIsNone(result['eligible_normalized_color_de2000'])


if __name__=='__main__':unittest.main()
