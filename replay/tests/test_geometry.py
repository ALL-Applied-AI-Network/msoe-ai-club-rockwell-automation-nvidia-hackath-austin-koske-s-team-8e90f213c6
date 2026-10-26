import unittest
from unittest.mock import patch

import cv2
import numpy as np

from replay.config import Config
from replay.geometry import VialGeometry, decode_mask, encode_mask
from replay.measurement import Extractor, rgb_to_lab


FIXTURE_BOXES = [dict(class_id=0,box=[278,65,340,430],confidence=.99),
                 dict(class_id=1,box=[288,340,330,390],confidence=.99)]


class GeometryChecks(unittest.TestCase):
    def scene(self,dx=0):
        rgb=np.full((480,640,3),160,np.uint8)
        rgb[231:253,264:338]=20
        rgb[340:390,288+dx:330+dx]=(40,120,70)
        return rgb

    def test_exact_mask_sampling_excludes_wall_glass_and_base(self):
        extractor=Extractor(Config())
        rgb=self.scene()
        with patch.object(VialGeometry,'detect',return_value=FIXTURE_BOXES):
            for _ in range(3):
                result=extractor.measure(rgb)
        self.assertTrue(result['measurement_valid'])
        mask=decode_mask(result['geometry']['mask']).astype(bool)
        self.assertEqual(int(mask.sum()),result['geometry']['mask_pixels'])
        truth=np.zeros((480,640),bool); truth[340:390,288:330]=True
        self.assertFalse(np.any(mask & ~truth))
        self.assertEqual(result['rgb'],np.median(rgb[mask],axis=0).tolist())
        np.testing.assert_allclose(result['lab'],rgb_to_lab([40,120,70]))
        np.testing.assert_array_equal(decode_mask(encode_mask(mask)).astype(bool),mask)

    def test_no_detection_drops_mask_and_new_track_cannot_reuse_old_pixels(self):
        g=VialGeometry(Config())
        with patch.object(g,'detect',return_value=FIXTURE_BOXES):
            rows=[g.update(self.scene()) for _ in range(3)]
        old=rows[-1]['track_id']
        with patch.object(g,'detect',return_value=[]):
            missing=g.update(self.scene())
        self.assertIsNone(missing['mask'])
        self.assertIsNone(missing['track_id'])
        shifted=[dict(b,box=[v+(10 if i%2==0 else 0) for i,v in enumerate(b['box'])]) for b in FIXTURE_BOXES]
        with patch.object(g,'detect',return_value=shifted):
            new=g.update(self.scene(10))
        self.assertNotEqual(new['track_id'],old)
        self.assertIn('vial_track_not_stable',new['quality_flags'])
        mask=decode_mask(new['mask']).astype(bool)
        self.assertFalse(mask[:,:298].any())
        restarted=VialGeometry(Config())
        self.assertIsNone(restarted.previous)
        self.assertEqual(restarted.age,0)

    def test_clear_neutral_holder_and_wrong_vial_liquid_are_rejected(self):
        g=VialGeometry(Config())
        neutral=np.full((480,640,3),80,np.uint8)
        with patch.object(g,'detect',return_value=FIXTURE_BOXES):
            row=g.update(neutral)
        self.assertIsNone(row['mask'])
        self.assertIn('insufficient_coherent_colored_liquid',row['quality_flags'])
        wrong=[FIXTURE_BOXES[0],dict(FIXTURE_BOXES[1],box=[400,340,440,390])]
        with patch.object(g,'detect',return_value=wrong):
            row=g.update(self.scene())
        self.assertIsNone(row['mask'])
        self.assertIn('liquid_missing_or_ambiguous_inside_vial',row['quality_flags'])

    def test_missing_or_changed_model_fails_closed(self):
        row=Extractor(Config()).measure(self.scene())
        self.assertFalse(row['measurement_valid'])
        self.assertIsNone(row['lab'])
        self.assertIn('vial_detector_not_configured',row['quality_flags'])
        with self.assertRaises(ValueError):
            Config(vision_model_path='model.onnx')
        with self.assertRaises(ValueError):
            decode_mask(dict(origin=[0,0],size=[2,2],runs=[[3,5]]))


if __name__=='__main__':
    unittest.main()
