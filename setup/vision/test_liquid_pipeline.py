"""Causal frame and ambiguity boundaries for the offline SAM demonstration."""
import unittest

import numpy as np

from render_liquid_pipeline import FrameAdapter
from replay.config import Config
from replay.geometry import VialGeometry, encode_mask


class PipelineBoundaries(unittest.TestCase):
    def setUp(self):
        self.rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        self.vial = dict(class_id=0, box=[270, 10, 350, 450], confidence=.9)
        self.adapter = FrameAdapter()

    def test_refuses_predictions_from_different_pixels(self):
        self.adapter.set(self.rgb, [self.vial], [])
        changed = self.rgb.copy(); changed[100, 300, 0] = 1
        for operation in (self.adapter.detect, lambda rgb: self.adapter.propose(rgb, self.vial['box'])):
            with self.assertRaisesRegex(ValueError, 'stale'):
                operation(changed)

    def test_copies_and_clears_previous_mask_proposals(self):
        proposal = dict(class_id=1, box=[285, 300, 325, 400], confidence=.8)
        self.adapter.set(self.rgb, [self.vial], [proposal])
        result = self.adapter.propose(self.rgb, self.vial['box'])
        result[0]['box'][0] = 0
        self.assertEqual(self.adapter.propose(self.rgb, self.vial['box'])[0]['box'][0], 285)
        self.adapter.set(self.rgb, [self.vial], [])
        self.assertEqual(self.adapter.propose(self.rgb, self.vial['box']), [])

    def test_existing_geometry_refuses_multiple_raw_masks(self):
        masks = []
        for left in (285, 290):
            mask = np.zeros((480, 640), dtype=np.uint8)
            mask[300:400, left:left+30] = 1
            masks.append(dict(class_id=1, box=[left, 300, left+30, 400],
                              confidence=.8, candidate_mask=encode_mask(mask)))
        self.adapter.set(self.rgb, [self.vial], masks)
        geometry = VialGeometry(Config(), detector=self.adapter)
        geometry.segmenter = self.adapter
        evidence = geometry.update(self.rgb)
        self.assertIn('liquid_missing_or_ambiguous_inside_vial', evidence['quality_flags'])
        self.assertIsNone(evidence['mask'])


if __name__ == '__main__':
    unittest.main()
