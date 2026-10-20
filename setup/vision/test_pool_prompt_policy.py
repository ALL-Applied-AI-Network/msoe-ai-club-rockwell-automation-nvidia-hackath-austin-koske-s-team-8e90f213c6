"""Prompt recovery, no invented presence, ambiguity and tall-fill boundaries."""
import unittest

import numpy as np

from pool_prompt_policy import pool_prompts, CausalPromptBridge


class PoolPromptTests(unittest.TestCase):
    def setUp(self):
        self.rgb = np.full((480, 640, 3), 180, np.uint8)
        self.vial = dict(class_id=0, box=[272.46, 60.09, 339.63, 417.34], confidence=.9)

    def prediction(self, box, confidence=.8):
        return dict(class_id=1, box=box, confidence=confidence)

    def test_recovers_edge_disagreement_and_prefers_basal_box(self):
        self.rgb[380:411, 286:335] = [0, 80, 150]
        predictions = [self.vial,
            self.prediction([280.86, 352.57, 346.05, 419.85], .987),
            self.prediction([285.82, 60.81, 336.26, 378.86], .399)]
        result = pool_prompts(self.rgb, predictions)
        self.assertEqual(len(result['association_boxes']), 2)
        self.assertLess(result['association'][0]['original_containment'], .9)
        self.assertEqual(result['selected_prediction_index'], 1)
        self.assertEqual(len(result['guided_boxes']), 1)
        self.assertTrue(result['point_labels'][0][0] == 1)
        self.assertTrue(result['guided_boxes'][0][2] < self.vial['box'][2])

    def test_no_colored_seed_does_not_invent_a_pool(self):
        result = pool_prompts(self.rgb, [self.vial, self.prediction([280, 70, 337, 415])])
        self.assertEqual(result['guided_boxes'], [])
        self.assertEqual(result['status'], 'no_colored_basal_seed')

    def test_colored_pixels_without_detection_do_not_supply_presence(self):
        self.rgb[370:411, 285:335] = [0, 80, 150]
        result = pool_prompts(self.rgb, [self.vial])
        self.assertEqual(result['guided_boxes'], [])

    def test_tall_filled_region_has_no_blanket_height_rejection(self):
        self.rgb[110:411, 285:335] = [0, 80, 150]
        result = pool_prompts(self.rgb, [self.vial, self.prediction([280, 100, 337, 415])])
        self.assertEqual(len(result['guided_boxes']), 1)
        self.assertLess(result['guided_boxes'][0][1], 150)

    def test_equal_localizations_remain_ambiguous(self):
        self.rgb[370:411, 285:335] = [0, 80, 150]
        box = [280, 350, 337, 415]
        result = pool_prompts(self.rgb, [self.vial, self.prediction(box), self.prediction(box.copy(), .99)])
        self.assertEqual(result['guided_boxes'], [])
        self.assertEqual(result['status'], 'ambiguous_pool_localization')


class CausalBridgeTests(unittest.TestCase):
    def policy(self, box):
        return dict(vial=[270, 60, 340, 420], guided_boxes=[box], seed=[310, 400],
                    points=[[[310, 400]]], point_labels=[[1]])

    def test_bridge_expires_without_refreshing_source_age(self):
        bridge = CausalPromptBridge()
        bridge.update(self.policy([280, 380, 338, 418]), 330, 11., 0.)
        for frame, seconds in ((336, 11.2), (342, 11.4), (348, 11.6)):
            result = bridge.update(self.policy([280, 65, 338, 418]), frame, seconds, 0.)
            self.assertEqual(result['mode'], 'past_box_with_current_seed')
            self.assertEqual(result['prompt_source_frame'], 330)
        result = bridge.update(self.policy([280, 65, 338, 418]), 354, 11.8, 0.)
        self.assertEqual(result['mode'], 'unresolved_box_jump')
        self.assertEqual(result['boxes'], [])

    def test_missing_current_seed_cannot_reuse_prior_box(self):
        bridge = CausalPromptBridge()
        bridge.update(self.policy([280, 380, 338, 418]), 330, 11., 0.)
        current = self.policy([280, 65, 338, 418]); current['guided_boxes'] = []
        self.assertEqual(bridge.update(current, 336, 11.2, 0.)['boxes'], [])

    def test_motion_clears_prior_and_noncausal_time_is_rejected(self):
        bridge = CausalPromptBridge()
        bridge.update(self.policy([280, 380, 338, 418]), 330, 11., 0.)
        self.assertEqual(bridge.update(self.policy([280, 65, 338, 418]), 336, 11.2, 4.)['boxes'], [])
        self.assertIsNone(bridge.last)
        with self.assertRaisesRegex(ValueError, 'increasing'):
            bridge.update(self.policy([280, 65, 338, 418]), 330, 11., 0.)

    def test_new_station_geometry_clears_prior(self):
        bridge = CausalPromptBridge()
        bridge.update(self.policy([280, 380, 338, 418]), 330, 11., 0.)
        current = self.policy([300, 65, 420, 418]); current['vial'] = [300, 60, 430, 420]
        bridge.update(current, 336, 11.2, 0.)
        self.assertEqual(bridge.last['frame'], 336)

    def test_seed_outside_past_box_cannot_bridge(self):
        bridge = CausalPromptBridge()
        bridge.update(self.policy([280, 380, 338, 418]), 330, 11., 0.)
        current = self.policy([280, 65, 338, 418]); current['seed'] = [310, 300]
        result = bridge.update(current, 336, 11.2, 0.)
        self.assertEqual(result['mode'], 'unresolved_box_jump')
        self.assertEqual(result['boxes'], [])


if __name__ == '__main__':
    unittest.main()
