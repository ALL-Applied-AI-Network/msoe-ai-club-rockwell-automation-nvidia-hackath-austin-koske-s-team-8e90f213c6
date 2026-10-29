import unittest
import numpy as np

from probe_prompted_liquid import prompts_from_predictions, mask_diagnostics, ARMS


class PromptChecks(unittest.TestCase):
    def test_ambiguous_vial_cannot_seed_a_mask(self):
        vials = [dict(class_id=0, confidence=.9, box=b) for b in ([280, 50, 335, 430], [290, 70, 340, 425])]
        vial, prompts = prompts_from_predictions(vials)
        self.assertIsNone(vial)
        self.assertEqual(prompts, {name: [] for name in ARMS})

    def test_liquid_prompt_uses_only_eligible_current_prediction(self):
        inside = [285, 390, 330, 420]
        rows = [dict(class_id=0, confidence=.9, box=[280, 50, 335, 430]),
                dict(class_id=1, confidence=.8, box=inside),
                dict(class_id=1, confidence=.9, box=[400, 390, 450, 420]),
                dict(class_id=1, confidence=.2, box=[285, 370, 330, 420])]
        _, prompts = prompts_from_predictions(rows)
        self.assertEqual(prompts['liquid_box'], [inside])
        _, missing = prompts_from_predictions(rows[:1])
        self.assertEqual(missing['liquid_box'], [])
        self.assertEqual(len(missing['lower_vial_box']), 1)

    def test_empty_mask_has_no_fabricated_geometry_fraction(self):
        report = mask_diagnostics(np.zeros((480, 640), np.uint8), [280, 50, 335, 430])
        self.assertEqual(report['pixels'], 0)
        self.assertIsNone(report['fraction_outside_vial'])


if __name__ == '__main__':
    unittest.main()
