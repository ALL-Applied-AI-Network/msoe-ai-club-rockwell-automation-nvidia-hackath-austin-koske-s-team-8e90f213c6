import json
from pathlib import Path
import tempfile
import unittest

from audit_liquid_specialists import at_threshold, development_rows
from liquid_specialists_data import sha


class DevelopmentChecks(unittest.TestCase):
    def test_threshold_rematches_after_low_confidence_box_is_removed(self):
        rows = [dict(source_id='test', targets=[dict(class_id=1, box=[0, 0, 10, 10])],
                     predictions=[dict(class_id=1, box=[0, 0, 8, 10], confidence=.8),
                                  dict(class_id=1, box=[0, 0, 10, 10], confidence=.1)])]
        report, _ = at_threshold(rows, .35)
        self.assertEqual((report['tp'], report['unmatched_boxes'], report['missed_targets']), (1, 0, 0))
        self.assertEqual(report['target_height_strata']['height_le_32'], dict(targets=1, detected=1))

    def test_missing_label_is_provisional_and_nested_boxes_are_counted(self):
        boxes = [dict(class_id=1, box=b, confidence=.8) for b in ([0, 0, 20, 100], [2, 80, 18, 98])]
        report, _ = at_threshold([dict(source_id='test', targets=[], predictions=boxes)], .35)
        self.assertEqual(report['provisional_negative_frames_with_predictions'], 1)
        self.assertEqual(report['nested_prediction_pairs'], 1)
        self.assertEqual(report['unmatched_boxes'], 2)

    def test_run5_rejected_before_any_image_is_opened(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); data = root/'data/all'; data.mkdir(parents=True)
            rows = [dict(source_id=str(i), split='val', source_run=5, source_sha256='held-out',
                         image_sha256='held-out') for i in range(157)]
            (data/'provenance.json').write_text(json.dumps(dict(samples=rows, held_out_source_hashes=['held-out'])))
            (root/'experiment-plan.json').write_text(json.dumps(dict(excluded_from_fit_and_selection=[5],
                data={'all': dict(provenance_sha256=sha(data/'provenance.json'))})))
            with self.assertRaisesRegex(ValueError, 'Only original Run 4'):
                development_rows(root)


if __name__ == '__main__':
    unittest.main()
