import json
import unittest
from cosmos_event_review import parse_review
from benchmark_vial_tensorrt import compare


class ExplanationContract(unittest.TestCase):
    def example(self):
        return dict(visible_summary='Liquid is visible in a vial.', quality_assessment='unavailable',
                    limitations=['Intended recipe is missing.'], operator_message='Review evidence.')

    def test_accepts_abstaining_explanation(self):
        value = self.example()
        self.assertEqual(parse_review(json.dumps(value)), (value, None))

    def test_rejects_quality_verdict(self):
        value = self.example()
        value['quality_assessment'] = 'pass'
        self.assertEqual(parse_review(json.dumps(value))[1], 'unsupported_quality_verdict')

    def test_rejects_action_fields(self):
        value = self.example()
        value['release_vial'] = True
        self.assertEqual(parse_review(json.dumps(value))[1], 'wrong_schema')

    def test_rejects_malformed_output(self):
        for text in ('', '[]', 'null', '{', 'A vial passes.'):
            self.assertIsNone(parse_review(text)[0])

    def test_requires_limitations(self):
        value = self.example()
        value['limitations'] = []
        self.assertEqual(parse_review(json.dumps(value))[1], 'missing_limitations')

    def test_display_contract_is_not_semantic_validation(self):
        value = self.example()
        value['visible_summary'] = 'An unsupported visual statement.'
        self.assertIsNone(parse_review(json.dumps(value))[1])


class ParityContract(unittest.TestCase):
    def test_missing_box_is_not_counted_as_agreement(self):
        result = compare([[0, 0, 10, 10, .9, 0]], [])
        self.assertEqual(result['unmatched_torch'], 1)

    def test_matches_reordered_boxes(self):
        boxes = [[0, 0, 10, 10, .9, 0], [20, 20, 30, 30, .8, 0]]
        result = compare(boxes, boxes[::-1])
        self.assertEqual(result['matches_iou50'], 2)
        self.assertTrue(all(p['iou'] == 1 for p in result['pairs']))


if __name__ == '__main__':
    unittest.main()
