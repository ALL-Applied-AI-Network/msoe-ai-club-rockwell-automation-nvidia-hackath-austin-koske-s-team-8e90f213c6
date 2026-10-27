"""Integrity checks for evaluation reviews; fixtures are not approved labels."""
import copy
import importlib.util
import unittest

from annotation_review.score_mask_review import validated_decisions


@unittest.skipUnless(importlib.util.find_spec('numpy'), 'CPU scoring extra requires NumPy')
class MaskScoringTests(unittest.TestCase):
    def fixture(self):
        source = dict(sample_id='test-frame', image_sha256='image', source_rgb_sha256='rgb',
            source_video_sha256='video', frame_index=1, pts=512, time_base='1/15360')
        event = dict(kind='frame', seq=1, reviewer='Test fixture', source=source,
            status='empty', mask=dict(width=640, height=480, counts=[640*480]),
            notes='', draft_model_sha256=None, training_eligible=False)
        review = dict(schema='visible-pool-human-review/v1', demonstration=False,
            source_run=4, training_eligible=False, history=[event])
        return review, {'test-frame': dict(source, case_id='test-case')}

    def test_demonstration_is_never_evaluation_truth(self):
        review, tasks = self.fixture(); review['demonstration'] = True
        with self.assertRaisesRegex(ValueError, 'non-demonstration'):
            validated_decisions(review, tasks)

    def test_source_mismatch_rejected(self):
        review, tasks = self.fixture()
        review['history'][0]['source'] = dict(review['history'][0]['source'], image_sha256='changed')
        with self.assertRaisesRegex(ValueError, 'identity differs'):
            validated_decisions(review, tasks)

    def test_pool_requires_nonempty_mask(self):
        review, tasks = self.fixture(); review['history'][0]['status'] = 'pool'
        with self.assertRaisesRegex(ValueError, 'mask disagree'):
            validated_decisions(review, tasks)

    def test_rle_must_cover_full_source(self):
        review, tasks = self.fixture(); review['history'][0]['mask']['counts'] = [1]
        with self.assertRaisesRegex(ValueError, 'Invalid reviewed'):
            validated_decisions(review, tasks)

    def test_latest_revision_preserves_source(self):
        review, tasks = self.fixture(); event = copy.deepcopy(review['history'][0])
        event.update(seq=2, status='uncertain', notes='Visibility is ambiguous')
        review['history'].append(event)
        states, intervals = validated_decisions(review, tasks)
        self.assertEqual(states['test-frame']['status'], 'uncertain')
        self.assertEqual(len(review['history']), 2)
        self.assertFalse(intervals)
