import copy
import unittest
from annotation_review.early_fill import validate_request


class EarlyFillRequestTests(unittest.TestCase):
    def request(self):
        return dict(schema='first-visible-liquid-review-request/v1', review_status='unreviewed',
            approved_labels=False, source=dict(run_id=4,camera='vial_fill',sha256='a'*64,
            relative_path='vial_fill/videos/vial_fill_video_4.mp4',frame_count=9482),
            cycles=[dict(case_id='run4-fill1',provisional_anchor_frame=324,
                anchor_status='image_detector_proposal_not_reviewed_onset')])

    def test_navigation_does_not_become_an_approved_onset(self):
        d=self.request();before=copy.deepcopy(d)
        self.assertEqual(validate_request(d),before)
        self.assertNotIn('reviewed_first_visible_frame',d['cycles'][0])

    def test_refuses_run5_or_approved_labels(self):
        for change in ('run5','approved','anchor'):
            d=self.request()
            if change=='run5':d['source']['run_id']=5
            elif change=='approved':d['approved_labels']=True
            else:d['cycles'][0]['anchor_status']='reviewed'
            with self.subTest(change=change),self.assertRaises(ValueError):validate_request(d)

    def test_refuses_unsafe_or_duplicate_identity(self):
        d=self.request();d['source']['relative_path']='../other.mp4'
        with self.assertRaises(ValueError):validate_request(d)
        d=self.request();d['cycles'].append(copy.deepcopy(d['cycles'][0]))
        with self.assertRaises(ValueError):validate_request(d)

    def test_refuses_censored_anchor_as_complete_neighborhood(self):
        for anchor in (0,20,9481,True):
            d=self.request();d['cycles'][0]['provisional_anchor_frame']=anchor
            with self.subTest(anchor=anchor),self.assertRaises(ValueError):validate_request(d)


if __name__=='__main__':unittest.main()
