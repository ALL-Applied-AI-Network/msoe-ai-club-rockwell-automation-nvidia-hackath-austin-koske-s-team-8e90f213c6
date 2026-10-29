from copy import deepcopy
import unittest

from train_sampling_segmentation import validate_split


class SegmentationSplitChecks(unittest.TestCase):
    def test_per_sample_runs_and_duplicate_crops_are_checked(self):
        p=dict(train_runs=[1,2,3],development_runs=[4],excluded_runs=[5],samples=[
            dict(split='train',source_run=1,crop_rgb_sha256='a'),dict(split='val',source_run=4,crop_rgb_sha256='b')])
        validate_split(p)
        for split,run in (('train',5),('val',5),('train',4),('val',1)):
            invalid=deepcopy(p);invalid['samples'].append(dict(split=split,source_run=run,crop_rgb_sha256='c'))
            with self.assertRaises(ValueError): validate_split(invalid)
        invalid=deepcopy(p);invalid['samples'][1]['crop_rgb_sha256']='a'
        with self.assertRaises(ValueError): validate_split(invalid)


if __name__=='__main__': unittest.main()
