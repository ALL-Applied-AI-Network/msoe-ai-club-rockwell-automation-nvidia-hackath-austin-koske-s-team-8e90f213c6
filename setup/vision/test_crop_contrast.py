from copy import deepcopy
import unittest
import numpy as np
from crop_contrast import crop_labels,enhance_crop,project_boxes,read_labels
from train_crop_contrast import validate_split


class CropContrastChecks(unittest.TestCase):
    def test_label_and_prediction_coordinate_roundtrip_and_no_source_mutation(self):
        roi=[230,20,415,455];labels=read_labels('0 .48 .49 .1 .75\n1 .48 .75 .07 .18')
        encoded=crop_labels(labels,roi);cropped=read_labels(encoded,(435,185))
        restored=project_boxes(cropped,roi)
        for a,b in zip(labels,restored):np.testing.assert_allclose(a['box'],b['box'],atol=1e-6)
        rgb=np.random.default_rng(19).integers(0,256,(480,640,3),dtype=np.uint8);original=rgb.copy()
        processed=enhance_crop(rgb,roi);self.assertEqual(processed.shape,(435,185,3))
        np.testing.assert_array_equal(rgb,original)
        with self.assertRaises(ValueError):crop_labels(labels,[290,20,415,455])
        with self.assertRaises(ValueError):project_boxes([dict(box=[-1,0,20,40])],roi)

    def test_run5_recording_duplicates_and_augmented_descendants_cannot_cross_splits(self):
        rows=[dict(source_id='a',split='train',source_run=1,source_sha256='a'),dict(source_id='b',split='val',source_run=4,source_sha256='b')]
        validate_split(rows,{'held'})
        for run in [4,5]:
            wrong=deepcopy(rows);wrong[0]['source_run']=run
            with self.assertRaises(ValueError):validate_split(wrong,{'held'})
        for digest in ['held','b']:
            wrong=deepcopy(rows);wrong[0]['source_sha256']=digest
            with self.assertRaises(ValueError):validate_split(wrong,{'held'})
        wrong=deepcopy(rows);wrong.append(dict(source_id='aug',parent_id='a',split='val',source_run=4,source_sha256='a'))
        with self.assertRaises(ValueError):validate_split(wrong,{'held'})


if __name__=='__main__':unittest.main()
