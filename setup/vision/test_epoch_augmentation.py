from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np

from detector_photometric import Photometric
from train_epoch_augmentation import plan
from evaluate_epoch_augmentation import training_evidence


class ControlledChecks(unittest.TestCase):
    def test_photo_preserves_labels_pixels_and_global_rng(self):
        image=np.arange(48*64*3,dtype=np.uint8).reshape(48,64,3)
        labels=dict(img=image,cls=np.array([[0],[1]]),boxes=np.array([[1,2,30,40],[4,5,15,20]]),im_file='example')
        original=deepcopy(labels);np.random.seed(123);before=np.random.get_state()
        photo=Photometric();result=photo(labels);after=np.random.get_state()
        for k in ('cls','boxes'):np.testing.assert_array_equal(result[k],original[k])
        np.testing.assert_array_equal(image,original['img'])
        self.assertEqual(result['img'].shape,image.shape);self.assertEqual(result['img'].dtype,np.uint8)
        self.assertEqual(before[0],after[0]);np.testing.assert_array_equal(before[1],after[1]);self.assertEqual(before[2:],after[2:])

    def test_separate_seeded_stream_is_reproducible_and_executes(self):
        a,b=Photometric(),Photometric();im=np.full((48,64,3),120,np.uint8)
        for _ in range(30):
            np.testing.assert_array_equal(a(dict(img=im.copy()))['img'],b(dict(img=im.copy()))['img'])
        self.assertGreater(a.counts['changed_images'],0)
        for key in ('contrast_factor','blur_sigma','noise_sigma_8bit'):self.assertGreater(a.counts[key],0)

    def test_frozen_budget_and_split(self):
        p=plan();self.assertEqual(p['excluded_from_fit_and_selection'],[5])
        self.assertEqual(p['train_runs'],[1,2,3]);self.assertEqual(p['development_runs'],[4])
        self.assertFalse(p['automatic_promotion']);self.assertFalse(p['measurement_qa_changed'])
        self.assertEqual(p['arms']['plain12']['epochs'],12)
        self.assertEqual(p['arms']['plain50']['epochs'],50)
        self.assertTrue(p['arms']['augmented50']['extra_augmentation'])
        self.assertFalse(p['arms']['plain50']['extra_augmentation'])

    def test_rejects_contaminated_control_or_validation(self):
        with TemporaryDirectory() as folder:
            path=Path(folder);(path/'train').mkdir()
            (path/'train/results.csv').write_text('epoch,metrics/mAP50-95(B)\n1,0.3\n')
            (path/'train/args.yaml').write_text('epochs: 12\n')
            model=dict(arm='plain12',extra_augmentation=False,epochs_completed=1,trainer_sha256='a',photo_sha256='b')
            prepared={'train_epoch_augmentation.py':'a','detector_photometric.py':'b'}
            evidence=dict(counts={},label_geometry_changed=False,validation_augmented=False)
            def save():(path/'augmentation-execution.json').write_text(json.dumps(evidence))
            save();self.assertEqual(training_evidence(path,'plain12',model,prepared)['best_run4_epoch'],1)
            evidence['counts']['calls']=1;save()
            with self.assertRaisesRegex(ValueError,'Plain control'):training_evidence(path,'plain12',model,prepared)
            evidence['counts'].clear();evidence['validation_augmented']=True;save()
            with self.assertRaisesRegex(ValueError,'boundary'):training_evidence(path,'plain12',model,prepared)
            evidence['validation_augmented']=False;save();model['trainer_sha256']='changed'
            with self.assertRaisesRegex(ValueError,'helper changed'):training_evidence(path,'plain12',model,prepared)


if __name__=='__main__':unittest.main()
