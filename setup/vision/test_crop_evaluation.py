import unittest
import numpy as np
from evaluate_crop_contrast import matched


class CropEvaluationChecks(unittest.TestCase):
    def test_source_box_matching_is_class_specific_and_one_to_one(self):
        targets=[dict(class_id=0,box=[10,20,60,100]),dict(class_id=1,box=[20,60,50,90])]
        predictions=[dict(targets[0],confidence=.9),dict(targets[0],confidence=.8),dict(targets[1],confidence=.7),
                     dict(class_id=0,box=targets[1]['box'],confidence=.6)]
        flags=matched(predictions,targets)
        self.assertTrue(flags[0].all());self.assertFalse(flags[1].any())
        self.assertTrue(flags[2].all());self.assertFalse(flags[3].any())
        self.assertEqual(matched([],targets).shape,(0,10))
        self.assertFalse(matched(predictions,[]).any())


if __name__=='__main__':unittest.main()
