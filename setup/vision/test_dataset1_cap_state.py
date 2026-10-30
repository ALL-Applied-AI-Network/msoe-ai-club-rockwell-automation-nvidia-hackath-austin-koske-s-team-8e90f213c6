"""Cap-state evidence cannot silently become whole-vial quality evidence."""
import unittest

from experiment_dataset1_cap_state import CAP_LABEL,metrics


class CapScopeTests(unittest.TestCase):
    def test_correct_cap_state_leaves_body_and_product_unassessed(self):
        records=[dict(scene='synthetic',slot='G01',truth=CAP_LABEL[3],prediction=0,probabilities=[.99,.003,.003,.004])]
        result=metrics(records,.5)
        self.assertEqual(result['accuracy'],1)
        self.assertEqual(result['body_condition'],'unassessed')
        self.assertFalse(result['quality_release_authorized'])
        self.assertTrue(result['cap_warning']['body_damage_not_assessed'])

    def test_wrong_cap_defect_name_and_escaped_warning_are_distinct(self):
        records=[dict(scene='synthetic',slot='G01',truth=2,prediction=1,probabilities=[.01,.8,.18,.01]),
            dict(scene='synthetic',slot='G02',truth=1,prediction=0,probabilities=[.99,.005,.004,.001])]
        result=metrics(records,.5)
        self.assertEqual(result['accuracy'],0)
        self.assertEqual(result['cap_warning']['per_class']['crooked_cap']['missed'],0)
        self.assertEqual(result['cap_warning']['per_class']['missing_cap']['missed'],1)
        self.assertEqual(result['cap_warning']['seated_targets'],0)


if __name__=='__main__':unittest.main()
