"""Synthetic split and escaped-defect accounting tests, not real annotations."""
import unittest
from dataset1_fiveclass import CLASSES,SLOTS,families
from train_dataset1_fiveclass import summarize,validate


class FiveClassTests(unittest.TestCase):
    def row(self,split='train'):
        return dict(group=1,split=split,image_sha256='TEST',golden_sha256='GOLDEN',
            labels={s:'pass' for s in SLOTS},boxes={s:[1,2,97,98] for s in SLOTS})

    def test_plan_covers_five_classes_and_both_fixtures_without_real_inputs(self):
        plan=families();self.assertEqual(len(plan),120)
        for row in plan:
            self.assertEqual(set(row['labels']),set(SLOTS))
            for prefix in ('G','S'):
                for label in CLASSES:self.assertEqual(sum(v==label and k.startswith(prefix) for k,v in row['labels'].items()),6)

    def test_scene_and_golden_cannot_cross_split_boundary(self):
        with self.assertRaises(ValueError):validate([self.row(),self.row('test')])
        a=self.row();b=self.row('test');b.update(group=2,image_sha256='OTHER')
        with self.assertRaises(ValueError):validate([a,b])

    def test_missing_second_fixture_target_is_rejected(self):
        row=self.row();del row['labels']['S30']
        with self.assertRaises(ValueError):validate([row])

    def test_escaped_defects_are_counted_per_class_and_empty_separately(self):
        records=[dict(scene='TEST',slot='G01',truth=0,prediction=0,probabilities=[.95,.01,.01,.01,.02]),
            dict(scene='TEST',slot='G02',truth=1,prediction=0,probabilities=[.9,.04,.02,.02,.02]),
            dict(scene='TEST',slot='G03',truth=2,prediction=2,probabilities=[.1,.1,.6,.1,.1]),
            dict(scene='TEST',slot='G04',truth=3,prediction=4,probabilities=[.2,.1,.1,.2,.4]),
            dict(scene='TEST',slot='G05',truth=4,prediction=4,probabilities=[.01,.01,.01,.01,.96])]
        result=summarize(records,.5);diagnostic=result['occupied_defect_diagnostic']
        self.assertEqual(diagnostic['per_class']['missing_cap']['missed'],1)
        self.assertEqual(diagnostic['per_class']['damaged_vial']['missed'],0)
        self.assertEqual(result['per_class']['damaged_vial']['missed'],1)
        self.assertEqual(diagnostic['acceptable_targets'],1);self.assertEqual(result['all_60_correct_scenes'],0)


if __name__=='__main__':unittest.main()
