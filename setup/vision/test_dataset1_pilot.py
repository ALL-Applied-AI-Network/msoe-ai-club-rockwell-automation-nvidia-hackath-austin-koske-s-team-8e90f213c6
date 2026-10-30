"""Critical split, target attribution and occupancy-scoring checks (CPU only)."""
import unittest

from train_dataset1_pilot import summarize, validate_records


class Dataset1PilotTests(unittest.TestCase):
    def row(self, group=0, split='train'):
        slots={f'G{i:02d}':'pass' for i in range(1,31)}
        return dict(group=group,split=split,image_sha256=f'image-{group}',golden_sha256=f'golden-{group}',
                    labels=slots,boxes={slot:[10,10,58,58] for slot in slots})

    def test_related_scene_crops_cannot_cross_splits(self):
        with self.assertRaisesRegex(ValueError,'cross'):
            validate_records([self.row(),self.row(split='test')])

    def test_shared_golden_cannot_cross_splits(self):
        a=self.row();b=self.row(1,'test');b['golden_sha256']=a['golden_sha256']
        with self.assertRaisesRegex(ValueError,'cross'):
            validate_records([a,b])

    def test_missing_target_is_rejected(self):
        row=self.row();del row['labels']['G15']
        with self.assertRaises(ValueError):
            validate_records([row])

    def test_empty_slot_is_excluded_from_occupied_defect_metric(self):
        records=[dict(scene='a',truth=0,prediction=0,probabilities=[.9,.05,.05]),
                 dict(scene='a',truth=1,prediction=2,probabilities=[.1,.2,.7]),
                 dict(scene='a',truth=2,prediction=2,probabilities=[.01,.01,.98])]
        metrics=summarize(records,.5)
        self.assertEqual(metrics['per_class']['missing_cap']['missed'],1)
        self.assertEqual(metrics['occupied_cap_diagnostic']['missing_cap_cases'],1)
        self.assertEqual(metrics['occupied_cap_diagnostic']['missing_cap_missed'],0)
        self.assertEqual(metrics['occupied_cap_diagnostic']['acceptable_cases'],1)
        self.assertEqual(metrics['all_30_correct_scenes'],0)


if __name__=='__main__':
    unittest.main()
