"""Coordinate, class and leakage boundaries for the team data snapshot."""
import unittest

from package_team_data import parse_yolo, coco, validate_splits, starter_record


class TeamDataTests(unittest.TestCase):
    def sample(self, name, split='train', run=1, source='source-a', **extra):
        return dict(source_id=name, split=split, source_run=run, source_sha256=source,
                    image_sha256='encoded-'+name, image='images/'+split+'/'+name+'.jpg', **extra)

    def test_coco_keeps_source_extent_and_remaps_liquid_only_to_one(self):
        sample=self.sample('a');boxes={'a':parse_yolo('0 .5 .5 .2 .8\n1 .5 .8 .1 .1\n')}
        result=coco([sample],boxes,[1]);annotation=result['annotations'][0]
        self.assertEqual(annotation['category_id'],1)
        self.assertEqual(result['categories'][0]['name'],'colored_liquid')
        x,y,w,h=annotation['bbox'];original=boxes['a'][1]['xyxy_pixels']
        self.assertAlmostEqual(x,original[0]);self.assertAlmostEqual(y,original[1])
        self.assertAlmostEqual(x+w,original[2]);self.assertAlmostEqual(y+h,original[3])
        self.assertNotIn('segmentation',annotation)

    def test_missing_liquid_is_explicitly_provisional(self):
        result=coco([self.sample('a')],{'a':parse_yolo('0 .5 .5 .2 .8\n')},[1])
        self.assertEqual(len(result['images']),1)
        self.assertEqual(result['annotations'],[])
        self.assertFalse(result['images'][0]['no_liquid_target_is_confirmed_empty'])
        self.assertEqual(result['images'][0]['annotation_status'],'unreviewed_provisional')

    def test_invalid_or_new_review_classes_cannot_enter_legacy_export(self):
        for text in ('2 .5 .5 .2 .2','1 .99 .5 .2 .2','1 nan .5 .2 .2'):
            with self.assertRaises(ValueError):parse_yolo(text)

    def test_run5_and_cross_split_source_duplicates_are_rejected(self):
        with self.assertRaises(ValueError):validate_splits([self.sample('a',run=5)],set())
        with self.assertRaises(ValueError):validate_splits([self.sample('a')],{'source-a'})
        with self.assertRaises(ValueError):
            validate_splits([self.sample('a'),self.sample('b',split='val',run=4)],set())

    def test_translation_stays_with_its_parent(self):
        parent=self.sample('a');child=self.sample('a-shift',parent_id='a')
        validate_splits([parent,child],set())
        child['source_sha256']='other-source'
        with self.assertRaises(ValueError):validate_splits([parent,child],set())

    def test_derived_vial_label_identity_keeps_legacy_ancestry_separate(self):
        sample=dict(self.sample('a'),labels='labels/train/a.txt',label_sha256='donor-label',actual_label_sha256='legacy-current')
        row=starter_record(sample,parse_yolo('0 .5 .5 .2 .8\n1 .5 .8 .1 .1\n'),'derived-vial')
        self.assertEqual(row['actual_label_sha256'],'derived-vial')
        self.assertEqual(row['legacy_label_sha256'],'legacy-current')
        self.assertEqual(row['legacy_labels_path'],'source_legacy_labels/labels/train/a.txt')
        self.assertTrue(row['legacy_liquid_target_present'])


if __name__=='__main__':unittest.main()
