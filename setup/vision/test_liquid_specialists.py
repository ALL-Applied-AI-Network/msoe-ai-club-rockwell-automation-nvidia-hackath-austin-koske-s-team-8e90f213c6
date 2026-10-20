import unittest
from liquid_specialists_data import liquid_text,select_units


class DatasetChecks(unittest.TestCase):
    def test_only_class_changes_and_sources_are_not_adjudicated(self):
        source='0 0.5 0.5 0.1 0.7\n1 0.51 0.8 0.08 0.10\n'
        self.assertEqual(liquid_text(source),'0 0.51 0.8 0.08 0.10\n')
        self.assertEqual(liquid_text('0 0.5 0.5 0.1 0.7\n'),'')
        with self.assertRaises(ValueError):liquid_text('2 0.5 0.5 0.1 0.1')

    def test_focused_arm_preserves_step_count_and_source_family(self):
        rows=[dict(source_id=str(i),split='train',source_run=1+i%3) for i in range(721)]
        positive={str(i):i<354 for i in range(721)}
        normal,focused=select_units(rows,positive)
        self.assertEqual(len(normal),len(focused));self.assertEqual(len(focused),721)
        self.assertEqual(sum(positive[r['source_id']] for r in focused),576)
        self.assertEqual(len({r['source_id'] for r in focused}),499)
        self.assertEqual({r['source_id'] for r in focused if positive[r['source_id']]},{str(i) for i in range(354)})
        self.assertEqual(select_units(rows,positive)[1],focused)
        self.assertTrue(all(r['source_run'] in [1,2,3] for r in focused))


if __name__=='__main__':unittest.main()
