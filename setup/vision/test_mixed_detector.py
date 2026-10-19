from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

from train_mixed_detector import synthetic_targets, validate_split, prepare_mixed, write
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'isaac'))
from make_training_fixture import make


class MixedDetectorChecks(unittest.TestCase):
    def test_source_grouping_rejects_run5_dev_synthetic_and_augmented_leaks(self):
        rows=[dict(source_id='a',origin='real',source_run=1,source_sha256='a',image_sha256='a',split='train'),
              dict(source_id='b',origin='real',source_run=4,source_sha256='b',image_sha256='b',split='val'),
              dict(source_id='s',origin='synthetic',source_run=None,source_sha256='s',image_sha256='s',split='train')]
        validate_split(rows, {'held'})
        for index,key,value in [(0,'source_run',5),(0,'source_run',4),(2,'split','val'),
                                (0,'source_sha256','held'),(2,'image_sha256','held'),(0,'source_sha256','b')]:
            wrong=deepcopy(rows);wrong[index][key]=value
            with self.assertRaises(ValueError): validate_split(wrong,{'held'})
        wrong=deepcopy(rows);wrong.append(dict(rows[0],source_id='aug',parent_id='a',source_run=4,split='val'))
        with self.assertRaises(ValueError):validate_split(wrong,{'held'})

    def test_seeded_fixture_is_independent_of_real_data_and_balances_negatives(self):
        a,b=make(768,19),make(768,19)
        self.assertEqual(a,b);self.assertEqual(a['real_data_inputs'],[])
        self.assertEqual(len(a['cases']),768)
        self.assertEqual(sum(r['level']==0 and r['stream']=='off' for r in a['cases']),96)
        self.assertEqual(sum(r['level']==0 and r['stream']!='off' for r in a['cases']),96)
        self.assertNotEqual(a,make(768,20))

    def test_raw_render_objects_map_only_to_existing_active_vial_and_body_classes(self):
        # Executed 12-image GB10 export: real native masks, not a mocked geometry.
        batch=Path.home()/'ai-club-tools/results/isaac-camera-boxes-20261019/gb10-review'
        if not batch.is_dir():self.skipTest('Local GB10 review fixture not installed')
        rows=[json.loads(v) for v in (batch/'manifest.jsonl').read_text(encoding='utf-8').splitlines()]
        empty=synthetic_targets(rows[0],batch);filled=synthetic_targets(rows[2],batch)
        self.assertEqual([s.split()[0] for s in empty],['0'])
        self.assertEqual([s.split()[0] for s in filled],['0','1'])
        reversed_row=deepcopy(rows[2]);reversed_row['annotations'].reverse()
        self.assertEqual(filled,synthetic_targets(reversed_row,batch))
        wrong=deepcopy(rows[2]);next(o for o in wrong['annotations'] if o['instance_id']==201)['bbox_xyxy'][0]+=1
        with self.assertRaisesRegex(ValueError,'mask-derived'):synthetic_targets(wrong,batch)

    def test_partial_render_batch_and_implicit_provisional_admission_are_rejected(self):
        batch=Path.home()/'ai-club-tools/results/real-synthetic-detector-20261019/local-spp32-capture-check'
        if not (batch/'verification.json').is_file():self.skipTest('Local native pilot not installed')
        with tempfile.TemporaryDirectory(prefix='mixed-admission-') as directory:
            root=Path(directory)
            write(root/'experiment-plan.json',dict(expected_synthetic_images=768,synthetic_fixture_sha256='not-the-pilot'))
            with self.assertRaisesRegex(ValueError,'Explicit provisional-render admission'):
                prepare_mixed(root,batch,False)
            with self.assertRaisesRegex(ValueError,'Render count/fixture differs'):
                prepare_mixed(root,batch,True)
            self.assertFalse((root/'data/mixed').exists())


if __name__=='__main__':unittest.main()
