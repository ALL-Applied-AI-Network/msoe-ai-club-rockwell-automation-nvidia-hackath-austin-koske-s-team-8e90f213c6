import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from bootstrap_detector import prepare


class BootstrapChecks(unittest.TestCase):
    def test_heldout_and_descendants_cannot_enter_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); source=root/'teacher'; images=source/'review-pack/vial_fill/images'; labels=source/'review-pack/vial_fill/labels'
            images.mkdir(parents=True); labels.mkdir()
            manifest=[]; gallery=[]
            for name,run,value,status in [('train',1,50,'review_candidates'),('dev',4,80,'review_candidates'),
                                          ('held',5,100,'review_candidates'),('leaked',1,100,'review_candidates'),
                                          ('manual',2,120,'manual_full_image'),('empty',3,140,'review_candidates')]:
                path=images/f'{name}.jpg'; cv2.imwrite(str(path),np.full((480,640,3),value,np.uint8))
                (labels/f'{name}.txt').write_text('0 0.5 0.5 0.1 0.7\n')
                manifest.append(dict(id=name,source=path.name,camera='vial_fill',video_id=str(run),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
                gallery.append(dict(id=name,camera='vial_fill',status=status,boxes=0 if name=='empty' else 1))
            (source/'manifest.json').write_text(json.dumps(manifest)); (source/'gallery.json').write_text(json.dumps(gallery))
            output=root/'prepared'; prepare(source,output)
            p=json.loads((output/'provenance.json').read_text())
            self.assertEqual(p['counts'],{'train':2,'val':1})
            self.assertEqual({s['source_run'] for s in p['samples']},{1,4})
            self.assertEqual({s['source_id'] for s in p['samples']},{'train','train-shift','dev'})
            child=next(s for s in p['samples'] if 'parent_id' in s)
            self.assertEqual(child['split'],'train')
            values=list(map(float,(output/child['labels']).read_text().split()))
            self.assertAlmostEqual(values[1],.5+child['augmentation']['dx']/640,places=7)
            self.assertAlmostEqual(values[2],.5+child['augmentation']['dy']/480,places=7)
            self.assertEqual(values[3:],[.1,.7])


if __name__=='__main__':
    unittest.main()
