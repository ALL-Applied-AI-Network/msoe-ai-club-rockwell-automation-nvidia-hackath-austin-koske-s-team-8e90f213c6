import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from annotation_review.core import Catalog, Reviews


class ReviewFixture:
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.data=self.root/'data';self.pre=self.root/'pre'
        self.pre.mkdir();self.image=self.data/'vial_fill/images/a.jpg';self.image.parent.mkdir(parents=True)
        self.image.write_bytes(b'unchanged source fixture')
        row=dict(id='a',source='a.jpg',camera='vial_fill',video_id='3',source_sha256=hashlib.sha256(self.image.read_bytes()).hexdigest(),timestamp='capture clock only',source_size=[640,480])
        (self.pre/'manifest.json').write_text(json.dumps([row]))
        for camera in ('vial_fill','tank_level','cap_seal'):
            folder=self.pre/camera;folder.mkdir();(folder/'classes.txt').write_text('vial\nliquid\n')
            records=[dict(sample_id='a',class_id=0,box=dict(left=.2,top=.1,right=.4,bottom=.9),presence=.99,confidence=.99)] if camera=='vial_fill' else []
            (folder/'annotations.json').write_text(json.dumps(records))
        self.catalog=Catalog(self.data,self.pre);self.reviews=Reviews(self.root/'work/reviews.sqlite',self.catalog)

    def tearDown(self):self.temp.cleanup()

    def decision(self,status='changed'):
        return dict(sample_id='a',sample_fingerprint=self.catalog.samples['a']['fingerprint'],candidate_id='machine:0',class_id=0,status=status,reviewer='Aleyna',notes='corrected glass base',proposed_box=[.2,.1,.41,.91],expected_revision=0)


class ReviewTests(ReviewFixture, unittest.TestCase):
    def test_roundtrip_and_idempotence(self):
        original=copy.deepcopy(self.catalog.samples['a']['candidates'])
        saved=self.reviews.save(self.decision());bundle=self.reviews.export()
        fresh=Reviews(self.root/'fresh/reviews.sqlite',Catalog(self.data,self.pre))
        fresh.import_bundle(bundle);fresh.import_bundle(bundle)
        self.assertEqual(fresh.history(),[saved]);self.assertEqual(self.catalog.samples['a']['candidates'],original)

    def test_source_tamper_blocks_save(self):
        self.image.write_bytes(b'different')
        with self.assertRaises(ValueError):self.reviews.save(self.decision())

    def test_optimistic_concurrency(self):
        self.reviews.save(self.decision())
        with self.assertRaises(ValueError):self.reviews.save(self.decision())

    def test_accept_cannot_change_box(self):
        with self.assertRaises(ValueError):self.reviews.save(self.decision('accepted'))

    def test_import_rejects_forged_original(self):
        self.reviews.save(self.decision());bundle=self.reviews.export();bundle['revisions'][0]['original_box']=[0,0,1,1]
        fresh=Reviews(self.root/'fresh/reviews.sqlite',self.catalog)
        with self.assertRaises(ValueError):fresh.import_bundle(bundle)
        self.assertEqual(fresh.history(),[])

    def test_demo_does_not_approve_labels(self):
        demo=Reviews(self.root/'demo/reviews.sqlite',self.catalog,True)
        result=demo.save(self.decision());self.assertFalse(result['eligible_as_reviewed_label'])
        with self.assertRaises(ValueError):self.reviews.import_bundle(demo.export())

    def test_uncertainty_requires_note(self):
        d=self.decision('uncertain');d['notes']=''
        with self.assertRaises(ValueError):self.reviews.save(d)

    def test_demo_cannot_import_disguised_human_approval(self):
        demo=Reviews(self.root/'demo/reviews.sqlite',self.catalog,True)
        demo.save(self.decision());bundle=demo.export()
        bundle['revisions'][0]['reviewer_kind']='human_self_reported'
        bundle['revisions'][0]['eligible_as_reviewed_label']=True
        with self.assertRaises(ValueError):demo.import_bundle(bundle)


if __name__=='__main__':unittest.main()
