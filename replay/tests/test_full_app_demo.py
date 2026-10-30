import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from replay.operator.full_demo import DemoStore


class DemoWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.bundle=self.root/'bundle';self.bundle.mkdir()
        data=dict(fill=[],cap=[dict(id='synthetic-frame',station='final_fixture',domain='synthetic',source_sha256='test-hash',slots=[dict(id='G01')])],sources={})
        p=self.bundle/'data.json';p.write_text(json.dumps(data));(self.bundle/'manifest.json').write_text(json.dumps(dict(files={'data.json':hashlib.sha256(p.read_bytes()).hexdigest()})))
        self.store=DemoStore(self.bundle,self.root/'workspace');self.count=0
    def tearDown(self):self.temp.cleanup()
    def send(self,kind,**kw):
        self.count+=1
        return self.store.command(dict(kind=kind,command_id='TEST-COMMAND-'+str(self.count),actor='AUTOMATED TEST',note='Synthetic unit fixture only',revision=self.store.snapshot()['revision'],**kw))
    def request(self):return self.send('request',frame_id='synthetic-frame',slot='G01')
    def test_response_is_required_and_never_approves_product(self):
        s=self.request();review=s['reviews'][0]['id']
        with self.assertRaisesRegex(ValueError,'Acknowledge'):self.send('close',review_id=review)
        self.send('acknowledge',review_id=review)
        with self.assertRaisesRegex(ValueError,'Acknowledge'):self.send('close',review_id=review)
        self.send('respond',review_id=review);s=self.send('close',review_id=review)
        self.assertEqual(s['reviews'][0]['status'],'closed');self.assertFalse(s['reviews'][0]['product_release_authorized']);self.assertIsNone(s['quality_decision'])
        self.assertTrue(self.store.export()['demonstration']);self.assertFalse(self.store.export()['eligible_for_team_review'])
    def test_unknown_source_or_slot_cannot_create_review(self):
        with self.assertRaises(ValueError):self.send('request',frame_id='OTHER',slot='G01')
        with self.assertRaises(ValueError):self.send('request',frame_id='synthetic-frame',slot='OTHER')
        self.assertEqual(self.store.snapshot()['revision'],0)
    def test_restart_preserves_source_bound_history(self):
        self.request();restored=DemoStore(self.bundle,self.root/'workspace');self.assertEqual(self.store.snapshot(),restored.snapshot())
        self.assertEqual(restored.export()['events'][0]['payload']['review']['source_identity'],{'sha256':'test-hash'})
    def test_changed_bundle_is_rejected(self):
        with (self.bundle/'data.json').open('a') as f:f.write(' ')
        with self.assertRaises(ValueError):DemoStore(self.bundle,self.root/'other-workspace')


if __name__=='__main__':unittest.main()
