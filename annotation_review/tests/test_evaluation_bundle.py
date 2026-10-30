"""Evaluation-only orchestration must retain unavailable truth and reject demos."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from annotation_review.evaluation_bundle import build


class EvaluationBundleTests(unittest.TestCase):
    def prepare(self, root):
        slot = root/'dataset1-real-transfer-20261022/review-v1'
        pool = root/'synthetic-iterations-20261022/visible-pool-review-v2'
        slot.mkdir(parents=True); pool.mkdir(parents=True)
        task = dict(task_id='TEST/G01', slot='G01', priority=True,
                    source_sha256='TEST', box_xyxy=[1,2,49,50], draft={'proposed_class':'no_vial'})
        packet = dict(training_eligible=False, tasks=[task])
        packet['packet_id'] = hashlib.sha256(json.dumps(packet,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        (slot/'packet.json').write_text(json.dumps(packet))
        (pool/'manifest.json').write_text(json.dumps(dict(packet_id='TEST-POOL',frames=1)))
        return packet

    def test_no_export_keeps_accuracy_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.prepare(root)
            result=build(SimpleNamespace(results=root,out=root/'out',slot_review=None,pool_review=None))
            for station in ('fixture','liquid'):
                self.assertIsNone(result[station]['accuracy'])
                self.assertFalse(result[station]['metrics_available'])
                self.assertEqual(result[station]['reviewed_targets'],0)
            self.assertEqual(result['exports_used'],{})

    def test_demonstration_export_cannot_become_reviewed_accuracy(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);packet=self.prepare(root)
            export=root/'demo.json'
            export.write_text(json.dumps(dict(schema='dataset1-slot-human-review/v1',
                packet_id=packet['packet_id'],training_eligible=False,demonstration=True,
                eligible_for_team_review=False,history=[])))
            with self.assertRaisesRegex(ValueError,'Demonstration'):
                build(SimpleNamespace(results=root,out=root/'out',slot_review=export,pool_review=None))
            self.assertFalse((root/'out/evaluation-status.json').exists())


if __name__=='__main__':unittest.main()
