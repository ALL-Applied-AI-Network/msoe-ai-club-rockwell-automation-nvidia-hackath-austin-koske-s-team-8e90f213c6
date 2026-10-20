import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from package_early_artifacts import package, ARMS


class ArtifactChecks(unittest.TestCase):
    def fixture(self, root):
        plan = b'{"frozen":true}'
        (root/'experiment-plan.json').write_bytes(plan)
        (root/'preparation.json').write_text(json.dumps({'experiment-plan.json': hashlib.sha256(plan).hexdigest()}))
        for arm in ARMS:
            folder = root/'models'/arm; (folder/'train').mkdir(parents=True)
            weights = ('test-weights-'+arm).encode(); (folder/'best.pt').write_bytes(weights)
            (folder/'model.json').write_text(json.dumps(dict(checkpoint_sha256=hashlib.sha256(weights).hexdigest(),
                excluded_runs=[5], epochs_completed=1, best_run4_epoch=1, best_run4_liquid_ap50_95=.5)))
            key = 'run4_liquid_ap50_95' if arm == 'fasterrcnn-resnet50' else 'metrics/mAP50-95(B)'
            curve = folder/('training.csv' if arm == 'fasterrcnn-resnet50' else 'train/results.csv')
            curve.write_text('epoch,'+key+'\n1,0.5\n')
            (folder/('model-check.json' if arm == 'fasterrcnn-resnet50' else 'train/args.yaml')).write_text('{}')

    def test_nul_csv_rejected_without_modifying_source_or_publishing_archive(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); self.fixture(root)
            source = root/'models/yolo11n-960/train/results.csv'
            damaged = source.read_bytes()+b'\0'; source.write_bytes(damaged)
            out = root/'bundle.tar.gz'
            with self.assertRaisesRegex(ValueError, 'NUL'):
                package(root, out)
            self.assertEqual(source.read_bytes(), damaged)
            self.assertFalse(out.exists())
            self.assertFalse(Path(str(out)+'.partial').exists())

    def test_changed_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); self.fixture(root)
            (root/'models/yolo11n-960/best.pt').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'checkpoint'):
                package(root, root/'bundle.tar.gz')


if __name__ == '__main__':
    unittest.main()
