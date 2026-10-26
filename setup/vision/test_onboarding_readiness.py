"""CPU fixtures: incomplete evidence must never become deployment authorization."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from onboarding_readiness import inspect,run,REQUIRED


class OnboardingChecks(unittest.TestCase):
    def request(self):
        return dict(schema='station-onboarding-request/v1',station_id='fixture-camera',task='vial_fill',goal='Fixture input check',assets=[])

    def test_all_files_and_hashes_still_do_not_authorize_deployment(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);request=self.request()
            for role in REQUIRED:
                p=base/(role+'.txt');p.write_text('demonstration fixture only')
                request['assets'].append(dict(role=role,path=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                    data_domain='synthetic' if role=='synthetic_manifest' else 'not_applicable'))
            report=inspect(request,base)
            self.assertEqual(report['status'],'asset_checks_passed')
            self.assertFalse(report['deployment_authorized']);self.assertFalse(report['qualification_checks_executed'])
            self.assertIsNone(report['real_accuracy']);self.assertTrue(report['pending_validation'])

    def test_real_training_declaration_and_bad_hash_are_blocked_without_modifying_input(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);p=base/'source.bin';p.write_bytes(b'original pixels');request=self.request()
            request['assets']=[dict(role='synthetic_manifest',path=p.name,sha256='0'*64,data_domain='real')]
            report=inspect(request,base);reasons={r['reason'] for r in report['issues']}
            self.assertIn('new_CSI_training_requires_synthetic_data',reasons);self.assertIn('sha256_mismatch',reasons)
            self.assertIn('required_asset_not_supplied',reasons);self.assertEqual(p.read_bytes(),b'original pixels')

    def test_outputs_cannot_overwrite_or_enter_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);inputs=base/'inputs';inputs.mkdir();p=inputs/'request.json';p.write_text(json.dumps(self.request()))
            with self.assertRaises(ValueError):run(p,inputs/'generated')
            out=base/'report';run(p,out)
            self.assertEqual(json.loads((out/'readiness.json').read_text())['status'],'asset_checks_incomplete')
            before=(out/'readiness.json').read_bytes()
            with self.assertRaises(ValueError):run(p,out)
            self.assertEqual(before,(out/'readiness.json').read_bytes())

    def test_duplicate_roles_and_unsupported_tasks_are_rejected(self):
        request=self.request();request['task']='arbitrary_factory'
        with self.assertRaises(ValueError):inspect(request,Path('.'))
        request=self.request();asset=dict(role='checkpoint',path='missing.pt',data_domain='not_applicable');request['assets']=[asset,asset]
        with self.assertRaises(ValueError):inspect(request,Path('.'))


if __name__=='__main__':unittest.main()
