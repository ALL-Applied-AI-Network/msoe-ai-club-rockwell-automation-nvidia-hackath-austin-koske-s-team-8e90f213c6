"""Offline contract checks for the starter; no model downloads or GPU work."""
import base64
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('cosmos_api', ROOT / 'cosmos' / 'api.py')
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)


class OnboardingChecks(unittest.TestCase):
    def test_image_request_and_authenticated_endpoint_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / 'image.png'
            image.write_bytes(b'local-test-image')
            observed = []
            class Response:
                def __enter__(self): return self
                def __exit__(self, *_): return False
                def read(self): return b'{"choices":[{"message":{"content":"two objects"}}]}'
            def capture(request, timeout):
                observed.append(request)
                return Response()
            with mock.patch.dict(os.environ, {'NVIDIA_API_KEY': 'test-only-placeholder'}), \
                 mock.patch.object(sys, 'argv', ['api.py', '--image', str(image)]), \
                 mock.patch.object(api.urllib.request, 'urlopen', side_effect=capture):
                self.assertEqual(api.main(), 0)
            request, = observed
            self.assertEqual(request.full_url, 'https://integrate.api.nvidia.com/v1/chat/completions')
            self.assertEqual(request.get_header('Authorization'), 'Bearer test-only-placeholder')
            body = json.loads(request.data)
            self.assertFalse(body['stream'])
            data_url = body['messages'][0]['content'][1]['image_url']['url']
            self.assertEqual(base64.b64decode(data_url.split(',', 1)[1]), image.read_bytes())

    def test_dry_run_never_requires_a_key_or_calls_network(self):
        with mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch.object(sys, 'argv', ['api.py', '--dry-run']), \
             mock.patch.object(api.urllib.request, 'urlopen', side_effect=AssertionError('network used')):
            self.assertEqual(api.main(), 0)

    def test_missing_key_fails_before_network(self):
        with mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch.object(sys, 'argv', ['api.py']), \
             mock.patch.object(api.urllib.request, 'urlopen', side_effect=AssertionError('network used')):
            with self.assertRaises(SystemExit) as caught:
                api.main()
            self.assertEqual(caught.exception.code, 2)

    def test_known_fixture_has_unique_identifiers(self):
        rows = [json.loads(line) for line in (ROOT / 'fixtures' / 'render_samples.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows), 12)
        self.assertEqual(len({row['sample_id'] for row in rows}), len(rows))
        self.assertTrue(all(row.get('physics_status') == 'uncalibrated_hypothesis' for row in rows))

    def test_hosted_client_cli_can_validate_without_optional_dependencies(self):
        result = subprocess.run([sys.executable, '-S', str(ROOT / 'cosmos' / 'api.py'), '--dry-run'],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(json.loads(result.stdout)['network_request'])


if __name__ == '__main__':
    unittest.main()
