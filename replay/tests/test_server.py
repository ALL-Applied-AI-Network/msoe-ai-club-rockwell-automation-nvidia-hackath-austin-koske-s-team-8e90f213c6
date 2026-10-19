import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from replay.config import Config
from replay.dataset import Catalog
from replay.server import Manager, ReplayServer, byte_range
from replay.tests.test_pipeline import fixture_video


class ServerChecks(unittest.TestCase):
    def test_image_default_has_no_label_dependency_and_exports_only_available_events(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            videos = root / 'data/vial_fill/videos'
            videos.mkdir(parents=True)
            fixture_video(videos / 'vial_fill_video_5.mp4', length=180, with_fill=True)
            manager = Manager(Catalog(root / 'data', include_annotations=False), Config(), root / 'out')
            manager.ensure(5)
            manager.jobs[5].result(timeout=15)
            server = ReplayServer(('127.0.0.1', 0), manager)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f'http://127.0.0.1:{server.server_port}'
            try:
                with urlopen(base + '/api/fills?run=5') as response:
                    navigation = json.load(response)
                self.assertEqual(navigation['purpose'], 'replay_navigation_only')
                self.assertEqual(len(navigation['fills']), 1)
                available = navigation['fills'][0]['available_frame']
                with urlopen(base + f'/api/state?run=5&frame={available - 1}') as response:
                    self.assertEqual(json.load(response)['completed'], [])
                with urlopen(base + f'/api/state?run=5&frame={available}') as response:
                    state = json.load(response)
                self.assertEqual(state['fill_source'], 'image')
                self.assertEqual(len(state['completed']), 1)
                self.assertTrue(state['completed'][0]['measurement_valid'])
                self.assertIsNone(state['completed'][0]['recipe_id'])
                with urlopen(base + f'/api/events.jsonl?run=5&frame={available - 1}') as response:
                    events = [json.loads(line) for line in response.read().splitlines()]
                self.assertTrue(all(e['frame_index'] < available for e in events))
                self.assertFalse(any(e['kind'] in ('fill_completed', 'annotated_fill_end', 'recipe_context_available') for e in events))
                with self.assertRaises(HTTPError) as rejected:
                    urlopen(Request(base + '/api/references', data=json.dumps(dict(run_id=5, vial_id=1, frame_index=available)).encode(),
                                    headers={'Content-Type': 'application/json'}))
                self.assertEqual(rejected.exception.code, 400)
            finally:
                server.shutdown()
                server.server_close()
                manager.close()
                thread.join(timeout=5)

    def test_byte_ranges(self):
        self.assertEqual(byte_range('bytes=0-9', 100), (0, 9, True))
        self.assertEqual(byte_range('bytes=-10', 100), (90, 99, True))
        self.assertEqual(byte_range('bytes=90-', 100), (90, 99, True))
        self.assertEqual(byte_range(None, 100), (0, 99, False))
        for value in ('bytes=101-', 'bytes=0-1,4-5', 'bytes=-0', 'bytes=-', 'bytes=8-2'):
            with self.assertRaises(ValueError):
                byte_range(value, 100)

    def test_media_seek_and_reference_api_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            videos = root / 'data/vial_fill/videos'
            videos.mkdir(parents=True)
            video = videos / 'vial_fill_video_5.mp4'
            fixture_video(video)
            labels = root / 'data/vial_fill/labels'
            labels.mkdir()
            (labels / 'video5_vial_recipes.csv').write_text('vial,recipe_id,start_frame,end_frame,t100_ml,dispense_order\n1,A,0,30,30,T100\n')
            manager = Manager(Catalog(root / 'data'), Config(), root / 'out', fill_source='annotations')
            manager.ensure(5)
            manager.jobs[5].result(timeout=15)
            server = ReplayServer(('127.0.0.1', 0), manager)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f'http://127.0.0.1:{server.server_port}'
            try:
                with urlopen(Request(base + '/media/5/vial_fill.mp4', headers={'Range': 'bytes=0-63'})) as response:
                    self.assertEqual(response.status, 206)
                    self.assertEqual(response.read(), video.read_bytes()[:64])
                with urlopen(base + '/api/state?run=5&frame=59') as response:
                    self.assertEqual(json.load(response)['completed'], [])
                with urlopen(base + '/api/state?run=5&frame=60') as response:
                    self.assertEqual(len(json.load(response)['completed']), 1)
                def publish(frame, origin=None):
                    headers = {'Content-Type': 'application/json'}
                    if origin:
                        headers['Origin'] = origin
                    return urlopen(Request(base + '/api/references', data=json.dumps(dict(run_id=5, vial_id=1, frame_index=frame)).encode(), headers=headers))
                with self.assertRaises(HTTPError) as rejected:
                    publish(30)
                self.assertEqual(rejected.exception.code, 400)
                with publish(60) as response:
                    self.assertEqual(response.status, 201)
                with urlopen(base + '/api/state?run=5&frame=60') as response:
                    state = json.load(response)
                    self.assertEqual(state['completed'][0]['assessment']['status'], 'unavailable')
                with self.assertRaises(HTTPError) as rejected:
                    publish(60, 'http://another-origin.invalid')
                self.assertEqual(rejected.exception.code, 403)
                with urlopen(base + '/api/events.jsonl?run=5&frame=29') as response:
                    events = [json.loads(line) for line in response.read().splitlines()]
                    self.assertTrue(all(item['frame_index'] <= 29 for item in events))
                    self.assertFalse(any(item['kind'] == 'completed_vial_assessment' for item in events))
            finally:
                server.shutdown()
                server.server_close()
                manager.close()
                thread.join(timeout=5)


if __name__ == '__main__':
    unittest.main()
