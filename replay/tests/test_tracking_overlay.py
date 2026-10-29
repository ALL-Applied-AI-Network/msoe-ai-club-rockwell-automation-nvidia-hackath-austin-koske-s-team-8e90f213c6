from concurrent.futures import Future
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import urlopen

import av

from replay.config import Config
from replay.dataset import Catalog
from replay.pipeline import RunData
from replay.server import Manager, ReplayServer
from replay.tests.test_geometry import FIXTURE_BOXES
from replay.tests.test_pipeline import fixture_video
from replay.tracking_overlay import prepare_overlay, TrackingUnavailable


class OverlayChecks(unittest.TestCase):
    def fixture(self, root):
        videos=root/'data/vial_fill/videos';videos.mkdir(parents=True)
        path=videos/'vial_fill_video_5.mp4';fixture_video(path,length=60)
        catalog=Catalog(root/'data',include_annotations=False)
        config=Config(vision_model_path='fixture-unused.onnx',vision_model_sha256='a'*64)
        rows=[]
        with av.open(str(path)) as container:
            for i,f in enumerate(container.decode(video=0)):
                if i%6:continue
                rows.append(dict(frame_index=i,source_rgb_sha256=hashlib.sha256(f.to_ndarray(format='rgb24').tobytes()).hexdigest(),
                    geometry=dict(model_sha256=config.vision_model_sha256,proposals=deepcopy(FIXTURE_BOXES)),
                    measurement_valid=False,quality_flags=['unchanged_test_qa'],lab=None))
        payload=dict(manifest=dict(run_id=5,fill_source='image',frame_count=60,fps=30.,video_sha256=catalog.runs[5]['cameras']['vial_fill']['sha256']),
                     features=rows,observations=[],annotations=[],detection_events=[],session_events=[])
        return catalog,config,RunData(payload,config)

    def test_source_bound_cache_restart_and_measurement_immutability(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);catalog,config,data=self.fixture(root);before=deepcopy(data.payload)
            camera=catalog.runs[5]['cameras']['vial_fill']
            overlay=prepare_overlay(camera,data,config,root/'tracking')
            self.assertEqual(data.payload,before)
            self.assertEqual(overlay.snapshot(13)['row']['frame_index'],12)
            self.assertFalse(overlay.snapshot(13)['sample_matches_playhead'])
            self.assertTrue(overlay.snapshot(12)['sample_matches_playhead'])
            self.assertEqual(overlay.snapshot(0)['row']['liquids'][0]['association'],'track_not_confirmed')
            self.assertEqual(overlay.snapshot(12)['row']['liquids'][0]['association'],'candidate_link')
            with patch('replay.tracking_overlay.av.open',side_effect=AssertionError('Cache must not decode again')):
                restarted=prepare_overlay(camera,data,config,root/'tracking')
            self.assertEqual(restarted.snapshot(0),overlay.snapshot(0))
            self.assertEqual(restarted.snapshot(48),overlay.snapshot(48))
            self.assertEqual(data.payload,before)
            cache=next((root/'tracking').glob('*.json'));p=json.loads(cache.read_text())
            p['rows'][0]['motion']=99;cache.write_text(json.dumps(p))
            with self.assertRaisesRegex(ValueError,'cache identity/content'):
                prepare_overlay(camera,data,config,root/'tracking')

    def test_missing_original_rgb_or_incompatible_profile_fails_closed(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);catalog,config,data=self.fixture(root);camera=catalog.runs[5]['cameras']['vial_fill']
            with self.assertRaises(TrackingUnavailable):prepare_overlay(camera,data,Config(),root/'off')
            with self.assertRaises(TrackingUnavailable):prepare_overlay(dict(camera,fps=60.),data,config,root/'unsupported')
            data.features[0]['source_rgb_sha256']='0'*64
            with self.assertRaisesRegex(ValueError,'pixels differ'):
                prepare_overlay(camera,data,config,root/'bad-rgb')

    def test_http_is_lazy_causal_and_keeps_state_and_exports_identical(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);catalog,config,data=self.fixture(root)
            manager=Manager(catalog,config,root/'out');done=Future();done.set_result(None)
            manager.jobs[5]=done;manager.data[5]=data
            server=ReplayServer(('127.0.0.1',0),manager)
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            base=f'http://127.0.0.1:{server.server_port}'
            def get(path):
                with urlopen(base+path) as r:return r.read()
            paths=['/api/state?run=5&frame=48','/api/events.jsonl?run=5&frame=48','/api/observations.csv?run=5&frame=48','/api/manifest?run=5']
            try:
                before=[get(p) for p in paths];self.assertEqual(manager.tracking_jobs,{})
                json.loads(get('/api/tracking?run=5&frame=48'))
                manager.tracking_jobs[5].result(timeout=10)
                later=json.loads(get('/api/tracking?run=5&frame=48'))
                earlier=json.loads(get('/api/tracking?run=5&frame=0'))
                self.assertEqual(later['sample_frame_index'],48)
                self.assertEqual(earlier['row']['frame_index'],0)
                self.assertEqual(earlier['row']['events'][0]['available_frame'],0)
                self.assertTrue(all(e['available_frame']<=0 for e in earlier['row']['events']))
                self.assertFalse(earlier['measurement_authorized'])
                self.assertEqual(before,[get(p) for p in paths])
                self.assertEqual(json.loads(get('/api/tracking?run=5&frame=49'))['sample_age_frames'],1)
                for frame in (-1,60):
                    with self.assertRaises(HTTPError) as e:get(f'/api/tracking?run=5&frame={frame}')
                    self.assertEqual(e.exception.code,400)
            finally:
                server.shutdown();server.server_close();manager.close();thread.join(timeout=5)


if __name__=='__main__':unittest.main()
