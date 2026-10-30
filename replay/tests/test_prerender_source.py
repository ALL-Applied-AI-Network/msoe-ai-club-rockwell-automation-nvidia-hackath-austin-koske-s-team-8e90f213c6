"""Playback time, input-cache identity, and safe source switching."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from replay.operator.prerender import PlaybackClock,PrerenderBundle
from replay.operator.station import InspectionStation
from replay.operator.store import Conflict
from replay.tests.test_inspection_station import fixture
from replay.operator.twin_process import fill_states


class PrerenderSourceChecks(unittest.TestCase):
    def test_clock_is_one_to_one_and_pause_does_not_advance(self):
        time=[100.];clock=PlaybackClock('loop-1',10.,now=lambda:time[0])
        time[0]+=2.5;self.assertEqual(clock.seconds(),2.5)
        clock.pause(True);time[0]+=7;self.assertEqual(clock.seconds(),2.5)
        clock.pause(False);time[0]+=1.5;self.assertEqual(clock.seconds(),4.)
        time[0]+=20;self.assertEqual(clock.seconds(),10.)
        self.assertEqual(clock.snapshot()['speed'],1.)
        for bad in (-1.,float('nan')):
            with self.assertRaises(ValueError):PlaybackClock('loop',bad)

    def test_cache_is_input_only_and_refuses_changed_media(self):
        root=Path(self.enterContext(tempfile.TemporaryDirectory()));(root/'video.mp4').write_bytes(b'unit identity fixture')
        (root/'sample.png').write_bytes(b'unit sample fixture')
        sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
        manifest=dict(schema_version=1,kind='prerendered_isaac_input_only',cached_predictions=False,fps=30,width=640,height=480,
            video='video.mp4',video_sha256=sha(root/'video.mp4'),duration_s=2.,samples=[dict(seconds=0.,png='sample.png',png_sha256=sha(root/'sample.png'))])
        (root/'manifest.json').write_text(json.dumps(manifest));bundle=PrerenderBundle(root)
        self.assertEqual(bundle.duration,2.)
        with self.assertRaises(ValueError):bundle.file('../outside.png')
        (root/'video.mp4').write_bytes(b'changed')
        with self.assertRaises(ValueError):PrerenderBundle(root)
        (root/'manifest.json').write_text(json.dumps(manifest|{'cached_predictions':True}))
        with self.assertRaises(ValueError):PrerenderBundle(root)

    def test_mode_switch_retires_old_frames_and_preserves_cases(self):
        root=Path(self.enterContext(tempfile.TemporaryDirectory()))
        station=InspectionStation(root/'station','test-source','isaac');station.processor=object()
        frame,row,images,_=fixture(kind='isaac');station.accept(frame,row,images,generation=0)
        old_id=station.current_id
        command=dict(command='source_mode',mode='prerender',actor='Test operator',request_id='switch-1')
        with self.assertRaises(Conflict):station.command(command)
        station.bundle=SimpleNamespace(duration=21.7)
        station.command(command)
        self.assertTrue(station.switching);self.assertFalse(station.fresh())
        self.assertEqual(station.source_generation,1)
        self.assertTrue(station.command(command)['duplicate'])
        with self.assertRaises(Conflict):station.accept(frame,row,images,generation=0)
        frame.metadata['stream_id']='new-loop';station.active_mode='prerender'
        station.accept(frame,row,images,generation=1)
        self.assertFalse(station.switching);self.assertIn(old_id,station.cases)
        self.assertNotEqual(old_id,station.current_id)

    def test_dense_render_cadence_preserves_existing_process_samples(self):
        sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'setup/vision'))
        from twin_contract import SharedPath,default_plan
        plan=default_plan()
        low=list(fill_states(plan,SharedPath(2.,plan['prior_fraction'])))
        high=list(fill_states(plan,SharedPath(2.,plan['prior_fraction']),1/30))
        self.assertEqual((len(low),len(high)),(37,217))
        for index,sample in enumerate(low):
            for key in ('metered_ml','actual_components_ml','path_components_ml'):
                self.assertTrue(np.allclose(sample[key],high[index*6][key],atol=1e-8,rtol=0))


if __name__=='__main__':unittest.main()
