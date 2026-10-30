import csv
from fractions import Fraction
import json
from pathlib import Path
import tempfile
import unittest

import av
import numpy as np

from inspect_csi_capture import inspect, sha
from evaluate_real_camera_preprocessing import preprocess


class CaptureTests(unittest.TestCase):
    def test_native_clock_and_original_bytes_survive_inventory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = root/'originals'; source.mkdir()
            path = source/'pilot.mp4'
            with av.open(str(path), 'w') as c:
                stream = c.add_stream('mpeg4', rate=25)
                stream.width = 64; stream.height = 48; stream.pix_fmt = 'yuv420p'
                for i in range(5):
                    frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), i*20, np.uint8), format='rgb24')
                    frame.pts = i; frame.time_base = Fraction(1, 25)
                    for packet in stream.encode(frame):
                        c.mux(packet)
                for packet in stream.encode():
                    c.mux(packet)
            (source/'process.csv').write_text('actual_meter_ml\n\n')
            before = {p.name: sha(p) for p in source.iterdir()}
            result = inspect(source, root/'inventory')
            self.assertEqual(before, {p.name: sha(p) for p in source.iterdir()})
            video = next(r for r in result['source_files'] if r['kind'] == 'video')
            self.assertEqual(video['decoded_frames'], 5)
            self.assertTrue(video['native_clock_usable'])
            with (root/'inventory'/video['native_frame_trace']).open() as f:
                trace = list(csv.DictReader(f))
            self.assertAlmostEqual(float(trace[-1]['pts_seconds']), .16)
            self.assertEqual(result['station_clock_mapping'], 'unreviewed')
            self.assertIsNone(result['accuracy'])
            single = inspect(path, root/'single-video-inventory')
            self.assertEqual(len(single['source_files']), 1)
            self.assertEqual(single['source_files'][0]['decoded_frames'], 5)
            with self.assertRaises(ValueError):
                inspect(source, root/'inventory')
            with self.assertRaises(ValueError):
                inspect(source, source/'inventory')

    def test_unavailable_camera_input_never_becomes_candidate_pixels(self):
        rgb = np.full((480, 640, 3), 150, np.uint8)
        calibration = dict(selected=['x192-y224', 'x112-y224', 'x112-y160'],
                           reference_rgb=[[150]*3]*3, maximum_disagreement=.05)
        corrected, reason = preprocess(rgb, 4.8, calibration)
        self.assertIsNone(corrected)
        self.assertEqual(reason['reason'], 'startup_reference_not_yet_available')
        corrected, reason = preprocess(rgb, 5., calibration)
        np.testing.assert_array_equal(corrected, rgb)
        rgb[160:208, 112:160] = 90
        corrected, reason = preprocess(rgb, 5., calibration)
        self.assertIsNone(corrected)
        self.assertEqual(reason['reason'], 'unhealthy_background_reference')


if __name__ == '__main__':
    unittest.main()
