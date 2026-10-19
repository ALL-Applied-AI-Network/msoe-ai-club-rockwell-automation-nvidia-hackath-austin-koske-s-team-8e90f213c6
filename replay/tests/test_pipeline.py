import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import av
import numpy as np

from replay.config import Config
from replay.dataset import Annotation, Catalog, read_annotations, sha256
from replay.measurement import Extractor, rgb_to_lab
from replay.pipeline import (RunData, analyze, assess, build_observations,
                             validate_references)


def feature(frame, valid=True, lab=(40., 10., 15.)):
    return dict(frame_index=frame, video_s=frame / 30, measurement_valid=valid,
                quality_flags=[] if valid else ['carrier_motion'], lab=list(lab), rgb=[110, 80, 65])


def annotation(vial=1, start=0, end=30):
    return Annotation(vial, 'A', start, end, {'T100': 30., 'T200': 0., 'T400': 0.}, 'T100', vial, None)


def fixture_video(path, length=90, change_after=None, with_fill=False):
    with av.open(str(path), 'w') as output:
        stream = output.add_stream('libx264', rate=30)
        stream.width, stream.height = 640, 480
        stream.pix_fmt = 'yuv444p'
        stream.options = {'crf': '0', 'g': '1', 'bf': '0', 'preset': 'ultrafast'}
        for index in range(length):
            rgb = np.full((480, 640, 3), 100, dtype=np.uint8)
            rgb[231:253, 264:338] = 20
            rgb[340:390, 288:330] = (210, 110, 60) if change_after is None or index < change_after else (40, 80, 210)
            if with_fill:
                phase = index % 180
                if 12 <= phase < 42 or 60 <= phase < 84:
                    rgb[75:130, 302:311] = (30, 70, 30)
                if 126 <= phase < 156:
                    rgb[200:460, 220:420] = 220 if (index // 6) % 2 else 30
            frame = av.VideoFrame.from_ndarray(rgb, format='rgb24')
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)


def make_run(path, annotations=None):
    return dict(run_id=5, cameras={'vial_fill': dict(path=str(path), fps=30., frames=90,
                width=640, height=480, start_pts_s=0., duration_s=3., sha256=sha256(path))},
                annotations=annotations or [annotation()], annotation_path=None,
                annotation_sha256='test-labels', clock_path=None, clock_sha256=None)


class PipelineChecks(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        self.features = [feature(frame) for frame in range(0, 90, 6)]
        self.observation = build_observations(self.features, [annotation()], self.config, 30., 90, 5)[0]
        self.observation.update(source_video_sha256='video', source_annotation_sha256='labels')

    def reference(self, **updates):
        result = dict(reference_id='r1', recipe_id='A', lab=[40., 10., 15.],
                      scope='same_run_development', measurement_config_id=self.config.fingerprint,
                      source_run_id=5, source_video_sha256='video', source_annotation_sha256='labels',
                      source_event_id='earlier-vial', available_frame=0, source_measurement_available_frame=0,
                      source_description='Explicit synthetic test reference')
        result.update(updates)
        return result

    def test_post_fill_window_is_hidden_until_available(self):
        data = RunData(dict(manifest={'run_id': 5, 'fps': 30., 'frame_count': 90},
                            features=self.features, observations=[self.observation],
                            annotations=[annotation().to_dict()]), self.config)
        self.assertEqual(data.snapshot(59, [])['completed'], [])
        self.assertEqual(len(data.snapshot(60, [])['completed']), 1)
        self.assertEqual(data.snapshot(12, [])['completed'], [])  # backward seek
        self.assertTrue(all(frame <= self.observation['available_frame'] for frame in self.observation['sample_frames']))

    def test_reference_cannot_use_current_or_future_fill(self):
        observation = dict(self.observation, start_frame=70)
        for reference in [self.reference(available_frame=70), self.reference(available_frame=80),
                          self.reference(source_event_id=observation['event_id'])]:
            self.assertEqual(assess(observation, [reference], self.config)['status'], 'unavailable')
        self.assertEqual(assess(observation, [self.reference(available_frame=60)], self.config)['status'], 'within_reference_band')

    def test_references_must_match_roi_and_recording_provenance(self):
        observation = dict(self.observation, start_frame=70)
        for reference in [self.reference(measurement_config_id='another-profile'),
                          self.reference(source_video_sha256='different-video'),
                          self.reference(source_annotation_sha256='revised-labels')]:
            self.assertEqual(assess(observation, [reference], self.config)['status'], 'unavailable')

    def test_invalid_measurement_cannot_become_match(self):
        invalid = dict(self.observation, start_frame=70, measurement_valid=False, quality_flags=['carrier_motion'])
        self.assertEqual(assess(invalid, [self.reference()], self.config)['status'], 'unavailable')
        damaged = [dict(item, measurement_valid=False, quality_flags=['carrier_motion']) for item in self.features]
        observation = build_observations(damaged, [annotation()], self.config, 30, 90, 5)[0]
        self.assertFalse(observation['measurement_valid'])
        self.assertIn('insufficient_valid_post_fill_samples', observation['quality_flags'])

    def test_external_reference_is_explicit_and_finite(self):
        external = self.reference(scope='external_commissioning', source_description='Independent commissioning block', available_frame=-1)
        self.assertEqual(assess(self.observation, [external], self.config)['status'], 'within_reference_band')
        with self.assertRaises(ValueError):
            validate_references([dict(external, lab=[float('nan'), 0, 0])])
        with self.assertRaises(ValueError):
            validate_references([dict(external, source_description='')])

    def test_color_difference_requests_review_without_causal_claim(self):
        observation = dict(self.observation, start_frame=70, lab=[70., -25., -20.])
        decision = assess(observation, [self.reference()], self.config)
        self.assertEqual(decision['status'], 'review')
        self.assertGreater(decision['delta_e2000'], self.config.review_threshold_de2000)
        self.assertNotIn('carryover', decision['explanation'])

    def test_black_frame_is_unavailable_and_rgb_convention_is_correct(self):
        measurement = Extractor(self.config).measure(np.zeros((480, 640, 3), dtype=np.uint8))
        self.assertFalse(measurement['measurement_valid'])
        self.assertIsNone(measurement['lab'])
        np.testing.assert_allclose(rgb_to_lab([255, 255, 255]), [100, 0, 0], atol=.02)

    def test_annotations_are_validated_and_missing_labels_stay_unknown(self):
        self.assertEqual(read_annotations(None, 90), [])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'labels.csv'
            path.write_text('vial,recipe_id,start_frame,end_frame,t100_ml,dispense_order\n1,A,0,30,30,T100\n2,A,20,50,30,T100\n')
            with self.assertRaises(ValueError):
                read_annotations(path, 90)

    def test_invalid_config_and_missing_override_are_rejected(self):
        for options in ({'sample_stride': 0}, {'sample_stride': 1.5},
                        {'review_threshold_de2000': float('nan')}, {'valid_fraction_min': 2}):
            with self.assertRaises(ValueError):
                Config(**options)
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(FileNotFoundError):
                Catalog(temporary, labels_dir=Path(temporary) / 'missing')

    def test_previous_recipe_means_previous_batch_not_previous_vial(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'labels.csv'
            path.write_text('vial,recipe_id,start_frame,end_frame,t100_ml,dispense_order\n1,A,0,20,30,T100\n2,A,25,45,30,T100\n3,B,50,70,30,T100\n')
            rows = read_annotations(path, 90)
            self.assertEqual([row.previous_recipe_id for row in rows], [None, None, 'A'])
            self.assertEqual([row.position_in_batch for row in rows], [1, 2, 1])

    @patch('replay.geometry.VialGeometry.detect',return_value=[dict(class_id=0,box=[278,65,340,430],confidence=.99),dict(class_id=1,box=[288,340,330,390],confidence=.99)])
    def test_decode_prefix_is_independent_of_future_images(self, _detector):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            left, right = root / 'a.mp4', root / 'b.mp4'
            fixture_video(left)
            fixture_video(right, change_after=60)
            a = analyze(make_run(left), self.config, root / 'cache')
            b = analyze(make_run(right), self.config, root / 'cache')
            prefix_a = [{k: row[k] for k in ('frame_index', 'lab', 'motion', 'stream_contrast', 'fill_detection', 'quality_flags')} for row in a.features if row['frame_index'] < 60]
            prefix_b = [{k: row[k] for k in ('frame_index', 'lab', 'motion', 'stream_contrast', 'fill_detection', 'quality_flags')} for row in b.features if row['frame_index'] < 60]
            self.assertEqual(prefix_a, prefix_b)
            self.assertNotEqual(a.features[-1]['lab'], b.features[-1]['lab'])
            cached = analyze(make_run(left), self.config, root / 'cache')
            self.assertEqual(a.payload, cached.payload)


if __name__ == '__main__':
    unittest.main()
