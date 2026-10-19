from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from replay.config import Config
from replay.dataset import Catalog
from replay.detection import FillDetector, detect
from replay.pipeline import RunData, analyze, build_detected_observations
from replay.tests.test_pipeline import annotation, feature, fixture_video, make_run
from replay.validation import evaluate


def samples(end=300, streams=((12, 42), (126, 162)), moves=((228, 258),)):
    rows = []
    for frame in range(0, end + 1, 6):
        moving = any(a <= frame <= b for a, b in moves)
        row = feature(frame, valid=not moving)
        row.update(motion=25. if moving else .3, stream_contrast=15. if any(a <= frame <= b for a, b in streams) else 1.,
                   holder_dark_fraction=.8)
        rows.append(row)
    return rows


class DetectorChecks(unittest.TestCase):
    def setUp(self):
        self.config = Config()

    def test_ingredient_pauses_do_not_finish_a_vial(self):
        rows = samples()
        detections, events = detect(rows, self.config, 30, 5)
        self.assertEqual(len(detections), 1)
        d = detections[0]
        self.assertEqual(len(d['episodes']), 2)
        self.assertEqual((d['start_frame'], d['end_frame'], d['departure_frame'], d['confirmed_frame']), (12, 168, 228, 234))
        self.assertEqual(rows[15]['fill_detection']['phase'], 'ingredient_pause')
        self.assertEqual([e['frame_index'] for e in events if e['kind'] == 'fill_completed'], [234])

    def test_prefix_is_unchanged_and_eof_does_not_complete_a_fill(self):
        rows = samples()
        prefix = deepcopy(rows[:38])
        pd, pe = detect(prefix, self.config, 30, 5)
        fd, fe = detect(rows, self.config, 30, 5)
        self.assertEqual(pd, [])
        self.assertEqual(prefix, rows[:38])
        self.assertEqual(pe, [e for e in fe if e['frame_index'] <= prefix[-1]['frame_index']])
        changed = samples(streams=((12, 42),), moves=())
        cd, ce = detect(changed, self.config, 30, 5)
        self.assertEqual(cd, [])
        self.assertEqual([r['fill_detection'] for r in changed[:20]], [r['fill_detection'] for r in rows[:20]])

    def test_single_motion_spike_and_empty_cycles_do_not_create_fills(self):
        rows = samples()
        rows[12]['motion'] = 30.
        detections, _ = detect(rows, self.config, 30, 5)
        self.assertEqual(len(detections), 1)
        detections, _ = detect(samples(streams=()), self.config, 30, 5)
        self.assertEqual(detections, [])

    def test_partial_start_and_debounce(self):
        detections, _ = detect(samples(streams=((0, 42),)), self.config, 30, 5)
        self.assertTrue(detections[0]['partial_start_possible'])
        self.assertEqual(detections[0]['start_available_frame'], 12)
        detections, _ = detect(samples(streams=((12, 18),)), self.config, 30, 5)
        self.assertEqual(detections, [])

    def test_missing_evidence_fails_measurement_and_gap_resets_confirmation(self):
        rows = samples()
        del rows[8]
        detections, _ = detect(rows, self.config, 30, 5)
        self.assertIn('detection_sample_gap', detections[0]['quality_flags'])
        observations = build_detected_observations(rows, detections, self.config, 30, 5)
        self.assertFalse(observations[0]['measurement_valid'])
        d = FillDetector(self.config, 30, 5)
        d.update(rows[0])
        with self.assertRaises(ValueError):
            d.update(rows[0])

    def test_tail_measurement_never_uses_departure_or_future_samples(self):
        rows = samples()
        detections, events = detect(rows, self.config, 30, 5)
        observations = build_detected_observations(rows, detections, self.config, 30, 5)
        o = observations[0]
        self.assertTrue(o['measurement_valid'])
        self.assertIsNone(o['recipe_id'])
        self.assertTrue(all(f < o['departure_frame'] <= o['available_frame'] for f in o['sample_frames']))
        payload = dict(manifest=dict(run_id=5, frame_count=301, fps=30, fill_source='image'),
                       features=rows, annotations=[], observations=observations, detection_events=events)
        data = RunData(payload, self.config)
        self.assertEqual(data.snapshot(233, [])['completed'], [])
        self.assertEqual(len(data.snapshot(234, [])['completed']), 1)
        self.assertEqual(data.snapshot(200, [])['completed'], [])
        for f in (0, 42, 228, 233, 234):
            self.assertTrue(all(event['frame_index'] <= f for event in data.events(f, [])))
        self.assertEqual(data.completed(234, [])[0]['assessment']['status'], 'unavailable')

    def test_annotation_removal_or_poisoning_does_not_change_image_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            videos = root / 'data/vial_fill/videos'
            videos.mkdir(parents=True)
            path = videos / 'vial_fill_video_5.mp4'
            fixture_video(path, length=180, with_fill=True)
            run = make_run(path)
            run['cameras']['vial_fill'].update(frames=180, duration_s=6.)
            a = analyze(run, self.config, root / 'cache')
            self.assertEqual(len(a.observations), 1)
            self.assertTrue(a.observations[0]['measurement_valid'])
            poisoned = dict(run, annotations=[annotation(start=70, end=80)], annotation_sha256='poisoned')
            b = analyze(poisoned, self.config, root / 'cache')
            removed = dict(run, annotations=[], annotation_sha256=None)
            c = analyze(removed, self.config, root / 'cache')
            for key in ('detections', 'detection_events', 'observations'):
                self.assertEqual(a.payload[key], b.payload[key])
                self.assertEqual(a.payload[key], c.payload[key])
            labels = root / 'data/vial_fill/labels'
            labels.mkdir()
            (labels / 'video5_vial_recipes.csv').write_text('vial,recipe_id,start_frame,end_frame\nbad,unknown,nan,nan\n')
            with self.assertRaises(ValueError):
                Catalog(root / 'data', include_annotations=True)
            self.assertEqual(Catalog(root / 'data', include_annotations=False).runs[5]['annotations'], [])

    def test_evaluation_is_one_to_one_and_cannot_match_a_duplicate(self):
        detections, _ = detect(samples(), self.config, 30, 5)
        manifest = dict(run_id=5, fps=30)
        report = evaluate(detections * 2, [annotation(start=12, end=168)], manifest)
        self.assertEqual(report['summary']['matched_fills'], 1)
        self.assertEqual(report['summary']['extra_detections'], 1)
        report = evaluate(detections, [], manifest)
        self.assertIsNone(report['summary']['precision'])


if __name__ == '__main__':
    unittest.main()
