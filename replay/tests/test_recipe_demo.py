from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from replay.config import Config
from replay.inputs import FramePacket, RecordingSource
from replay.monitor import InputInterrupted, MonitorSession
from replay.pipeline import analyze, assess, validate_references
from replay.recipes import RecipePlan
from replay.review import export_review
from replay.server import Manager
from replay.tests.test_pipeline import feature, fixture_video, make_run


def selection(frame=0, recipe='A', volume=30., video='video', available=None):
    return dict(selection_id=f's-{frame}', run_id=5, source_video_sha256=video,
                active_from_frame=frame, available_frame=frame if available is None else available,
                recipe_id=recipe, volumes_ml={'T100': volume} if recipe else {}, dispense_order='T100' if recipe else None,
                recorded_by='test operator', recorded_at_utc='2026-10-19T00:00:00Z',
                provenance=dict(kind='operator_plan', description='Explicit test input'))


def plan(*rows):
    return RecipePlan(dict(schema_version=1, plan_id='test', source_description='Explicit test plan', selections=list(rows)))


def observation():
    return dict(feature(150), run_id=5, event_id='run5-fill2', vial_id=2, start_frame=100,
                end_frame=140, departure_frame=155, available_frame=160, source_video_sha256='video',
                source_annotation_sha256=None, measurement_config_id=Config().fingerprint, measurement_profile_id='profile')


def reference(o, **changes):
    r = dict(reference_id='ref1', recipe_id=o['recipe_id'], recipe_definition_id=o['recipe_definition_id'],
             measurement_config_id=o['measurement_config_id'], measurement_profile_id='profile', lab=o['lab'],
             scope='same_run_development', source_event_id='run5-fill1', source_run_id=5,
             source_video_sha256='video', source_annotation_sha256=None, available_frame=90,
             source_measurement_available_frame=80, source_description='Manually selected test example; not certified clean')
    r.update(changes)
    return r


class RecipeDemoChecks(unittest.TestCase):
    def test_selection_activation_unknown_and_definition_change(self):
        p = plan(selection(), selection(170, volume=40), selection(250, recipe=None))
        a = p.bind(observation())
        b = p.bind(dict(observation(), start_frame=200, departure_frame=240))
        self.assertEqual(a['recipe_id'], 'A')
        self.assertNotEqual(a['recipe_definition_id'], b['recipe_definition_id'])
        self.assertIsNone(p.bind(dict(observation(), start_frame=270, departure_frame=300))['recipe_id'])
        self.assertIsNone(plan(selection(101)).bind(observation())['recipe_id'])
        self.assertIsNone(plan(selection(video='another recording')).bind(observation())['recipe_id'])

    def test_change_during_fill_is_unavailable_without_changing_measurement_qa(self):
        o = plan(selection(), selection(120, recipe='B')).bind(observation())
        self.assertTrue(o['measurement_valid'])
        self.assertEqual(o['context_quality_flags'], ['recipe_changed_during_fill'])
        self.assertEqual(assess(o, [reference(o)], Config())['status'], 'unavailable')

    def test_recipe_plan_rejects_backdating_bad_volumes_and_evaluation_ids(self):
        for change in (dict(available_frame=1), dict(volumes_ml={'T100':float('nan')}),
                       dict(dispense_order='T100>T100'), dict(vial_id=4), dict(recorded_by=''),
                       dict(provenance={'kind':'annotation','description':'Not operator context'})):
            with self.assertRaises(ValueError):
                plan(dict(selection(), **change))
        with self.assertRaises(ValueError):
            plan(selection(), selection())

    def test_reference_rejects_future_missing_incompatible_and_current_sources(self):
        o = plan(selection()).bind(observation())
        self.assertEqual(assess(o, [reference(o)], Config())['status'], 'within_reference_band')
        for changes in (dict(available_frame=100), dict(available_frame=120), dict(recipe_definition_id='wrong'),
                        dict(measurement_profile_id='wrong'), dict(source_video_sha256='wrong'),
                        dict(source_event_id=o['event_id']), dict(source_measurement_available_frame=95)):
            result = assess(o, [reference(o, **changes)], Config())
            self.assertEqual(result['status'], 'unavailable')
            self.assertTrue(result['reference_rejections'])
        for missing in ('source_event_id', 'source_measurement_available_frame', 'source_description'):
            r = reference(o)
            del r[missing]
            with self.assertRaises(ValueError):
                validate_references([r])
        r = reference(o, scope='external_commissioning')
        del r['available_frame']
        with self.assertRaises(ValueError):
            validate_references([r])

    def test_future_plan_changes_do_not_change_earlier_binding(self):
        a = plan(selection()).bind(observation())
        b = plan(selection(), selection(200, recipe='B')).bind(observation())
        self.assertEqual(a, b)

    def test_malformed_reference_is_rejected_at_monitor_and_assessment_boundaries(self):
        o = plan(selection()).bind(observation())
        r = reference(o, lab=[float('nan'),0.,0.])
        self.assertEqual(assess(o,[r],Config())['status'],'unavailable')
        with self.assertRaises(ValueError):
            MonitorSession(Config(),30,5,'video',references=[r])

    def test_same_processing_path_and_diagnostic_export_preserve_qa(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / 'source.mp4'
            fixture_video(path, length=180, with_fill=True)
            run = make_run(path)
            run['cameras']['vial_fill'].update(frames=180, duration_s=6.)
            c = Config(minimum_window_samples=10)
            replay = analyze(run, c, root / 'cache')
            monitor = MonitorSession(c, 30, 5, run['cameras']['vial_fill']['sha256'], session_id=replay.payload['manifest']['session_id'])
            monitor.start()
            for packet in RecordingSource(run['cameras']['vial_fill'], c.sample_stride):
                monitor.ingest(packet)
            monitor.stop('end_of_recording', 179)
            self.assertEqual(monitor.observations, replay.observations)
            before = deepcopy(replay.observations)
            report = json.loads(export_review(replay, run, root / 'review').read_text())
            self.assertEqual(replay.observations, before)
            self.assertEqual(report['counts'], dict(detected=1, measurement_available=0, rejected=1))
            self.assertTrue(report['cases'][0]['representative_frames'])
            for frame in report['cases'][0]['representative_frames']:
                self.assertTrue((root / 'review' / frame['path']).is_file())
            self.assertIsNone(report['cases'][0]['reviewed_label'])

    def test_interruption_stop_and_restart_do_not_bridge_or_carry_context(self):
        rgb = np.full((480,640,3),100,dtype=np.uint8)
        rgb[231:253,264:338]=20
        rgb[75:130,302:311]=(30,70,30)
        first = MonitorSession(Config(),30,5,'video',recipe_plan=plan(selection()))
        first.start()
        for i in (0,6,12):
            first.ingest(FramePacket(i,i/30,rgb))
        self.assertIsNotNone(first.detector.active)
        with self.assertRaises(InputInterrupted):
            first.ingest(FramePacket(30,1.,rgb))
        self.assertFalse(first.running)
        self.assertEqual(first.observations, [])
        self.assertTrue(any(e['kind']=='incomplete_fill_abandoned' for e in first.session_events))
        with self.assertRaises(ValueError):
            first.start()
        second = MonitorSession(Config(),30,5,'video')
        second.start()
        for i in (600,606,612):
            second.ingest(FramePacket(i,i/30,rgb))
        self.assertTrue(second.detector.active['partial_start_possible'])
        self.assertIsNone(second.recipe_plan.at(5,'video',612))
        self.assertNotEqual(first.session_id, second.session_id)
        second.stop()
        with self.assertRaises(ValueError):
            second.ingest(FramePacket(618,20.6,rgb))

    def test_timestamp_discontinuity_stops_input(self):
        m = MonitorSession(Config(),30,5,'video')
        m.start()
        rgb = np.zeros((480,640,3),dtype=np.uint8)
        m.ingest(FramePacket(0,0.,rgb))
        with self.assertRaises(InputInterrupted):
            m.ingest(FramePacket(6,2.,rgb))
        self.assertFalse(m.running)

    def test_monitor_manual_recipe_uses_owned_cursor_and_preserves_unknown(self):
        m = MonitorSession(Config(),30,5,'video')
        m.start()
        first = m.select_recipe('A', {'T100':30}, 'T100', 'operator', 'manual intended recipe')
        self.assertEqual(first['available_frame'],0)
        rgb = np.zeros((480,640,3),dtype=np.uint8)
        m.ingest(FramePacket(0,0.,rgb))
        m.ingest(FramePacket(6,.2,rgb))
        with self.assertRaises(ValueError):
            m.select_recipe('B', {'T100':30}, 'T100', 'operator', 'cannot backdate', active_from_frame=1)
        clear = m.select_recipe(None, {}, None, 'operator', 'recipe unknown')
        self.assertEqual(clear['available_frame'],6)
        self.assertIsNone(m.recipe_plan.at(5,'video',6)['recipe_id'])


if __name__ == '__main__':
    unittest.main()
