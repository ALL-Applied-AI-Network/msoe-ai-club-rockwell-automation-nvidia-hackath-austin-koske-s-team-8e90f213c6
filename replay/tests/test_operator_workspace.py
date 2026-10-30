"""Operator workflow, provenance, playback causality and local HTTP boundaries."""
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import uuid

import av
import cv2
import numpy as np

from replay.geometry import encode_mask
from replay.measurement import rgb_to_lab
from replay.operator.evidence import Evidence, sha
from replay.operator.store import OperatorStore, Conflict
from replay.operator.server import OperatorServer


def fixture(root):
    video = root/'source.mp4'
    with av.open(str(video), 'w') as c:
        stream = c.add_stream('mpeg4', rate=30)
        stream.width = 64; stream.height = 48; stream.pix_fmt = 'yuv420p'
        for i in range(96):
            color = [80, 125, 165] if i < 18 else [170, 90, 50]
            f = av.VideoFrame.from_ndarray(np.full((48, 64, 3), color, np.uint8), format='rgb24')
            f.pts = i; f.time_base = Fraction(1, 30)
            for packet in stream.encode(f):
                c.mux(packet)
        for packet in stream.encode():
            c.mux(packet)
    rows = []
    mask = np.zeros((48,64), np.uint8); mask[20:40,20:40] = 1
    with av.open(str(video)) as c:
        for i, f in enumerate(c.decode(video=0)):
            if i % 6:
                continue
            rgb = f.to_ndarray(format='rgb24'); median = np.median(rgb[mask.astype(bool)],axis=0)
            rows.append(dict(source_frame=i, seconds=float(f.pts*f.time_base), pts=f.pts, time_base=str(f.time_base),
                source_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                measurement=dict(rgb=median.tolist(), lab=rgb_to_lab(median).tolist(), quality_flags=[],
                    measurement_valid=True, geometry=dict(mask=encode_mask(mask),mask_pixels=int(mask.sum()),
                        track_id=1 if i < 18 else 2,vial_box=[10,5,55,46])),
                fill_detection=dict(phase='visible_stream')))
    evidence = root/'evidence'; arm = evidence/'pytorch-pass0'; arm.mkdir(parents=True)
    (arm/'frames.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (arm/'events.json').write_text(json.dumps([dict(kind='fill_started',video_s=0,frame_index=0)]))
    (arm/'completed.json').write_text(json.dumps(dict(artifacts={n:sha(arm/n) for n in ['frames.jsonl','events.json']})))
    (evidence/'completed.json').write_text(json.dumps(dict(runs={'pytorch-pass0':sha(arm/'completed.json')})))
    (evidence/'plan.json').write_text(json.dumps(dict(source_run=4,source_sha256=sha(video),
        qa_config=dict(image_size=[64,48],review_threshold_de2000=5,max_temporal_de2000=5),
        qa_code_sha256={'fixture':'synthetic-test-only'},frozen_models={'fixture':'synthetic-test-only'})))
    (evidence/'completed.json').write_text(json.dumps(dict(runs={'pytorch-pass0':sha(arm/'completed.json')},plan_sha256=sha(evidence/'plan.json'))))
    return Evidence(evidence,video)


class OperatorChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.evidence = fixture(self.root)
        self.store = OperatorStore(self.root/'workspace', self.evidence)

    def command(self, kind, frame, case=None, **kwargs):
        body = dict(command=kind, source_frame=frame, actor='Test operator', request_id=uuid.uuid4().hex, **kwargs)
        if case:
            cases, _ = self.store.reduce(self.store.events())
            body.update(case_id=case, expected_revision=cases[case]['revision'])
        return self.store.command(body)

    def start(self, frame=0, physical='vial-1', recipe='A', confirmed=True):
        return self.command('start', frame, physical_vial_id=physical, batch_id='test-batch',recipe_id=recipe,
                            identity_confirmed=confirmed,note='Explicit synthetic workflow fixture')['case_id']

    def test_endpoint_reference_causality_restart_and_source_preservation(self):
        original_video, original_frames = sha(self.evidence.video), sha(self.evidence.root/'pytorch-pass0/frames.jsonl')
        first = self.start()
        self.command('complete', 12, first, confirmed=True)
        self.command('save_reference', 12, first, note='Illustrative reference', lab=[999,999,999], approved_reference=True)
        first_view = self.store.snapshot(.4)['current']
        self.assertIsNone(first_view['assessment']['diagnostic_status']) # Same-fill reference never eligible.
        second = self.start(18,'vial-2')
        self.command('complete', 30, second, confirmed=True)
        current = self.store.snapshot(1.)['current']
        self.assertEqual(current['assessment']['diagnostic_status'],'deviation')
        self.assertEqual(current['assessment']['status'],'unavailable')
        self.assertFalse(current['assessment']['reference']['approved_reference'])
        self.assertNotEqual(current['assessment']['reference']['lab'],[999,999,999])
        again = OperatorStore(self.root/'workspace', self.evidence)
        self.assertEqual(again.session_id,self.store.session_id)
        self.assertEqual(again.snapshot(1.)['cases'],self.store.snapshot(1.)['cases'])
        self.assertEqual(sha(self.evidence.video),original_video)
        self.assertEqual(sha(self.evidence.root/'pytorch-pass0/frames.jsonl'),original_frames)

    def test_completion_is_manual_and_invalid_samples_stay_unavailable(self):
        case = self.start()
        with self.assertRaises(ValueError):
            self.command('complete',12,case,confirmed=False)
        self.assertIsNone(self.store.snapshot(.4)['current']['endpoint'])
        self.evidence.by_frame[12]['measurement']['measurement_valid'] = False
        self.command('complete',12,case,confirmed=True)
        current = self.store.snapshot(.4)['current']
        self.assertFalse(current['endpoint']['measurement_valid'])
        with self.assertRaises(ValueError):
            self.command('save_reference',12,case,note='Cannot approve a bad sample')
        self.assertEqual(self.store.snapshot(.4)['hardware_commands_sent'],0)

    def test_no_recipe_or_identity_is_inferred(self):
        case = self.start(recipe='',confirmed=False,physical='')
        self.command('complete',12,case,confirmed=True)
        current = self.store.snapshot(.4)['current']
        self.assertEqual(current['recipe_id'],'')
        self.assertIn('unknown',current['assessment']['reason'])
        with self.assertRaises(ValueError):
            self.command('save_reference',12,case,note='Unknown context')

    def test_review_latches_acknowledgement_and_distinct_action_and_resolution(self):
        case = self.start()
        self.command('request_review',6,case,note='Operator cannot establish expected appearance')
        self.command('acknowledge',6,case)
        self.assertEqual(self.store.snapshot(.2)['current']['review_state'],'acknowledged')
        with self.assertRaises(Conflict):
            self.command('close_review',6,case,note='Acknowledgement is not resolution')
        with self.assertRaises(ValueError):
            self.command('record_action',6,case,action='reviewed_evidence',note='Not performed',performed=False)
        self.command('record_action',6,case,action='set_aside_for_review',note='Operator practice response',performed=True,reported_station_route='labeled_good')
        with self.assertRaises(Conflict):
            self.command('close_review',6,case,note='Still an open fill')
        self.command('complete',12,case,confirmed=True)
        self.command('close_review',12,case,note='Follow-up documented; independent outcome unknown')
        current = self.store.snapshot(.4)['current']
        self.assertEqual(current['review_state'],'closed')
        self.assertFalse(current['resolution']['product_release_authorized'])
        self.assertEqual(current['actions'][0]['route_semantics'],'unconfirmed')
        self.assertEqual(current['actions'][0]['outcome_status'],'not_independently_verified')
        self.command('request_review',12,case,note='Reopened question')
        self.command('acknowledge',12,case)
        with self.assertRaises(Conflict):
            self.command('close_review',12,case,note='Prior-cycle action cannot resolve this one')

    def test_reference_saved_after_later_fill_started_is_ineligible(self):
        first = self.start(); self.command('complete',12,first,confirmed=True)
        second = self.start(18,'vial-2'); self.command('complete',30,second,confirmed=True)
        self.command('save_reference',30,first,note='Late reference publication')
        view = self.store.snapshot(1.,second)['current']
        self.assertIsNone(view['assessment']['diagnostic_status'])
        self.assertIn('before this vial',view['assessment']['reason'])

    def test_request_idempotency_revision_conflicts_and_append_only_database(self):
        body = dict(command='start',source_frame=0,actor='Test',request_id='retry-id',recipe_id='A',physical_vial_id='1',identity_confirmed=True,note='Manual')
        first = self.store.command(body); repeat = self.store.command(body)
        self.assertTrue(repeat['duplicate']); self.assertEqual(first['event_seq'],repeat['event_seq'])
        with self.assertRaises(Conflict):
            self.store.command(dict(body,recipe_id='B'))
        case = first['case_id']
        self.command('request_review',6,case,note='Follow up')
        with self.assertRaises(Conflict):
            self.store.command(dict(command='acknowledge',source_frame=6,actor='Other',request_id=uuid.uuid4().hex,case_id=case,expected_revision=1))
        with self.store.connect() as db:
            with self.assertRaises(sqlite3.IntegrityError):
                db.execute('UPDATE events SET actor=?',('changed',))
            with self.assertRaises(sqlite3.IntegrityError):
                db.execute('DELETE FROM events')

    def test_backward_playback_hides_future_records_and_cannot_backdate_actions(self):
        case = self.start(6)
        self.command('request_review',18,case,note='A later review')
        self.assertEqual(self.store.snapshot(0)['cases'],[])
        past = self.store.snapshot(.2)['current']
        self.assertEqual(past['review_state'],'none')
        self.assertFalse(past['editable'])
        with self.assertRaises(Conflict):
            self.command('complete',12,case,confirmed=True)
        exported = self.store.export(.2)
        self.assertEqual(len(exported['events']),1)
        self.assertEqual(exported['hardware_commands_sent'],0)

    def test_workspace_and_frozen_evidence_binding(self):
        other = deepcopy(self.evidence); other.identity['source_video_sha256']='other-source'
        with self.assertRaises(ValueError):
            OperatorStore(self.root/'workspace',other)
        (self.evidence.root/'pytorch-pass0/frames.jsonl').write_text('changed')
        with self.assertRaises(ValueError):
            Evidence(self.evidence.root,self.evidence.video)

    def test_local_http_security_range_exact_images_and_audit_exports(self):
        server = OperatorServer(('127.0.0.1',0),self.store)
        thread = threading.Thread(target=server.serve_forever,daemon=True); thread.start()
        self.addCleanup(thread.join,3); self.addCleanup(server.server_close); self.addCleanup(server.shutdown)
        base = f'http://127.0.0.1:{server.server_port}'
        with urlopen(base+'/api/session') as r:
            csrf = json.load(r)['csrf_token']
        body = dict(command='start',source_frame=0,actor='Test',request_id=uuid.uuid4().hex,recipe_id='A',physical_vial_id='1',identity_confirmed=True,note='HTTP fixture')
        def post(token=csrf,origin=None):
            headers={'Content-Type':'application/json','X-Operator-CSRF':token}
            if origin:headers['Origin']=origin
            return urlopen(Request(base+'/api/command',data=json.dumps(body).encode(),headers=headers))
        for token,origin in [('',None),(csrf,'http://other.invalid')]:
            with self.assertRaises(HTTPError) as rejected:post(token,origin)
            self.assertEqual(rejected.exception.code,403)
        with post() as r:self.assertEqual(r.status,201)
        with urlopen(Request(base+'/video.mp4',headers={'Range':'bytes=0-63'})) as r:
            self.assertEqual(r.status,206); self.assertEqual(r.read(),self.evidence.video.read_bytes()[:64])
        with urlopen(base+'/api/evidence?frame=0&at_s=0&kind=mask') as r:
            png=cv2.imdecode(np.frombuffer(r.read(),np.uint8),cv2.IMREAD_UNCHANGED)
            self.assertEqual(int((png>0).sum()),400)
        with self.assertRaises(HTTPError):urlopen(base+'/api/evidence?frame=12&at_s=0')
        with urlopen(base+'/api/export.json?at_s=0') as r:
            report=json.load(r); self.assertFalse(report['independent_outcomes_available']); self.assertEqual(report['hardware_commands_sent'],0)
        with urlopen(base+'/api/export.csv?at_s=0') as r:
            self.assertIn('reported_station_route',r.read().decode())


if __name__ == '__main__':
    unittest.main()
