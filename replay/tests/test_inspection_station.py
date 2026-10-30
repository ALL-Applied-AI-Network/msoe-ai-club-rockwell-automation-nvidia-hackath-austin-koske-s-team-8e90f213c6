"""Frame input, review persistence and action isolation; fixtures are not accuracy evidence."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request,urlopen
import uuid

import cv2
import numpy as np

from replay.geometry import encode_mask
from replay.operator.frame_input import IncomingFrame
from replay.operator.station import InspectionStation,StationServer,SIM_RECIPE
from replay.operator.store import Conflict


def fixture(sequence=0,segment='window-1',kind='frames',lab=None):
    rgb=np.full((480,640,3),100,np.uint8);png=cv2.imencode('.png',cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))[1].tobytes()
    source=dict(schema_version=1,source_id='test-source',stream_id='stream-1',kind=kind,sequence=sequence,
        pts=sequence,time_base='1/5',width=640,height=480,rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
        segment_id=segment,telemetry=dict(phase_key='1',metered_ml=[1.,0.,0.],dispense_complete=False,recipe_key=SIM_RECIPE) if kind=='isaac' else None)
    mask=np.zeros((480,640),np.uint8);mask[300:310,300:310]=1
    row=dict(source_rgb_sha256=source['rgb_sha256'],inference_ms=1.,predictions=[],fill_detection=dict(phase='waiting',detection_id=None),
        measurement=dict(measurement_valid=True,quality_flags=[],lab=lab or [20.,0.,0.],rgb=[100.,100.,100.],
            geometry=dict(mask=encode_mask(mask),mask_pixels=100,track_id=1)))
    return IncomingFrame(source,rgb),row,dict(source=png,overlay=png,mask=png),png


class InspectionStationChecks(unittest.TestCase):
    def station(self,kind='frames'):
        temp=self.enterContext(tempfile.TemporaryDirectory())
        station=InspectionStation(Path(temp)/'station','test-source',kind);station.processor=object();return station

    def command(self,s,command,c=None,**kwargs):
        case=c or s.cases[s.current_id]
        return dict(command=command,actor='Test operator',request_id=uuid.uuid4().hex,
                    case_id=case['case_id'],expected_revision=case['revision'],**kwargs)

    def test_frame_contract_rejects_private_truth_wrong_pixels_and_unqualified_progress(self):
        frame,row,images,png=fixture()
        decoded=IncomingFrame.decode(frame.metadata,png,'test-source','frames')
        self.assertTrue(np.array_equal(decoded.rgb,frame.rgb))
        for update in [dict(rgb_sha256='0'*64),dict(width=641),dict(sequence=True),dict(actual_components_ml=[1,2,3]),
                       dict(telemetry={'phase_key':'1'}),dict(source_id='different')]:
            with self.assertRaises((ValueError,TypeError)):
                IncomingFrame.decode(frame.metadata|update,png,'test-source','frames')

    def test_unknown_context_and_missing_progress_cannot_become_recipe_decisions(self):
        s=self.station();f,r,i,_=fixture();s.accept(f,r,i)
        case=s.cases[s.current_id]
        self.assertFalse(case['identity_confirmed']);self.assertEqual(case['recipe_id'],'')
        self.assertEqual(s.last['diagnostic']['status'],'unavailable')
        s.command(self.command(s,'complete',confirmed=True))
        self.assertFalse(case['endpoint']['measurement_valid'])
        self.assertEqual(case['endpoint']['recipe_acceptance'],'unavailable')
        self.assertEqual(s.snapshot()['operational_recipe_assessment'],'unavailable')

    def test_capture_persists_while_feed_advances_and_response_never_controls_source(self):
        s=self.station();f,r,i,_=fixture();s.accept(f,r,i)
        s.command(self.command(s,'request_review',note='Inspect this source frame'))
        old=s.cases[s.current_id];captured=deepcopy(old['evidence']);evidence=s.out/captured['paths']['overlay']
        for seq in range(1,104):
            f,r,i,_=fixture(seq,'window-'+str(seq));s.accept(f,r,i)
        self.assertNotEqual(s.current_id,old['case_id']);self.assertNotIn(captured['image_key'],s.images)
        self.assertTrue(evidence.exists());self.assertEqual(old['evidence'],captured)
        self.assertIn(old['case_id'],[c['case_id'] for c in s.snapshot()['cases']])
        s.command(self.command(s,'acknowledge',old))
        with self.assertRaises(Conflict):s.command(self.command(s,'close',old,note='Too early'))
        s.command(self.command(s,'action',old,action='reviewed_evidence',confirmed=True,note='Inspected the retained frame'))
        s.command(self.command(s,'close',old,note='Human response documented'))
        self.assertEqual(old['review'],'closed');self.assertFalse(old['resolution']['product_release_authorized'])
        self.assertEqual(s.counts['frames'],104);self.assertFalse(s.source_paused)
        with self.assertRaises(ValueError):s.command(self.command(s,'flush',old))
        self.assertEqual(s.snapshot()['hardware_commands_sent'],0)

    def test_sample_gaps_break_consecutive_warning_and_alert_uses_actual_samples(self):
        s=self.station('isaac');s.reference=dict(recipe_key=SIM_RECIPE,knots={('1',1.):[10.,0.,0.]},approved=False)
        for seq in (0,1,3,4):
            f,r,i,_=fixture(seq,kind='isaac',lab=[90.,0.,0.]);s.accept(f,r,i)
        case=s.cases[s.current_id];self.assertEqual(case['review'],'none')
        f,r,i,_=fixture(5,kind='isaac',lab=[90.,0.,0.]);s.accept(f,r,i)
        self.assertEqual(case['review'],'requested');self.assertTrue(case['diagnostic_warning'])
        self.assertEqual(s.counts['sequence_gaps'],1)
        self.assertEqual(s.events[-1]['payload']['origin'],'actual_model_color_diagnostic')

    def test_manual_review_keeps_clicked_frame_after_source_changes_and_buffer_expires(self):
        s=self.station();f,r,i,_=fixture();s.accept(f,r,i);old=s.cases[s.current_id]
        body=self.command(s,'capture',image_key=s.last['image_key']);result=s.command(body)
        self.assertEqual(s.command(body)['capture_id'],result['capture_id'])
        for seq in range(1,52):
            f,r,i,_=fixture(seq,'new-source-window');s.accept(f,r,i)
        self.assertNotEqual(s.current_id,old['case_id']);self.assertEqual(len(s.images),48)
        s.command(self.command(s,'request_review',old,capture_id=result['capture_id'],note='Review the original clicked frame'))
        self.assertEqual(old['evidence']['source']['sequence'],0)
        self.assertEqual(s.last['source']['sequence'],51)
        self.assertEqual(s.events[-1]['source']['sequence'],0)

    def test_above_band_pending_is_distinct_from_good_and_failed_QA_keeps_confirmed_alert(self):
        s=self.station('isaac');s.reference=dict(recipe_key=SIM_RECIPE,knots={('1',1.):[10.,0.,0.]},approved=False)
        f,r,i,_=fixture(0,kind='isaac',lab=[90.,0.,0.]);s.accept(f,r,i)
        self.assertEqual(s.last['diagnostic']['status'],'outside_development_band_pending')
        self.assertFalse(s.last['diagnostic']['warning']);self.assertEqual(s.cases[s.current_id]['review'],'none')
        f,r,i,_=fixture(2,kind='isaac',lab=[90.,0.,0.]);s.accept(f,r,i)
        self.assertEqual(s.last['diagnostic']['consecutive'],1) # Skipping a sample still breaks confirmation.
        for seq in (3,4):
            f,r,i,_=fixture(seq,kind='isaac',lab=[90.,0.,0.]);s.accept(f,r,i)
        case=s.cases[s.current_id];self.assertEqual(case['review'],'requested');captured=deepcopy(case['evidence'])
        f,r,i,_=fixture(5,kind='isaac');r['measurement']['measurement_valid']=False
        r['measurement']['quality_flags']=['insufficient_final_liquid_pixels'];s.accept(f,r,i)
        self.assertTrue(s.last['diagnostic']['warning']);self.assertIsNone(s.last['diagnostic']['delta_e2000'])
        self.assertFalse(s.last['measurement']['measurement_valid']);self.assertEqual(case['evidence'],captured)

    def test_dashboard_polling_omits_historical_masks_but_preserves_full_evidence(self):
        s=self.station();f,r,i,_=fixture();s.accept(f,r,i)
        s.command(self.command(s,'request_review',note='Keep the full evidence'))
        full=s.snapshot();dashboard=s.snapshot(compact=True)
        self.assertNotIn('mask',dashboard['latest']['measurement']['geometry'])
        self.assertNotIn('last_measurement',dashboard['cases'][0])
        self.assertNotIn('mask',dashboard['cases'][0]['evidence']['measurement']['geometry'])
        self.assertEqual(full['latest']['measurement']['geometry']['mask'],r['measurement']['geometry']['mask'])
        self.assertEqual(full['cases'][0]['evidence']['measurement'],r['measurement'])
        self.assertEqual(s.images[dashboard['latest']['image_key']],i)
        self.assertEqual(full['cases'][0]['review_id'],dashboard['cases'][0]['review_id'])

    def test_duplicate_input_stale_commands_and_request_retries_have_no_extra_effect(self):
        s=self.station();f,r,i,_=fixture();s.accept(f,r,i)
        with self.assertRaises(Conflict):s.accept(f,r,i)
        body=self.command(s,'request_review',note='Manual inspection');s.command(body)
        self.assertTrue(s.command(body)['duplicate'])
        with self.assertRaises(Conflict):s.command(body|{'actor':'Different'})
        with self.assertRaises(Conflict):s.command(self.command(s,'acknowledge')|{'expected_revision':1})
        s.last_received-=10;self.assertEqual(s.snapshot()['source']['status'],'stale')
        self.assertFalse(s.fresh())
        with self.assertRaises(Conflict):s.command(self.command(s,'complete',confirmed=True))
        s.command(self.command(s,'acknowledge')) # Historical evidence remains reviewable.
        self.assertEqual(s.cases[s.current_id]['review'],'acknowledged')

    def test_http_frame_producer_and_operator_credentials_are_separate(self):
        s=self.station();frame,row,images,png=fixture()
        class Processor:
            def process(self,station,incoming):return station.accept(incoming,row,images)
        s.processor=Processor();server=StationServer(('127.0.0.1',0),s)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        self.addCleanup(server.server_close);self.addCleanup(server.shutdown)
        base='http://127.0.0.1:'+str(server.server_port)
        def publish(token):
            return urlopen(Request(base+'/api/frames',data=png,headers={'Content-Type':'image/png','X-Frame-Token':token,'X-Frame-Metadata':json.dumps(frame.metadata)}))
        with self.assertRaises(HTTPError) as error:publish(server.csrf)
        self.assertEqual(error.exception.code,403)
        with publish(s.frame_token) as response:self.assertEqual(json.load(response)['accepted_sequence'],0)
        with self.assertRaises(HTTPError) as error:publish(s.frame_token)
        self.assertEqual(error.exception.code,409)
        body=self.command(s,'request_review',note='Operator token required')
        request=Request(base+'/api/command',data=json.dumps(body).encode(),headers={'Content-Type':'application/json','X-Operator-CSRF':s.frame_token})
        with self.assertRaises(HTTPError) as error:urlopen(request)
        self.assertEqual(error.exception.code,403)
        with urlopen(base+'/api/session') as response:self.assertNotIn(s.frame_token,response.read().decode())


if __name__=='__main__':unittest.main()
