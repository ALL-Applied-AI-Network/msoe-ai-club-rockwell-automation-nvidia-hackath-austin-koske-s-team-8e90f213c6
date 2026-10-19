import hashlib
import json
from pathlib import Path
import threading
import urllib.request
import unittest

from annotation_review.batch import append_replay_qa
from annotation_review.server import create_server
from annotation_review.alignment import load_mappings
from test_review import ReviewFixture


class HandoffTests(ReviewFixture, unittest.TestCase):
    def test_http_decision_export_import(self):
        server=create_server(self.data,self.pre,self.root/'http',port=0,demo=True)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        base=f'http://127.0.0.1:{server.server_port}'
        def request(path,data=None):
            req=urllib.request.Request(base+path,json.dumps(data).encode() if data else None,headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(req) as response:return json.load(response)
        try:
            detail=request('/api/sample?id=a');self.assertEqual(detail['sample']['association']['status'],'unmatched')
            saved=request('/api/review',self.decision());self.assertFalse(saved['eligible_as_reviewed_label'])
            bundle=request('/api/export');request('/api/import',bundle)
            self.assertEqual(request('/api/sample?id=a')['current']['machine:0'],saved)
        finally:server.shutdown();server.server_close();thread.join()

    def test_proposed_mapping_is_not_reviewed(self):
        path=self.root/'mapping.csv'
        path.write_text('sample_id,still_sha256,status,video_sha256,frame_index,reviewer,evidence,notes\n'
            f'a,{self.catalog.samples["a"]["source"]["sha256"]},proposed,,123,,,unconfirmed\n')
        load_mappings(self.catalog,path)
        self.assertEqual(self.catalog.samples['a']['association']['status'],'proposed')

    def test_reviewed_mapping_requires_human_evidence(self):
        path=self.root/'mapping.csv'
        path.write_text('sample_id,still_sha256,status,video_sha256,frame_index,reviewer,evidence,notes\n'
            f'a,{self.catalog.samples["a"]["source"]["sha256"]},reviewed,,123,,,\n')
        with self.assertRaises(ValueError):load_mappings(self.catalog,path)

    def test_public_qa_adapter_preserves_clock_and_hash(self):
        video=self.data/'vial_fill/videos/vial_fill_video_5.mp4';video.parent.mkdir();video.write_bytes(b'source video identity')
        export=self.root/'qa';(export/'frames').mkdir(parents=True);image=export/'frames/evidence.jpg';image.write_bytes(b'exact exported image')
        doc=dict(schema_version=1,artifact_kind='measurement_qa_review',source=dict(camera='vial_fill',run_id=5,video_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),frame_count=100),
            config_id='config',implementation_id='implementation',cases=[dict(detection_id='candidate3',measurement_available=False,review_status='unreviewed',reviewed_label=None,
            rejection_reasons=['motion'],representative_frames=[dict(frame_index=2,mp4_s=.08,path='frames/evidence.jpg',role='window_sample',sha256=hashlib.sha256(image.read_bytes()).hexdigest(),replay_url='http://127.0.0.1:8765/?run=5&frame=2')])])
        path=export/'manifest.json';path.write_text(json.dumps(doc));before=path.read_bytes()
        batch=self.root/'batch';batch.mkdir();(batch/'cases.json').write_text('{"cases":[]}')
        append_replay_qa(path,self.data,batch)
        c=json.loads((batch/'cases.json').read_text())['cases'][0]
        self.assertEqual(c['source']['frame_index'],2);self.assertEqual(c['source']['export_reported_mp4_seconds'],.08)
        self.assertIsNone(c['source']['video_pts_seconds']);self.assertEqual(path.read_bytes(),before)
        self.assertEqual((batch/c['source']['image_relative_path']).read_bytes(),image.read_bytes())
        append_replay_qa(path,self.data,batch)
        self.assertEqual(len(json.loads((batch/'cases.json').read_text())['cases']),1)
        doc['config_id']='different';path.write_text(json.dumps(doc))
        with self.assertRaises(ValueError):append_replay_qa(path,self.data,batch)
        self.assertEqual((batch/'qa-source/manifest.json').read_bytes(),before)


if __name__=='__main__':unittest.main()
