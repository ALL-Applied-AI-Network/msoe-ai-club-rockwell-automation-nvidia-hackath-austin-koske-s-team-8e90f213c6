"""Real-time cached Isaac input with live CV and bounded latest-sample delivery."""
import argparse
from bisect import bisect_right
import json
from pathlib import Path
import sys
import time
from urllib.error import HTTPError
from urllib.request import Request,urlopen
import uuid

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from replay.operator.frame_input import FramePublisher


def main(a):
    manifest=json.loads((a.cache/'manifest.json').read_text());samples=manifest['samples']
    publisher=FramePublisher(a.url,a.token_file,generation=a.generation)
    token=json.loads(a.token_file.read_text())['token']
    a.out.mkdir(parents=True,exist_ok=False)
    def control():
        with urlopen(Request(a.url+'/api/source',headers={'X-Frame-Token':token,'X-Source-Generation':str(a.generation)}),timeout=10) as response:return json.load(response)
    with (a.out/'delivery.jsonl').open('w') as log:
        while True:
            status=control()
            if status['stop']:return
            if status['paused']:time.sleep(.05);continue
            stream='prerender-'+uuid.uuid4().hex
            request=Request(a.url+'/api/playback',data=json.dumps(dict(stream_id=stream)).encode(),
                headers={'Content-Type':'application/json','X-Frame-Token':token,'X-Source-Generation':str(a.generation)})
            with urlopen(request,timeout=10):pass
            last=-1
            while True:
                status=control()
                if status['stop']:return
                if status['paused']:time.sleep(.04);continue
                elapsed=status['playback']['seconds']
                index=max(0,bisect_right([s['seconds'] for s in samples],elapsed)-1)
                if index<=last:
                    if elapsed>=manifest['duration_s']:break
                    time.sleep(.025);continue
                sample=samples[index]
                metadata=dict(schema_version=1,source_id=a.source_id,stream_id=stream,kind='isaac',sequence=index,
                    pts=sample['pts'],time_base=sample['time_base'],width=640,height=480,rgb_sha256=sample['rgb_sha256'],
                    segment_id=sample['segment_id'],telemetry=sample['telemetry'])
                started=time.perf_counter()
                try:reply=publisher.publish(metadata,(a.cache/sample['png']).read_bytes())
                except HTTPError as exc:
                    if exc.code==409 and control()['stop']:return
                    if exc.code==429:time.sleep(.025);continue
                    raise
                if reply['accepted_sequence']!=index or reply['rgb_sha256']!=sample['rgb_sha256']:raise ValueError('Video/CV acknowledgement differs')
                log.write(json.dumps(dict(stream_id=stream,sample_index=index,video_frame=sample['video_frame'],
                    pts=sample['pts'],time_base=sample['time_base'],rgb_sha256=sample['rgb_sha256'],
                    skipped_samples=index-last-1,publish_wall_s=time.perf_counter()-started,ack=reply))+'\n');log.flush()
                last=index


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url',required=True);p.add_argument('--token-file',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--source-id',required=True);p.add_argument('--generation',type=int,required=True)
    main(p.parse_args())
