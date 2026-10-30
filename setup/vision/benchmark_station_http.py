"""Measure the existing per-request inference path and an isolated worker candidate.

This changes only the benchmark server; the deployed operator station is untouched.
"""
import argparse
import bisect
import json
from pathlib import Path
import queue
import sys
import threading
import time
from urllib.request import Request,urlopen
import uuid

from benchmark_station_compute import prepare,station,cache_frames,frame_for,read,write,summary


class PersistentProcessor:
    def __init__(self,processor):
        self.processor=processor;self.queue=queue.Queue(maxsize=1)
        self.thread=threading.Thread(target=self.work,daemon=True,name='benchmark-inference')
        self.thread.start()

    def work(self):
        while True:
            item=self.queue.get()
            if item is None:return
            args,generation,finished,result=item
            try:result['value']=self.processor.process(*args,generation=generation)
            except BaseException as error:result['error']=error
            finally:finished.set()

    def process(self,station,frame,generation=None):
        finished=threading.Event();result={}
        self.queue.put(((station,frame),generation,finished,result))
        finished.wait()
        if 'error' in result:raise result['error']
        return result['value']

    def close(self):self.queue.put(None);self.thread.join(timeout=10)


def serve(a):
    from replay.operator.station import StationServer
    instance,processor=station(a)
    if a.persistent_worker:instance.processor=PersistentProcessor(processor)
    server=StationServer(('127.0.0.1',0),instance)
    write(a.out/'endpoint.json',dict(url='http://127.0.0.1:'+str(server.server_port),
        token_file=str(instance.out/'frame-input-token.json'),persistent_worker=a.persistent_worker))
    print('Benchmark HTTP observer ready:',server.server_port,flush=True)
    try:server.serve_forever(poll_interval=.1)
    except KeyboardInterrupt:pass
    finally:
        server.server_close()
        if a.persistent_worker:instance.processor.close()


def client(a):
    endpoint=read(a.endpoint);token=read(a.token_file or Path(endpoint['token_file']))['token']
    url=a.url or endpoint['url'];bundle,samples=cache_frames(a)
    # Decode validation is outside timing; POST still carries the pinned PNG.
    pngs={s['png']:(bundle.root/s['png']).read_bytes() for s,_ in samples}
    measured=[];total=0
    with (a.out/'timings.jsonl').open('w') as log:
        for repetition in range(2 if a.paced else 3):
            warmup=repetition==0;stream='http-bench-'+uuid.uuid4().hex;index=0;last=-1;clock=time.monotonic()
            while index<len(samples):
                if a.paced and not warmup:
                    elapsed=time.monotonic()-clock
                    if elapsed>=bundle.duration:break
                    due=bisect.bisect_right([s['seconds'] for s,_ in samples],elapsed)-1
                    if due<=last:time.sleep(.005);continue
                    index=due
                sample,rgb=samples[index];metadata=frame_for(sample,rgb,stream,index).metadata
                headers={'Content-Type':'image/png','X-Frame-Metadata':json.dumps(metadata,separators=(',',':')),
                    'X-Frame-Token':token,'X-Source-Generation':'0'}
                if a.host_port:headers['Host']='127.0.0.1:'+str(a.host_port)
                request=Request(url+'/api/frames',data=pngs[sample['png']],headers=headers)
                started=time.perf_counter()
                with urlopen(request,timeout=180) as response:ack=json.load(response)
                wall_ms=(time.perf_counter()-started)*1000
                assert ack['accepted_sequence']==index and ack['rgb_sha256']==sample['rgb_sha256']
                record=dict(repetition=repetition,warmup=warmup,sample_index=index,stream_id=stream,
                    source_rgb_sha256=sample['rgb_sha256'],post_to_ack_ms=wall_ms,png_bytes=len(pngs[sample['png']]),
                    source_to_ack_ms=max(0.,(time.monotonic()-clock-sample['seconds'])*1000) if a.paced and not warmup else None,
                    skipped_samples=index-last-1 if a.paced and not warmup else 0)
                log.write(json.dumps(record)+'\n');log.flush()
                if not warmup:measured.append(record)
                total+=1;last=index;index+=1
                if total%25==0:print('HTTP_BENCH',total,'warmup',warmup,round(wall_ms,1),flush=True)
    result=dict(status='passed',mode='paced_http' if a.paced else 'offline_http',
        persistent_worker=endpoint['persistent_worker'],warmup_samples=len(samples),measured_samples=len(measured),
        input_samples=len(samples),post_to_ack_ms=summary([r['post_to_ack_ms'] for r in measured]),
        throughput_samples_s=len(measured)/(sum(r['post_to_ack_ms'] for r in measured)/1000),
        source_to_ack_ms=summary([r['source_to_ack_ms'] for r in measured if r['source_to_ack_ms'] is not None]),
        skipped_input_samples=sum(r['skipped_samples'] for r in measured),
        unprocessed_tail_samples=len(samples)-1-measured[-1]['sample_index'] if a.paced else 0,
        video_sha256=bundle.manifest['video_sha256'],hardware_commands=0,cached_predictions=False,
        limits=['POST timing includes request transport, PNG decode, actual model processing, overlays, journal writes and acknowledgement.',
            'No browser video/mask display is timed; any SSH/VPN transport is identified separately in the experiment report.',
            'A persistent inference thread is an experimental benchmark adapter; app defaults are unchanged.'])
    write(a.out/'completed.json',result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--mode',choices=('serve','client'),required=True)
    parser.add_argument('--persistent-worker',action='store_true');parser.add_argument('--endpoint',type=Path)
    parser.add_argument('--url');parser.add_argument('--token-file',type=Path);parser.add_argument('--host-port',type=int)
    parser.add_argument('--paced',action='store_true');args=parser.parse_args()
    prepare(args)
    if args.mode=='serve':serve(args)
    else:client(args)
