"""Time a paced cached source through an SSH-forwarded worker to a browser image.

Original source clock and RGB hashes are preserved. No device or line I/O.
Uses the existing optional Playwright verification environment.
"""
import argparse
import asyncio
import bisect
import hashlib
import json
from pathlib import Path
import statistics
import time
from urllib.request import Request,urlopen
import uuid

from playwright.async_api import async_playwright


def stats(values):
    values=sorted(values)
    def percentile(f):
        at=(len(values)-1)*f;low=int(at);high=min(low+1,len(values)-1)
        return values[low]+(values[high]-values[low])*(at-low)
    return dict(count=len(values),p50=percentile(.5),p95=percentile(.95),maximum=values[-1])


async def main(a):
    a.out.mkdir(parents=True,exist_ok=False)
    endpoint=json.loads(a.endpoint.read_text());manifest=json.loads((a.cache/'manifest.json').read_text())
    token=json.loads(Path(endpoint['token_file']).read_text())['token']
    samples=manifest['samples'];times=[s['seconds'] for s in samples]
    pngs=[(a.cache/s['png']).read_bytes() for s in samples]
    stream='browser-link-'+uuid.uuid4().hex;anchor={};delivered=[];shown=[];errors=[]
    def publish():
        clock=time.monotonic();anchor['utc_ms']=time.time()*1000;last=-1
        while time.monotonic()-clock<manifest['duration_s']:
            index=bisect.bisect_right(times,time.monotonic()-clock)-1
            if index<=last:time.sleep(.005);continue
            s=samples[index]
            metadata=dict(schema_version=1,source_id='benchmark-fillcam',stream_id=stream,kind='isaac',sequence=index,
                pts=s['pts'],time_base=s['time_base'],width=640,height=480,rgb_sha256=s['rgb_sha256'],segment_id=s['segment_id'],telemetry=s['telemetry'])
            request=Request(endpoint['url']+'/api/frames',data=pngs[index],headers={'Content-Type':'image/png',
                'X-Frame-Metadata':json.dumps(metadata,separators=(',',':')),'X-Frame-Token':token,'X-Source-Generation':'0'})
            started=time.perf_counter()
            with urlopen(request,timeout=30) as response:ack=json.load(response)
            assert ack['accepted_sequence']==index and ack['rgb_sha256']==s['rgb_sha256']
            delivered.append(dict(index=index,rgb_sha256=s['rgb_sha256'],post_to_ack_ms=(time.perf_counter()-started)*1000,
                source_to_ack_ms=time.time()*1000-anchor['utc_ms']-s['seconds']*1000,skipped_samples=index-last-1))
            last=index
    async with async_playwright() as p:
        browser=await p.chromium.launch(channel='chrome',headless=True,args=['--disable-gpu','--disable-background-networking'])
        page=await browser.new_page(viewport=dict(width=1440,height=1080))
        page.on('pageerror',lambda error:errors.append(str(error)))
        try:
            await page.goto(endpoint['url'],wait_until='domcontentloaded')
            await page.wait_for_function('()=>state!==null')
            task=asyncio.create_task(asyncio.to_thread(publish));seen=set();screenshot=False
            while not task.done():
                item=await page.evaluate('''()=>state?.latest?({source:state.latest.source,key:state.latest.image_key,
                    displayed:document.querySelector('#live-image').dataset.key,
                    loaded:document.querySelector('#live-image').complete && document.querySelector('#live-image').naturalWidth===640,
                    utc_ms:Date.now(),mask_pixels:state.latest.measurement.geometry.mask_pixels,
                    qa_valid:state.latest.measurement.measurement_valid}):null''')
                if item and item['source']['stream_id']==stream and item['loaded'] and item['key']==item['displayed']:
                    index=item['source']['sequence']
                    if index not in seen:
                        assert item['source']['rgb_sha256']==samples[index]['rgb_sha256'];seen.add(index)
                        shown.append(dict(index=index,rgb_sha256=samples[index]['rgb_sha256'],
                            source_to_loaded_image_ms=item['utc_ms']-anchor['utc_ms']-samples[index]['seconds']*1000,
                            mask_pixels=item['mask_pixels'],qa_valid=item['qa_valid']))
                        if item['mask_pixels'] and not screenshot:
                            await page.screenshot(path=str(a.out/'remote-sampled-mask.png'),full_page=True);screenshot=True
                await page.wait_for_timeout(40)
            await task;assert shown and delivered and not errors
        finally:await browser.close()
    (a.out/'delivered.json').write_text(json.dumps(delivered,indent=2)+'\n')
    (a.out/'displayed.json').write_text(json.dumps(shown,indent=2)+'\n')
    result=dict(status='passed',input_samples=len(samples),delivered_samples=len(delivered),displayed_unique_samples=len(shown),
        skipped_input_samples=sum(r['skipped_samples'] for r in delivered),
        post_to_ack_ms=stats([r['post_to_ack_ms'] for r in delivered]),
        source_to_ack_ms=stats([r['source_to_ack_ms'] for r in delivered]),
        source_to_loaded_browser_image_ms=stats([r['source_to_loaded_image_ms'] for r in shown]),
        stream_id=stream,source_video_sha256=manifest['video_sha256'],hardware_commands=0,browser_errors=errors,
        limits=['Cached input scheduled at native source times; actual inference runs remotely on GB10.',
            'PNG input upload, server processing, VPN/SSH, dashboard polling and decoded overlay image load are included.',
            'Source schedule and browser clocks are on the same Windows machine; no remote clock synchronization is assumed.',
            'Image loaded in a headless browser is measured, not camera exposure-to-physical-monitor latency.',
            'This is the sampled-CV view, not a 30 fps encoded video streaming benchmark.'])
    result['artifacts_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in a.out.iterdir() if p.is_file()}
    (a.out/'completed.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('endpoint','cache','out'):parser.add_argument('--'+name,type=Path,required=True)
    asyncio.run(main(parser.parse_args()))
