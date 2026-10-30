"""Observe real CV samples: visible masks, pending deviations, QA and playback."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import time

from playwright.async_api import async_playwright


async def main(a):
    a.out.mkdir(parents=True,exist_ok=False);errors=[];trace=[];shots=set()
    async with async_playwright() as p:
        browser=await p.chromium.launch(channel='chrome',headless=True,args=['--disable-gpu','--disable-background-networking'])
        page=await browser.new_page(viewport=dict(width=1500,height=1080))
        page.on('pageerror',lambda error:errors.append(str(error)))
        try:
            await page.goto(a.url,wait_until='domcontentloaded')
            await page.wait_for_function('()=>state?.source.status==="live" && state.playback',timeout=60000)
            assert await page.locator('#cv-preview').evaluate('(element)=>element.open')
            assert await page.locator('#cv-preview').is_visible()
            await page.wait_for_function('()=>document.querySelector("#cv-image").naturalWidth===640')
            start=time.monotonic();last=None
            while time.monotonic()-start<a.seconds:
                row=await page.evaluate('''()=>({key:state.latest.image_key,source:state.latest.source,
                    valid:state.latest.measurement.measurement_valid,mask_pixels:state.latest.measurement.geometry.mask_pixels,
                    flags:state.latest.measurement.quality_flags,diagnostic:state.latest.diagnostic,
                    title:document.querySelector('#assessment-title').textContent,
                    reason:document.querySelector('#assessment-reason').textContent,
                    preview_visible:!document.querySelector('#cv-preview').hidden && document.querySelector('#cv-preview').open,
                    mask_status:document.querySelector('#cv-mask-status').textContent})''')
                if row['key']!=last:
                    last=row['key'];trace.append(row);d=row['diagnostic']
                    if d['warning']:
                        assert row['title']=='Appearance differs'
                        kind='confirmed-alert-with-failed-QA' if not row['valid'] else 'confirmed-alert'
                    elif row['valid'] and d['delta_e2000'] is not None and d['delta_e2000']>d['threshold_de2000']:
                        assert row['title']=='Outside development band';kind='pending-deviation'
                    elif not row['valid']:
                        assert row['title']=='Measurement unavailable';kind='unavailable'
                    else:kind='sampling-usable'
                    if kind not in shots and (kind!='sampling-usable' or row['mask_pixels']>0):
                        await page.screenshot(path=str(a.out/(kind+'.png')),full_page=True);shots.add(kind)
                await page.wait_for_timeout(80)
            assert any(r['valid'] and r['mask_pixels']>0 for r in trace)
            assert any(r['title']=='Outside development band' for r in trace),'No actual pending deviation observed'
            assert all(r['preview_visible'] for r in trace)
            await page.wait_for_function('()=>state.playback.seconds>1 && state.playback.seconds<state.playback.duration_s-6')
            before=await page.evaluate('({wall:performance.now()/1000,time:document.querySelector("#source-video").currentTime,frames:document.querySelector("#source-video").getVideoPlaybackQuality().totalVideoFrames,cv:state.counts.frames})')
            await page.wait_for_timeout(3000)
            after=await page.evaluate('({wall:performance.now()/1000,time:document.querySelector("#source-video").currentTime,frames:document.querySelector("#source-video").getVideoPlaybackQuality().totalVideoFrames,cv:state.counts.frames})')
            elapsed=after['wall']-before['wall'];ratio=(after['time']-before['time'])/elapsed
            assert .88<ratio<1.12 and after['frames']-before['frames']>=70
            response=await page.request.get(a.url+'/api/state');payload=await response.body()
            dashboard=json.loads(payload);assert 'mask' not in dashboard['latest']['measurement']['geometry']
            await page.set_viewport_size(dict(width=390,height=844));await page.wait_for_timeout(500)
            assert await page.evaluate('document.documentElement.scrollWidth<=innerWidth')
            await page.screenshot(path=str(a.out/'mobile-visible-mask.png'),full_page=True)
            export=await page.request.get(a.url+'/api/export.json');exported=await export.json()
            assert 'mask' in exported['state']['latest']['measurement']['geometry']
            assert not errors,errors
            result=dict(status='passed',observed_live_samples=len(trace),
                actual_pending_readings=sum(r['title']=='Outside development band' for r in trace),
                confirmed_alert_samples=sum(r['diagnostic']['warning'] for r in trace),
                confirmed_alerts_retained_on_failed_QA=sum(r['diagnostic']['warning'] and not r['valid'] for r in trace),
                samples_with_masks=sum(r['mask_pixels']>0 for r in trace),
                masks_visible_without_expanding=True,dashboard_poll_bytes=len(payload),
                full_mask_evidence_preserved_in_export=True,mobile_no_overflow=True,
                playback=dict(wall_seconds=elapsed,speed_ratio=ratio,presented_frames=after['frames']-before['frames'],CV_samples=after['cv']-before['cv']),
                browser_errors=errors,hardware_commands_sent=exported['state']['hardware_commands_sent'])
            (a.out/'observed-samples.json').write_text(json.dumps(trace,indent=2)+'\n')
        except BaseException as exc:
            await page.screenshot(path=str(a.out/'failed.png'),full_page=True)
            (a.out/'failed.json').write_text(json.dumps(dict(error=str(exc),browser_errors=errors),indent=2));raise
        finally:await browser.close()
    result['artifacts_sha256']={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in a.out.iterdir() if f.is_file()}
    (a.out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',default='http://127.0.0.1:8767');parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--seconds',type=float,default=45.)
    asyncio.run(main(parser.parse_args()))
