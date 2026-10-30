"""Measure real-time video plus live CV; exercise both source modes in the UI."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from playwright.async_api import async_playwright


async def main(a):
    a.out.mkdir(parents=True,exist_ok=False);errors=[];checks={}
    async with async_playwright() as p:
        browser=await p.chromium.launch(channel='chrome',headless=True,args=['--disable-gpu','--disable-background-networking'])
        page=await browser.new_page(viewport=dict(width=1440,height=1080))
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.on('console',lambda e:errors.append(e.text) if e.type=='error' else None)
        async def wait(expression):
            await page.wait_for_function('()=>'+expression,timeout=180000)
        async def shot(name):
            await page.locator('h1').click();await page.screenshot(path=str(a.out/name),full_page=True)
        async def mode(value):
            await page.locator('#source-mode').select_option(value)
            await wait(f'state.source.delivery_mode==="{value}" && state.source.status==="live"')
        try:
            await page.goto(a.url,wait_until='domcontentloaded')
            await wait('state && state.source.status==="live" && state.playback')
            await page.locator('#actor').fill('Pre-render verification')
            await wait('document.querySelector("#source-video").readyState>=2 && !document.querySelector("#source-video").paused')
            await wait('state.playback.seconds>1 && state.playback.seconds<state.playback.duration_s-6')
            before=await page.evaluate('({wall:performance.now()/1000,time:document.querySelector("#source-video").currentTime,frames:document.querySelector("#source-video").getVideoPlaybackQuality().totalVideoFrames,cv:state.counts.frames})')
            await page.wait_for_timeout(3000)
            after=await page.evaluate('({wall:performance.now()/1000,time:document.querySelector("#source-video").currentTime,frames:document.querySelector("#source-video").getVideoPlaybackQuality().totalVideoFrames,cv:state.counts.frames})')
            elapsed=after['wall']-before['wall'];advance=after['time']-before['time'];ratio=advance/elapsed
            assert .88<ratio<1.12,(elapsed,advance,ratio)
            assert after['frames']-before['frames']>=70
            assert after['cv']>before['cv']
            checks['video_plays_at_one_times_wall_clock']=True;checks['live_CV_progresses_independently']=True
            if not await page.locator('#cv-preview').evaluate('(element)=>element.open'):
                await page.locator('#cv-preview summary').click()
            await wait('document.querySelector("#cv-image").naturalWidth===640')
            assert await page.locator('#live-image').is_hidden()
            checks['source_video_and_timestamped_CV_mask_are_separate']=True
            await shot('01-prerender-real-time.png')
            range_response=await page.request.get(a.url+'/api/video.mp4',headers={'Range':'bytes=0-511'})
            assert range_response.status==206 and len(await range_response.body())==512
            checks['browser_video_range_requests']=True
            old_cases=await page.evaluate('state.cases.map(c=>c.case_id)')
            await page.locator('#source-settings summary').click();await page.locator('#pause-source').click()
            await wait('state.source.paused && document.querySelector("#source-video").paused')
            await page.wait_for_timeout(700);paused=await page.locator('#source-video').evaluate('(v)=>v.currentTime')
            await page.wait_for_timeout(1100);assert abs(await page.locator('#source-video').evaluate('(v)=>v.currentTime')-paused)<.05
            checks['pause_freezes_video_clock']=True
            await mode('isaac')
            assert await page.locator('#source-video').is_hidden()
            assert await page.evaluate('state.latest.source.stream_id.startsWith("isaac-")')
            await shot('02-live-isaac-mode.png');checks['toggle_runs_native_Isaac_and_actual_inference']=True
            await mode('prerender')
            await wait('document.querySelector("#source-video").readyState>=2 && !document.querySelector("#source-video").paused')
            assert await page.evaluate('(ids)=>ids.every(id=>state.cases.some(c=>c.case_id===id))',old_cases)
            checks['toggle_preserves_cases_and_operator_journal']=True
            await shot('03-return-to-real-time.png')
            await page.reload(wait_until='domcontentloaded');await wait('state && state.source.status==="live" && state.source.delivery_mode==="prerender"')
            await wait('document.querySelector("#source-video").readyState>=2 && !document.querySelector("#source-video").paused')
            checks['reload_resumes_current_playback_clock']=True
            await page.set_viewport_size(dict(width=390,height=844));await page.wait_for_timeout(500)
            assert await page.evaluate('document.documentElement.scrollWidth<=window.innerWidth')
            await shot('04-mobile-prerender.png');checks['mobile_no_page_overflow']=True
            response=await page.request.get(a.url+'/api/export.json');export=await response.json()
            assert export['state']['hardware_commands_sent']==0 and export['state']['operational_recipe_assessment']=='unavailable'
            assert not errors,errors
            (a.out/'session-export.json').write_text(json.dumps(export,indent=2))
            result=dict(status='passed',checks=checks,video_playback=dict(wall_seconds=elapsed,video_seconds=advance,speed_ratio=ratio,
                presented_frames=after['frames']-before['frames'],CV_samples=after['cv']-before['cv']),browser_errors=errors,
                hardware_commands_sent=0,operational_recipe_assessment='unavailable')
        except BaseException as exc:
            await page.screenshot(path=str(a.out/'failed.png'),full_page=True)
            (a.out/'failed.json').write_text(json.dumps(dict(error=str(exc),browser_errors=errors),indent=2));raise
        finally:await browser.close()
    result['artifacts']={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in a.out.iterdir() if f.is_file()}
    (a.out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--url',default='http://127.0.0.1:8767');p.add_argument('--out',type=Path,required=True)
    asyncio.run(main(p.parse_args()))
