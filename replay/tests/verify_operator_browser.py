"""Headless real-recording operator walkthrough; private browser-test environment.

Requires Playwright and an existing Chrome installation. Starts and stops only its
own loopback server. This creates demonstration journal entries, never QA labels.
"""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from playwright.async_api import async_playwright


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def main(a):
    if a.out.exists() or a.workspace.exists():
        raise ValueError('Use fresh browser evidence and demonstration workspace paths')
    a.out.mkdir(parents=True)
    shutil.copy2(__file__,a.out/'executed_browser_check.py')
    command=[str(a.python),'-m','replay.operator','--port','0','--out',str(a.workspace),
             '--evidence',str(a.evidence),'--video',str(a.video)]
    kwargs=dict(stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE)
    if os.name=='nt':
        kwargs['creationflags']=subprocess.CREATE_NEW_PROCESS_GROUP
    process=await asyncio.create_subprocess_exec(*command,**kwargs)
    errors=[]; network=[]; screenshots=[]; checks={}
    try:
        first=(await asyncio.wait_for(process.stdout.readline(),30)).decode().strip()
        match=re.search(r'http://127\.0\.0\.1:\d+',first)
        if not match:
            raise ValueError('Operator server did not start: '+first)
        base=match[0]
        async with async_playwright() as p:
            browser=await p.chromium.launch(channel='chrome',headless=True,args=['--disable-gpu','--disable-background-networking'])
            try:
                context=await browser.new_context(viewport=dict(width=1440,height=1100),device_scale_factor=1)
                page=await context.new_page()
                page.on('pageerror',lambda e:errors.append(str(e)))
                page.on('request',lambda r:network.append(r.url))
                page.on('console',lambda message:errors.append(message.text) if message.type=='error' else None)
                await page.goto(base,wait_until='networkidle')
                await page.wait_for_function('() => snapshot && snapshot.sample.source_frame === 0')
                await page.wait_for_function('() => video.readyState >= 2 && video.videoWidth === 640')
                await page.locator('#operator-name').fill('UI verification')
                async def seek(seconds):
                    await page.evaluate('(t)=>{video.pause();video.currentTime=t;}',seconds)
                    await page.wait_for_function('(t)=>snapshot && Math.abs(snapshot.sample.seconds-t)<.01',arg=seconds)
                    await page.wait_for_function('() => !video.seeking && video.readyState >= 2')
                    await page.wait_for_timeout(400)
                async def screenshot(name):
                    await page.locator('h1').click()
                    await page.screenshot(path=str(a.out/name),full_page=True)
                    screenshots.append(name)
                async def start(physical):
                    await page.locator('#new-vial').click()
                    await page.locator('#start-dialog').wait_for(state='visible')
                    await page.locator('#start-actor').fill('UI verification')
                    await page.locator('#start-recipe').fill('DEMO-A')
                    await page.locator('#start-batch').fill('BROWSER-PRACTICE')
                    await page.locator('#start-physical').fill(physical)
                    await page.locator('#start-identity').check()
                    await page.locator('#start-note').fill('Illustrative operator context for browser verification, not physical recipe truth.')
                    await page.get_by_role('button',name='Save vial context',exact=True).click()
                    await page.locator('#start-dialog').wait_for(state='hidden')
                    await page.wait_for_function('(v)=>snapshot.current?.physical_vial_id===v',arg=physical)
                async def complete():
                    await page.locator('#complete-vial').click()
                    await page.locator('#command-confirmed').check()
                    await page.get_by_role('button',name='Confirm & capture endpoint',exact=True).click()
                    await page.locator('#command-dialog').wait_for(state='hidden')
                    await page.wait_for_function('() => snapshot.current?.lifecycle==="complete"')
                async def note_command(button,note):
                    await page.locator(button).click()
                    await page.locator('#command-note').fill(note)
                    await page.locator('#command-save').click()
                    await page.locator('#command-dialog').wait_for(state='hidden')
                await seek(20)
                await screenshot('operator-workspace-overview.png')
                checks['actual_h264_video_decoded']=await page.evaluate('video.videoWidth===640 && video.videoHeight===480 && video.readyState>=2')
                await seek(11.2);await start('DEMO-001');await seek(26);await complete()
                assert await page.evaluate('snapshot.current.endpoint.measurement_valid')
                await page.locator('#record-tools > summary').click()
                await note_command('#save-reference','Development reference for the UI walkthrough only; not CSI-approved.')
                await page.wait_for_function('() => snapshot.references.length===1')
                await seek(56.8);await start('DEMO-002');await seek(70);await complete()
                assert await page.evaluate('snapshot.current.assessment.status==="unavailable"')
                checks['operational_assessment_not_promoted']=True
                await note_command('#request-review','Practice review: independent recipe acceptance is not verified. Operator should inspect the evidence.')
                await page.wait_for_function('() => snapshot.current.review_state==="requested"')
                await page.locator('#sample-details').evaluate('(element)=>{element.open=true;}')
                await page.wait_for_function('() => document.querySelector("#evidence-image").naturalWidth===640 && !document.querySelector("#evidence-image").hidden')
                checks['exact_source_sample_image_displayed']=await page.evaluate('evidenceKey.startsWith(snapshot.sample.source_frame+"/")')
                await screenshot('operator-review-requested.png')
                await page.locator('#ack-review').click()
                await page.locator('#command-save').click()
                await page.locator('#command-dialog').wait_for(state='hidden')
                await page.wait_for_function('() => snapshot.current.review_state==="acknowledged"')
                checks['acknowledgement_does_not_close_review']=True
                await page.locator('#record-action').click()
                await page.locator('#action-type').select_option('reviewed_evidence')
                await page.locator('#command-note').fill('Reviewed the displayed source/mask evidence in a browser test. No physical intervention or station command.')
                await page.locator('#command-confirmed').check()
                await page.locator('#command-save').click()
                await page.locator('#command-dialog').wait_for(state='hidden')
                await page.wait_for_function('() => snapshot.current.actions.length===1')
                await screenshot('operator-response-recorded.png')
                await note_command('#close-review','Browser follow-up documented. Product outcome remains independently unverified.')
                await page.wait_for_function('() => snapshot.current.review_state==="closed"')
                async with page.expect_download() as item:
                    await page.locator('#export-header').click()
                download=await item.value;await download.save_as(str(a.out/'operator-session.json'))
                report=json.loads((a.out/'operator-session.json').read_text())
                assert len(report['cases'])==2 and report['hardware_commands_sent']==0
                assert report['cases'][-1]['review_state']=='closed'
                assert report['cases'][-1]['assessment']['status']=='unavailable'
                assert report['cases'][-1]['resolution']['product_release_authorized'] is False
                checks['browser_download_preserves_manual_context_history_and_limits']=True
                async with page.expect_download() as item:
                    await page.locator('#export-csv').click()
                download=await item.value;await download.save_as(str(a.out/'operator-vials.csv'))
                # A delayed earlier request cannot restore a future snapshot after a seek.
                async def delayed(route):
                    if 'at_s=80' in route.request.url:
                        await asyncio.sleep(.7)
                    try:await route.continue_()
                    except Exception:pass # Aborted by the browser generation change.
                await page.route('**/api/state*',delayed)
                await page.evaluate('video.currentTime=80')
                await page.wait_for_timeout(100)
                await seek(1)
                await page.wait_for_timeout(1000)
                assert await page.evaluate('snapshot.sample.seconds===1 && snapshot.cases.length===0 && document.querySelector("#case-context").hidden')
                checks['backward_seek_and_delayed_response_hide_future_context']=True
                await page.unroute('**/api/state*',delayed)
                await page.reload(wait_until='networkidle');await seek(70)
                assert await page.evaluate('snapshot.cases.length===2 && snapshot.current.review_state==="closed" && snapshot.references.length===1')
                checks['browser_reload_restores_journal']=True
                await page.set_viewport_size(dict(width=390,height=844))
                await page.wait_for_timeout(300)
                assert await page.evaluate('document.documentElement.scrollWidth<=window.innerWidth')
                await screenshot('operator-mobile.png')
                checks['mobile_no_page_overflow']=True
                checks['all_page_requests_loopback_only']=all(url.startswith(base+'/') for url in network)
                assert checks['all_page_requests_loopback_only']
                assert not errors,errors
            finally:
                await browser.close()
    finally:
        if process.returncode is None:
            if os.name=='nt':
                killer=await asyncio.create_subprocess_exec('taskkill','/PID',str(process.pid),'/T','/F',stdout=asyncio.subprocess.DEVNULL,stderr=asyncio.subprocess.DEVNULL)
                await asyncio.wait_for(killer.wait(),10)
            else:process.terminate()
        stdout,stderr=await asyncio.wait_for(process.communicate(),10)
        (a.out/'server-stdout.txt').write_text(stdout.decode(errors='replace'))
        (a.out/'server-stderr.txt').write_text(stderr.decode(errors='replace'))
    result=dict(status='passed',source_mode='recorded_demonstration',checks=checks,
        browser_errors=errors,independently_verified_outcomes=False,hardware_commands_sent=0,
        model_inference_rerun=False,workspace=str(a.workspace),
        artifacts={p.name:sha(p) for p in a.out.iterdir() if p.is_file()},screenshots=screenshots)
    (a.out/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('python','evidence','video','workspace','out'):p.add_argument('--'+name,type=Path,required=True)
    asyncio.run(main(p.parse_args()))
