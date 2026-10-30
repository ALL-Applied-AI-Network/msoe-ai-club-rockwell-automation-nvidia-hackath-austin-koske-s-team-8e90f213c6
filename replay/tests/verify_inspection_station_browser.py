"""Verify a running continuous feed while operators work on retained exceptions."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from playwright.async_api import async_playwright


async def main(a):
    a.out.mkdir(parents=True,exist_ok=False);errors=[];urls=[];checks={}
    async with async_playwright() as p:
        browser=await p.chromium.launch(channel='chrome',headless=True,args=['--disable-gpu','--disable-background-networking'])
        page=await browser.new_page(viewport=dict(width=1440,height=1080))
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.on('console',lambda e:errors.append(e.text) if e.type=='error' else None)
        page.on('request',lambda e:urls.append(e.url))
        async def wait(expression):await page.wait_for_function('()=>'+expression,timeout=180000)
        async def shot(name):
            await page.locator('h1').click();await page.wait_for_timeout(400)
            await page.screenshot(path=str(a.out/name),full_page=True)
        try:
            await page.goto(a.url,wait_until='domcontentloaded')
            await wait('state && state.source.status==="live" && state.counts.frames>=2')
            await wait('document.querySelector("#live-image").naturalWidth===640 && !document.querySelector("#live-image").hidden')
            await page.locator('#actor').fill('Station verification')
            assert await page.locator('h1').inner_text()=='Live inspection'
            assert not await page.get_by_role('button',name='Capture clean reference',exact=True).count()
            checks['frames_arrive_without_operator_start_or_scenario_controls']=True
            await shot('01-live-inspection.png')
            await wait('state.open_reviews>0')
            await page.locator('.queue-item').first.click()
            await wait('document.querySelector("#review-image").naturalWidth===640')
            cid=await page.evaluate('selected');image=await page.locator('#review-image').get_attribute('src')
            count=await page.evaluate('state.counts.frames')
            await page.wait_for_function('(n)=>state.counts.frames>=n+4',arg=count,timeout=30000)
            assert await page.locator('#review-image').get_attribute('src')==image
            checks['captured_review_stays_fixed_while_live_feed_advances']=True
            await shot('02-live-feed-and-retained-review.png')
            await page.locator('#acknowledge').click();await wait('reviewCase().review==="acknowledged"')
            assert not await page.locator('#close-review').is_visible()
            checks['acknowledgement_does_not_clear_review']=True
            count=await page.evaluate('state.counts.frames')
            await page.locator('#record-action').click();await page.locator('#action-type').select_option('reviewed_evidence')
            await page.locator('#action-note').fill('Reviewed the captured source-aligned evidence in this local verification. No physical intervention was performed.')
            await page.locator('#action-confirm').check();await page.locator('#save-action').click()
            await page.locator('#action-dialog').wait_for(state='hidden')
            await wait('reviewCase().actions.length===1')
            await page.locator('#close-review').click();await page.locator('#action-note').fill('Follow-up documented from the retained evidence. Product quality and release remain unverified.')
            await page.locator('#save-action').click();await page.locator('#action-dialog').wait_for(state='hidden')
            await wait('reviewCase().review==="closed"')
            await page.wait_for_function('(n)=>state.counts.frames>n',arg=count,timeout=15000)
            assert await page.evaluate('!state.source.paused && state.hardware_commands_sent===0')
            checks['operator_response_does_not_pause_or_command_frame_source']=True
            await shot('03-human-response-feed-continues.png')
            await page.locator('#context').click()
            context_id=await page.evaluate('dialogContext.case_id')
            await page.locator('#context-vial').fill('DEMO-CV-001')
            await page.locator('#context-note').fill('Manual practice association; physical identity deliberately unconfirmed.')
            await page.get_by_role('button',name='Save context',exact=True).click()
            await page.locator('#context-dialog').wait_for(state='hidden')
            await page.wait_for_function('(id)=>state.cases.find(c=>c.case_id===id)?.physical_vial_id==="DEMO-CV-001"',arg=context_id)
            assert await page.evaluate('(id)=>!state.cases.find(c=>c.case_id===id).identity_confirmed',context_id)
            checks['operator_context_is_explicit_and_separate_from_source_identity']=True
            await page.wait_for_function('(id)=>current() && current().case_id!==id && current().review==="none"',arg=cid,timeout=90000)
            await page.locator('#request-review').click();await page.locator('#action-dialog').wait_for(state='visible')
            capture_id=await page.evaluate('dialogContext.capture_id')
            capture_src=await page.locator('#manual-capture-image').get_attribute('src')
            before=await page.evaluate('state.counts.frames')
            await page.wait_for_function('(n)=>state.counts.frames>=n+3',arg=before,timeout=15000)
            assert await page.locator('#manual-capture-image').get_attribute('src')==capture_src
            await page.locator('#action-note').fill('Manual request for the frame captured when this form opened; the incoming feed continued while notes were entered.')
            await page.locator('#save-action').click();await page.locator('#action-dialog').wait_for(state='hidden')
            await wait('reviewCase()?.review==="requested"')
            captured=await page.evaluate('reviewCase().evidence')
            assert capture_id in captured['paths']['overlay']
            checks['manual_request_pins_the_clicked_frame_while_notes_are_entered']=True
            await page.locator('#source-settings summary').click();await page.locator('#pause-source').click()
            await wait('state.source.paused')
            await page.wait_for_timeout(1700);paused=await page.evaluate('state.counts.frames')
            await page.wait_for_timeout(1200);assert await page.evaluate('state.counts.frames')==paused
            assert await page.locator('#request-review').is_disabled()
            assert await page.locator('#complete').is_disabled()
            checks['paused_source_is_not_presented_as_fresh_evidence']=True
            await page.locator('#pause-source').click();await wait('state.source.status==="live"')
            await page.wait_for_function('(n)=>state.counts.frames>n',arg=paused,timeout=15000)
            await page.reload(wait_until='domcontentloaded');await wait('state && state.source.status==="live"')
            assert await page.evaluate('(id)=>state.cases.find(c=>c.case_id===id)?.review==="closed"',cid)
            checks['browser_reload_preserves_operator_history']=True
            await page.set_viewport_size(dict(width=390,height=844));await page.wait_for_timeout(400)
            assert await page.evaluate('document.documentElement.scrollWidth<=window.innerWidth')
            await shot('04-mobile-inspection.png');checks['mobile_no_page_overflow']=True
            # Freeze this test source after verification, preserving a finite capture.
            await page.locator('#source-settings summary').click();await page.locator('#pause-source').click();await wait('state.source.paused')
            await page.wait_for_timeout(1500)
            response=await page.request.get(a.url+'/api/export.json');exported=await response.json()
            assert exported['cases'] and not exported['product_release_authorized']
            assert not exported['state']['hardware_commands_sent']
            assert all(url.startswith(a.url+'/') for url in urls)
            checks['all_browser_requests_loopback']=True
            assert not errors,errors
            (a.out/'session-export.json').write_text(json.dumps(exported,indent=2))
            result=dict(status='passed',mode='continuous_inspection',checks=checks,browser_errors=errors,
                captured_frames=exported['state']['counts']['frames'],reviewed_case_id=cid,hardware_commands_sent=0,
                snapshot_case_count=len(exported['cases']))
        except BaseException as exc:
            await page.screenshot(path=str(a.out/'failed.png'),full_page=True)
            (a.out/'failed.json').write_text(json.dumps(dict(error=str(exc),browser_errors=errors),indent=2));raise
        finally:await browser.close()
    result['artifacts']={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in a.out.iterdir() if f.is_file()}
    (a.out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--url',default='http://127.0.0.1:8767');p.add_argument('--out',type=Path,required=True)
    asyncio.run(main(p.parse_args()))
