"""Exercise the actual running Isaac/operator app; retain screenshots and video."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from playwright.async_api import async_playwright


async def main(a):
    a.out.mkdir(parents=True,exist_ok=False)
    errors=[];urls=[];screenshots=[]
    async with async_playwright() as p:
        browser=await p.chromium.launch(channel='chrome',headless=True,args=['--disable-gpu','--disable-background-networking'])
        context=await browser.new_context(viewport=dict(width=1440,height=1080),
            record_video_dir=str(a.out/'browser-video'),record_video_size=dict(width=1440,height=1080))
        page=await context.new_page()
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.on('console',lambda message:errors.append(message.text) if message.type=='error' else None)
        page.on('request',lambda request:urls.append(request.url))
        async def wait(step):
            await page.wait_for_function('(step)=>state && (state.step===step || state.error)',arg=step,timeout=240000)
            error=await page.evaluate('state.error')
            if error:raise RuntimeError(error)
        async def shot(name):
            await page.locator('h1').click();await page.wait_for_timeout(500)
            await page.screenshot(path=str(a.out/name),full_page=True);screenshots.append(name)
        try:
            await page.goto(a.url,wait_until='domcontentloaded')
            await wait('ready_reference');await page.locator('#actor').fill('Simulation walkthrough')
            await shot('01-ready.png')
            await page.locator('#reference').click();await wait('ready_test')
            await page.wait_for_function('()=>document.querySelector("#live-image").naturalWidth===640 && !document.querySelector("#live-image").hidden')
            await shot('02-reference-captured.png')
            await page.locator('#test').click()
            await page.wait_for_function('()=>state && (state.review==="requested" || state.error)',timeout=240000)
            await shot('03-review-requested.png');await wait('awaiting_response')
            await page.locator('#acknowledge').click()
            await page.wait_for_function('()=>state.review==="acknowledged"')
            assert await page.evaluate('state.step==="awaiting_response"')
            await page.locator('#flush').click();await page.locator('#action-note').fill('Operator requests a hypothetical 8 mL prime to drain in this local simulation only. No physical procedure is being performed.')
            await page.locator('#action-confirm').check();await page.locator('#save-action').click()
            await wait('ready_recovery');await shot('04-response-completed.png')
            await page.locator('#recovery').click();await wait('recovery_complete')
            assert await page.evaluate('state.review==="acknowledged"')
            await shot('05-recovery-measured.png')
            await page.locator('#close').click();await page.locator('#action-note').fill('Reviewed newly rendered recovery evidence and measured diagnostics. Simulated follow-up documented; no real product acceptance or release is verified.')
            await page.locator('#action-confirm').check();await page.locator('#save-action').click();await wait('complete')
            await shot('06-walkthrough-complete.png')
            state=await page.evaluate('state')
            assert len(state['episodes'])==3 and all(e['frames']==37 for e in state['episodes'])
            assert state['review']=='closed' and state['hardware_commands_sent']==0
            assert state['operational_recipe_assessment']=='unavailable'
            events=(await page.request.get(a.url+'/api/export.json'));data=await events.json()
            assert any(e['kind']=='simulated_flush_completed' for e in data['events'])
            assert data['events'][-1]['payload']['product_release_authorized'] is False
            await page.locator('#view').select_option('source');await page.wait_for_timeout(700)
            assert await page.locator('#live-image').get_attribute('data-frame')==state['latest']['image_key']
            await page.locator('#view').select_option('mask');await page.wait_for_timeout(700)
            await shot('07-model-mask.png')
            await page.locator('#view').select_option('overlay');await page.wait_for_timeout(500)
            await page.reload(wait_until='domcontentloaded');await wait('complete')
            assert await page.evaluate('state.episodes.length===3 && state.review==="closed"')
            await page.set_viewport_size(dict(width=390,height=844));await page.wait_for_timeout(400)
            assert await page.evaluate('document.documentElement.scrollWidth<=window.innerWidth')
            await shot('08-mobile.png')
            assert all(url.startswith(a.url+'/') for url in urls),urls
            assert not errors,errors
            (a.out/'session-export.json').write_text(json.dumps(data,indent=2))
            result=dict(status='passed',episodes=state['episodes'],actual_live_frames=111,hardware_commands_sent=0,
                review_closed=True,operational_recipe_assessment='unavailable',uncaught_browser_errors=errors,
                checks=dict(live_RGB_display=True,source_overlay_mask_views=True,acknowledgement_keeps_review_open=True,
                    intervention_receipt_before_recovery=True,review_latched_through_recovery=True,
                    reload_preserves_state=True,mobile_no_page_overflow=True,all_page_requests_loopback=True),
                screenshots=screenshots,mode='live_isaac_simulation')
        except BaseException as exc:
            await page.screenshot(path=str(a.out/'failed.png'),full_page=True)
            (a.out/'failed.json').write_text(json.dumps(dict(error=str(exc),browser_errors=errors),indent=2))
            raise
        finally:
            await context.close();await browser.close()
    result['artifacts']={str(f.relative_to(a.out)):hashlib.sha256(f.read_bytes()).hexdigest() for f in a.out.rglob('*') if f.is_file()}
    (a.out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--url',default='http://127.0.0.1:8767');p.add_argument('--out',type=Path,required=True)
    asyncio.run(main(p.parse_args()))
