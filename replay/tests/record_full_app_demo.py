"""Record the actual browser/API workflow; every response is a demonstration."""
import argparse
import asyncio
import json
from pathlib import Path

from playwright.async_api import async_playwright


async def record(a):
    a.out.mkdir(parents=True,exist_ok=False);errors=[];requests=[];checks={}
    async with async_playwright() as p:
        browser=await p.chromium.launch(executable_path=a.browser,headless=True,args=['--no-sandbox','--disable-gpu','--disable-background-networking'])
        context=await browser.new_context(viewport=dict(width=1600,height=1200),record_video_dir=str(a.out/'recording'),record_video_size=dict(width=1600,height=1200),accept_downloads=True)
        page=await context.new_page();page.on('pageerror',lambda e:errors.append(str(e)));page.on('request',lambda r:requests.append(r.url))
        try:
            await page.goto(a.url);await page.wait_for_function('() => window.demo && document.getElementById("fill-image").naturalWidth===640')
            await page.locator('#actor').fill('Demonstration operator')
            await page.get_by_role('button',name='Processing pipeline').click();await page.wait_for_timeout(6500)
            await page.screenshot(path=str(a.out/'01-pipeline.png'),full_page=True)
            await page.get_by_role('button',name='Station monitor').click()
            await page.locator('#fill-play').click();await page.wait_for_timeout(10500);await page.locator('#fill-play').click()
            await page.locator('#context-open').click();await page.locator('#context-recipe').fill('DEMO-A')
            await page.locator('#context-note').fill('Demonstration job context only. No approved recipe reference or physical vial identity supplied.')
            await page.get_by_role('button',name='Save context',exact=True).click();await page.wait_for_function('() => demo.state.context?.recipe==="DEMO-A"')
            index=await page.evaluate('demo.data.fill.findIndex(r=>r.modes.sam.qa_pass&&!r.modes.direct.qa_pass)')
            assert index>=0;await page.evaluate('(i)=>demo.fill(i)',index)
            await page.locator('#sam').uncheck();await page.wait_for_function('() => demo.displayed.mode==="direct"');await page.wait_for_timeout(3000)
            await page.locator('#sam').check();await page.wait_for_function('() => demo.displayed.mode==="sam"');await page.wait_for_timeout(4000)
            assert await page.locator('#fill-kpi').text_content()=='Color sample available'
            await page.screenshot(path=str(a.out/'02-two-station-monitor.png'),full_page=True)
            await page.evaluate('()=>demo.cap(1)')
            target=a.audit_slot or await page.evaluate('demo.data.cap[1].slots.find(r=>r.class_name==="crooked_cap"&&r.cap_warning).id')
            await page.locator('#slot').select_option(target);await page.locator('#review-note').fill('Inspect the provisional cap-state classification at '+target+'. This is synthetic demonstration evidence; classification may be wrong.')
            await page.wait_for_function('() => document.getElementById("cap-crop-caption").textContent.includes("fixture aperture")')
            checks['fixture_crop_is_distinct_from_learned_cap_localization']=await page.evaluate('demo.data.cap_display.learned_cap_localization===false')
            await page.wait_for_timeout(3000);await page.locator('#cap-play').click();await page.locator('#request-review').click()
            await page.wait_for_function('() => demo.state.reviews.length===1');await page.wait_for_selector('#review-image:not([hidden])')
            source=await page.locator('#review-image').get_attribute('src');await page.wait_for_timeout(3200)
            checks['review_frame_pinned_while_source_advances']=source==await page.locator('#review-image').get_attribute('src')
            checks['closure_blocked_before_response']=await page.locator('#close').is_disabled()
            await page.screenshot(path=str(a.out/'03-captured-cap-review.png'),full_page=True)
            await page.locator('#response-note').fill('Demonstration: opened the captured evidence for inspection.')
            await page.locator('#ack').click();await page.wait_for_function('() => demo.state.reviews[0].status==="acknowledged"');assert await page.locator('#close').is_disabled()
            await page.wait_for_timeout(2000);await page.locator('#response-note').fill('Demonstration: operator would inspect the cap under an approved procedure. No physical action or equipment command was performed.')
            await page.locator('#respond').click();await page.wait_for_function('() => demo.state.reviews[0].status==="responded"');await page.wait_for_timeout(3000)
            await page.screenshot(path=str(a.out/'04-operator-response.png'),full_page=True)
            await page.locator('#response-note').fill('Demonstration follow-up recorded. Product acceptance remains unavailable.')
            await page.locator('#close').click();await page.wait_for_function('() => demo.state.reviews[0].status==="closed"');await page.wait_for_timeout(2200)
            await page.get_by_role('button',name='Station monitor').click();await page.locator('#cap-play').click()
            await page.locator('#inspect-fill').click();await page.locator('#review-note').fill('Color sample is available, but the intended-recipe reference is unqualified. Operator/CSI confirmation is still required.')
            await page.locator('#request-review').click();await page.wait_for_function('() => demo.state.reviews.length===2');await page.wait_for_timeout(3000)
            async with page.expect_download() as event:await page.locator('#export').click()
            await (await event.value).save_as(a.out/'demonstration-session.json')
            exported=json.loads((a.out/'demonstration-session.json').read_text());reviews=exported['state']['reviews']
            assert len(reviews)==2 and reviews[0]['status']=='closed' and reviews[1]['status']=='requested'
            assert exported['demonstration'] and not exported['eligible_for_team_review'] and not exported['product_release_authorized'] and exported['hardware_commands']==0
            assert {r['domain'] for r in reviews}=={'real','synthetic'}
            assert reviews[1]['source_identity']['clock_kind']=='native'
            checks['actual_backend_export_has_two_domains_and_native_source_identity']=True
            checks['acknowledgement_not_resolution_and_no_release']=True
            await page.reload();await page.wait_for_function('() => window.demo && demo.state.reviews.length===2')
            checks['browser_reload_preserves_server_journal']=True
            await page.locator('#actor').fill('Demonstration operator');await page.evaluate('(i)=>demo.fill(i)',index);await page.evaluate('()=>demo.cap(1)')
            await page.locator('#fill-play').click();await page.wait_for_timeout(5000);await page.locator('#fill-play').click()
            await page.screenshot(path=str(a.out/'05-final-monitor.png'),full_page=True)
            # Responsive inspection is outside the recorded desktop session.
            mobile=await browser.new_page(viewport=dict(width=390,height=844));await mobile.goto(a.url)
            await mobile.wait_for_function('() => window.demo');assert not await mobile.evaluate('document.documentElement.scrollWidth>innerWidth')
            await mobile.screenshot(path=str(a.out/'mobile.png'),full_page=True);await mobile.close();checks['mobile_no_horizontal_overflow']=True
            assert not errors,errors;assert all(url.startswith(a.url) for url in requests if url.startswith('http')),requests
            checks['no_browser_errors_or_external_requests']=True
            (a.out/'verification.json').write_text(json.dumps(dict(checks=checks,counts=await page.evaluate('demo.data.counts'),
                browser_errors=errors,network_requests=len(requests),new_inference=False,training=False,physical_vials_linked=False,
                demonstration=True,product_release_authorized=False,hardware_commands=0),indent=2)+'\n')
        except BaseException as error:
            await page.screenshot(path=str(a.out/'failed.png'),full_page=True);(a.out/'failed.json').write_text(json.dumps(dict(error=str(error),browser_errors=errors),indent=2));raise
        finally:
            await context.close();await browser.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--url',default='http://127.0.0.1:8871');p.add_argument('--out',type=Path,required=True)
    p.add_argument('--browser',default='/home/austin/.cache/ms-playwright/chromium-1248/chrome-linux64/chrome')
    p.add_argument('--audit-slot',help='Explicit demonstration target; does not relabel its prediction')
    asyncio.run(record(p.parse_args()))
