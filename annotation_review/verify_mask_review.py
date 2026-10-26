"""Exercise the offline mask editor using explicitly ineligible demonstration labels."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from playwright.async_api import async_playwright


async def verify(a):
    a.out.mkdir(parents=True, exist_ok=False)
    checks = {}
    errors = []
    urls = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(executable_path=str(a.chromium), headless=True,
            args=['--disable-gpu', '--disable-background-networking'])
        context = await browser.new_context(viewport={'width': 1400, 'height': 1100})
        page = await context.new_page()
        page.on('pageerror', lambda e: errors.append(str(e)))
        page.on('request', lambda r: urls.append(r.url))
        url = a.html.resolve().as_uri() + '?demonstration=1'
        try:
            await page.goto(url)
            await page.wait_for_function('currentImage !== null')
            assert await page.evaluate('tasks.length===72 && packet.cases.length===6 && history.length===0')
            assert await page.evaluate('overlay===false && demonstration===true')
            checks['raw_first_all_drafts_unreviewed'] = True
            await page.locator('#reviewer').fill('Demonstration verification')
            await page.screenshot(path=str(a.out / '01-raw-first.png'), full_page=True)
            await page.locator('#status').select_option('empty')
            await page.locator('#save').click()
            assert await page.evaluate('history.length===1 && history[0].mask.counts[0]===640*480')
            checks['explicit_empty_saves_zero_mask'] = True
            # A visible-pool sample is used only to exercise editing, never as approved truth.
            await page.locator('#next').click()
            await page.wait_for_function('index===1 && currentImage!==null')
            await page.locator('#status').select_option('pool')
            await page.locator('#save').click()
            assert await page.evaluate('history.length===1')
            assert 'mask disagree' in await page.locator('#message').inner_text()
            checks['pool_without_mask_rejected'] = True
            await page.locator('#zoom').check()
            box = await page.locator('#image').bounding_box()
            for x, y in [(0.40, 0.70), (0.64, 0.70), (0.61, 0.78), (0.43, 0.78)]:
                await page.mouse.click(box['x'] + box['width'] * x, box['y'] + box['height'] * y)
            await page.locator('#close').click()
            await page.locator('#notes').fill('Demonstration geometry only; not team-approved labels.')
            await page.locator('#save').click()
            assert await page.evaluate('history.length===2 && history[1].mask.counts.length>1')
            checks['zoom_polygon_full_image_mask_saved'] = True
            await page.locator('#last-empty').select_option('run4-fill1-f000264')
            await page.locator('#first-pool').select_option('run4-fill1-f000294')
            await page.locator('#save-onset').click()
            assert await page.evaluate('history.length===3 && exportReview().onset_intervals["run4-fill1"].exact_native_first_present===false')
            checks['sparse_endpoint_interval_not_exact_onset'] = True
            await page.locator('#draft').click()
            await page.wait_for_function('loadedDraft!==null')
            assert await page.evaluate('history.length===3 && dirty')
            checks['loading_draft_does_not_approve'] = True
            await page.locator('#tool').select_option('erase')
            before = await page.evaluate('mask.reduce((a,b)=>a+b,0)')
            await page.mouse.click(box['x'] + box['width'] * .5, box['y'] + box['height'] * .75)
            await page.locator('#undo').click()
            assert await page.evaluate('mask.reduce((a,b)=>a+b,0)') == before
            checks['brush_edit_and_undo'] = True
            # Restore the saved decision before the round-trip; demonstration history remains isolated.
            await page.evaluate('dirty=false; load(index)')
            async with page.expect_download() as event:
                await page.locator('#download').click()
            download = await event.value
            exported = a.out / 'demonstration-review.json'
            await download.save_as(exported)
            value = json.loads(exported.read_text())
            assert value['demonstration'] and not value['eligible_for_team_review'] and not value['team_approved']
            assert not value['training_eligible']
            checks['demonstration_ineligible_for_team_labels'] = True
            original = await page.evaluate('JSON.stringify(history)')
            await page.evaluate('localStorage.clear()')
            await page.reload()
            await page.wait_for_function('currentImage!==null')
            assert await page.evaluate('history.length===0')
            await page.locator('#import').set_input_files(str(exported))
            await page.wait_for_function('history.length===3')
            assert await page.evaluate('JSON.stringify(history)') == original
            checks['download_import_round_trip_preserves_geometry_and_history'] = True
            rejected = await page.evaluate('''() => {
              const reasons={}; for(const kind of ['source','packet','history','mask','demonstration']){
                const v=JSON.parse(JSON.stringify(exportReview()));
                if(kind==='source')v.history[0].source.image_sha256='0'.repeat(64);
                if(kind==='packet')v.packet_id='another-packet';
                if(kind==='history')v.history[0].notes='divergent history';
                if(kind==='mask')v.history[1].mask.counts=[1];
                if(kind==='demonstration')v.demonstration=false;
                try{importReview(v);reasons[kind]=false;}catch(e){reasons[kind]=true;}
              } return reasons;
            }''')
            assert all(rejected.values()), rejected
            checks['source_packet_mask_history_and_demo_guards'] = rejected
            await page.locator('#frame').select_option('run4-fill1-f000294')
            await page.wait_for_function('index===1 && currentImage!==null')
            await page.screenshot(path=str(a.out / '02-reviewed-demonstration.png'), full_page=True)
            await page.set_viewport_size({'width': 390, 'height': 844})
            assert await page.evaluate('document.documentElement.scrollWidth<=innerWidth')
            await page.screenshot(path=str(a.out / '03-mobile.png'), full_page=True)
            checks['mobile_without_page_overflow'] = True
            assert not errors, errors
            assert all(u.startswith(('file:', 'data:', 'blob:')) for u in urls), urls
            checks['offline_no_network_requests'] = True
        finally:
            await browser.close()
    result = dict(status='passed', checks=checks, browser_errors=errors,
                  approved_labels_created=False, local_gpu_used=False,
                  artifacts={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in a.out.iterdir() if p.is_file()})
    (a.out / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--html', type=Path, required=True)
    parser.add_argument('--chromium', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    asyncio.run(verify(parser.parse_args()))
