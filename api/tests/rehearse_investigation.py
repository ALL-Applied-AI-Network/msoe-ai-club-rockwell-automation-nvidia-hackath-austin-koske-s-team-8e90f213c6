"""Headless checks against the running app; synthetic RCA stays separate from runs."""
import argparse
import asyncio
import json
from pathlib import Path

from playwright.async_api import async_playwright, Error as BrowserError


async def main(args):
    args.out.mkdir(parents=True, exist_ok=False)
    report = {"physical_camera": False, "GPU_used": False, "training": False}
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, executable_path=args.chromium, args=["--disable-gpu"])
        context = await browser.new_context(viewport={"width": 1440, "height": 1100},
                                           record_video_dir=str(args.out / "browser-video"),
                                           record_video_size={"width": 1440, "height": 1100})
        page = await context.new_page()
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        try:
            health = await (await page.request.get(args.base_url + "/health")).json()
            assert health["service"] == "linewatch-inspection-api"
            await page.goto(args.base_url, wait_until="networkidle")
            await page.get_by_role("link", name="INVESTIGATION", exact=True).click()
            panel = page.get_by_role("region", name="Investigation", exact=True)
            await panel.get_by_role("button", name="SYNTHETIC EXAMPLE", exact=True).click()
            await panel.get_by_text("SYNTHETIC EXAMPLE · NOT THIS CAMERA/RUN", exact=True).wait_for()
            await page.wait_for_function("document.querySelectorAll('.investigation-table tbody tr').length===4")
            assert await panel.get_by_text("Confirmed cause:", exact=False).inner_text() == "Confirmed cause: not identified"
            assert await panel.get_by_role("heading", name="Suspected carryover:", exact=False).is_visible()
            assert await panel.locator('.investigation-causes li').count() == 3
            await panel.screenshot(path=str(args.out / "synthetic-transient.png"))
            report["transient_example"] = "four completed synthetic vials, hypothesis plus alternatives, no confirmed cause"
            await panel.locator('select').select_option('persistent')
            await panel.get_by_role("heading", name="Persistent recipe-appearance deviation:", exact=False).wait_for()
            assert await panel.get_by_role("heading", name="Suspected carryover:", exact=False).count() == 0
            await panel.screenshot(path=str(args.out / "synthetic-persistent.png"))
            await panel.locator('select').select_option('dose-record')
            await panel.get_by_role("heading", name="Appearance is within the research band", exact=False).wait_for()
            assert await panel.get_by_text("Delivery record disagrees", exact=True).is_visible()
            async with page.expect_download() as captured:
                await panel.get_by_role("button", name="DOWNLOAD EVIDENCE", exact=True).click()
            download = await captured.value
            assert download.suggested_filename.startswith('SYNTHETIC-')
            await download.save_as(args.out / download.suggested_filename)
            exported = json.loads((args.out / download.suggested_filename).read_text())
            assert exported['mode'] == 'synthetic_example' and exported['session_id'] is None
            assert exported['diagnosis']['quality_decision'] is None
            assert all(row['delta_e2000'] <= 5 for row in exported['diagnosis']['evidence'])
            report["dose_record"] = "record discrepancy survives normal-band appearance; downloaded evidence remains synthetic"
            await panel.screenshot(path=str(args.out / "synthetic-dose-record.png"))
            await panel.locator('select').select_option('camera')
            await panel.get_by_role("heading", name="Diagnosis unavailable:", exact=False).wait_for()
            assert await panel.get_by_text("SYNTHETIC EXAMPLE · NOT THIS CAMERA/RUN", exact=True).is_visible()
            await panel.get_by_role("button", name="THIS RUN", exact=True).click()
            await panel.get_by_text("Start an inspection run", exact=False).wait_for()
            assert await panel.locator('.synthetic-banner').count() == 0
            assert await panel.locator('tbody tr').count() == 0
            report["mode_switch"] = "synthetic values cleared when returning to this run"

            runs = await (await page.request.get(args.base_url + '/api/v1/sessions')).json()
            run = next(row for row in runs if row['source_mode'] == 'recording' and row['frames_processed'] >= 30)
            await page.goto(args.base_url + '/runs/' + run['session_id'], wait_until='networkidle')
            panel = page.get_by_role("region", name="Investigation", exact=True)
            await panel.get_by_text("No qualified completed-vial records supplied.", exact=False).wait_for()
            body = await (await page.request.get(args.base_url + '/api/v1/sessions/' + run['session_id'] + '/investigation')).json()
            assert body['completed_vials'] == 0 and body['frame_count'] >= 30
            assert body['diagnosis']['status'] == 'unavailable'
            await panel.screenshot(path=str(args.out / "run-unavailable.png"))
            report["real_run"] = {"frames":body['frame_count'], "completed_vials":0, "status":"unavailable"}
            path = args.out / 'rejected-synthetic-import.json'
            path.write_text(json.dumps({'reviewer':'DEMONSTRATION ONLY','evidence_id':'not-real',
                                        'document':{'data_domain':'synthetic'}}))
            await panel.locator('input[type=file]').set_input_files(path)
            await panel.get_by_role('alert').filter(has_text='Run imports require real evidence').wait_for()
            after = await (await page.request.get(args.base_url + '/api/v1/sessions/' + run['session_id'] + '/investigation')).json()
            assert after['completed_vials'] == 0
            report['synthetic_import'] = 'rejected; real run unchanged'

            held, release = asyncio.Event(), asyncio.Event()
            async def delayed(route):
                response = await route.fetch()
                held.set()
                await release.wait()
                try:
                    await route.fulfill(response=response)
                except BrowserError:
                    pass
            await page.route('**/investigation/examples/transient', delayed)
            await panel.get_by_role('button',name='SYNTHETIC EXAMPLE',exact=True).click()
            await asyncio.wait_for(held.wait(), 10)
            await panel.get_by_role('button',name='THIS RUN',exact=True).click()
            release.set()
            await panel.get_by_text('No qualified completed-vial records supplied.',exact=False).wait_for()
            await asyncio.sleep(.2)
            assert await panel.locator('.synthetic-banner').count() == 0
            assert await panel.locator('tbody tr').count() == 0
            report['late_example_response'] = 'does not replace real-run evidence'
            assert not errors, errors
            report['page_errors'] = errors
            report['passed'] = True
        finally:
            report['page_errors'] = errors
            (args.out / 'browser-checks.json').write_text(json.dumps(report, indent=2))
            await context.close()
            await browser.close()
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8788')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--chromium')
    asyncio.run(main(parser.parse_args()))
