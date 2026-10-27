"""Exercise a running Compose demo with real models and controlled browser races.

Requires Playwright and its Chromium browser in the review environment. Does not
request a physical camera or change model weights, thresholds or source labels.
"""
import argparse
import asyncio
import json
from pathlib import Path

from playwright.async_api import async_playwright, Error as BrowserError


async def rehearse(args):
    args.out.mkdir(parents=True, exist_ok=False)
    result = {"base_url": args.base_url, "physical_camera": False, "GPU_used": False,
              "training": False, "run5_used": False}
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, executable_path=args.chromium, args=["--disable-gpu"])
        context = await browser.new_context(viewport={"width": 1440, "height": 1000},
                                           record_video_size={"width": 1440, "height": 1000},
                                           record_video_dir=str(args.out / "browser-video"))
        await context.add_init_script("window.reviewSockets=[]; const WS=window.WebSocket; window.WebSocket=class extends WS {constructor(...args){super(...args);window.reviewSockets.push(this)}}")
        page = await context.new_page()
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        async def get(path):
            response = await page.request.get(args.base_url + path)
            assert response.ok, (path, response.status)
            return await response.json()
        try:
            health = await get("/health")
            assert health["service"] == "linewatch-inspection-api" and health["pipeline_mode"] == "model"
            await page.goto(args.base_url, wait_until="networkidle")
            await page.get_by_role("button", name="IMAGES", exact=False).click()
            await page.locator('input[type=file]').first.set_input_files(str(args.assets / "examples/run4-fill1-f000339.png"))
            await page.get_by_text("1 source image persisted and queued", exact=True).wait_for(timeout=120000)
            runs = await get("/api/v1/sessions")
            run = runs[0]
            first = (await get(f"/api/v1/sessions/{run['session_id']}/events"))[0]
            assert first["source_timestamp_ms"] is None
            assert first["pipeline_mode"] == "model" and first["visuals"]["status"] == "available"
            assert first["decision"]["evidence"]["models"]["liquid_sha256"]
            assert first["decision"]["status"] == "unavailable"
            result["independent_still"] = {"has_mask": bool(first["visuals"]["mask"]), "qa": first["decision"]["evidence"]["quality_flags"], "time": first["source_timestamp_ms"]}
            await page.get_by_role("button", name="SOURCE", exact=True).click()
            await page.wait_for_function("document.querySelector('.pipeline-visual img')?.naturalWidth===640")
            assert await page.locator('.pipeline-visual img').get_attribute("alt") == "Original source frame"
            result["source_toggle"] = "original source visible"
            await page.get_by_role("button", name="POST-PIPELINE", exact=True).click()
            await page.screenshot(path=str(args.out / "still-qa.png"))

            await page.get_by_role("button", name="RECORDING", exact=False).click()
            await page.locator('.advanced-setup summary').click()
            await page.locator('.sample-control input').fill("0.2")
            await page.locator('input[type=file]').first.set_input_files(str(args.clip))
            await page.wait_for_function("document.querySelector('.message-bar')?.textContent.includes('Recording complete')", timeout=180000)
            runs = await get("/api/v1/sessions")
            run = next(item for item in runs if item.get("source_name") == args.clip.name)
            events = await get(f"/api/v1/sessions/{run['session_id']}/events?limit=500")
            assert len(events) >= 30
            assert any(event["decision"]["evidence"].get("measurement_valid") for event in events)
            assert all(event["decision"]["status"] != "running_ok" for event in events)
            assert all(event["decision"]["evidence"]["quality_decision"] is None for event in events)
            assert all(event["visuals"]["mask"] for event in events)
            result["recording"] = {"session_id": run["session_id"], "frames": len(events),
                "qa_available": sum(event["decision"]["evidence"].get("measurement_valid", False) for event in events),
                "proposal_frames": sum(event["decision"]["evidence"].get("proposal_pixels", 0) > 0 for event in events),
                "source_clock": sorted({event["decision"]["evidence"]["source_clock"] for event in events})}
            (args.out / "recording-events.json").write_text(json.dumps(events, indent=2))
            await page.evaluate("window.scrollTo(0,0)")
            await page.locator(".decision-body .measurements").wait_for()
            assert await page.locator("[role=progressbar]").count() == 0  # no invented progress
            await page.screenshot(path=str(args.out / "recording-monitor.png"))
            if args.presentation_only:
                result["presentation_only"] = True
                result["passed"] = True
                await asyncio.sleep(4)
                return
            await page.locator('.response-panel input').first.fill("Demonstration reviewer")
            await page.get_by_role("button", name="SAVE RESPONSE", exact=True).first.click()
            await page.get_by_text("Operator response persisted; this does not approve product release", exact=True).wait_for()
            saved = await get(f"/api/v1/sessions/{run['session_id']}/events?limit=500")
            assert any(event.get("acknowledgement", {}).get("reviewer") == "Demonstration reviewer" for event in saved if event.get("acknowledgement"))
            result["operator_response"] = "attributed response saved separately from product approval"
            acknowledged = next(event for event in saved if event.get("acknowledgement"))
            delayed_duplicate = dict(acknowledged, acknowledged_at=None, acknowledgement=None)
            pending_before = await page.locator('.review-queue-item').count()
            await page.evaluate("event=>window.reviewSockets.filter(s=>s.url.includes('/sessions/')).forEach(s=>s.dispatchEvent(new MessageEvent('message',{data:JSON.stringify(event)})))", delayed_duplicate)
            await asyncio.sleep(.1)
            assert await page.locator('.review-queue-item').count() == pending_before
            result["late_duplicate_after_response"] = "acknowledged event stays out of review queue"
            await page.get_by_role("link", name="OPEN FULL FRAME REVIEW →").click()
            await page.wait_for_function("document.querySelectorAll('.artifact-grid img').length===3 && Array.from(document.querySelectorAll('.artifact-grid img')).every(i=>i.naturalWidth>0)")
            await page.screenshot(path=str(args.out / "recording-review.png"))
            result["history"] = "source, rendered and mask artifacts decoded in review"

            # Delay a valid old-session response across NEW RUN. No actual inference is mocked above.
            await page.goto(args.base_url, wait_until="networkidle")
            await page.get_by_role("button", name="IMAGES", exact=False).click()
            held, release = asyncio.Event(), asyncio.Event()
            async def delayed(route):
                event = dict(first)
                event["session_id"] = route.request.url.split("/sessions/")[1].split("/")[0]
                held.set()
                await release.wait()
                try:
                    await route.fulfill(status=202, json=event)
                except BrowserError:
                    pass  # Aborting the obsolete request is an expected successful boundary.
            await page.route("**/sessions/*/frames?**", delayed)
            await page.locator('input[type=file]').first.set_input_files(str(args.assets / "examples/run4-fill1-f000339.png"))
            await asyncio.wait_for(held.wait(), 10)
            await page.get_by_role("button", name="NEW RUN", exact=True).click()
            release.set()
            await asyncio.sleep(.5)
            assert await page.locator('.event-row').count() == 0
            assert await page.locator('.pipeline-visual img').count() == 0
            assert await page.locator('.state-chip').inner_text() == "STOPPED"
            result["late_response_after_reset"] = "obsolete result discarded"
            await page.unroute("**/sessions/*/frames?**", delayed)

            # Use a generated canvas stream instead of a physical camera.
            await page.get_by_role("button", name="LIVE CAMERA", exact=False).click()
            await page.evaluate("""() => {
                const canvas=document.createElement('canvas');canvas.width=640;canvas.height=480;
                const ctx=canvas.getContext('2d');ctx.fillStyle='rgb(210,20,30)';ctx.fillRect(0,0,320,480);
                ctx.fillStyle='rgb(10,220,50)';ctx.fillRect(320,0,320,480);
                window.reviewCanvas=canvas;window.reviewStream=canvas.captureStream(5);
                navigator.mediaDevices.getUserMedia=async()=>window.reviewStream;
            }""")
            await page.get_by_role("button", name="START CAMERA", exact=False).click()
            await page.wait_for_function("document.querySelectorAll('.event-row').length>0", timeout=120000)
            await page.get_by_role("button", name="STOP CAMERA", exact=True).click()
            runs = await get("/api/v1/sessions")
            live = next(item for item in runs if item["source_mode"] == "live_camera")
            event = (await get(f"/api/v1/sessions/{live['session_id']}/events"))[0]
            left = await page.evaluate("""async(url)=>{const im=new Image();im.src=url;await im.decode();const c=document.createElement('canvas');c.width=640;c.height=480;const ctx=c.getContext('2d');ctx.drawImage(im,0,0);return Array.from(ctx.getImageData(100,240,1,1).data).slice(0,3)}""", args.base_url + event["image"]["url"])
            assert left == [210,20,30], left
            await page.evaluate("window.reviewSockets.forEach(s=>s.close())")
            await asyncio.sleep(6)
            assert await page.locator('.state-chip').inner_text() != "RUNNING WELL"
            assert await page.locator('.decision-body h2').inner_text() == "Live inspection stopped"
            result["live_source"] = {"pixels_preserved": left, "stopped_state": await page.locator('.state-chip').inner_text()}
            await page.route("**/health", lambda route: route.abort())
            await page.get_by_text("BACKEND OFFLINE", exact=True).wait_for(timeout=12000)
            assert await page.locator('.state-chip').inner_text() != "RUNNING WELL"
            result["offline_state"] = "no current normal verdict"
            await page.screenshot(path=str(args.out / "offline-state.png"))
            assert not errors, errors
            result["page_errors"] = errors
            result["passed"] = True
        finally:
            result["page_errors"] = errors
            (args.out / "browser-checks.json").write_text(json.dumps(result, indent=2))
            await context.close()
            await browser.close()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8788")
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--chromium")
    parser.add_argument("--presentation-only", action="store_true", help="Record the actual model/image/video path without simulated fault probes")
    asyncio.run(rehearse(parser.parse_args()))
