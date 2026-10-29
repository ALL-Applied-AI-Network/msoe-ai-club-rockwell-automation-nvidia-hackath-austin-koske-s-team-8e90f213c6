"""Record a paced fallback video of the actual operator flow against a running app.

Presentation only: no fault injection, races or offline probes. Requires Playwright
with Chromium. Uses the real model pipeline; does not request a physical camera.
"""
import argparse
import asyncio
import json
from pathlib import Path

from playwright.async_api import async_playwright


async def record(args):
    args.out.mkdir(parents=True, exist_ok=False)
    report = {"physical_camera": False, "GPU_used": args.gpu_used, "training": False, "run5_used": False}
    errors = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, executable_path=args.chromium, args=["--disable-gpu"])
        context = await browser.new_context(viewport={"width": 1440, "height": 1000},
                                           record_video_dir=str(args.out / "browser-video"),
                                           record_video_size={"width": 1440, "height": 1000})
        page = await context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        pause = lambda s: page.wait_for_timeout(int(s * 1000))
        shot = lambda name: page.screenshot(path=str(args.out / f"{name}.png"))

        async def get(path):
            response = await page.request.get(args.base_url + path)
            assert response.ok, (path, response.status)
            return await response.json()

        async def show(locator, seconds=2.5):
            await locator.scroll_into_view_if_needed()
            await pause(seconds)

        try:
            health = await get("/health")
            assert health["pipeline_mode"] == "model"
            await page.goto(args.base_url, wait_until="networkidle")
            await pause(3)
            await shot("01-monitor-empty")

            await page.get_by_role("button", name="RECORDING", exact=False).click()
            await page.locator(".advanced-setup summary").click()
            await page.locator(".sample-control input").fill("0.2")
            await page.locator(".advanced-setup summary").click()
            await pause(1.5)
            await page.locator("input[type=file]").first.set_input_files(str(args.clip))
            await page.wait_for_function(
                "document.querySelector('.message-bar')?.textContent.includes('Recording complete')", timeout=300000)
            await page.evaluate("window.scrollTo(0,0)")
            await pause(3)
            await shot("02-recording-post-pipeline")

            runs = await get("/api/v1/sessions")
            run = next(item for item in runs if item.get("source_name") == args.clip.name)
            sid = run["session_id"]
            events = await get(f"/api/v1/sessions/{sid}/events?limit=500")
            evidence = lambda e: e["decision"]["evidence"]
            report["session_id"] = sid
            report["frames"] = len(events)
            report["proposal_frames"] = sum(evidence(e).get("proposal_pixels", 0) > 0 for e in events)
            report["qa_available"] = sum(bool(evidence(e).get("measurement_valid")) for e in events)
            (args.out / "recording-events.json").write_text(json.dumps(events, indent=2))

            await page.get_by_role("button", name="SOURCE", exact=True).click()
            await pause(3)
            await shot("03-recording-source")
            await page.get_by_role("button", name="POST-PIPELINE", exact=True).click()
            await pause(2.5)

            await page.locator(".response-panel input").first.fill("Demonstration reviewer")
            await show(page.locator(".response-panel").first, 2)
            await page.get_by_role("button", name="SAVE RESPONSE", exact=True).first.click()
            await page.get_by_text("Operator response persisted; this does not approve product release", exact=True).wait_for()
            await pause(3)
            await shot("04-operator-response")

            await show(page.get_by_role("link", name="OPEN FULL FRAME REVIEW →"), 1.5)
            await page.get_by_role("link", name="OPEN FULL FRAME REVIEW →").click()
            await page.wait_for_function("document.querySelectorAll('.artifact-grid img').length===3 && Array.from(document.querySelectorAll('.artifact-grid img')).every(i=>i.naturalWidth>0)")
            await pause(3)

            # Proposal exists but sampling QA is unavailable, then a usable measurement.
            proposal_only = next(e for e in events if evidence(e).get("proposal_pixels", 0) > 0 and not evidence(e).get("measurement_valid"))
            measured = next(e for e in events if evidence(e).get("measurement_valid"))
            report["shown_proposal_only_frame"] = proposal_only["frame_number"]
            report["shown_measured_frame"] = measured["frame_number"]
            for name, event in (("05-review-proposal-qa-unavailable", proposal_only), ("06-review-measurement-available", measured)):
                button = page.locator(".frame-browser-list button").filter(has_text=f"FRAME {event['frame_number']}").first
                await button.scroll_into_view_if_needed()
                await button.click()
                await page.wait_for_function("Array.from(document.querySelectorAll('.artifact-grid img')).every(i=>i.complete&&i.naturalWidth>0)")
                await page.evaluate("window.scrollTo(0,0)")
                await pause(2)
                await show(page.locator(".frame-metadata"), 3)
                await shot(name)

            panel = page.get_by_role("region", name="Investigation", exact=True)
            await show(panel, 1)
            await panel.get_by_text("No qualified completed-vial records supplied.", exact=False).wait_for()
            await pause(3)
            await panel.screenshot(path=str(args.out / "07-investigation-this-run.png"))
            await panel.get_by_role("button", name="SYNTHETIC EXAMPLE", exact=True).click()
            await panel.get_by_text("SYNTHETIC EXAMPLE · NOT THIS CAMERA/RUN", exact=True).wait_for()
            await page.wait_for_function("document.querySelectorAll('.investigation-table tbody tr').length===4")
            await show(panel, 4)
            await panel.screenshot(path=str(args.out / "08-investigation-transient.png"))
            await panel.locator("select").select_option("dose-record")
            await panel.get_by_role("heading", name="Appearance is within the research band", exact=False).wait_for()
            await show(panel, 4)
            await panel.screenshot(path=str(args.out / "09-investigation-dose-record.png"))
            await panel.get_by_role("button", name="THIS RUN", exact=True).click()
            await pause(2)
            assert not errors, errors
            report["passed"] = True
        finally:
            report["page_errors"] = errors
            (args.out / "browser-checks.json").write_text(json.dumps(report, indent=2))
            await context.close()
            await browser.close()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8788")
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--chromium")
    parser.add_argument("--gpu-used", action="store_true", help="Record that the served app ran inference on a GPU")
    asyncio.run(record(parser.parse_args()))
