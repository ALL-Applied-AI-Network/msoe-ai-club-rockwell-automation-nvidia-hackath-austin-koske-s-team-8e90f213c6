"""Drive the recipe-first vial flow in a browser against a running app.

Enters recipe 5, 6, 6 for Run5 vial clips 2, 4, 5 and checks each vial verdict and its
simulated line action. Requires Playwright with Chromium and the clips from
replay/make_run5_vial_clips.py. Nothing is sent to any equipment.
"""
import argparse
import asyncio
import json
from pathlib import Path

from playwright.async_api import async_playwright

PLAN = [("5", "run5-vial02-recipe5.mp4", "IN SPEC"),
        ("6", "run5-vial04-recipe6.mp4", "OUT OF SPEC"),
        ("6", "run5-vial05-recipe6.mp4", "OUT OF SPEC")]


async def main(args):
    args.out.mkdir(parents=True, exist_ok=False)
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=["--disable-gpu"])
        context = await browser.new_context(viewport={"width": 1440, "height": 1000})
        page = await context.new_page()
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        await page.goto(args.base_url, wait_until="networkidle")
        await page.get_by_role("button", name="RECORDING", exact=False).click()
        for index, (recipe, clip, expected) in enumerate(PLAN, 1):
            await page.locator(".recipe-card select").select_option(recipe)
            await page.get_by_role("button", name="SET RECIPE", exact=True).click()
            await page.locator("input[type=file]").first.set_input_files(str(args.clips / clip))
            await page.wait_for_function(f"document.querySelectorAll('.vial-result').length==={index}", timeout=600000)
            chip = await page.locator(".vial-result .vial-chip").first.inner_text()
            assert chip == expected, (clip, chip)
        results = await page.evaluate("Array.from(document.querySelectorAll('.vial-result')).map(e => e.innerText)")
        assert "toward recipe 5" in results[0] and "toward recipe 5" in results[1], results
        await page.locator(".vial-panel input").fill("Demonstration operator")
        await page.get_by_role("button", name="CONFIRM", exact=True).first.click()
        await page.wait_for_selector(".vial-decided")
        await page.screenshot(path=str(args.out / "vial-flow.png"), full_page=True)
        assert not errors, errors
        report = {"passed": True, "vials": results, "page_errors": errors, "simulated_actions_only": True}
        (args.out / "vial-flow.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        await context.close()
        await browser.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8788")
    parser.add_argument("--clips", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    asyncio.run(main(parser.parse_args()))
