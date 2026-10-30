import sys; sys.path.insert(0, r"C:\Users\Austin\AppData\Local\Temp\linewatch-live\pw")
from pathlib import Path
from playwright.sync_api import sync_playwright
here = Path(__file__).parent
jobs = [("slides.html?png&fig=" + f, "fig-" + f + ".png") for f in ["run", "changeover", "live", "sim2real", "results"]] + [("arch-slide.html", "fig-architecture.png")]
with sync_playwright() as p:
    b = p.chromium.launch(channel="msedge")
    for src, out in jobs:
        pg = b.new_page(viewport={"width": 1660, "height": 1000}, device_scale_factor=2)
        errs = []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.on("console", lambda m: m.type == "error" and errs.append(m.text))
        pg.goto((here / src.split("?")[0]).as_uri() + ("?" + src.split("?")[1] if "?" in src else ""))
        pg.wait_for_selector(".canvas"); pg.wait_for_timeout(1200)
        pg.locator(".canvas").screenshot(path=str(here / out), omit_background=False)
        print(out, errs)
        pg.close()
    b.close()
