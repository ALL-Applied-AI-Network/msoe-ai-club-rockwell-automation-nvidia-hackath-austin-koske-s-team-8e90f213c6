"""End-to-end check of a running Linewatch server on mock cameras (START.sh --mock-fill ...).

Does what the operator does on the LINE page, through the same API: enters the recipe order
(an inspection run with a recipe plan, as RECIPE ORDER does), switches the cameras to MOCK and
starts them (source=camera on run "live", simulate=mock, following that order; --input webcams
uses the station's real cameras instead), reads both MJPEG previews while it runs,
waits for the session to end with the recording, then opens the saved run in Review. With
--compare RUN it compares the vial verdicts with that run's stored results (for Run 5, the batch
line runner's model outputs through the same engine).

Fails (exit 1) on a session error, no decided vials, frames dropped, lag above --max-lag, a
missing preview or saved run, or (with --compare) a verdict that differs.

    python setup/gb10/e2e_mock.py --schedule replay/examples/run5-schedule.csv --compare run5 \\
        --report e2e-report.json [--base http://127.0.0.1:8788] [--seconds 300]
"""
import argparse
import asyncio
import json
import time
import urllib.request
from pathlib import Path

import websockets


def call(base, method, path, body=None):
    request = urllib.request.Request(base + path, method=method, data=None if body is None else json.dumps(body).encode(),
                                     headers={"content-type": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def enter_order(base, schedule):
    recipes = [line.split(",")[1].strip() for line in Path(schedule).read_text().splitlines()[1:] if line.strip()]
    plan = [{"current_recipe_id": f"recipe-{int(r):02d}", "current_recipe_name": f"Recipe {r}", "vial_index": i}
            for i, r in enumerate(recipes, 1)]
    session = call(base, "POST", "/api/v1/sessions", {"source_mode": "live_camera", "run_name": "Recipe order (e2e mock)",
                                                      "recipe": plan[0]})
    call(base, "PUT", f"/api/v1/sessions/{session['session_id']}/recipe-plan", {"recipes": plan})
    return session["session_id"], recipes


def preview_frames(base, key, camera, nbytes=200_000):
    with urllib.request.urlopen(f"{base}/api/v1/line/preview/{key}/{camera}", timeout=10) as response:
        return response.read(nbytes).count(b"\xff\xd8")


async def session(base, order, seconds, mock):
    url = (base.replace("http", "ws", 1) + "/api/v1/line/runs/live/live?source=camera" + (f"&order={order}" if order else "")
           + ("&simulate=mock" if mock else ""))
    out = dict(events=[], clocks=[], error=None, saved=None, previews={})
    start = time.monotonic()
    key, cameras, stopping = None, [], False
    async with websockets.connect(url, max_size=None) as ws:
        while True:
            try:
                message = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
            except asyncio.TimeoutError:
                message = {"type": "timeout"}
            kind = message["type"]
            if kind == "preview":
                key, cameras = message["key"], message["cameras"]
            elif kind == "events":
                out["events"] += message["events"]
            elif kind == "clock":
                out["clocks"].append(message)
            elif kind == "error":
                out["error"] = message.get("message")
                break
            elif kind == "saved":
                out["saved"] = message["run_id"]
            elif kind == "end":
                break
            # Read each MJPEG preview once the feeds are running, as the page shows them.
            if key and not out["previews"] and time.monotonic() - start > 30:
                for camera in cameras:
                    try:
                        out["previews"][camera] = await asyncio.to_thread(preview_frames, base, key, camera)
                    except Exception as error:  # reported as a failure below
                        out["previews"][camera] = f"error: {error}"
            if seconds and time.monotonic() - start > seconds and not stopping:
                stopping = True
                await ws.send(json.dumps({"stop": True}))
    out["wall_s"] = time.monotonic() - start
    return out


def vials(events):
    return {e["vial"]["seq"]: e["vial"] for e in events if e["type"] == "vial"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="http://127.0.0.1:8788")
    parser.add_argument("--schedule", help="recipe order CSV (seq,recipe_id) entered as the operator's recipe plan")
    parser.add_argument("--compare", help="a recorded run id whose stored verdicts the session should reproduce")
    parser.add_argument("--seconds", type=float, default=0, help="stop the session after this long (0: until the video ends)")
    parser.add_argument("--input", choices=("mock", "webcams"), default="mock")
    parser.add_argument("--max-lag", type=float, default=2.0)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    base = args.base.rstrip("/")

    health = call(base, "GET", "/health")
    runs = call(base, "GET", "/api/v1/line/runs")
    live = next((r for r in runs if r["id"] == "live"), None)
    failures = []
    if live is None or not live.get("models"):
        failures.append("no live station with models (START.sh --mock-fill ... and the model assets)")
    elif args.input == "mock" and not live.get("mock"):
        failures.append("the station has no mock cameras (START.sh --mock-fill VIDEO)")
    order, recipes = enter_order(base, args.schedule) if args.schedule else (None, None)
    result = asyncio.run(session(base, order, args.seconds, args.input == "mock")) if not failures else {}

    decided = vials(result.get("events", []))
    clocks = result.get("clocks", [])
    last = clocks[-1] if clocks else {}
    report = dict(base=base, health=health, live=live, order=order, recipe_order=recipes, error=result.get("error"),
                  wall_s=result.get("wall_s"), video_s=last.get("t"), max_lag_s=max((c["lag_s"] for c in clocks), default=None),
                  cameras=last.get("cameras"), previews=result.get("previews"), saved=result.get("saved"),
                  samples=sum(e["type"] == "sample" for e in result.get("events", [])),
                  actions={}, vials=[dict(seq=s, recipe=v.get("recipe_id"), verdict=v.get("verdict"), de=v.get("de_intended"),
                                          decision_s=v.get("decision_s")) for s, v in sorted(decided.items())])
    for e in result.get("events", []):
        if e["type"] == "action":
            kind = e["kind"]
            report["actions"][kind] = report["actions"].get(kind, 0) + 1
    if result.get("error"):
        failures.append(f"session error: {result['error']}")
    if result and not decided:
        failures.append("no vial was decided")
    if report["cameras"] and report["cameras"].get("dropped"):
        failures.append(f"{report['cameras']['dropped']} frames dropped")
    if report["max_lag_s"] is not None and report["max_lag_s"] > args.max_lag:
        failures.append(f"lag {report['max_lag_s']} s above {args.max_lag}")
    for camera, frames in (result.get("previews") or {}).items():
        if not isinstance(frames, int) or frames < 1:
            failures.append(f"{camera} preview: {frames}")
    if result and not result.get("previews"):
        failures.append("previews not read")
    if result and not result.get("saved"):
        failures.append("session was not saved as a run")
    elif result:
        saved_events = call(base, "GET", f"/api/v1/line/runs/{result['saved']}/events")["events"]
        report["review_vials"] = len(vials(saved_events))
        if report["review_vials"] != len(decided):
            failures.append(f"Review of the saved run has {report['review_vials']} vials, the session {len(decided)}")
    if args.compare and decided:
        reference = vials(call(base, "GET", f"/api/v1/line/runs/{args.compare}/events")["events"])
        compared = [s for s in decided if s in reference]
        differ = [dict(seq=s, session=decided[s]["verdict"], reference=reference[s]["verdict"],
                       session_de=decided[s].get("de_intended"), reference_de=reference[s].get("de_intended"))
                  for s in compared if decided[s]["verdict"] != reference[s]["verdict"]]
        report["compare"] = dict(run=args.compare, compared=len(compared), same_verdict=len(compared) - len(differ), differ=differ)
        if differ:
            failures.append(f"{len(differ)}/{len(compared)} verdicts differ from {args.compare}")
    report["failures"] = failures
    report["passed"] = not failures
    args.report.write_text(json.dumps(report, indent=1))
    print(f"{'PASSED' if not failures else 'FAILED'}: {len(decided)} vials decided, video {report['video_s']} s, "
          f"max lag {report['max_lag_s']} s, cameras {report['cameras']}, saved {report['saved']}")
    for v in report["vials"]:
        print(f"  vial {v['seq']:>2} recipe {v['recipe']}: {v['verdict']} dE {v['de']}")
    for failure in failures:
        print("FAIL:", failure)
    raise SystemExit(0 if not failures else 1)


if __name__ == "__main__":
    main()
