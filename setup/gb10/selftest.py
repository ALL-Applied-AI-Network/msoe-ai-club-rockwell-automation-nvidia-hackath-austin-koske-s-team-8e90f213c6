"""Reproducibility self-test: post the reference frames and compare per-frame decisions.

Standard library only. Expected values were produced by the released models on the x86 CPU
Docker path; they check that this machine reproduces them, not that they are accurate.
"""
import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

LAB_TOLERANCE = 1.0  # Device numerics; GB10 CUDA vs x86 CPU measured 0.59.


def call(base, method, path, body=None, content_type="application/json"):
    request = urllib.request.Request(base + path, data=body, method=method,
                                     headers={"content-type": content_type})
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.loads(response.read() or b"null")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8788")
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    expected = json.loads(args.expected.read_text())
    health = call(args.base_url, "GET", "/health")
    if health.get("pipeline_mode") != "model":
        sys.exit("Self-test needs model mode")
    session = call(args.base_url, "POST", "/api/v1/sessions", json.dumps({
        "run_name": "GB10 reproducibility self-test", "source_mode": "recording",
        "source_name": expected["source_name"],
        "recipe": {"current_recipe_id": "SELFTEST", "current_recipe_name": "Self-test (no recipe)"},
    }).encode())
    rows, timings = [], []
    for index, item in enumerate(expected["frames"]):
        content = (args.frames / item["file"]).read_bytes()
        if hashlib.sha256(content).hexdigest() != item["file_sha256"]:
            sys.exit(f"Reference frame changed: {item['file']}")
        start = time.perf_counter()
        event = call(args.base_url, "POST",
                     f"/api/v1/sessions/{session['session_id']}/frames?frame_id=selftest-{index:04d}"
                     f"&width=640&height=480&source_timestamp_ms={item['timestamp_ms']}&source_clock=recording_seek",
                     content, "image/png")
        timings.append((time.perf_counter() - start) * 1000)
        evidence = event["decision"]["evidence"]
        got = {"proposal": evidence["proposal_pixels"] > 0, "qa": bool(evidence["measurement_valid"]),
               "proposal_pixels": evidence["proposal_pixels"], "lab": evidence["lab"]}
        lab_delta = (max(abs(a - b) for a, b in zip(got["lab"], item["lab"]))
                     if got["lab"] and item["lab"] else None)
        ok = (got["proposal"] == item["proposal"] and got["qa"] == item["qa"]
              and (lab_delta is None or lab_delta <= LAB_TOLERANCE))
        rows.append({"file": item["file"], "ok": ok, "expected": {k: item[k] for k in got}, "got": got,
                     "proposal_pixel_delta": abs(got["proposal_pixels"] - item["proposal_pixels"]),
                     "lab_max_delta": lab_delta})
    agree = sum(row["ok"] for row in rows)
    deltas = [row["lab_max_delta"] for row in rows if row["lab_max_delta"] is not None]
    ordered = sorted(timings[1:] or timings)
    report = {
        "schema": "linewatch-gb10-selftest/v1", "passed": agree == len(rows),
        "frames": len(rows), "frames_agreeing": agree,
        "proposals": sum(row["got"]["proposal"] for row in rows),
        "qa_available": sum(row["got"]["qa"] for row in rows),
        "expected_proposals": sum(item["proposal"] for item in expected["frames"]),
        "expected_qa_available": sum(item["qa"] for item in expected["frames"]),
        "max_proposal_pixel_delta": max(row["proposal_pixel_delta"] for row in rows),
        "max_lab_delta": max(deltas) if deltas else None,
        "request_ms_median_after_first": ordered[len(ordered) // 2],
        "health": health, "session_id": session["session_id"],
        "note": "Reproducibility against frozen reference outputs; not accuracy or camera latency.",
        "rows": rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print(f"Self-test {'PASSED' if report['passed'] else 'FAILED'}: {agree}/{len(rows)} frames agree; "
          f"proposals {report['proposals']}/{report['expected_proposals']} expected, "
          f"QA {report['qa_available']}/{report['expected_qa_available']} expected; "
          f"max dLab {report['max_lab_delta']}; median {report['request_ms_median_after_first']:.0f} ms/frame")
    print(f"Report: {args.report}")
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
