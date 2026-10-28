"""Stage 5: fill the `explanation` of each RCA evidence packet and score the explainers.

    python -m replay.explain --run DIR --backend rule
    python -m replay.explain --run DIR --backend openai --url http://HOST:8000/v1 --model M [--video V]

Backends see the same packets. `rule` is the deterministic baseline. `openai` calls any
OpenAI-compatible chat endpoint: vLLM serving Cosmos Reason2-2B (+ LoRA) on the GB10, or a text
LLM. With --video, the empty / mid-fill / fill-end keyframes (vial region) are attached as images.
The explainer never changes a verdict or action; output that fails the schema is recorded as
cause "unknown". Results go to DIR/rca-<backend>/ (packets are not modified) with a metrics file.
"""
import argparse
import base64
import json
import os
import re
import time
import urllib.request
from collections import Counter
from pathlib import Path

CAUSES = ("carryover", "wrong_recipe", "dose_error", "missed_dose", "short_fill", "occlusion",
          "vial_position", "lighting", "bubbles", "no_reference", "unknown")
SECTIONS = ("context", "measurement", "comparison", "neighbours", "keyframes", "actions")
IMAGES = ("reference_frame", "mid_fill_frame", "fill_end_frame")
ROI = (256, 240, 368, 432)

SYSTEM = f"""You are the root-cause analyst for a vial filling line. Each vial is filled from three
dye tanks according to a recipe; the line switches recipes without cleaning, so the first vials
after a change can carry colour from the previous recipe. A water vial flushes the line.
You receive an evidence packet: deterministic measurements, ΔE2000 colour distances, a direction
cosine (1 = colour shifted exactly toward the previous recipe), neighbouring vials, QA flags and
optionally keyframe images of the vial. The verdict and actions are already decided; do not change
them. Explain the most likely cause.
Answer with one JSON object and nothing else:
{{"cause": one of {list(CAUSES)},
 "confidence": number 0-1,
 "evidence_ids": list of the packet sections or images you relied on, from {list(SECTIONS) + ['image:' + i for i in IMAGES]},
 "explanation": two or three sentences for an operator, citing numbers,
 "next_check": one concrete thing the operator should check}}"""


def rule_explain(packet):
    """Deterministic baseline from the stage-4 pattern and QA evidence."""
    c, m = packet["comparison"], packet["measurement"]
    ctx = packet["context"]
    pattern = c["pattern"]
    flags = m.get("qa_flags") or {}
    if pattern == "carryover":
        return {"cause": "carryover", "confidence": min(0.95, 0.5 + 0.5 * c["direction_cos"]),
                "evidence_ids": ["comparison", "context"],
                "explanation": f"Endpoint is ΔE {c['de_intended']:.1f} from recipe {ctx['intended_recipe']} and shifted "
                               f"toward the previous recipe {ctx['previous_recipe']} (cosine {c['direction_cos']:.2f}).",
                "next_check": "Confirm the water flush ran before the next fill."}
    if pattern == "persistent_deviation":
        return {"cause": "wrong_recipe", "confidence": 0.6, "evidence_ids": ["comparison", "neighbours"],
                "explanation": f"Several consecutive vials miss recipe {ctx['intended_recipe']} without moving toward "
                               f"the previous recipe (ΔE {c['de_intended']:.1f}).",
                "next_check": "Check the recipe loaded on the HMI and the tank volumes."}
    if pattern == "deviation":
        return {"cause": "dose_error", "confidence": 0.4, "evidence_ids": ["comparison"],
                "explanation": f"Endpoint is ΔE {c['de_intended']:.1f} from recipe {ctx['intended_recipe']} but not "
                               f"toward the previous recipe.",
                "next_check": "Check the dosed volumes for this vial."}
    if pattern == "no_reference":
        return {"cause": "no_reference", "confidence": 0.9, "evidence_ids": ["context"],
                "explanation": f"Recipe {ctx['intended_recipe']} has no approved reference colour, so the vial "
                               f"cannot be verified.",
                "next_check": "Record a reference vial for this recipe."}
    top = max(flags, key=flags.get) if flags else None
    cause = "occlusion" if top and "occlu" in top else "short_fill" if (m.get("final_liquid_px") or 0) < 1500 else "unknown"
    return {"cause": cause, "confidence": 0.3, "evidence_ids": ["measurement"],
            "explanation": f"No QA-valid colour in the endpoint window ({m['qa_valid_samples']}/{m['samples']} valid "
                           f"samples{', most common flag ' + top if top else ''}).",
            "next_check": "Inspect the vial by eye before release."}


def keyframes(video, packet):
    import cv2
    capture = cv2.VideoCapture(str(video))
    images = {}
    for name in IMAGES:
        frame = packet["keyframes"].get(name)
        if frame is None:
            continue
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, bgr = capture.read()
        if ok:
            x0, y0, x1, y1 = ROI
            crop = cv2.resize(bgr[y0:y1, x0:x1], None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            images[name] = base64.b64encode(cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])[1]).decode()
    capture.release()
    return images


def openai_explain(packet, args, images):
    body = {k: packet[k] for k in SECTIONS if k in packet}
    content = [{"type": "text", "text": "Evidence packet:\n" + json.dumps(body, indent=1)}]
    for name, data in images.items():
        content.append({"type": "text", "text": f"image:{name}"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}})
    request = urllib.request.Request(
        args.url.rstrip("/") + "/chat/completions",
        data=json.dumps({"model": args.model, "temperature": 0, "max_tokens": args.max_tokens,
                         "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]}).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {os.environ.get(args.key_env, 'none')}"})
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        return json.loads(response.read())["choices"][0]["message"]["content"]


def parse(text, allowed_ids):
    """Validate model output against the schema. Returns (explanation, problems)."""
    if isinstance(text, dict):
        obj = text
    else:
        text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
        match = re.search(r"\{.*\}", text, re.S)
        try:
            obj = json.loads(match.group(0)) if match else None
        except json.JSONDecodeError:
            obj = None
    if not isinstance(obj, dict):
        return {"cause": "unknown", "confidence": 0.0, "evidence_ids": [], "explanation": None, "next_check": None}, ["not_json"]
    problems = []
    if obj.get("cause") not in CAUSES:
        problems.append("cause_not_in_taxonomy")
        obj["cause"] = "unknown"
    try:
        obj["confidence"] = min(1.0, max(0.0, float(obj.get("confidence", 0))))
    except (TypeError, ValueError):
        problems.append("bad_confidence")
        obj["confidence"] = 0.0
    ids = obj.get("evidence_ids") if isinstance(obj.get("evidence_ids"), list) else []
    if not ids or any(i not in allowed_ids for i in ids):
        problems.append("ungrounded_evidence")
    obj["evidence_ids"] = [i for i in ids if i in allowed_ids]
    for key in ("explanation", "next_check"):
        if not isinstance(obj.get(key), str) or not obj[key].strip():
            problems.append(f"missing_{key}")
    return {k: obj.get(k) for k in ("cause", "confidence", "evidence_ids", "explanation", "next_check")}, problems


def score(results, truth=None):
    n = len(results)
    rule = [r["rule_baseline"]["cause"] for r in results]
    causes = [r["explanation"]["cause"] for r in results]
    summary = {
        "packets": n,
        "schema_valid": sum(not r["explanation_meta"]["problems"] for r in results),
        "grounded": sum("ungrounded_evidence" not in r["explanation_meta"]["problems"] for r in results),
        "causes": dict(Counter(causes)),
        "agrees_with_stage4_pattern": "{}/{}".format(
            sum(c == rule_map(r) for c, r in zip(causes, results) if rule_map(r)), sum(bool(rule_map(r)) for r in results)),
        "latency_s_median": sorted(r["explanation_meta"]["latency_s"] for r in results)[n // 2] if n else None,
        "stage4_rule_causes": dict(Counter(rule)),
    }
    if truth:
        scored = [(r["context"]["seq"], c) for r, c in zip(results, causes) if r["context"]["seq"] in truth]
        summary["cause_accuracy"] = f"{sum(truth[s] == c for s, c in scored)}/{len(scored)}"
        summary["confusion"] = dict(Counter(f"{truth[s]}->{c}" for s, c in scored))
    return summary


def rule_map(packet):
    """The cause implied by the stage-4 pattern alone (what an explainer should not contradict without evidence)."""
    return {"carryover": "carryover", "persistent_deviation": "wrong_recipe", "no_reference": "no_reference"}.get(
        packet["comparison"]["pattern"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True, help="output directory of python -m replay.line")
    parser.add_argument("--backend", choices=("rule", "openai"), default="rule")
    parser.add_argument("--name", help="output suffix (default: backend, or backend-model)")
    parser.add_argument("--url", help="OpenAI-compatible base URL, e.g. http://gb10:8000/v1")
    parser.add_argument("--model")
    parser.add_argument("--key-env", default="OPENAI_API_KEY", help="environment variable holding the API key")
    parser.add_argument("--video", type=Path, help="attach keyframe crops (for VLMs such as Cosmos)")
    parser.add_argument("--truth", type=Path, help="CSV seq,cause (synthetic sessions) for cause accuracy")
    parser.add_argument("--max-tokens", type=int, default=600)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    if args.backend == "openai" and not (args.url and args.model):
        parser.error("--url and --model are required for --backend openai")

    name = args.name or (args.backend if args.backend == "rule" else f"{args.backend}-{Path(args.model).name}")
    out = args.run / f"rca-{name}"
    out.mkdir(exist_ok=False)
    results = []
    for path in sorted((args.run / "rca").glob("vial-*.json")):
        packet = json.loads(path.read_text())
        images = keyframes(args.video, packet) if args.video and args.backend != "rule" else {}
        allowed = set(SECTIONS) | {f"image:{k}" for k in images}
        started = time.perf_counter()
        try:
            raw = rule_explain(packet) if args.backend == "rule" else openai_explain(packet, args, images)
            explanation, problems = parse(raw, allowed)
        except Exception as error:  # endpoint down or timeout: record, keep going
            raw, explanation, problems = repr(error), parse(None, allowed)[0], ["request_failed"]
        packet["explanation"] = explanation
        packet["explanation_meta"] = {"backend": args.backend, "model": args.model, "images": sorted(images),
                                      "latency_s": round(time.perf_counter() - started, 3), "problems": problems,
                                      "raw": raw if isinstance(raw, str) else None}
        (out / path.name).write_text(json.dumps(packet, indent=1))
        results.append(packet)
        print(f"vial {packet['context']['seq']:2d}: {explanation['cause']:<13} {problems or ''}", flush=True)

    truth = None
    if args.truth:
        import csv
        with open(args.truth) as handle:
            truth = {int(r["seq"]): r["cause"] for r in csv.DictReader(handle)}
    summary = score(results, truth)
    (out / "metrics.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
