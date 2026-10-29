"""Stage 5: fill the `explanation` of each RCA evidence packet and score the explainers.

    python -m replay.explain --run DIR --backend rule
    python -m replay.explain --run DIR --backend openai --url http://HOST:8000/v1 --model M [--video V]
    python -m replay.explain --run DIR --export-keyframes DIR/keyframes --video V     # needs cv2
    python -m replay.explain --run DIR --backend hf --model PATH_OR_ID [--keyframes DIR/keyframes]
    TYPESAFE_API_KEY=... python -m replay.explain --run DIR --backend jev [--keyframes DIR/keyframes]
    OPENAI_API_KEY=... python -m replay.explain --run DIR --backend decisions [--keyframes DIR/keyframes]

Backends see the same packets. `rule` is the deterministic baseline. `openai` calls any
OpenAI-compatible chat endpoint (vLLM serving Cosmos, or a text LLM). `hf` loads a Qwen2.5-VL
family model (Cosmos-Reason1-7B, Qwen2.5-VL-7B) with transformers on the local GPU; it needs no
cv2, so keyframes are exported first where cv2 exists. With --video or --keyframes, the empty /
mid-fill / fill-end keyframes (vial region) are attached as images.
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


def export_keyframes(run, video, out):
    out.mkdir(parents=True, exist_ok=True)
    for path in sorted((run / "rca").glob("vial-*.json")):
        for name, data in keyframes(video, json.loads(path.read_text())).items():
            (out / f"{path.stem}-{name}.jpg").write_bytes(base64.b64decode(data))


def load_keyframes(directory, path):
    return {name: base64.b64encode((directory / f"{path.stem}-{name}.jpg").read_bytes()).decode()
            for name in IMAGES if (directory / f"{path.stem}-{name}.jpg").is_file()}


class LocalVLM:
    """Qwen2.5-VL-architecture model (Cosmos-Reason1-7B, Qwen2.5-VL-7B-Instruct) via transformers."""

    def __init__(self, model, max_tokens):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        self.torch, self.max_tokens = torch, max_tokens
        self.processor = AutoProcessor.from_pretrained(model)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model, dtype=torch.bfloat16, device_map="cuda").eval()

    def __call__(self, packet, images):
        import io
        from PIL import Image
        body = {k: packet[k] for k in SECTIONS if k in packet}
        content, pictures = [{"type": "text", "text": "Evidence packet:\n" + json.dumps(body, indent=1)}], []
        for name, data in images.items():
            content += [{"type": "text", "text": f"image:{name}"}, {"type": "image"}]
            pictures.append(Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB"))
        messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]}, {"role": "user", "content": content}]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[prompt], images=pictures or None, return_tensors="pt").to("cuda")
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=self.max_tokens, do_sample=False)
        return self.processor.batch_decode(output[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]


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


JEV_URL = "https://api.typesafe.ai/v1/systemone"
CAUSE_CRITERIA = {
    "carryover": "Colour from the previous recipe remains in the line: endpoint shifted toward the previous recipe soon after a recipe change.",
    "wrong_recipe": "The line dosed a different recipe than intended: several consecutive vials miss the intended colour the same way.",
    "dose_error": "One tank dosed too much or too little: endpoint off the intended colour but not toward the previous recipe.",
    "missed_dose": "One dye was not dosed at all: the colour lacks one component entirely.",
    "short_fill": "Too little liquid in the vial to measure or release.",
    "occlusion": "The vial or liquid was hidden (arm, nozzle, glare) so the colour could not be measured.",
    "vial_position": "The vial was not in the expected position under the nozzle.",
    "lighting": "Lighting or exposure changed, so the measured colour is unreliable.",
    "bubbles": "Bubbles or foam disturbed the colour measurement.",
    "no_reference": "The intended recipe has no approved reference colour, so the vial cannot be verified.",
    "unknown": "The evidence does not support any single cause.",
}


def jev_image(data, minimum=280):
    """JPEG padded with black to Jev's minimum side (280 px); the vial crop itself is not resampled."""
    import io
    from PIL import Image
    image = Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")
    if min(image.size) >= minimum:
        return base64.b64decode(data)
    canvas = Image.new("RGB", (max(minimum, image.width), max(minimum, image.height)))
    canvas.paste(image, ((canvas.width - image.width) // 2, (canvas.height - image.height) // 2))
    out = io.BytesIO()
    canvas.save(out, "JPEG", quality=95)
    return out.getvalue()


def jev_explain(packet, args, images):
    """TypeSafe Jev System One: the cause as a `choice` question over CAUSE_CRITERIA.
    Jev returns a decision, not prose, so explanation/next_check stay empty (scored as missing)."""
    import uuid
    import requests
    body = {k: packet[k] for k in SECTIONS if k in packet}
    state, assets, files = {"system": SYSTEM.split("Answer with")[0].strip(), "packet": json.dumps(body, indent=1)}, {}, []
    for name, data in images.items():
        reference = str(uuid.uuid4())  # Jev replaces any text equal to a reference, so never reuse packet words
        state[name] = reference
        assets[reference] = {"type": "image", "part": name + "-file"}
        files.append((name + "-file", (name + ".jpg", jev_image(data), "image/jpeg")))
    seen = " and the keyframes " + ", ".join(f"`{n}`" for n in images) if images else ""
    questions = {"cause": {"type": "choice", "criteria": CAUSE_CRITERIA,
                           "instructions": f"Following `system`, what is the most likely cause of this vial's verdict, "
                                           f"given the evidence packet `packet`{seen}?"}}
    request = {"model": args.model, "state": state, "questions": questions}
    headers = {"Authorization": "Bearer " + os.environ[args.key_env]}
    if images:
        request["assets"] = assets
        response = requests.post(args.url or JEV_URL, timeout=(15, args.timeout), allow_redirects=False,
                                 headers={**headers, "TypeSafe-Danger-Zone-Images": "true"},
                                 files=[("request", (None, json.dumps(request), "application/json"))] + files)
    else:
        response = requests.post(args.url or JEV_URL, timeout=(15, args.timeout), allow_redirects=False,
                                 headers=headers, json=request)
    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:500]}")
    answer = response.json()["answers"]["cause"]
    return json.dumps({"cause": answer.get("choice"), "confidence": answer.get("confidence", 0.0),
                       "evidence_ids": [k for k in body] + [f"image:{n}" for n in images],
                       "explanation": None, "next_check": None, "jev_answer": answer})


def decisions_explain(packet, args, images):
    """OpenAI Decisions API (gpt-6-luna): the same cause choice as Jev, through POST /v1/decisions."""
    body = {k: packet[k] for k in SECTIONS if k in packet}
    content = [{"type": "input_text", "text": "Evidence packet:\n" + json.dumps(body, indent=1)}]
    for name, data in images.items():
        content.append({"type": "input_text", "text": f"image:{name}"})
        content.append({"type": "input_image", "image_url": f"data:image/jpeg;base64,{data}", "detail": "high"})
    question = {"type": "choice", "name": "cause",
                "instructions": "What is the most likely cause of this vial's verdict, given the evidence packet"
                                + (" and the keyframes?" if images else "?"),
                "choices": [{"value": k, "description": v} for k, v in CAUSE_CRITERIA.items()]}
    request = urllib.request.Request(
        (args.url or "https://api.openai.com/v1").rstrip("/") + "/decisions",
        data=json.dumps({"model": args.model, "questions": [question],
                         "input": [{"role": "system", "content": SYSTEM.split("Answer with")[0].strip()},
                                   {"role": "user", "content": content}]}).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {os.environ[args.key_env]}"})
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        answer = json.loads(response.read())["answers"][0]
    if answer["type"] == "refusal":
        return json.dumps({"cause": None, "decision_answer": answer})
    return json.dumps({"cause": answer["choice"], "confidence": answer["confidence"],
                       "evidence_ids": [k for k in body] + [f"image:{n}" for n in images],
                       "explanation": None, "next_check": None, "decision_answer": answer})


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


def contradictions(packet, explanation):
    """Claims the packet's own numbers rule out."""
    ctx, c, cause = packet["context"], packet["comparison"], explanation["cause"]
    found = []
    if cause == "carryover" and not ctx.get("previous_recipe"):
        found.append("carryover_without_previous_recipe")
    elif cause == "carryover" and c.get("direction_cos") is None:
        found.append("carryover_without_colour_comparison")
    if cause == "carryover" and c.get("direction_cos") is not None and c["direction_cos"] < 0.5:
        found.append("carryover_but_colour_not_toward_previous")
    if cause == "no_reference" and c.get("de_intended") is not None:
        found.append("no_reference_but_reference_used")
    return found


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
        "consistent_with_packet": sum(not contradictions(r, r["explanation"]) for r in results),
        "contradictions": {f"vial {r['context']['seq']}": contradictions(r, r["explanation"])
                           for r in results if contradictions(r, r["explanation"])},
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
    parser.add_argument("--backend", choices=("rule", "openai", "hf", "jev", "decisions"), default="rule")
    parser.add_argument("--name", help="output suffix (default: backend, or backend-model)")
    parser.add_argument("--url", help="OpenAI-compatible base URL, e.g. http://gb10:8000/v1")
    parser.add_argument("--model")
    parser.add_argument("--key-env", default="OPENAI_API_KEY", help="environment variable holding the API key")
    parser.add_argument("--video", type=Path, help="attach keyframe crops (for VLMs such as Cosmos)")
    parser.add_argument("--keyframes", type=Path, help="directory from --export-keyframes (instead of --video)")
    parser.add_argument("--export-keyframes", type=Path, help="write keyframe crops for every packet and exit")
    parser.add_argument("--text-only", action="store_true", help="send no images (numbers only)")
    parser.add_argument("--rescore", type=Path, help="recompute metrics.json for an existing rca-<name> directory")
    parser.add_argument("--truth", type=Path, help="CSV seq,cause (synthetic sessions) for cause accuracy")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    if args.rescore:
        results = [json.loads(p.read_text()) for p in sorted(args.rescore.glob("vial-*.json"))]
        summary = score(results)
        (args.rescore / "metrics.json").write_text(json.dumps(summary, indent=1))
        print(json.dumps(summary, indent=1))
        return
    if args.export_keyframes:
        export_keyframes(args.run, args.video, args.export_keyframes)
        return
    if args.backend == "openai" and not (args.url and args.model):
        parser.error("--url and --model are required for --backend openai")
    if args.backend == "hf" and not args.model:
        parser.error("--model is required for --backend hf")
    if args.backend == "jev":
        args.model = args.model or "jev-1.14.0"
        if args.key_env == "OPENAI_API_KEY":
            args.key_env = "TYPESAFE_API_KEY"
        if not os.environ.get(args.key_env):
            parser.error(f"--backend jev needs the API key in ${args.key_env}")
    if args.backend == "decisions":
        args.model = args.model or "gpt-6-luna"
        if not os.environ.get(args.key_env):
            parser.error(f"--backend decisions needs the API key in ${args.key_env}")

    name = args.name or (args.backend if args.backend == "rule" else f"{args.backend}-{Path(args.model).name}")
    local = LocalVLM(args.model, args.max_tokens) if args.backend == "hf" else None
    out = args.run / f"rca-{name}"
    out.mkdir(exist_ok=False)
    results = []
    for path in sorted((args.run / "rca").glob("vial-*.json")):
        packet = json.loads(path.read_text())
        images = {}
        if args.backend != "rule" and not args.text_only:
            images = load_keyframes(args.keyframes, path) if args.keyframes else \
                keyframes(args.video, packet) if args.video else {}
        allowed = set(SECTIONS) | {f"image:{k}" for k in images}
        started = time.perf_counter()
        try:
            raw = rule_explain(packet) if args.backend == "rule" else local(packet, images) if local \
                else jev_explain(packet, args, images) if args.backend == "jev" \
                else decisions_explain(packet, args, images) if args.backend == "decisions" else openai_explain(packet, args, images)
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
