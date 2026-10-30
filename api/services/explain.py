"""AI explanation of one decided vial, on request (the LINE page's Explain button).

The evidence packet is the one the offline explainer evaluation used (replay/explain.py): context,
measurement, comparison, neighbours (vials decided so far) and actions. It goes to an
OpenAI-compatible chat endpoint, by default Nemotron 3 Ultra on build.nvidia.com (18/20 consistent
with the packet on Run 5, docs/evaluation.md). The answer is advisory: it never changes a verdict
or action, and claims the packet's own numbers rule out are returned as `contradictions`.

The API key comes from NVIDIA_API_KEY, or from APP__EXPLAIN_KEY_FILE (a KEY=VALUE file), and is
never logged or returned. Without a key or network, a run whose library folder holds
explanations.json (recorded answers, e.g. Run 5) returns the recorded answer for the same vial.
"""
from __future__ import annotations

import json
import os
import threading
import time
from argparse import Namespace
from pathlib import Path

KEY_ENV = "NVIDIA_API_KEY"


class Explainer:
    def __init__(self, url: str, model: str, key_file: Path | None = None, timeout: float = 120):
        self.url, self.model, self.key_file, self.timeout = url, model, key_file, timeout
        self._cache: dict[str, dict] = {}
        self._lock = threading.Lock()

    def _key(self):
        if os.environ.get(KEY_ENV):
            return os.environ[KEY_ENV]
        if self.key_file and self.key_file.is_file():
            for line in self.key_file.read_text().splitlines():
                name, _, value = line.strip().removeprefix("export ").partition("=")
                if name == KEY_ENV and value:
                    return value.strip().strip("'\"")
        return None

    def available(self):
        return {"model": self.model, "live": self._key() is not None}

    def explain(self, packet: dict, recorded: Path | None = None) -> dict:
        cache = json.dumps(packet, sort_keys=True)
        with self._lock:
            if cache in self._cache:
                return self._cache[cache]
        from replay.explain import SECTIONS, contradictions, openai_explain, parse  # the line modules, not in every image
        key, problem = self._key(), None
        if key:
            started = time.perf_counter()
            try:
                raw = openai_explain(packet, Namespace(url=self.url, model=self.model, max_tokens=6000,
                                                       timeout=self.timeout, key=key, key_env=KEY_ENV), {})
                explanation, problems = parse(raw, set(SECTIONS))
                result = {"source": "live", "model": self.model, "latency_s": round(time.perf_counter() - started, 1),
                          "explanation": explanation, "problems": problems,
                          "contradictions": contradictions(packet, explanation)}
                if "not_json" not in problems:
                    with self._lock:
                        self._cache[cache] = result
                    return result
                problem = "The model's answer was not valid JSON"
            except Exception as error:  # network down, timeout, quota: fall back to a recorded answer
                problem = f"{self.model} unavailable ({type(error).__name__})"
        else:
            problem = f"No {KEY_ENV} on this server"
        stored = _recorded(recorded, packet)
        if stored:
            return {**stored, "source": "recorded", "problems": [], "note": problem,
                    "contradictions": contradictions(packet, stored["explanation"])}
        raise LookupError(problem)


def _recorded(path: Path | None, packet: dict):
    """The recorded answer for this vial, if the file has the same vial with the same recipe and verdict."""
    if not path or not path.is_file():
        return None
    data = json.loads(path.read_text())
    vial = data["vials"].get(str(packet["context"]["seq"]))
    if not vial or vial["recipe_id"] != packet["context"]["intended_recipe"] \
            or vial["verdict"] != packet["comparison"]["verdict"]:
        return None
    return {"model": data["model"], "recorded": data.get("recorded"), "latency_s": vial.get("latency_s"),
            "explanation": vial["explanation"]}


def packet(vial: dict, actions: list[dict], decided: list[dict], next_recipe: str | None) -> dict:
    """The replay.line RCA packet for a vial event, from what the live engine has emitted so far."""
    lab = vial.get("endpoint_lab")
    return {
        "context": {"seq": vial["seq"], "intended_recipe": vial["recipe_id"],
                    "previous_recipe": vial.get("previous_recipe_id"), "next_recipe": next_recipe},
        "measurement": {"endpoint_lab": lab, "endpoint_samples": vial.get("endpoint_samples"),
                        "fill_start_s": vial.get("fill_start_s"), "fill_end_s": vial.get("fill_end_s"),
                        "final_liquid_px": vial.get("final_liquid_px")},
        "comparison": {k: vial.get(k) for k in ("de_intended", "de_previous", "direction_cos", "verdict", "pattern")},
        "neighbours": [{k: v.get(k) for k in ("seq", "recipe_id", "verdict", "de_intended")}
                       for v in decided if v["seq"] != vial["seq"] and abs(v["seq"] - vial["seq"]) <= 2],
        "actions": [{k: a.get(k) for k in ("kind", "seq", "issued_s", "deadline_s", "reason", "plc_command")}
                    for a in actions if a.get("seq") == vial["seq"]],
    }
