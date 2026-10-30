import json
from pathlib import Path

import pytest

from fastapi.testclient import TestClient

from dependencies import get_explainer, get_line_runs
from main import app
import replay.explain as explain_module
from services.explain import KEY_ENV, Explainer
from services.line import LineRuns

VIAL = {"seq": 4, "recipe_id": "6", "previous_recipe_id": "5", "verdict": "OUT_OF_SPEC", "pattern": "carryover",
        "de_intended": 24.0, "de_previous": 8.2, "direction_cos": 0.93, "endpoint_lab": [60.0, 30.0, 40.0]}
ACTIONS = [{"kind": "SCRAP", "seq": 4, "issued_s": 300.0, "deadline_s": 420.0, "reason": "ΔE 24.0 to recipe 6",
            "plc_command": "DIVERT_SCRAP"},
           {"kind": "PASS", "seq": 3, "issued_s": 250.0, "deadline_s": None, "reason": "", "plc_command": "ROUTE_GOOD"}]
ANSWER = {"cause": "carryover", "confidence": 0.9, "evidence_ids": ["comparison"],
          "explanation": "Shifted toward recipe 5.", "next_check": "Run a flush."}


@pytest.fixture
def setup(tmp_path: Path, monkeypatch):
    monkeypatch.delenv(KEY_ENV, raising=False)
    (tmp_path / "library.json").write_text(json.dumps({"recipes": []}))
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "demo.json").write_text(json.dumps({"title": "Demo", "library": "../library.json"}))
    explainer = Explainer("http://model.invalid/v1", "nvidia/test-model")
    app.dependency_overrides[get_line_runs] = lambda: LineRuns(runs)
    app.dependency_overrides[get_explainer] = lambda: explainer
    with TestClient(app) as client:
        yield client, tmp_path
    app.dependency_overrides.clear()


def ask(client, vial=VIAL):
    return client.post("/api/v1/line/explain", json={"run_id": "demo", "vial": vial, "actions": ACTIONS,
                                                       "decided": [{**VIAL, "seq": 3, "verdict": "IN_SPEC"}, vial]})


def test_no_key_and_no_recording_is_unavailable(setup):
    client, _ = setup
    assert client.get("/api/v1/line/explain").json() == {"model": "nvidia/test-model", "live": False}
    response = ask(client)
    assert response.status_code == 503 and KEY_ENV in response.json()["detail"]


def test_recorded_answer_only_for_the_same_vial(setup):
    client, folder = setup
    (folder / "explanations.json").write_text(json.dumps({"model": "nvidia/recorded", "recorded": "2026-10-30", "vials": {
        "4": {"recipe_id": "6", "verdict": "OUT_OF_SPEC", "latency_s": 20.0, "explanation": ANSWER}}}))
    body = ask(client).json()
    assert body["source"] == "recorded" and body["explanation"]["cause"] == "carryover" and body["contradictions"] == []
    assert ask(client, {**VIAL, "recipe_id": "7"}).status_code == 503


def test_live_call_gets_packet_and_key(setup, monkeypatch):
    client, _ = setup
    monkeypatch.setenv(KEY_ENV, "test-key")
    seen = {}

    def fake(packet, args, images):
        seen.update(packet=packet, key=args.key, model=args.model)
        return json.dumps({**ANSWER, "cause": "carryover"})

    monkeypatch.setattr(explain_module, "openai_explain", fake)
    body = ask(client).json()
    assert body["source"] == "live" and body["problems"] == [] and seen["key"] == "test-key"
    assert seen["model"] == "nvidia/test-model"
    packet = seen["packet"]
    assert packet["context"] == {"seq": 4, "intended_recipe": "6", "previous_recipe": "5", "next_recipe": None}
    assert [a["kind"] for a in packet["actions"]] == ["SCRAP"]
    assert [n["seq"] for n in packet["neighbours"]] == [3]
    assert "test-key" not in json.dumps(body)


def test_answer_the_numbers_rule_out_is_flagged(setup, monkeypatch):
    client, _ = setup
    monkeypatch.setenv(KEY_ENV, "test-key")
    monkeypatch.setattr(explain_module, "openai_explain", lambda *a: json.dumps(ANSWER))
    body = ask(client, {**VIAL, "direction_cos": 0.1, "pattern": "deviation"}).json()
    assert body["contradictions"] == ["carryover_but_colour_not_toward_previous"]
