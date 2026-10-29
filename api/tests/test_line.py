import json
from pathlib import Path

import pytest

pytest.importorskip("cv2")
pytest.importorskip("skimage")

from fastapi.testclient import TestClient

from databridge.sqlite_inspections import SQLiteInspectionBridge
from dependencies import get_inspection_service, get_line_runs
from main import app
from pipelines.mock import MockInspectionPipeline
from services.artifacts import ArtifactStore
from services.event_broker import EventBroker
from services.inspection import InspectionService
from services.line import LineRuns

FPS = 30.0


def dwell(start_s, lab):
    start, end = int(start_s * FPS), int((start_s + 20) * FPS)
    samples = [{"frame": f, "t": f / FPS, "presence": 1.0, "vial_boxes": 1, "liquid_px": min(5000, 400 * k),
                "qa_valid": True, "flags": [], "lab": lab, "rgb": None}
               for k, f in enumerate(range(start + 23, end, 15))]
    return {"start_frame": start, "end_frame": end, "reference_frame": start + 8, "samples": samples, "fps": FPS}


@pytest.fixture
def client(tmp_path: Path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "frames.jsonl").write_text("\n".join(json.dumps(dwell(s, [50.0, 10.0, 10.0])) for s in (0, 30)))
    (data / "schedule.csv").write_text("seq,recipe_id\n1,1\n2,1\n")
    (data / "library.json").write_text(json.dumps({"recipes": [
        {"id": "1", "t100_ml": 20.0, "t400_ml": 50.0, "order": "T400>T100", "reference": {"lab": [50.0, 10.0, 10.0]}},
        {"id": "2", "t200_ml": 30.0, "order": "T200", "reference": {"lab": [50.0, -40.0, 30.0]}}]}))
    (data / "fill.mp4").write_bytes(b"\x00" * 1000)
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "demo.json").write_text(json.dumps({
        "title": "Demo", "fill_video": "../data/fill.mp4", "schedule": "../data/schedule.csv",
        "library": "../data/library.json", "fill_samples": "../data/frames.jsonl"}))
    bridge = SQLiteInspectionBridge(tmp_path / "linewatch.sqlite3")
    service = InspectionService(bridge, MockInspectionPipeline(), EventBroker(), ArtifactStore(tmp_path / "artifacts"))
    app.dependency_overrides[get_line_runs] = lambda: LineRuns(runs)
    app.dependency_overrides[get_inspection_service] = lambda: service
    with TestClient(app) as test_client:
        yield test_client
        test_client.portal.call(service.stop)
        test_client.portal.call(bridge.close)
    app.dependency_overrides.clear()


def test_lists_runs_and_reviews_events(client):
    assert client.get("/api/v1/line/runs").json() == [
        {"id": "demo", "title": "Demo", "sources": ["recorded"], "has_tray": False}]
    events = client.get("/api/v1/line/runs/demo/events").json()["events"]
    assert events[0]["type"] == "meta"
    assert [e["vial"]["verdict"] for e in events if e["type"] == "vial"] == ["IN_SPEC", "IN_SPEC"]
    assert client.get("/api/v1/line/runs/missing/events").status_code == 404
    assert client.get("/api/v1/line/runs/../demo/events").status_code == 404


def test_video_supports_range_requests(client):
    response = client.get("/api/v1/line/runs/demo/video/fill", headers={"Range": "bytes=0-9"})
    assert response.status_code == 206 and len(response.content) == 10
    assert client.get("/api/v1/line/runs/demo/video/tray").status_code == 404


def test_live_session_paces_events_and_ends(client):
    with client.websocket_connect("/api/v1/line/runs/demo/live?source=recorded&speed=32") as ws:
        messages = []
        while not messages or messages[-1]["type"] not in ("end", "error"):
            messages.append(ws.receive_json())
    assert messages[0]["type"] == "meta"
    assert messages[-1]["type"] == "end"
    events = [e for m in messages if m["type"] == "events" for e in m["events"]]
    assert [e["vial"]["seq"] for e in events if e["type"] == "vial"] == [1, 2]
    assert any(m["type"] == "clock" for m in messages)


def test_live_rejects_unavailable_source(client):
    with client.websocket_connect("/api/v1/line/runs/demo/live?source=video") as ws:
        assert ws.receive_json()["type"] == "error"


def recipe(name, **formula):
    return {"current_recipe_id": name, "current_recipe_name": name, "vial_index": 1, **formula}


def plan_run(client, recipes):
    session = client.post("/api/v1/sessions", json={"source_mode": "live_camera", "recipe": recipes[0]}).json()
    resource = f"/api/v1/sessions/{session['session_id']}/recipe-plan"
    assert client.put(resource, json={"recipes": recipes}).status_code == 200
    return session["session_id"], resource


def test_review_follows_an_inspection_run_recipe_plan(client):
    # Vial 1 is ordered by formula under another name (-> recipe 1), vial 2 is recipe 2.
    order, _ = plan_run(client, [recipe("pale-blue", t100_ml=20, t400_ml=50, dispense_order="T400>T100"),
                                 recipe("recipe-02")])
    assert [o["session_id"] for o in client.get("/api/v1/line/orders").json()] == [order]
    events = client.get(f"/api/v1/line/runs/demo/events?order={order}").json()["events"]
    assert events[0]["schedule"] == ["1", "2"]
    vials = [e["vial"] for e in events if e["type"] == "vial"]
    assert [(v["recipe_id"], v["verdict"]) for v in vials] == [("1", "IN_SPEC"), ("2", "OUT_OF_SPEC")]
    assert client.get("/api/v1/line/runs/demo/events?order=not-a-run").status_code == 404


def test_live_session_picks_up_vials_ordered_while_running(client):
    order, resource = plan_run(client, [recipe("1")])
    with client.websocket_connect(f"/api/v1/line/runs/demo/live?source=recorded&speed=1&order={order}") as ws:
        assert ws.receive_json()["schedule"] == ["1"]
        assert client.put(resource, json={"recipes": [recipe("1"), recipe("2")]}).status_code == 200
        ws.send_json({"speed": 32})
        messages = []
        while not messages or messages[-1]["type"] not in ("end", "error"):
            messages.append(ws.receive_json())
    events = [e for m in messages if m["type"] == "events" for e in m["events"]]
    assert any(e["type"] == "schedule" and e["schedule"] == ["1", "2"] for e in events)
    assert [e["vial"]["recipe_id"] for e in events if e["type"] == "vial"] == ["1", "2"]
