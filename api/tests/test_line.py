import json
from pathlib import Path

import pytest

pytest.importorskip("cv2")
pytest.importorskip("skimage")

from fastapi.testclient import TestClient

from dependencies import get_line_runs
from main import app
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
    (data / "library.json").write_text(json.dumps({"recipes": [{"id": "1", "reference": {"lab": [50.0, 10.0, 10.0]}}]}))
    (data / "fill.mp4").write_bytes(b"\x00" * 1000)
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "demo.json").write_text(json.dumps({
        "title": "Demo", "fill_video": "../data/fill.mp4", "schedule": "../data/schedule.csv",
        "library": "../data/library.json", "fill_samples": "../data/frames.jsonl"}))
    app.dependency_overrides[get_line_runs] = lambda: LineRuns(runs)
    with TestClient(app) as test_client:
        yield test_client
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
