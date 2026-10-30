import json
from pathlib import Path
from types import SimpleNamespace

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


def station_client(tmp_path, station):
    (tmp_path / "station.json").write_text(json.dumps(station))
    runs = LineRuns(None, station=tmp_path / "station.json", site=tmp_path / "site" / "site-recipes.json")
    app.dependency_overrides[get_line_runs] = lambda: runs
    return TestClient(app)


def test_station_offers_mock_cameras(client, tmp_path):
    library = str(tmp_path / "data" / "library.json")
    with station_client(tmp_path, {"title": "CSI", "fill_camera": "/dev/video9", "library": library,
                                   "mock": {"fill_camera": "file:/videos/run5.mp4"}}) as test_client:
        live = test_client.get("/api/v1/line/runs").json()[0]
        assert (live["id"], live["mock"], live["models"]) == ("live", True, False)
        # The mock path is taken (and needs the models, like the station cameras).
        with test_client.websocket_connect("/api/v1/line/runs/live/live?source=camera&simulate=mock") as ws:
            messages = [ws.receive_json()]
            while messages[-1]["type"] not in ("end", "error"):
                messages.append(ws.receive_json())
        assert "models" in messages[-1]["message"]


def test_mock_cameras_need_a_mock_entry(client, tmp_path):
    with station_client(tmp_path, {"title": "CSI", "fill_camera": "/dev/video9",
                                   "library": str(tmp_path / "data" / "library.json")}) as test_client:
        assert test_client.get("/api/v1/line/runs").json()[0]["mock"] is False
        with test_client.websocket_connect("/api/v1/line/runs/live/live?source=camera&simulate=mock") as ws:
            messages = [ws.receive_json()]
            while messages[-1]["type"] not in ("end", "error"):
                messages.append(ws.receive_json())
        assert "No mock cameras" in messages[-1]["message"]


def test_recipes_created_at_the_line_and_create_mode(client, tmp_path):
    library = str(tmp_path / "data" / "library.json")
    with station_client(tmp_path, {"title": "CSI", "fill_camera": "/dev/video9", "library": library}) as test_client:
        created = test_client.post("/api/v1/line/recipes", json={"name": "Blue 50", "t400_ml": 50, "t100_ml": 50})
        assert created.status_code == 201
        recipe = created.json()
        assert (recipe["id"], recipe["t100_ml"], recipe["t400_ml"], recipe["order"], recipe["reference"]) == (
            "custom-blue-50", 50.0, 50.0, "T100>T400", None)
        listed = test_client.get("/api/v1/line/recipes").json()
        assert [r["id"] for r in listed["recipes"]] == ["1", "2", "custom-blue-50"] and listed["capture"] is None
        assert test_client.post("/api/v1/line/recipes", json={"name": "Blue 50", "t400_ml": 50}).json()["id"] == "custom-blue-50-2"
        assert test_client.post("/api/v1/line/recipes", json={"name": "Nothing"}).status_code == 400
        assert test_client.put("/api/v1/line/recipes/capture", json={"recipe_id": "custom-blue-50"}).json() == {"capture": "custom-blue-50"}
        assert test_client.get("/api/v1/line/recipes").json()["capture"] == "custom-blue-50"
        assert test_client.put("/api/v1/line/recipes/capture", json={"recipe_id": "missing"}).status_code == 404
        assert test_client.put("/api/v1/line/recipes/capture", json={"recipe_id": None}).json() == {"capture": None}
    # The shipped library file is not changed; site recipes live in the data folder.
    assert len(json.loads(Path(library).read_text())["recipes"]) == 2
    assert (tmp_path / "site" / "site-recipes.json").is_file()


def test_create_mode_reference_is_saved_with_provenance(tmp_path):
    library = tmp_path / "library.json"
    library.write_text(json.dumps({"recipes": [{"id": "5", "reference": {"lab": [1.0, 2.0, 3.0], "from": "shipped"}}]}))
    (tmp_path / "station.json").write_text(json.dumps({"fill_camera": "/dev/video9", "library": str(library)}))
    runs = LineRuns(None, station=tmp_path / "station.json", site=tmp_path / "site-recipes.json")
    runs.arm("5")
    runs.save_reference({"recipe_id": "5", "seq": 3, "t": 101.234, "lab": [9.0, 8.0, 7.0],
                         "from": "vial 3 endpoint, operator CREATE mode"}, "live-20261030-090000")
    reference = runs.recipes()["recipes"][0]["reference"]
    assert (reference["lab"], reference["session"], reference["vial"], reference["mode"]) == (
        [9.0, 8.0, 7.0], "live-20261030-090000", 3, "operator CREATE")
    assert runs.capture is None
    assert json.loads(library.read_text())["recipes"][0]["reference"]["lab"] == [1.0, 2.0, 3.0]


def test_recreate_replaces_the_reference_and_keeps_the_old_one(tmp_path):
    library = tmp_path / "library.json"
    library.write_text(json.dumps({"recipes": [{"id": "5", "reference": None}]}))
    (tmp_path / "station.json").write_text(json.dumps({"fill_camera": "/dev/video9", "library": str(library)}))
    runs = LineRuns(None, station=tmp_path / "station.json", site=tmp_path / "site-recipes.json")
    for seq, lab in ((1, [1.0, 1.0, 1.0]), (7, [2.0, 2.0, 2.0])):
        runs.arm("5")
        runs.save_reference({"recipe_id": "5", "seq": seq, "t": 1.0, "lab": lab, "from": "x"}, "live-1")
    recipe = runs.recipes()["recipes"][0]
    assert recipe["reference"]["lab"] == [2.0, 2.0, 2.0] and recipe["reference"]["vial"] == 7
    assert [h["lab"] for h in recipe["reference_history"]] == [[1.0, 1.0, 1.0]]


def test_saved_session_keeps_its_library_and_reviews_after_it_moves(client, tmp_path):
    data = tmp_path / "data"
    (tmp_path / "station.json").write_text(json.dumps({"title": "CSI", "fill_camera": "/dev/video9", "library": str(data / "library.json")}))
    runs = LineRuns(None, station=tmp_path / "station.json", recordings=tmp_path / "recordings")
    live = SimpleNamespace(dwells=[json.loads(line) for line in (data / "frames.jsonl").read_text().splitlines()],
                           tray_samples=[], schedule=["1", "1"])
    runs._save_live("live-1", tmp_path / "recordings" / "live-1", live, runs.manifest("live"), runs.manifest("live"), None, 0)
    assert json.loads((tmp_path / "recordings" / "live-1" / "recipe-library.json").read_text())["recipes"][0]["id"] == "1"
    assert [e["vial"]["verdict"] for e in runs.review("live-1") if e["type"] == "vial"] == ["IN_SPEC", "IN_SPEC"]

    # A run saved before runs kept their own copy, whose library has since moved, uses the station's.
    (tmp_path / "recordings" / "live-0.json").write_text(json.dumps({
        "title": "old", "fill_samples": "live-1/frames.jsonl", "schedule": "live-1/schedule.csv",
        "library": str(tmp_path / "gone" / "library.json")}))
    assert sum(e["type"] == "vial" for e in runs.review("live-0")) == 2


def test_frame_returns_one_jpeg_of_the_recorded_video(client, tmp_path):
    cv2 = pytest.importorskip("cv2")
    import numpy as np
    writer = cv2.VideoWriter(str(tmp_path / "data" / "fill.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (64, 48))
    for k in range(20):
        writer.write(np.full((48, 64, 3), k * 10, np.uint8))
    writer.release()
    response = client.get("/api/v1/line/runs/demo/frame/fill", params={"t": 0.2})
    assert response.status_code == 200 and response.headers["content-type"] == "image/jpeg"
    assert cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR).shape == (48, 64, 3)
    assert client.get("/api/v1/line/runs/demo/frame/fill", params={"t": 99}).status_code == 404
    assert client.get("/api/v1/line/runs/demo/frame/side", params={"t": 0}).status_code == 404


def test_live_events_and_photos_of_the_running_camera_session(tmp_path):
    (tmp_path / "station.json").write_text(json.dumps({"title": "CSI", "fill_camera": "/dev/video9"}))
    runs = LineRuns(None, station=tmp_path / "station.json")
    app.dependency_overrides[get_line_runs] = lambda: runs
    with TestClient(app) as test_client:
        assert test_client.get("/api/v1/line/runs/live/events").json() == {"run_id": None, "active": False, "events": []}
        vial = {"type": "vial", "t": 40.0, "vial": {"seq": 1, "fill_end_s": 30.0}}
        runs.current = {"run_id": "live-1", "active": True, "events": [{"type": "meta"}, vial], "photos": {1: (30.1, b"jpeg")}}
        body = test_client.get("/api/v1/line/runs/live/events").json()
        assert (body["run_id"], body["active"], [e["type"] for e in body["events"]]) == ("live-1", True, ["meta", "vial"])
        assert test_client.get("/api/v1/line/runs/live/frame/fill", params={"t": 30.0}).content == b"jpeg"
        assert test_client.get("/api/v1/line/runs/live/frame/fill", params={"t": 50.0}).status_code == 404
    app.dependency_overrides.clear()
