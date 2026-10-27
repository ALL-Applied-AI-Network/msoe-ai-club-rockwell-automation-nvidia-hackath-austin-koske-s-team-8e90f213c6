"""Regressions for source/recipe identity and persisted operator responses."""
from pathlib import Path
from io import BytesIO

import pytest
from fastapi.testclient import TestClient

from databridge.sqlite_inspections import SQLiteInspectionBridge
from dependencies import get_inspection_service
from main import app
from pipelines.mock import MockInspectionPipeline
from pipelines.temporal import TemporalInspectionPipeline
from services.artifacts import ArtifactStore
from services.event_broker import EventBroker
from services.inspection import InspectionService


@pytest.fixture
def client(tmp_path):
    bridge = SQLiteInspectionBridge(tmp_path / "demo.sqlite3")
    service = InspectionService(bridge, MockInspectionPipeline(), EventBroker(), ArtifactStore(tmp_path / "artifacts"))
    app.dependency_overrides[get_inspection_service] = lambda: service
    with TestClient(app) as client:
        yield client, service
        client.portal.call(service.stop)
        client.portal.call(bridge.close)
    app.dependency_overrides.clear()


def session(client, mode="recording"):
    response = client.post("/api/v1/sessions", json={"source_mode": mode, "recipe": {
        "current_recipe_id": "manual", "current_recipe_name": "Manual context"}})
    assert response.status_code == 201
    return f"/api/v1/sessions/{response.json()['session_id']}"


def frame(client, resource, n, timestamp=None, simulate=None):
    return client.post(resource + "/frames", params={"frame_id": f"f-{n}", "width": 640, "height": 480,
        **({"source_timestamp_ms": timestamp} if timestamp is not None else {}),
        **({"simulate": simulate} if simulate else {})}, content=b"explicit mock fixture", headers={"content-type": "image/png"})


def test_frame_index_csv_requires_explicit_source_clock_mapping(client):
    api, _ = client
    resource = session(api)
    result = api.post(resource + "/recipe-segments/import", content="vial,recipe_id,start_frame,end_frame\n1,A,100,200\n")
    assert result.status_code == 400
    assert len(api.get(resource + "/recipe-segments").json()) == 1


def test_csv_import_is_atomic_and_rejects_overlaps(client):
    api, _ = client
    resource = session(api)
    for csv in ["recipe_id,start_s,end_s\nA,0,2\nB,1,3\n", "recipe_id,start_s,end_s\nA,0,1\nB,3,2\n"]:
        assert api.post(resource + "/recipe-segments/import", content=csv).status_code == 400
        assert api.get(resource + "/recipe-segments").json()[0]['recipe']['current_recipe_id'] == 'manual'


def test_expired_recipe_is_unmatched_and_end_is_exclusive(client):
    api, _ = client
    resource = session(api)
    assert api.post(resource + "/recipe-segments/import", content="recipe_id,start_s,end_s\nA,0,1\nB,2,3\n").status_code == 201
    for n, stamp in enumerate([1000, 1500, 3000]):
        event = frame(api, resource, n, stamp).json()
        assert event['decision']['evidence']['recipe_context_matched'] is False
    assert frame(api, resource, 4, 2000).json()['recipe']['current_recipe_id'] == 'B'


def test_stills_do_not_invent_time_or_resolve_timeline(client):
    api, _ = client
    resource = session(api, "images")
    assert frame(api, resource, 1, 1000).status_code == 422
    event = frame(api, resource, 1).json()
    assert event['source_timestamp_ms'] is None
    assert event['decision']['evidence']['recipe_context_matched'] is False


def test_old_event_acknowledgement_and_pagination(client):
    api, _ = client
    resource = session(api)
    first = frame(api, resource, 0, 0, "failure").json()
    for index in range(1, 103):
        assert frame(api, resource, index, index * 1000).status_code == 202
    page = api.get(resource + '/events', params={'limit': 100}).json()
    older = api.get(resource + '/events', params={'limit': 100, 'before_frame': page[-1]['frame_number']}).json()
    assert len(page) == 100 and len(older) == 3
    assert older[-1]['event_id'] == first['event_id']
    response = {'reviewer': 'Test operator', 'action': 'investigate', 'note': 'Synthetic test only'}
    saved = api.put(resource + f"/events/{first['event_id']}/acknowledgement", json=response)
    assert saved.status_code == 200
    assert saved.json()['acknowledgement'] == response
    assert api.get(resource + '/events', params={'review_only': True}).json() == []
    old = api.get(resource + '/events', params={'before_frame': 2}).json()[0]
    assert old['acknowledgement'] == response


def test_unconfigured_model_never_returns_normal_or_simulated_result(client):
    api, service = client
    service.pipeline = TemporalInspectionPipeline()
    resource = session(api)
    event = frame(api, resource, 0, 0).json()
    assert event['decision']['status'] == 'unavailable'
    assert event['decision']['recipe_progress_percent'] is None
    assert api.get(event['image']['url']).status_code == 200
    assert frame(api, resource, 1, 1, 'failure').status_code == 409


def test_reference_upload_is_unreviewed_and_rejects_wrong_geometry(client):
    image = pytest.importorskip('PIL.Image')
    api, service = client
    service.pipeline = TemporalInspectionPipeline()
    resource = session(api)
    def png(size):
        buffer = BytesIO()
        image.new('RGB', size, (10, 20, 30)).save(buffer, format='PNG')
        return buffer.getvalue()
    good = api.post(resource + '/reference', content=png((640, 480)), headers={'content-type': 'image/png'})
    assert good.status_code == 201
    assert good.json()['status'] == 'unreviewed_not_certified_empty'
    assert good.json()['recipe_qualified'] is False
    assert api.get(good.json()['artifact']['url']).status_code == 200
    bad = api.post(resource + '/reference', content=png((1280, 720)), headers={'content-type': 'image/png'})
    assert bad.status_code == 400
