import asyncio
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from databridge.sqlite_inspections import SQLiteInspectionBridge
from dependencies import get_inspection_service
from main import app
from pipelines.mock import MockInspectionPipeline
from services.artifacts import ArtifactStore
from services.event_broker import EventBroker
from services.inspection import InspectionService
from models.inspection import CreateSessionRequest, OperatorDecision, OperatorStatus, RecipeContext
from pipelines.base import PipelineResult


@pytest.fixture
def client(tmp_path: Path):
    bridge = SQLiteInspectionBridge(tmp_path / "linewatch.sqlite3")
    service = InspectionService(
        bridge,
        MockInspectionPipeline(),
        EventBroker(),
        ArtifactStore(tmp_path / "artifacts"),
    )
    app.dependency_overrides[get_inspection_service] = lambda: service
    with TestClient(app) as test_client:
        yield test_client
        test_client.portal.call(service.stop)
        test_client.portal.call(bridge.close)
    app.dependency_overrides.clear()


def create_session(client: TestClient, source_mode: str = "live_camera") -> dict:
    response = client.post(
        "/api/v1/sessions",
        json={
            "camera_id": "fillcam-1",
            "station_name": "Fluid Mixing Station 3",
            "run_name": "Run 5 replay",
            "source_mode": source_mode,
            "source_name": "vial-fill-video-5.mp4" if source_mode == "recording" else None,
            "recipe": {
                "current_recipe_id": "recipe-7",
                "current_recipe_name": "Recipe 7 · Ocean Blue",
                "previous_recipe_id": "recipe-6",
                "vial_index": 1,
                "is_first_vial_after_change": True,
            },
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_health_session_and_history_contract(client: TestClient):
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["pipeline_mode"] == "mock"
    assert health.json()["persistence"] == "sqlite+filesystem"
    session = create_session(client)
    assert session["status"] == "starting"
    assert session["source_mode"] == "live_camera"
    runs = client.get("/api/v1/sessions").json()
    assert runs[0]["session_id"] == session["session_id"]


def test_frame_is_queued_persisted_and_acknowledged(client: TestClient):
    session = create_session(client)
    resource = f"/api/v1/sessions/{session['session_id']}"
    frame = client.post(
        resource + "/frames?frame_id=frame-1&width=2&height=2&source_timestamp_ms=1250&original_filename=sample.jpg&simulate=failure",
        content=b"not-decoded-by-the-mock-adapter",
        headers={"content-type": "image/jpeg"},
    )
    assert frame.status_code == 202, frame.text
    event = frame.json()
    assert event["source_timestamp_ms"] == 1250
    assert event["recipe"]["current_recipe_id"] == "recipe-7"
    assert event["image"]["url"].startswith("/api/v1/artifacts/")
    assert event["visuals"]["status"] == "not_configured"
    artifact = client.get(event["image"]["url"])
    assert artifact.status_code == 200
    assert artifact.content == b"not-decoded-by-the-mock-adapter"
    events = client.get(resource + "/events").json()
    assert events[0]["event_id"] == event["event_id"]
    ack = client.put(resource + f"/events/{event['event_id']}/acknowledgement")
    assert ack.status_code == 200
    assert ack.json()["acknowledged_at"]


def test_recipe_csv_resolves_context_by_source_timestamp(client: TestClient):
    session = create_session(client, "recording")
    resource = f"/api/v1/sessions/{session['session_id']}"
    recipe_csv = (
        "recipe_id,type,start_s,end_s,vial\n"
        "5,Triple,0.0,9.9,1\n"
        "6,Double,10.0,20.0,2\n"
    )
    imported = client.post(
        resource + "/recipe-segments/import",
        content=recipe_csv,
        headers={"content-type": "text/csv"},
    )
    assert imported.status_code == 201, imported.text
    assert len(imported.json()) == 2
    repeated = client.post(
        resource + "/recipe-segments/import",
        content=recipe_csv,
        headers={"content-type": "text/csv"},
    )
    assert repeated.status_code == 201, repeated.text
    assert repeated.json() == []
    assert len(client.get(resource + "/recipe-segments").json()) == 3
    frame = client.post(
        resource + "/frames?frame_id=frame-2&width=2&height=2&source_timestamp_ms=15000",
        content=b"frame",
        headers={"content-type": "image/jpeg"},
    )
    assert frame.status_code == 202
    assert frame.json()["recipe"]["current_recipe_id"] == "6"


def test_operator_recipe_plan_preserves_order_and_expected_color(client: TestClient):
    session = create_session(client)
    resource = f"/api/v1/sessions/{session['session_id']}"
    recipes = [
        {
            "current_recipe_id": "blue-01",
            "current_recipe_name": "Blue recipe",
            "vial_index": 1,
            "is_first_vial_after_change": True,
            "expected_color_name": "Blue",
            "expected_color_hex": "#287fb5",
        },
        {
            "current_recipe_id": "orange-02",
            "current_recipe_name": "Orange recipe",
            "previous_recipe_id": "blue-01",
            "vial_index": 2,
            "is_first_vial_after_change": True,
            "expected_color_name": "Orange",
            "expected_color_hex": "#dc6b2f",
        },
    ]
    replaced = client.put(resource + "/recipe-plan", json={"recipes": recipes})
    assert replaced.status_code == 200, replaced.text
    assert [item["position"] for item in replaced.json()] == [1, 2]
    assert replaced.json()[0]["recipe"]["expected_color_name"] == "Blue"
    persisted = client.get(resource + "/recipe-plan")
    assert persisted.status_code == 200
    assert [item["recipe"]["current_recipe_id"] for item in persisted.json()] == [
        "blue-01", "orange-02",
    ]


def test_recipe_csv_keeps_recipe_formula_and_color_intent(client: TestClient):
    session = create_session(client, "recording")
    resource = f"/api/v1/sessions/{session['session_id']}"
    recipe_csv = (
        "recipe_id,type,start_s,end_s,vial,t100_ml,t200_ml,t300_ml,t400_ml,total_ml,dispense_order,color_purpose\n"
        "7,Pair 100+400,4.0,8.0,7,20,,0,50,70,T400>T100,pale blue\n"
    )
    imported = client.post(
        resource + "/recipe-segments/import",
        content=recipe_csv,
        headers={"content-type": "text/csv"},
    )
    assert imported.status_code == 201, imported.text
    recipe = imported.json()[0]["recipe"]
    assert recipe["expected_color_name"] == "pale blue"
    assert recipe["t100_ml"] == 20
    assert recipe["t300_ml"] == 0
    assert recipe["t400_ml"] == 50
    assert recipe["total_ml"] == 70
    assert recipe["dispense_order"] == "T400>T100"


def test_recording_upload_and_frame_validation(client: TestClient):
    session = create_session(client, "recording")
    resource = f"/api/v1/sessions/{session['session_id']}"
    upload = client.post(
        resource + "/recordings?filename=test.mp4",
        content=b"small-video-fixture",
        headers={"content-type": "video/mp4"},
    )
    assert upload.status_code == 201, upload.text
    assert upload.json()["kind"] == "recording"
    artifact = client.get(upload.json()["url"])
    assert artifact.content == b"small-video-fixture"
    invalid = client.post(
        resource + "/frames?frame_id=frame-1&width=2&height=2",
        content=b"text",
        headers={"content-type": "text/plain"},
    )
    assert invalid.status_code == 415


def test_queue_preserves_recipe_context_across_multiple_runs(tmp_path: Path):
    async def scenario():
        bridge = SQLiteInspectionBridge(tmp_path / "multi-run.sqlite3")
        service = InspectionService(
            bridge,
            MockInspectionPipeline(),
            EventBroker(),
            ArtifactStore(tmp_path / "multi-artifacts"),
            worker_count=2,
        )
        first = await service.create_session(
            CreateSessionRequest(
                run_name="Recipe A camera",
                source_mode="live_camera",
                recipe=RecipeContext(
                    current_recipe_id="A",
                    current_recipe_name="Recipe A",
                ),
            )
        )
        second = await service.create_session(
            CreateSessionRequest(
                run_name="Recipe B replay",
                source_mode="recording",
                recipe=RecipeContext(
                    current_recipe_id="B",
                    current_recipe_name="Recipe B",
                ),
            )
        )
        first_event, second_event = await asyncio.gather(
            service.process_frame(
                first.session_id, "a-1", b"a", "image/jpeg", 2, 2,
                100, "a.jpg", None,
            ),
            service.process_frame(
                second.session_id, "b-1", b"b", "image/jpeg", 2, 2,
                200, "b.jpg", None,
            ),
        )
        assert first_event and first_event.recipe.current_recipe_id == "A"
        assert second_event and second_event.recipe.current_recipe_id == "B"
        assert service.queue_depth == 0
        await service.stop()
        await bridge.close()

        reopened = SQLiteInspectionBridge(tmp_path / "multi-run.sqlite3")
        assert len(await reopened.list_sessions()) == 2
        assert (await reopened.list_events(first.session_id))[0].original_filename == "a.jpg"
        await reopened.close()

    asyncio.run(scenario())


def test_rendered_and_mask_artifacts_are_saved(tmp_path: Path):
    class VisualPipeline:
        mode = "model"

        async def analyze(self, image_bytes, session, frame_number, simulate=None):
            return PipelineResult(
                decision=OperatorDecision(
                    status=OperatorStatus.RUNNING_OK,
                    headline="Visuals ready",
                    detail="Rendered frame and mask were produced.",
                    recommended_action="Continue monitoring",
                    recipe_progress_percent=50,
                    model_state="VISUAL TEST",
                ),
                rendered_frame=b"rendered-frame",
                mask=b"mask-frame",
            )

    async def scenario():
        bridge = SQLiteInspectionBridge(tmp_path / "visual.sqlite3")
        store = ArtifactStore(tmp_path / "visual-artifacts")
        service = InspectionService(bridge, VisualPipeline(), EventBroker(), store)
        run = await service.create_session(
            CreateSessionRequest(
                run_name="Visual artifact run",
                recipe=RecipeContext(
                    current_recipe_id="A",
                    current_recipe_name="Recipe A",
                ),
            )
        )
        event = await service.process_frame(
            run.session_id, "visual-1", b"source", "image/jpeg", 2, 2,
            0, "source.jpg", None,
        )
        assert event and event.visuals.rendered_frame and event.visuals.mask
        for expected, image in (
            (b"rendered-frame", event.visuals.rendered_frame),
            (b"mask-frame", event.visuals.mask),
        ):
            artifact_id = image.url.rsplit("/", 1)[-1]
            record, relative_path = await bridge.get_artifact(artifact_id)
            assert store.resolve(relative_path).read_bytes() == expected
            assert record.event_id == event.event_id
        await service.stop()
        await bridge.close()

    asyncio.run(scenario())
