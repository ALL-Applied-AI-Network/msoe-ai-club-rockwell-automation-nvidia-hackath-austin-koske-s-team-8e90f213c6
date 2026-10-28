import asyncio
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from models.inspection import InspectionSession, OperatorStatus, RecipeContext
from pipelines.temporal_cnn_bridge import (
    TemporalCnnArtifacts,
    TemporalCnnInspectionBridge,
)


class FakeBackend:
    def __init__(self, observation):
        self.observation = observation
        self.calls = []

    def analyze(self, image_bytes, session_id, frame_number):
        self.calls.append((image_bytes, session_id, frame_number))
        return TemporalCnnArtifacts(
            observation=self.observation,
            rendered_frame=b"rendered-jpeg",
            mask=b"mask-png",
        )


def session() -> InspectionSession:
    now = datetime.now(timezone.utc)
    return InspectionSession(
        session_id=uuid4(),
        camera_id="fillcam-1",
        station_name="Fluid Mixing Station 3",
        run_name="Adapter test",
        source_mode="images",
        recipe=RecipeContext(
            current_recipe_id="recipe-7",
            current_recipe_name="Recipe 7",
        ),
        status=OperatorStatus.PROCESSING,
        pipeline_mode="model",
        created_at=now,
        updated_at=now,
    )


def pipeline(backend, reference_qualified=True):
    return TemporalCnnInspectionBridge(
        Path("unused-yolo.pt"),
        Path("unused-temporal.pt"),
        Path("unused-reference.png"),
        reference_qualified=reference_qualified,
        backend=backend,
    )


def test_adapter_returns_model_visuals_and_reviewable_measurements():
    backend = FakeBackend({
        "association_passed": True,
        "measurement_valid": True,
        "quality_flags": "[]",
        "presence": 0.92,
        "vial_detections": 1,
        "containment": 0.96,
    })
    run = session()
    result = asyncio.run(pipeline(backend).analyze(b"source", run, 3))

    assert result.rendered_frame == b"rendered-jpeg"
    assert result.mask == b"mask-png"
    assert result.decision.status == OperatorStatus.RUNNING_OK
    assert result.decision.recipe_progress_percent == 0
    assert result.decision.confidence == 0.92
    assert backend.calls == [(b"source", run.session_id, 3)]


def test_unreviewed_temporal_reference_cannot_produce_normal_status():
    backend = FakeBackend({
        "association_passed": True,
        "measurement_valid": True,
        "quality_flags": [],
        "presence": 0.88,
        "vial_detections": 1,
        "containment": 0.95,
    })
    result = asyncio.run(
        pipeline(backend, reference_qualified=False).analyze(b"source", session(), 1)
    )

    assert result.decision.status == OperatorStatus.WARNING
    assert "qualification" in result.decision.headline.lower()


def test_missing_proposal_is_warning_not_empty_or_failure_decision():
    backend = FakeBackend({
        "association_passed": False,
        "measurement_valid": False,
        "quality_flags": '["fill_position_vial_missing_or_ambiguous"]',
        "presence": 0.03,
        "vial_detections": 0,
        "containment": 0,
    })
    result = asyncio.run(pipeline(backend).analyze(b"source", session(), 1))

    assert result.decision.status == OperatorStatus.WARNING
    assert "does not confirm" in result.decision.detail
