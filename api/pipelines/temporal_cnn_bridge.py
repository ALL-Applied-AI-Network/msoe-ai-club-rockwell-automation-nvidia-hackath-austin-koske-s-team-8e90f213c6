from __future__ import annotations

import asyncio
import importlib
import json
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from uuid import UUID

from models.inspection import (
    InspectionSession,
    OperatorDecision,
    OperatorStatus,
    PipelineMeasurement,
)
from pipelines.base import PipelineResult


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@dataclass(slots=True)
class TemporalCnnArtifacts:
    observation: dict[str, Any]
    rendered_frame: bytes
    mask: bytes


class TemporalCnnBackend(Protocol):
    def analyze(
        self, image_bytes: bytes, session_id: UUID, frame_number: int
    ) -> TemporalCnnArtifacts: ...


class ModelPipelineBackend:
    """Lazy, serialized bridge to integration/model_pipeline.py."""

    def __init__(
        self,
        yolo_checkpoint: Path,
        temporal_checkpoint: Path,
        reference_image: Path | None,
    ) -> None:
        self.yolo_checkpoint = self._resolve(yolo_checkpoint)
        self.temporal_checkpoint = self._resolve(temporal_checkpoint)
        self.reference_image = self._resolve(reference_image) if reference_image else None
        self._lock = threading.RLock()
        self._module: Any | None = None
        self._yolo: Any | None = None
        self._temporal: Any | None = None
        self._qa_by_session: dict[UUID, Any] = {}

    @staticmethod
    def _resolve(path: Path) -> Path:
        return path.resolve() if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()

    def missing_assets(self) -> list[str]:
        candidates = [self.yolo_checkpoint, self.temporal_checkpoint]
        if self.reference_image is None:
            return ["qualified temporal reference image"]
        candidates.append(self.reference_image)
        return [str(path) for path in candidates if not path.is_file()]

    def _load(self) -> None:
        if self._module is not None:
            return
        missing = self.missing_assets()
        if missing:
            raise RuntimeError("Missing model assets: " + ", ".join(missing))

        for path in (REPOSITORY_ROOT,):
            value = str(path)
            if value not in sys.path:
                sys.path.insert(0, value)

        try:
            module = importlib.import_module("integration.model_pipeline")
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "Model dependencies are unavailable; install setup/requirements/temporal-pool.in"
            ) from error

        self._yolo, self._temporal = module.load_models(
            self.yolo_checkpoint, self.temporal_checkpoint
        )
        self._module = module

    def analyze(
        self, image_bytes: bytes, session_id: UUID, frame_number: int
    ) -> TemporalCnnArtifacts:
        # Model objects and MeasurementQA contain mutable state. Serialize inference
        # while retaining a separate QA history for each run.
        with self._lock:
            self._load()
            assert self._module is not None
            assert self.reference_image is not None
            qa = self._qa_by_session.setdefault(
                session_id, self._module.MeasurementQA()
            )

            with tempfile.TemporaryDirectory(prefix="linewatch-model-") as temp_name:
                temp_dir = Path(temp_name)
                current_path = temp_dir / "current-image"
                current_path.write_bytes(image_bytes)
                output_dir = temp_dir / "output"
                output_dir.mkdir()
                row = self._module.process_pair(
                    current_path=current_path,
                    reference_path=self.reference_image,
                    yolo=self._yolo,
                    temporal=self._temporal,
                    output_dir=output_dir,
                    number=frame_number,
                    qa=qa,
                )
                return TemporalCnnArtifacts(
                    observation=row,
                    rendered_frame=(output_dir / row["result_file"]).read_bytes(),
                    mask=(output_dir / row["mask_file"]).read_bytes(),
                )


class TemporalCnnInspectionBridge:
    """Bridge batch vision output into the FastAPI inspection contract."""

    mode = "model"

    def __init__(
        self,
        yolo_checkpoint: Path,
        temporal_checkpoint: Path,
        reference_image: Path | None,
        reference_qualified: bool = False,
        backend: TemporalCnnBackend | None = None,
    ) -> None:
        self.reference_qualified = reference_qualified
        self.backend = backend or ModelPipelineBackend(
            yolo_checkpoint, temporal_checkpoint, reference_image
        )

    @property
    def status(self) -> str:
        if isinstance(self.backend, ModelPipelineBackend):
            if self.backend.missing_assets():
                return "assets_required"
        if not self.reference_qualified:
            return "reference_review_required"
        return "configured"

    async def analyze(
        self,
        image_bytes: bytes,
        session: InspectionSession,
        frame_number: int,
        simulate: str | None = None,
    ) -> PipelineResult:
        del simulate  # Model mode never fabricates the dashboard's test outcomes.
        inference = await asyncio.to_thread(
            self.backend.analyze, image_bytes, session.session_id, frame_number
        )
        return PipelineResult(
            decision=self._decision(inference.observation, session),
            rendered_frame=inference.rendered_frame,
            rendered_frame_mime_type="image/jpeg",
            mask=inference.mask,
            mask_mime_type="image/png",
        )

    def _decision(
        self, observation: dict[str, Any], session: InspectionSession
    ) -> OperatorDecision:
        flags = observation.get("quality_flags", [])
        if isinstance(flags, str):
            try:
                flags = json.loads(flags)
            except json.JSONDecodeError:
                flags = [flags]
        flags = [str(flag) for flag in flags]
        association_passed = bool(observation.get("association_passed"))
        measurement_valid = bool(observation.get("measurement_valid"))
        presence = float(observation.get("presence", 0.0))
        vial_count = int(observation.get("vial_detections", 0))
        containment = float(observation.get("containment", 0.0))

        if measurement_valid and self.reference_qualified:
            status = OperatorStatus.RUNNING_OK
            headline = "Measurement ready for operator review"
            detail = (
                "Vial-associated liquid evidence passed measurement QA. "
                "Recipe acceptance still requires a separately qualified recipe reference."
            )
            action = "Review measurement against the active recipe"
        elif measurement_valid:
            status = OperatorStatus.WARNING
            headline = "Reference qualification required"
            detail = (
                "The frame passed measurement QA, but the configured empty-vial "
                "reference has not been marked independently reviewed."
            )
            action = "Review and qualify the aligned empty reference"
        elif association_passed:
            status = OperatorStatus.WARNING
            headline = "Measurement held by quality checks"
            detail = "QA flags: " + (", ".join(flags) if flags else "unspecified")
            action = "Review the rendered frame and QA flags"
        else:
            status = OperatorStatus.WARNING
            headline = "No usable liquid measurement"
            detail = (
                "No vial-associated liquid proposal passed the pipeline. This does not "
                "confirm that the vial is empty or that the recipe is acceptable."
            )
            action = "Inspect the source and reference frames"

        reference_state = "REVIEWED REFERENCE" if self.reference_qualified else "UNVERIFIED REFERENCE"
        return OperatorDecision(
            status=status,
            headline=headline,
            detail=detail,
            recommended_action=action,
            recipe_progress_percent=0,
            model_state=(
                f"YOLO + TEMPORAL CNN + MEASUREMENT QA · {reference_state} · "
                f"RECIPE {session.recipe.current_recipe_id}"
            ),
            confidence=presence,
            measurements=[
                PipelineMeasurement(
                    label="Vial",
                    value=f"{vial_count} detected" if vial_count else "Not detected",
                    state="good" if vial_count else "warning",
                ),
                PipelineMeasurement(
                    label="Fill",
                    value=f"{containment:.0%} contained" if association_passed else "Not measured",
                    state="good" if association_passed else "warning",
                ),
                PipelineMeasurement(
                    label="QA",
                    value="Passed" if measurement_valid else f"{len(flags)} flag(s)",
                    state="good" if measurement_valid else "warning",
                ),
            ],
        )
