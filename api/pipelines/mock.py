from __future__ import annotations

from models.inspection import (
    InspectionSession,
    OperatorDecision,
    OperatorStatus,
    PipelineMeasurement,
)
from pipelines.base import PipelineResult


class MockInspectionPipeline:
    """Deterministic integration adapter used until trained model adapters land."""

    mode = "mock"

    async def analyze(
        self,
        image_bytes: bytes,
        session: InspectionSession,
        frame_number: int,
        simulate: str | None = None,
        context=None,
    ) -> PipelineResult:
        if simulate == "failure":
            return PipelineResult(decision=OperatorDecision(
                status=OperatorStatus.FAILURE,
                headline="Possible fill or recipe mismatch",
                detail=(
                    "The integration simulator flagged this frame. An operator should "
                    "inspect the vial and confirm the active recipe before continuing."
                ),
                recommended_action="Hold for operator review",
                recipe_progress_percent=min(100, 28 + frame_number * 3),
                model_state="SIMULATED FAILURE — no production decision",
                confidence=0.91,
                measurements=[
                    PipelineMeasurement(label="Vial", value="Detected", state="good"),
                    PipelineMeasurement(label="Fill", value="Outside range", state="failure"),
                    PipelineMeasurement(label="Color", value="Review", state="warning"),
                ],
            ))

        if simulate == "warning":
            return PipelineResult(decision=OperatorDecision(
                status=OperatorStatus.WARNING,
                headline="Changeover settling",
                detail="Frame accepted; recipe transition evidence needs another vial.",
                recommended_action="Inspect the next vial",
                recipe_progress_percent=min(100, 30 + frame_number * 4),
                model_state="SIMULATED WARNING — awaiting model adapters",
                confidence=0.76,
                measurements=[
                    PipelineMeasurement(label="Vial", value="Detected", state="good"),
                    PipelineMeasurement(label="Fill", value="In range", state="good"),
                    PipelineMeasurement(label="Color", value="Transition", state="warning"),
                ],
            ))

        progress = min(100, 32 + frame_number * 7)
        return PipelineResult(decision=OperatorDecision(
            status=OperatorStatus.RUNNING_OK,
            headline="Recipe is progressing well",
            detail=(
                f"Frame {frame_number} completed the integration path for "
                f"{session.recipe.current_recipe_name}."
            ),
            recommended_action="Continue monitoring",
            recipe_progress_percent=progress,
            model_state="MOCK PIPELINE — replace with Team 1/2/3 adapters",
            confidence=0.94,
            measurements=[
                PipelineMeasurement(label="Vial", value="Detected", state="good"),
                PipelineMeasurement(label="Fill", value="In range", state="good"),
                PipelineMeasurement(label="Color", value="On recipe", state="good"),
            ],
        ))

