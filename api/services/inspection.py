from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import AsyncIterator
from uuid import UUID, uuid4

from databridge.sqlite_inspections import SQLiteInspectionBridge
from models.inspection import (
    ArtifactRecord,
    CreateSessionRequest,
    FrameImage,
    InspectionEvent,
    InspectionSession,
    OperatorDecision,
    OperatorStatus,
    PipelineVisuals,
    RecipeContext,
    RecipePlanItem,
    RecipeSegment,
    RecipeSegmentCreate,
)
from pipelines.base import FrameContext, InspectionPipeline, PipelineResult
from services.artifacts import ArtifactStore
from services.event_broker import EventBroker


@dataclass(slots=True)
class FrameWorkItem:
    session_id: UUID
    frame_id: str
    image_bytes: bytes
    mime_type: str
    width: int
    height: int
    source_timestamp_ms: int | None
    original_filename: str | None
    simulate: str | None
    source_clock: str
    future: asyncio.Future[InspectionEvent | None]


class InspectionService:
    def __init__(
        self,
        databridge: SQLiteInspectionBridge,
        pipeline: InspectionPipeline,
        broker: EventBroker,
        artifacts: ArtifactStore,
        worker_count: int = 1,
    ):
        self.databridge = databridge
        self.pipeline = pipeline
        self.broker = broker
        self.artifacts = artifacts
        self.worker_count = worker_count
        self._queue: asyncio.Queue[FrameWorkItem | None] = asyncio.Queue(maxsize=32)
        self._workers: list[asyncio.Task[None]] = []
        self._start_lock = asyncio.Lock()
        self._run_locks: dict[UUID, asyncio.Lock] = {}

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    async def start(self) -> None:
        async with self._start_lock:
            if self._workers:
                return
            self._workers = [
                asyncio.create_task(self._worker(), name=f"model-worker-{index + 1}")
                for index in range(self.worker_count)
            ]

    async def stop(self) -> None:
        workers = list(self._workers)
        if not workers:
            return
        for _ in workers:
            await self._queue.put(None)
        await asyncio.gather(*workers, return_exceptions=True)
        self._workers.clear()

    async def _worker(self) -> None:
        while True:
            item = await self._queue.get()
            try:
                if item is None:
                    return
                try:
                    run_lock = self._run_locks.setdefault(item.session_id, asyncio.Lock())
                    async with run_lock:
                        result = await self._process_frame_now(item)
                    if not item.future.cancelled():
                        item.future.set_result(result)
                except Exception as error:
                    if not item.future.cancelled():
                        item.future.set_exception(error)
            finally:
                self._queue.task_done()

    async def create_session(self, request: CreateSessionRequest) -> InspectionSession:
        return await self.databridge.create_session(request, self.pipeline.mode)

    async def get_session(self, session_id: UUID) -> InspectionSession | None:
        return await self.databridge.get_session(session_id)

    async def list_sessions(self, limit: int = 100) -> list[InspectionSession]:
        return await self.databridge.list_sessions(limit)

    async def add_recipe_segment(
        self, session_id: UUID, request: RecipeSegmentCreate
    ) -> RecipeSegment | None:
        return await self.databridge.add_recipe_segment(session_id, request)

    async def replace_recipe_plan(
        self, session_id: UUID, recipes: list[RecipeContext]
    ) -> list[RecipePlanItem] | None:
        return await self.databridge.replace_recipe_plan(session_id, recipes)

    async def list_recipe_plan(
        self, session_id: UUID
    ) -> list[RecipePlanItem] | None:
        return await self.databridge.list_recipe_plan(session_id)

    async def process_frame(
        self,
        session_id: UUID,
        frame_id: str,
        image_bytes: bytes,
        mime_type: str,
        width: int,
        height: int,
        source_timestamp_ms: int | None,
        original_filename: str | None,
        simulate: str | None,
        source_clock: str = "unknown",
    ) -> InspectionEvent | None:
        await self.start()
        future: asyncio.Future[InspectionEvent | None] = asyncio.get_running_loop().create_future()
        self._queue.put_nowait(
            FrameWorkItem(
                session_id=session_id,
                frame_id=frame_id,
                image_bytes=image_bytes,
                mime_type=mime_type,
                width=width,
                height=height,
                source_timestamp_ms=source_timestamp_ms,
                original_filename=original_filename,
                simulate=simulate,
                source_clock=source_clock,
                future=future,
            )
        )
        return await future

    async def _process_frame_now(self, item: FrameWorkItem) -> InspectionEvent | None:
        session = await self.databridge.get_session(item.session_id)
        if session is None:
            return None
        if item.original_filename and not session.source_name:
            await self.databridge.set_source_name(item.session_id, item.original_filename)
        resolved_recipe = await self.databridge.resolve_recipe(
            item.session_id, item.source_timestamp_ms
        )
        recipe = resolved_recipe or session.recipe
        frame_number = session.frames_processed + 1
        frame_session = session.model_copy(update={"recipe": recipe})
        event_id = uuid4()
        source = await self.artifacts.save_bytes(
            session_id=item.session_id,
            event_id=event_id,
            kind="source",
            content=item.image_bytes,
            mime_type=item.mime_type,
            filename=item.original_filename or f"{item.frame_id}.jpg",
            width=item.width,
            height=item.height,
            source_timestamp_ms=item.source_timestamp_ms,
        )
        await self.databridge.add_artifact(source)

        try:
            pipeline_result = await self.pipeline.analyze(
                item.image_bytes, frame_session, frame_number, simulate=item.simulate,
                context=FrameContext(item.frame_id, item.source_timestamp_ms, item.source_clock,
                                     (item.original_filename or item.frame_id) if session.source_mode == "images"
                                     else (session.source_name or str(session.session_id)),
                                     resolved_recipe is not None, item.width, item.height),
            )
        except Exception as error:
            pipeline_result = PipelineResult(
                decision=OperatorDecision(
                    status=OperatorStatus.UNAVAILABLE,
                    headline="Model pipeline unavailable",
                    detail="Source retained. " + str(error),
                    recommended_action="Review the saved source frame and pipeline logs",
                    recipe_progress_percent=None,
                    model_state=f"PIPELINE ERROR · {type(error).__name__}",
                    confidence=None,
                )
            )

        pipeline_result.decision.evidence.update(
            recipe_context_matched=resolved_recipe is not None,
            source_clock=item.source_clock,
            source_timestamp_ms=item.source_timestamp_ms,
        )

        async def save_visual(kind: str, content: bytes | None, mime_type: str):
            if content is None:
                return None
            suffix = ".png" if mime_type == "image/png" else ".jpg"
            stored = await self.artifacts.save_bytes(
                session_id=item.session_id,
                event_id=event_id,
                kind=kind,
                content=content,
                mime_type=mime_type,
                filename=f"{item.frame_id}-{kind}{suffix}",
                width=item.width,
                height=item.height,
                source_timestamp_ms=item.source_timestamp_ms,
            )
            await self.databridge.add_artifact(stored)
            return stored

        rendered = await save_visual(
            "rendered", pipeline_result.rendered_frame,
            pipeline_result.rendered_frame_mime_type,
        )
        mask = await save_visual(
            "mask", pipeline_result.mask, pipeline_result.mask_mime_type
        )

        def image(record) -> FrameImage:
            return FrameImage(
                mime_type=record.record.mime_type,
                url=record.record.url,
                width=record.record.width or item.width,
                height=record.record.height or item.height,
            )

        event = InspectionEvent(
            event_id=event_id,
            session_id=item.session_id,
            frame_id=item.frame_id,
            frame_number=frame_number,
            source_timestamp_ms=item.source_timestamp_ms,
            original_filename=item.original_filename,
            created_at=datetime.now(timezone.utc),
            pipeline_mode=self.pipeline.mode,
            recipe=recipe,
            image=image(source),
            visuals=PipelineVisuals(
                rendered_frame=image(rendered) if rendered else None,
                mask=image(mask) if mask else None,
                status="available" if rendered or mask else "not_configured",
            ),
            decision=pipeline_result.decision,
        )
        await self.databridge.add_event(
            event,
            source.record.artifact_id,
            rendered.record.artifact_id if rendered else None,
            mask.record.artifact_id if mask else None,
        )
        await self.broker.publish(event)
        return event

    async def store_recording(
        self,
        session_id: UUID,
        filename: str,
        mime_type: str,
        chunks: AsyncIterator[bytes],
        max_bytes: int,
    ) -> ArtifactRecord | None:
        if await self.get_session(session_id) is None:
            return None
        stored = await self.artifacts.save_stream(
            session_id=session_id,
            kind="recording",
            chunks=chunks,
            mime_type=mime_type,
            filename=filename,
            max_bytes=max_bytes,
        )
        await self.databridge.set_source_name(session_id, filename)
        return await self.databridge.add_artifact(stored)
