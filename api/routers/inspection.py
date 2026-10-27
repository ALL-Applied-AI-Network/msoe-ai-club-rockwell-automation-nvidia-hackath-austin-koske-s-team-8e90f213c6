from __future__ import annotations

import csv
import io
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from dependencies import get_inspection_service
from models.inspection import (
    AcknowledgeEventResponse,
    ArtifactRecord,
    CreateSessionRequest,
    InspectionEvent,
    InspectionSession,
    RecipeContext,
    RecipeSegment,
    RecipeSegmentCreate,
)
from services.inspection import InspectionService


router = APIRouter(prefix="/sessions", tags=["inspection"])
artifact_router = APIRouter(prefix="/artifacts", tags=["artifacts"])
MAX_FRAME_BYTES = 12 * 1024 * 1024
MAX_RECORDING_BYTES = 2 * 1024 * 1024 * 1024
SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
SUPPORTED_VIDEO_TYPES = {
    "video/mp4", "video/webm", "video/quicktime", "video/x-msvideo",
    "application/octet-stream",
}


@router.post("", response_model=InspectionSession, status_code=201)
async def create_session(
    body: CreateSessionRequest,
    service: InspectionService = Depends(get_inspection_service),
):
    return await service.create_session(body)


@router.get("", response_model=list[InspectionSession])
async def list_sessions(
    limit: int = Query(default=100, ge=1, le=500),
    service: InspectionService = Depends(get_inspection_service),
):
    return await service.list_sessions(limit)


@router.get("/{session_id}", response_model=InspectionSession)
async def get_session(
    session_id: UUID,
    service: InspectionService = Depends(get_inspection_service),
):
    session = await service.get_session(session_id)
    if session is None:
        raise HTTPException(404, "Inspection run not found")
    return session


@router.get("/{session_id}/events", response_model=list[InspectionEvent])
async def list_events(
    session_id: UUID,
    limit: int = Query(default=25, ge=1, le=500),
    service: InspectionService = Depends(get_inspection_service),
):
    events = await service.databridge.list_events(session_id, limit)
    if events is None:
        raise HTTPException(404, "Inspection run not found")
    return events


@router.get("/{session_id}/recipe-segments", response_model=list[RecipeSegment])
async def list_recipe_segments(
    session_id: UUID,
    service: InspectionService = Depends(get_inspection_service),
):
    segments = await service.databridge.list_recipe_segments(session_id)
    if segments is None:
        raise HTTPException(404, "Inspection run not found")
    return segments


@router.post("/{session_id}/recipe-segments", response_model=RecipeSegment, status_code=201)
async def add_recipe_segment(
    session_id: UUID,
    body: RecipeSegmentCreate,
    service: InspectionService = Depends(get_inspection_service),
):
    segment = await service.add_recipe_segment(session_id, body)
    if segment is None:
        raise HTTPException(404, "Inspection run not found")
    return segment


@router.post(
    "/{session_id}/recipe-segments/import",
    response_model=list[RecipeSegment],
    status_code=201,
)
async def import_recipe_segments(
    session_id: UUID,
    request: Request,
    service: InspectionService = Depends(get_inspection_service),
):
    if await service.get_session(session_id) is None:
        raise HTTPException(404, "Inspection run not found")
    try:
        text = (await request.body()).decode("utf-8-sig")
        rows = list(csv.DictReader(io.StringIO(text)))
    except (UnicodeDecodeError, csv.Error) as error:
        raise HTTPException(400, f"Invalid recipe CSV: {error}") from error
    if not rows:
        raise HTTPException(400, "Recipe CSV contains no rows")

    imported: list[RecipeSegment] = []
    previous_id: str | None = None
    for index, row in enumerate(rows, 1):
        recipe_id = (row.get("recipe_id") or row.get("current_recipe_id") or "").strip()
        if not recipe_id:
            raise HTTPException(400, f"Row {index} is missing recipe_id")
        try:
            start_ms = int(float(row.get("start_ms") or float(row.get("start_s") or 0) * 1000))
            end_value = row.get("end_ms")
            end_ms = (
                int(float(end_value)) if end_value
                else int(float(row["end_s"]) * 1000) if row.get("end_s") else None
            )
            vial_index = int(row.get("vial") or row.get("vial_index") or 1)
        except ValueError as error:
            raise HTTPException(400, f"Row {index} has an invalid timestamp or vial") from error
        recipe = RecipeContext(
            current_recipe_id=recipe_id,
            current_recipe_name=(
                row.get("recipe_name") or row.get("current_recipe_name")
                or row.get("type") or f"Recipe {recipe_id}"
            ).strip(),
            previous_recipe_id=(row.get("previous_recipe_id") or previous_id or None),
            vial_index=vial_index,
            is_first_vial_after_change=recipe_id != previous_id,
        )
        segment = await service.add_recipe_segment(
            session_id,
            RecipeSegmentCreate(start_ms=start_ms, end_ms=end_ms, recipe=recipe),
        )
        if segment:
            imported.append(segment)
        previous_id = recipe_id
    return imported


@router.post("/{session_id}/recordings", response_model=ArtifactRecord, status_code=201)
async def upload_recording(
    session_id: UUID,
    request: Request,
    filename: str = Query(min_length=1, max_length=260),
    service: InspectionService = Depends(get_inspection_service),
):
    mime_type = request.headers.get("content-type", "application/octet-stream").split(";")[0]
    if mime_type not in SUPPORTED_VIDEO_TYPES:
        raise HTTPException(415, "Recording must be MP4, WebM, MOV, or AVI")
    try:
        artifact = await service.store_recording(
            session_id, filename, mime_type, request.stream(), MAX_RECORDING_BYTES
        )
    except ValueError as error:
        raise HTTPException(413, str(error)) from error
    if artifact is None:
        raise HTTPException(404, "Inspection run not found")
    return artifact


@router.post("/{session_id}/frames", response_model=InspectionEvent, status_code=202)
async def ingest_frame(
    session_id: UUID,
    request: Request,
    frame_id: str = Query(min_length=1, max_length=160),
    width: int = Query(gt=0, le=8192),
    height: int = Query(gt=0, le=8192),
    source_timestamp_ms: int | None = Query(default=None, ge=0),
    original_filename: str | None = Query(default=None, max_length=260),
    simulate: str | None = Query(default=None, pattern="^(warning|failure)$"),
    service: InspectionService = Depends(get_inspection_service),
):
    mime_type = request.headers.get("content-type", "").split(";")[0]
    if mime_type not in SUPPORTED_IMAGE_TYPES:
        raise HTTPException(415, "Frame must be JPEG, PNG, or WebP")
    image_bytes = await request.body()
    if not image_bytes:
        raise HTTPException(400, "Frame body is empty")
    if len(image_bytes) > MAX_FRAME_BYTES:
        raise HTTPException(413, "Frame exceeds 12 MB limit")
    event = await service.process_frame(
        session_id, frame_id, image_bytes, mime_type, width, height,
        source_timestamp_ms, original_filename, simulate,
    )
    if event is None:
        raise HTTPException(404, "Inspection run not found")
    return event


@router.put(
    "/{session_id}/events/{event_id}/acknowledgement",
    response_model=AcknowledgeEventResponse,
)
async def acknowledge_event(
    session_id: UUID,
    event_id: UUID,
    service: InspectionService = Depends(get_inspection_service),
):
    event = await service.databridge.acknowledge_event(session_id, event_id)
    if event is None or event.acknowledged_at is None:
        raise HTTPException(404, "Inspection event not found")
    return AcknowledgeEventResponse(
        event_id=event.event_id, acknowledged_at=event.acknowledged_at
    )


@router.websocket("/{session_id}/events")
async def event_stream(
    websocket: WebSocket,
    session_id: UUID,
    service: InspectionService = Depends(get_inspection_service),
):
    if await service.get_session(session_id) is None:
        await websocket.close(code=4404, reason="Inspection run not found")
        return
    await service.broker.connect(session_id, websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await service.broker.disconnect(session_id, websocket)


@artifact_router.get("/{artifact_id}")
async def get_artifact(
    artifact_id: UUID,
    service: InspectionService = Depends(get_inspection_service),
):
    result = await service.databridge.get_artifact(artifact_id)
    if result is None:
        raise HTTPException(404, "Artifact not found")
    record, relative_path = result
    path = service.artifacts.resolve(relative_path)
    if not path.is_file():
        raise HTTPException(404, "Artifact file is unavailable")
    return FileResponse(path, media_type=record.mime_type, filename=record.original_filename)
