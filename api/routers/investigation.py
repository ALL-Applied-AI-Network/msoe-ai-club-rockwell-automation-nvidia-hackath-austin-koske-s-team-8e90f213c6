import json
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request

from dependencies import get_inspection_service
from services.investigation import current_investigation, example_index, import_investigation, synthetic_example

router = APIRouter(tags=["investigation"])


@router.get("/investigation/examples")
async def examples():
    return example_index()


@router.get("/investigation/examples/{case_id}")
async def example(case_id: str):
    try:
        return synthetic_example(case_id)
    except ValueError as error:
        raise HTTPException(404, str(error)) from error


@router.get("/sessions/{session_id}/investigation")
async def get_investigation(session_id: UUID, service=Depends(get_inspection_service)):
    session = await service.get_session(session_id)
    if session is None:
        raise HTTPException(404, "Inspection run not found")
    return await current_investigation(service, session)


@router.post("/sessions/{session_id}/investigation", status_code=201)
async def post_investigation(session_id: UUID, request: Request, service=Depends(get_inspection_service)):
    session = await service.get_session(session_id)
    if session is None:
        raise HTTPException(404, "Inspection run not found")
    content = bytearray()
    async for chunk in request.stream():
        content.extend(chunk)
        if len(content) > 256 * 1024:
            raise HTTPException(413, "Completed-vial input exceeds 256 KB")
    try:
        return await import_investigation(service, session, json.loads(content))
    except (ValueError, TypeError, KeyError, IndexError) as error:
        raise HTTPException(422, str(error)) from error
