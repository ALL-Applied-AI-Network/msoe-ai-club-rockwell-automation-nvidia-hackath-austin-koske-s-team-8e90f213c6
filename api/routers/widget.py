from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Response

from dependencies import get_widget_service
from models.widget import CreateWidgetRequest, UpdateWidgetRequest, Widget
from services.widget import WidgetService

router = APIRouter(prefix="/widgets", tags=["widgets"])


@router.get("", response_model=list[Widget])
async def list_widgets(limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0),
                       service: WidgetService = Depends(get_widget_service)):
    return await service.list_widgets(limit, offset)


@router.post("", response_model=Widget, status_code=201)
async def create_widget(body: CreateWidgetRequest, service: WidgetService = Depends(get_widget_service)):
    return await service.create_widget(body)


@router.get("/{widget_id}", response_model=Widget)
async def get_widget(widget_id: UUID, service: WidgetService = Depends(get_widget_service)):
    widget = await service.get_widget(widget_id)
    if widget is None:
        raise HTTPException(404, "Widget not found")
    return widget


@router.put("/{widget_id}", response_model=Widget)
async def update_widget(widget_id: UUID, body: UpdateWidgetRequest,
                        service: WidgetService = Depends(get_widget_service)):
    widget = await service.update_widget(widget_id, body)
    if widget is None:
        raise HTTPException(404, "Widget not found")
    return widget


@router.delete("/{widget_id}", status_code=204)
async def delete_widget(widget_id: UUID, service: WidgetService = Depends(get_widget_service)):
    if not await service.delete_widget(widget_id):
        raise HTTPException(404, "Widget not found")
    return Response(status_code=204)
