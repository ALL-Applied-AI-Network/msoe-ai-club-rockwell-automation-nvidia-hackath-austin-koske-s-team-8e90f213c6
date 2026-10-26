from uuid import UUID

from databridge.memory_widgets import MemoryWidgetBridge
from models.widget import CreateWidgetRequest, UpdateWidgetRequest


class WidgetService:
    def __init__(self, databridge: MemoryWidgetBridge):
        self.databridge = databridge

    async def list_widgets(self, limit=50, offset=0):
        return await self.databridge.list(limit, offset)

    async def get_widget(self, widget_id: UUID):
        return await self.databridge.get(widget_id)

    async def create_widget(self, body: CreateWidgetRequest):
        return await self.databridge.create(body.name, body.description)

    async def update_widget(self, widget_id: UUID, body: UpdateWidgetRequest):
        changes = body.model_dump(exclude_unset=True)
        if changes.get("name") is None:
            changes.pop("name", None)
        return await self.databridge.update(widget_id, changes)

    async def delete_widget(self, widget_id: UUID):
        return await self.databridge.delete(widget_id)
