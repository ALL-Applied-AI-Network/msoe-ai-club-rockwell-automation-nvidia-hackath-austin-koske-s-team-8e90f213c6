from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class Widget(BaseModel):
    """Example resource. Copy this shape — model + service + router — for
    your first real domain object, then delete widgets once you don't need
    the reference anymore."""

    widget_id: UUID
    name: str
    description: str | None = None
    created_at: datetime
    updated_at: datetime


class CreateWidgetRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    name: str = Field(..., min_length=1, max_length=200)
    description: str | None = Field(None, max_length=2000)


class UpdateWidgetRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    name: str | None = Field(None, min_length=1, max_length=200)
    description: str | None = Field(None, max_length=2000)
