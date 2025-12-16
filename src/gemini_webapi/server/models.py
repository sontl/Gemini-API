from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, HttpUrl


class StartSessionRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    image_urls: list[HttpUrl] = Field(default_factory=list)
    model: Optional[str] = None
    gem: Optional[str] = None


class ContinueSessionRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    image_urls: list[HttpUrl] = Field(default_factory=list)


class ImagePayload(BaseModel):
    title: str
    alt: str
    mime_type: str
    data: str
    path: str
    url: str | None = None


class ConversationResponse(BaseModel):
    session_id: str
    text: str
    metadata: list[str | None]
    images: list[ImagePayload]
    thoughts: str | None = None


class StartSessionAsyncRequest(BaseModel):
    """Request model for async session creation with webhook callback."""
    prompt: str = Field(..., min_length=1)
    image_urls: list[HttpUrl] = Field(default_factory=list)
    model: Optional[str] = None
    gem: Optional[str] = None
    webhook_url: HttpUrl = Field(..., description="URL to call when processing completes")


class TaskCreatedResponse(BaseModel):
    """Response returned immediately when an async task is created."""
    task_id: str
    status: str = "pending"
    message: str = "Task created successfully. Result will be sent to webhook URL."


class TaskStatusResponse(BaseModel):
    """Response for task status queries."""
    task_id: str
    status: str  # "pending", "processing", "completed", "failed"
    result: Optional[ConversationResponse] = None
    error: Optional[str] = None
    created_at: str
    updated_at: str
