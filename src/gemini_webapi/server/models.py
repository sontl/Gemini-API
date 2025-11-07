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
