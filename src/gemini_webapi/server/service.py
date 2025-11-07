from __future__ import annotations

import asyncio
import base64
import mimetypes
import secrets
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from httpx import AsyncClient, HTTPError

from ..client import GeminiClient, ChatSession
from ..constants import Model
from ..types import Image, GeneratedImage
from .models import ConversationResponse, ImagePayload


class SessionNotFoundError(KeyError):
    pass


class InvalidModelError(ValueError):
    pass


@dataclass
class SessionData:
    metadata: list[str | None]
    model: str
    gem: str | None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, SessionData] = {}
        self._lock = asyncio.Lock()

    async def create(self, session_id: str, data: SessionData) -> None:
        async with self._lock:
            self._sessions[session_id] = data

    async def get(self, session_id: str) -> SessionData:
        async with self._lock:
            if session_id not in self._sessions:
                raise SessionNotFoundError(session_id)
            data = self._sessions[session_id]
            return SessionData(
                metadata=list(data.metadata),
                model=data.model,
                gem=data.gem,
                created_at=data.created_at,
                updated_at=data.updated_at,
            )

    async def update_metadata(self, session_id: str, metadata: Sequence[str | None]) -> None:
        async with self._lock:
            if session_id not in self._sessions:
                raise SessionNotFoundError(session_id)
            stored = self._sessions[session_id]
            stored.metadata = list(metadata)
            stored.updated_at = datetime.utcnow()


async def _fetch_bytes(url: str, *, cookies: dict[str, str] | None, proxy: str | None) -> tuple[bytes, str]:
    async with AsyncClient(http2=True, follow_redirects=True, cookies=cookies, proxy=proxy) as client:
        response = await client.get(url)
        response.raise_for_status()
        mime = response.headers.get("content-type", "application/octet-stream")
        return response.content, mime


@asynccontextmanager
async def _files_from_urls(urls: Sequence[str], *, proxy: str | None) -> AsyncIterator[list[str]]:
    if not urls:
        yield []
        return

    with TemporaryDirectory() as tmp_dir:
        dest = Path(tmp_dir)
        async with AsyncClient(http2=True, follow_redirects=True, proxy=proxy) as client:
            async def _download(url: str) -> str:
                response = await client.get(url)
                response.raise_for_status()
                filename = Path(url.split("?")[0]).name or "image"
                target = dest / filename
                suffix = 0
                while target.exists():
                    suffix += 1
                    target = dest / f"{Path(filename).stem}_{suffix}{Path(filename).suffix or '.bin'}"
                target.write_bytes(response.content)
                return str(target)

            try:
                files = await asyncio.gather(*(_download(url) for url in urls))
            except HTTPError:
                raise

        yield files


async def _serialize_images(
    images: Sequence[Image],
    *,
    proxy: str | None,
    output_dir: Path,
) -> list[ImagePayload]:
    if not images:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)

    async def _serialize(image: Image) -> ImagePayload:
        url = image.url
        cookies = getattr(image, "cookies", None)
        if isinstance(image, GeneratedImage) and not url.endswith("=s2048"):
            url = f"{url}=s2048"

        content, mime = await _fetch_bytes(url, cookies=cookies, proxy=proxy)
        extension = mimetypes.guess_extension(mime) or ".bin"
        filename = (
            f"{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}_"
            f"{secrets.token_hex(4)}{extension}"
        )
        file_path = output_dir / filename
        file_path.write_bytes(content)
        return ImagePayload(
            title=image.title,
            alt=image.alt,
            mime_type=mime,
            data=base64.b64encode(content).decode("ascii"),
            path=str(file_path.resolve()),
        )

    return await asyncio.gather(*(_serialize(image) for image in images))


class ImageEditingService:
    def __init__(
        self,
        client: GeminiClient,
        store: SessionStore,
        output_dir: str,
        base_url: str | None,
    ) -> None:
        self._client = client
        self._store = store
        self._output_dir = Path(output_dir)
        self._base_url = base_url.rstrip("/") if base_url else None

    async def start_session(
        self,
        prompt: str,
        *,
        image_urls: Sequence[str],
        model: str | None = None,
        gem: str | None = None,
    ) -> ConversationResponse:
        chat = self._client.start_chat(model=self._resolve_model(model), gem=gem)
        output = await self._send(chat, prompt, image_urls)

        session_id = output.session_id
        await self._store.create(
            session_id,
            SessionData(
                metadata=list(chat.metadata),
                model=self._model_name(chat.model),
                gem=self._gem_identifier(chat.gem),
            ),
        )
        return output

    async def continue_session(
        self,
        session_id: str,
        *,
        prompt: str,
        image_urls: Sequence[str],
    ) -> dict[str, Any]:
        stored = await self._store.get(session_id)
        chat = self._client.start_chat(
            model=self._resolve_model(stored.model),
            gem=stored.gem,
            metadata=list(stored.metadata),
        )
        output = await self._send(chat, prompt, image_urls, session_id=session_id)
        await self._store.update_metadata(session_id, chat.metadata)
        return output

    def _resolve_model(self, model: str | None) -> Model:
        if model is None:
            return Model.UNSPECIFIED
        try:
            return Model.from_name(model)
        except ValueError as exc:  # noqa: B904
            raise InvalidModelError(model) from exc

    def _model_name(self, model: Model | str | None) -> str:
        if isinstance(model, Model):
            return model.model_name
        if isinstance(model, str) and model:
            return model
        return Model.UNSPECIFIED.model_name

    def _gem_identifier(self, gem: Any) -> str | None:
        return getattr(gem, "id", gem)

    async def _send(
        self,
        chat: ChatSession,
        prompt: str,
        image_urls: Sequence[str],
        *,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        if not prompt:
            raise ValueError("prompt cannot be empty")

        async with _files_from_urls(image_urls, proxy=self._client.proxy) as files:
            output = await chat.send_message(prompt, files=files or None)

        images = await _serialize_images(
            output.images,
            proxy=self._client.proxy,
            output_dir=self._output_dir,
        )

        if self._base_url:
            for image in images:
                absolute_path = Path(image.path)
                try:
                    relative = absolute_path.relative_to(self._output_dir)
                except ValueError:
                    image.url = f"{self._base_url}/{absolute_path.name}"
                else:
                    image.url = f"{self._base_url}/{relative.as_posix()}"
        else:
            for image in images:
                absolute_path = Path(image.path)
                try:
                    relative = absolute_path.relative_to(self._output_dir)
                except ValueError:
                    image.url = f"/images/{absolute_path.name}"
                else:
                    image.url = f"/images/{relative.as_posix()}"

        return ConversationResponse(
            session_id=session_id or self._generate_session_id(),
            text=output.text,
            metadata=list(output.metadata),
            images=images,
            thoughts=output.thoughts,
        )

    def _generate_session_id(self) -> str:
        import secrets

        return secrets.token_urlsafe(16)


__all__ = [
    "ImageEditingService",
    "SessionStore",
    "SessionData",
    "SessionNotFoundError",
    "InvalidModelError",
]
