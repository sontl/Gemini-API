import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from gemini_webapi.constants import Model
from gemini_webapi.server import service as service_module
from gemini_webapi.server.models import ConversationResponse, ImagePayload
from gemini_webapi.server.service import (
    ImageEditingService,
    InvalidModelError,
    SessionNotFoundError,
    SessionStore,
)
from gemini_webapi.types import Candidate, ModelOutput, GeneratedImage


class DummyChat:
    def __init__(self, output: ModelOutput, *, model: Model | None = None, gem: str | None = None, metadata=None):
        self._output = output
        self.model = model or Model.UNSPECIFIED
        self.gem = gem
        self.metadata = list(metadata) if metadata else [None, None, None]

    async def send_message(self, prompt: str, files=None):  # noqa: ARG002
        self.metadata = list(self._output.metadata)
        return self._output


class DummyClient:
    proxy = None

    def __init__(self, outputs: list[ModelOutput]):
        self._outputs = outputs

    def start_chat(self, **kwargs):
        if not self._outputs:
            raise RuntimeError("No more outputs available")
        output = self._outputs.pop(0)
        return DummyChat(output, model=kwargs.get("model"), gem=kwargs.get("gem"), metadata=kwargs.get("metadata"))


@pytest.fixture(autouse=True)
def patch_helpers(monkeypatch, tmp_path):
    @asynccontextmanager
    async def fake_files(urls, proxy=None):  # noqa: ARG001
        yield ["/tmp/fake.png" for _ in urls]

    async def fake_serialize(images, proxy=None, output_dir=None):  # noqa: ARG001
        target_dir = Path(output_dir) if output_dir else tmp_path
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / "image.png"
        path.write_bytes(b"data")
        return [
            ImagePayload(
                title=image.title,
                alt=image.alt,
                mime_type="image/png",
                data="ZGF0YQ==",
                path=str(path),
                url="http://localhost/images/image.png",
            )
            for image in images
        ]

    monkeypatch.setattr(service_module, "_files_from_urls", fake_files)
    monkeypatch.setattr(service_module, "_serialize_images", fake_serialize)


def make_output(text: str, metadata, with_image: bool = False):
    generated_images = []
    if with_image:
        generated_images = [
            GeneratedImage(
                url="https://example.com/image",
                title="Generated",
                alt="",
                proxy=None,
                cookies={"__Secure-1PSID": "cookie"},
            )
        ]

    return ModelOutput(
        metadata=metadata,
        candidates=[
            Candidate(
                rcid="rcid",
                text=text,
                web_images=[],
                generated_images=generated_images,
            )
        ],
    )


def test_start_session_creates_store_entry():
    output = make_output("Edited image", ["cid1", "rid1", "rcid1"], with_image=True)
    client = DummyClient([output])
    store = SessionStore()
    service = ImageEditingService(client, store, output_dir="/tmp", base_url="http://localhost/images")

    response = asyncio.run(service.start_session("prompt", image_urls=["https://example.com/image.png"]))

    assert isinstance(response, ConversationResponse)
    assert response.text == "Edited image"
    assert response.images[0].path
    assert response.images[0].url
    stored = asyncio.run(store.get(response.session_id))
    assert stored.metadata == ["cid1", "rid1", "rcid1"]


def test_start_session_generates_relative_url(tmp_path):
    output = make_output("Edited image", ["cid1", "rid1", "rcid1"], with_image=True)
    client = DummyClient([output])
    store = SessionStore()
    service = ImageEditingService(client, store, output_dir=str(tmp_path), base_url=None)

    response = asyncio.run(service.start_session("prompt", image_urls=[]))

    assert response.images[0].url.startswith("/images/")


def test_continue_session_updates_metadata():
    first_output = make_output("First", ["cid1", "rid1", "rcid1"])
    second_output = make_output("Second", ["cid2", "rid2", "rcid2"])
    client = DummyClient([first_output, second_output])
    store = SessionStore()
    service = ImageEditingService(client, store, output_dir="/tmp", base_url="http://localhost/images")

    first_response = asyncio.run(service.start_session("prompt", image_urls=[]))

    second_response = asyncio.run(
        service.continue_session(first_response.session_id, prompt="next", image_urls=[])
    )

    assert second_response.text == "Second"
    stored = asyncio.run(store.get(first_response.session_id))
    assert stored.metadata == ["cid2", "rid2", "rcid2"]


def test_continue_session_missing(monkeypatch):
    client = DummyClient([make_output("text", ["cid", "rid", "rcid"])])
    store = SessionStore()
    service = ImageEditingService(client, store, output_dir="/tmp", base_url="http://localhost/images")

    with pytest.raises(SessionNotFoundError):
        asyncio.run(service.continue_session("missing", prompt="p", image_urls=[]))


def test_invalid_model_raises_error():
    client = DummyClient([make_output("text", ["cid", "rid", "rcid"])])
    store = SessionStore()
    service = ImageEditingService(client, store, output_dir="/tmp", base_url="http://localhost/images")

    with pytest.raises(InvalidModelError):
        asyncio.run(service.start_session("prompt", image_urls=[], model="invalid-model"))
