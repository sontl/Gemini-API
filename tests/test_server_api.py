from fastapi.testclient import TestClient
from httpx import HTTPError

from gemini_webapi.server.app import create_app
from gemini_webapi.server.models import ConversationResponse, ImagePayload
from gemini_webapi.server.service import InvalidModelError, SessionNotFoundError


class StubService:
    def __init__(self):
        self.started_payload = None
        self.continued_payload = None
        self.response = ConversationResponse(
            session_id="session-1",
            text="ok",
            metadata=["cid", "rid", "rcid"],
            thoughts=None,
            images=[
                ImagePayload(
                    title="t",
                    alt="",
                    mime_type="image/png",
                    data="ZGF0YQ==",
                    path="/tmp/image.png",
                    url="http://localhost/images/image.png",
                )
            ],
        )

    async def start_session(self, prompt, *, image_urls, model=None, gem=None):
        self.started_payload = {
            "prompt": prompt,
            "image_urls": image_urls,
            "model": model,
            "gem": gem,
        }
        return self.response

    async def continue_session(self, session_id, *, prompt, image_urls):
        self.continued_payload = {
            "session_id": session_id,
            "prompt": prompt,
            "image_urls": image_urls,
        }
        return self.response


def test_start_session_endpoint_success():
    service = StubService()
    app = create_app(service=service)
    client = TestClient(app)

    payload = {"prompt": "edit", "image_urls": ["https://example.com/a.png"]}
    response = client.post("/sessions", json=payload)

    assert response.status_code == 201
    assert response.json()["text"] == "ok"
    assert service.started_payload["image_urls"] == ["https://example.com/a.png"]


def test_start_session_invalid_model():
    class FailingService(StubService):
        async def start_session(self, *args, **kwargs):  # noqa: ARG002
            raise InvalidModelError("bad")

    app = create_app(service=FailingService())
    client = TestClient(app)

    response = client.post("/sessions", json={"prompt": "edit"})
    assert response.status_code == 400


def test_start_session_image_error():
    class FailingService(StubService):
        async def start_session(self, *args, **kwargs):  # noqa: ARG002
            raise HTTPError("boom")

    app = create_app(service=FailingService())
    client = TestClient(app)

    response = client.post("/sessions", json={"prompt": "edit"})
    assert response.status_code == 400


def test_continue_session_not_found():
    class MissingService(StubService):
        async def continue_session(self, *args, **kwargs):  # noqa: ARG002
            raise SessionNotFoundError("missing")

    app = create_app(service=MissingService())
    client = TestClient(app)

    response = client.post("/sessions/abc/messages", json={"prompt": "next"})
    assert response.status_code == 404


def test_continue_session_success():
    service = StubService()
    app = create_app(service=service)
    client = TestClient(app)

    response = client.post("/sessions/xyz/messages", json={"prompt": "next"})
    assert response.status_code == 200
    assert service.continued_payload["session_id"] == "xyz"


def test_start_session_async_returns_task_id():
    """Test that /sessions/async returns 202 with task_id immediately."""
    service = StubService()
    app = create_app(service=service)
    client = TestClient(app)

    payload = {
        "prompt": "edit async",
        "image_urls": ["https://example.com/a.png"],
        "webhook_url": "https://example.com/webhook",
    }
    response = client.post("/sessions/async", json=payload)

    assert response.status_code == 202
    data = response.json()
    assert "task_id" in data
    assert data["status"] == "pending"
    assert "message" in data


def test_start_session_async_missing_webhook():
    """Test that /sessions/async requires webhook_url."""
    service = StubService()
    app = create_app(service=service)
    client = TestClient(app)

    payload = {"prompt": "edit async"}  # Missing webhook_url
    response = client.post("/sessions/async", json=payload)

    assert response.status_code == 422  # Validation error


def test_get_task_status_not_found():
    """Test that /tasks/{task_id} returns 404 for unknown task."""
    service = StubService()
    app = create_app(service=service)
    client = TestClient(app)

    response = client.get("/tasks/unknown-task-id")
    assert response.status_code == 404


def test_get_task_status_after_async_start():
    """Test that task status is retrievable after async start."""
    service = StubService()
    app = create_app(service=service)
    client = TestClient(app)

    # Start async session
    payload = {
        "prompt": "edit async",
        "webhook_url": "https://example.com/webhook",
    }
    response = client.post("/sessions/async", json=payload)
    assert response.status_code == 202
    task_id = response.json()["task_id"]

    # Check task status
    status_response = client.get(f"/tasks/{task_id}")
    assert status_response.status_code == 200
    status_data = status_response.json()
    assert status_data["task_id"] == task_id
    assert status_data["status"] in ["pending", "processing", "completed"]

