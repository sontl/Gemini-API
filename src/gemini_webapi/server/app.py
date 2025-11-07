from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from httpx import HTTPError

from ..client import GeminiClient
from ..exceptions import (
    APIError,
    ImageGenerationError,
    ModelInvalid,
    TimeoutError,
    UsageLimitExceeded,
    TemporarilyBlocked,
)
from .config import AppConfig
from .models import (
    ConversationResponse,
    ContinueSessionRequest,
    StartSessionRequest,
)
from .service import (
    ImageEditingService,
    InvalidModelError,
    SessionNotFoundError,
    SessionStore,
)


def create_app(service: ImageEditingService | None = None) -> FastAPI:
    app = FastAPI(title="Gemini Image Editing API", version="1.0.0")

    app.state.service = service
    app.state.gemini_client = None
    app.state.session_store = None

    async def get_service(request: Request) -> ImageEditingService:
        svc = request.app.state.service
        if svc is None:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Service not ready")
        return svc

    @app.post("/sessions", response_model=ConversationResponse, status_code=status.HTTP_201_CREATED)
    async def start_session(
        payload: StartSessionRequest,
        service: ImageEditingService = Depends(get_service),
    ) -> ConversationResponse:
        try:
            return await service.start_session(
                payload.prompt,
                image_urls=[str(url) for url in payload.image_urls],
                model=payload.model,
                gem=payload.gem,
            )
        except InvalidModelError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid model: {exc.args[0]}") from exc
        except HTTPError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to fetch image: {exc}") from exc
        except (APIError, ImageGenerationError, TimeoutError, UsageLimitExceeded, TemporarilyBlocked, ModelInvalid) as exc:
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    @app.post("/sessions/{session_id}/messages", response_model=ConversationResponse)
    async def continue_session(
        session_id: str,
        payload: ContinueSessionRequest,
        service: ImageEditingService = Depends(get_service),
    ) -> ConversationResponse:
        try:
            return await service.continue_session(
                session_id,
                prompt=payload.prompt,
                image_urls=[str(url) for url in payload.image_urls],
            )
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Session {exc.args[0]} not found") from exc
        except HTTPError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to fetch image: {exc}") from exc
        except (APIError, ImageGenerationError, TimeoutError, UsageLimitExceeded, TemporarilyBlocked, ModelInvalid) as exc:
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    @app.get("/healthz")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.on_event("startup")
    async def on_startup() -> None:
        nonlocal service
        if service is not None:
            return

        config = AppConfig.from_env()
        client = GeminiClient(config.secure_1psid, config.secure_1psidts, proxy=config.proxy)
        await client.init(
            timeout=config.timeout,
            auto_close=config.auto_close,
            close_delay=config.close_delay,
            auto_refresh=config.auto_refresh,
            refresh_interval=config.refresh_interval,
        )
        store = SessionStore()
        app.state.gemini_client = client
        app.state.session_store = store
        app.state.service = ImageEditingService(
            client,
            store,
            output_dir=config.image_output_dir,
            base_url=config.image_base_url,
        )

        if config.image_base_url is None:
            directory = config.image_output_dir
            app.mount("/images", StaticFiles(directory=directory, html=False), name="images")

    @app.on_event("shutdown")
    async def on_shutdown() -> None:
        client: GeminiClient | None = app.state.gemini_client
        if client is not None:
            await client.close()

    @app.exception_handler(ValueError)
    async def value_error_handler(_: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"detail": str(exc)})

    return app


app = create_app()


__all__ = ["create_app", "app"]
