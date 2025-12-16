from __future__ import annotations

import asyncio
import secrets

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from httpx import AsyncClient, HTTPError
from loguru import logger

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
    StartSessionAsyncRequest,
    TaskCreatedResponse,
    TaskStatusResponse,
)
from .service import (
    ImageEditingService,
    InvalidModelError,
    SessionNotFoundError,
    SessionStore,
    TaskStore,
    TaskData,
    TaskNotFoundError,
)


WEBHOOK_RETRY_ATTEMPTS = 3
WEBHOOK_RETRY_DELAY_SECONDS = 2


async def call_webhook(
    webhook_url: str,
    payload: dict,
    max_retries: int = WEBHOOK_RETRY_ATTEMPTS,
) -> bool:
    """Call a webhook URL with the given payload.
    
    Retries up to max_retries times with exponential backoff.
    Returns True if successful, False otherwise.
    """
    for attempt in range(max_retries):
        try:
            async with AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    webhook_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
                if response.status_code < 400:
                    logger.info(f"Webhook called successfully: {webhook_url}")
                    return True
                logger.warning(
                    f"Webhook returned status {response.status_code}, "
                    f"attempt {attempt + 1}/{max_retries}"
                )
        except Exception as e:
            logger.warning(
                f"Webhook call failed: {e}, attempt {attempt + 1}/{max_retries}"
            )
        
        if attempt < max_retries - 1:
            await asyncio.sleep(WEBHOOK_RETRY_DELAY_SECONDS * (2 ** attempt))
    
    logger.error(f"Failed to call webhook after {max_retries} attempts: {webhook_url}")
    return False


async def process_task_and_notify(
    task_id: str,
    task_store: TaskStore,
    service: ImageEditingService,
    request_data: dict,
    webhook_url: str,
) -> None:
    """Process a session task in the background and notify via webhook."""
    try:
        await task_store.update_status(task_id, "processing")
        
        result = await service.start_session(
            request_data["prompt"],
            image_urls=request_data["image_urls"],
            model=request_data.get("model"),
            gem=request_data.get("gem"),
        )
        
        await task_store.update_status(task_id, "completed", result=result)
        
        # Debug logging
        logger.debug(f"Task {task_id} result type: {type(result)}")
        logger.debug(f"Task {task_id} result images count: {len(result.images)}")
        logger.debug(f"Task {task_id} result text: {result.text[:100] if result.text else 'empty'}")
        
        result_dict = result.model_dump()
        logger.debug(f"Task {task_id} serialized images count: {len(result_dict.get('images', []))}")
        
        webhook_payload = {
            "task_id": task_id,
            "status": "completed",
            "result": result_dict,
        }
        await call_webhook(webhook_url, webhook_payload)
        
    except (InvalidModelError, HTTPError, APIError, ImageGenerationError, 
            TimeoutError, UsageLimitExceeded, TemporarilyBlocked, ModelInvalid) as exc:
        error_message = str(exc)
        logger.error(f"Task {task_id} failed: {error_message}")
        
        await task_store.update_status(task_id, "failed", error=error_message)
        
        webhook_payload = {
            "task_id": task_id,
            "status": "failed",
            "error": error_message,
        }
        await call_webhook(webhook_url, webhook_payload)
        
    except Exception as exc:
        error_message = f"Unexpected error: {exc}"
        logger.exception(f"Task {task_id} failed with unexpected error")
        
        await task_store.update_status(task_id, "failed", error=error_message)
        
        webhook_payload = {
            "task_id": task_id,
            "status": "failed",
            "error": error_message,
        }
        await call_webhook(webhook_url, webhook_payload)


def create_app(service: ImageEditingService | None = None) -> FastAPI:
    app = FastAPI(title="Gemini Image Editing API", version="1.0.0")

    app.state.service = service
    app.state.gemini_client = None
    app.state.session_store = None
    app.state.task_store = TaskStore()

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

    @app.post("/sessions/async", response_model=TaskCreatedResponse, status_code=status.HTTP_202_ACCEPTED)
    async def start_session_async(
        request: Request,
        payload: StartSessionAsyncRequest,
        service: ImageEditingService = Depends(get_service),
    ) -> TaskCreatedResponse:
        """Start a session asynchronously. Returns immediately with a task_id.
        
        The result will be sent to the provided webhook_url when processing completes.
        """
        task_id = secrets.token_urlsafe(16)
        task_store: TaskStore = request.app.state.task_store
        
        task_data = TaskData(
            status="pending",
            request={
                "prompt": payload.prompt,
                "image_urls": [str(url) for url in payload.image_urls],
                "model": payload.model,
                "gem": payload.gem,
            },
            webhook_url=str(payload.webhook_url),
        )
        await task_store.create(task_id, task_data)
        
        # Use asyncio.create_task instead of BackgroundTasks for proper async execution
        asyncio.create_task(
            process_task_and_notify(
                task_id=task_id,
                task_store=task_store,
                service=service,
                request_data=task_data.request,
                webhook_url=task_data.webhook_url,
            )
        )
        
        return TaskCreatedResponse(
            task_id=task_id,
            status="pending",
            message="Task created successfully. Result will be sent to webhook URL.",
        )

    @app.get("/tasks/{task_id}", response_model=TaskStatusResponse)
    async def get_task_status(
        request: Request,
        task_id: str,
    ) -> TaskStatusResponse:
        """Get the current status of an async task."""
        task_store: TaskStore = request.app.state.task_store
        try:
            task = await task_store.get(task_id)
            return TaskStatusResponse(
                task_id=task_id,
                status=task.status,
                result=task.result,
                error=task.error,
                created_at=task.created_at.isoformat(),
                updated_at=task.updated_at.isoformat(),
            )
        except TaskNotFoundError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task {task_id} not found",
            )

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
