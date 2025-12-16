# Docker Compose Usage Guide

## Prerequisites

1. Copy your Gemini cookies into environment variables:
   - `SECURE_1PSID` (required)
   - `SECURE_1PSIDTS` (optional but recommended)
2. Optionally define:
   - `GEMINI_PROXY` for outbound requests (e.g. `http://proxy:8080`).
   - `GEMINI_AUTO_REFRESH`, `GEMINI_REFRESH_INTERVAL`, `GEMINI_TIMEOUT`, `GEMINI_AUTO_CLOSE`, `GEMINI_CLOSE_DELAY` for runtime tuning.

Create a `.env` file alongside `docker-compose.yml` to store these values (all required unless noted):

```
SECURE_1PSID=your_cookie_value
SECURE_1PSIDTS=optional_cookie_value
GEMINI_AUTO_REFRESH=true
GEMINI_REFRESH_INTERVAL=540
GEMINI_TIMEOUT=300
GEMINI_AUTO_CLOSE=false
GEMINI_CLOSE_DELAY=300
GEMINI_IMAGE_BASE_URL=http://localhost:8000/images
# GEMINI_PROXY=http://proxy:8080
```

## Starting the Service

```bash
docker compose up --build -d
```

This command builds the API image, starts the container, and exposes port `8000` for HTTP access. The `gemini_cookies` volume automatically persists refreshed cookies at `/data/cookies` inside the container.

## Verifying Deployment

Check health:

```bash
curl http://localhost:8000/healthz
```

Send an initial image-editing request:

```bash
curl -X POST http://localhost:8000/sessions \
  -H "Content-Type: application/json" \
  -d '{
        "prompt": "Apply a watercolor style",
        "image_urls": ["https://sgl.com.vn/wp-content/uploads/2023/12/kien-truc-nha-xua-02.jpg"],
        "model": "gemini-2.5-flash"
      }'
```

Subsequent conversational turn:

```bash
curl -X POST http://localhost:8000/sessions/<SESSION_ID>/messages \
  -H "Content-Type: application/json" \
  -d '{
        "prompt": "Make the colors warmer"
      }'
```

Images are saved inside the container under `/data/outputs` and exposed via HTTP. If `GEMINI_IMAGE_BASE_URL` is unset, responses include relative URLs like `/images/<file>` that map to `http://localhost:8000/images/<file>` by default.

## Async Webhook Pattern (for Long-Running Requests)

For requests that may take 1-5 minutes to complete (which can cause timeouts when served through proxies like Cloudflare), use the async webhook endpoints:

### Start Async Session

```bash
curl -X POST http://localhost:8000/sessions/async \
  -H "Content-Type: application/json" \
  -d '{
        "prompt": "Apply a watercolor style",
        "image_urls": ["https://sgl.com.vn/wp-content/uploads/2023/12/kien-truc-nha-xua-02.jpg"],
        "model": "gemini-2.5-flash",
        "webhook_url": "https://your-server.com/webhook"
      }'
```

**Response (202 Accepted):**
```json
{
  "task_id": "abc123xyz",
  "status": "pending",
  "message": "Task created successfully. Result will be sent to webhook URL."
}
```

### Check Task Status (Optional)

```bash
curl http://localhost:8000/tasks/abc123xyz
```

**Response:**
```json
{
  "task_id": "abc123xyz",
  "status": "processing",
  "result": null,
  "error": null,
  "created_at": "2025-12-16T09:48:00",
  "updated_at": "2025-12-16T09:48:05"
}
```

### Webhook Callback

When processing completes, your webhook endpoint will receive a POST request:

**On Success:**
```json
{
  "task_id": "abc123xyz",
  "status": "completed",
  "result": {
    "session_id": "...",
    "text": "...",
    "images": [...],
    "metadata": [...],
    "thoughts": null
  }
}
```

**On Failure:**
```json
{
  "task_id": "abc123xyz",
  "status": "failed",
  "error": "Error message describing what went wrong"
}
```

> [!NOTE]
>
> The webhook is called with up to 3 retry attempts using exponential backoff (2s, 4s, 8s delays). Make sure your webhook endpoint can handle POST requests with JSON payloads.

## Shutdown

```bash
docker compose down
```

Add `--volumes` to remove cached cookies:

```bash
docker compose down --volumes
```
