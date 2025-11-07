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

## Shutdown

```bash
docker compose down
```

Add `--volumes` to remove cached cookies:

```bash
docker compose down --volumes
```
