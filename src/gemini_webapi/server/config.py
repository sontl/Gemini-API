from __future__ import annotations

import os
from dataclasses import dataclass


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _parse_float(value: str | None, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:  # noqa: B904
        raise ValueError(f"Invalid float value: {value}") from exc


@dataclass(slots=True)
class AppConfig:
    secure_1psid: str
    secure_1psidts: str | None
    proxy: str | None
    image_output_dir: str
    timeout: float
    auto_close: bool
    close_delay: float
    auto_refresh: bool
    refresh_interval: float

    @classmethod
    def from_env(cls) -> "AppConfig":
        env = os.environ
        secure = env.get("SECURE_1PSID")
        if not secure:
            raise RuntimeError("SECURE_1PSID environment variable is required")

        secure_ts = env.get("SECURE_1PSIDTS") or None
        proxy = env.get("GEMINI_PROXY") or None
        output_dir = env.get("GEMINI_IMAGE_OUTPUT_DIR", "/data/outputs")
        timeout = _parse_float(env.get("GEMINI_TIMEOUT"), 300)
        auto_close = _parse_bool(env.get("GEMINI_AUTO_CLOSE"), False)
        close_delay = _parse_float(env.get("GEMINI_CLOSE_DELAY"), 300)
        auto_refresh = _parse_bool(env.get("GEMINI_AUTO_REFRESH"), True)
        refresh_interval = _parse_float(env.get("GEMINI_REFRESH_INTERVAL"), 540)

        return cls(
            secure_1psid=secure,
            secure_1psidts=secure_ts,
            proxy=proxy,
            image_output_dir=output_dir,
            timeout=timeout,
            auto_close=auto_close,
            close_delay=close_delay,
            auto_refresh=auto_refresh,
            refresh_interval=refresh_interval,
        )


__all__ = ["AppConfig"]
