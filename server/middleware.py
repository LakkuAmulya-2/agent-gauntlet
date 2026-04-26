from __future__ import annotations

import os
import time
from collections import defaultdict, deque
from threading import Lock
from typing import Deque, Dict

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


def _is_exempt(path: str) -> bool:
    return (
        path in {"/health"}
        or path.startswith("/docs")
        or path.startswith("/openapi")
        or path.startswith("/web")
    )


def _is_openenv_core(path: str) -> bool:
    return path in {"/reset", "/step", "/state", "/ws", "/schema"}


def _get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    real_ip = request.headers.get("x-real-ip", "")
    if real_ip:
        return real_ip.strip()
    return request.client.host if request.client else "unknown"


class ApiKeyMiddleware(BaseHTTPMiddleware):
    """
    API key auth with fallback modes.

    Modes:
      - off: disable checks
      - optional: if header present, validate; if missing, allow
      - strict: require valid key on protected routes
    """

    def __init__(self, app):
        super().__init__(app)
        self._api_key = os.environ.get("OPSENV_API_KEY", "").strip()
        self._mode = os.environ.get("OPSENV_AUTH_MODE", "optional").strip().lower()
        self._require_for_openenv = os.environ.get("OPSENV_REQUIRE_AUTH_FOR_OPENENV", "false").strip().lower() == "true"

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if _is_exempt(path):
            return await call_next(request)
        if _is_openenv_core(path) and not self._require_for_openenv:
            return await call_next(request)
        if self._mode == "off" or not self._api_key:
            return await call_next(request)

        provided = request.headers.get("x-api-key", "")
        if self._mode == "optional" and not provided:
            return await call_next(request)
        if provided != self._api_key:
            return JSONResponse({"error": "unauthorized", "message": "Invalid or missing x-api-key"}, status_code=401)
        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding-window limiter with proxy/session/token-aware keys."""

    def __init__(self, app):
        super().__init__(app)
        self._max_requests = int(os.environ.get("OPSENV_RATE_LIMIT_MAX", "120"))
        self._window_s = int(os.environ.get("OPSENV_RATE_LIMIT_WINDOW_S", "60"))
        self._buckets: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = Lock()

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if _is_exempt(path):
            return await call_next(request)

        now = time.time()
        client_ip = _get_client_ip(request)
        session_id = request.headers.get("x-session-id", "").strip()
        token_fingerprint = request.headers.get("x-api-key", "").strip()
        scope = os.environ.get("OPSENV_RATE_LIMIT_SCOPE", "session_or_ip").strip().lower()
        if scope == "ip":
            identity = f"ip:{client_ip}"
        elif scope == "api_key" and token_fingerprint:
            identity = f"api:{token_fingerprint[:12]}"
        else:
            identity = f"session:{session_id}" if session_id else f"ip:{client_ip}"
        key = f"{identity}:{path}"
        with self._lock:
            bucket = self._buckets[key]
            while bucket and now - bucket[0] > self._window_s:
                bucket.popleft()
            if len(bucket) >= self._max_requests:
                return JSONResponse(
                    {
                        "error": "rate_limited",
                        "message": "Too many requests",
                        "max_requests": self._max_requests,
                        "window_s": self._window_s,
                    },
                    status_code=429,
                )
            bucket.append(now)

        return await call_next(request)
