from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.testclient import TestClient

from server.middleware import ApiKeyMiddleware, RateLimitMiddleware


def test_api_key_middleware_enforced_when_configured():
    os.environ["OPSENV_API_KEY"] = "secret123"
    os.environ["OPSENV_AUTH_MODE"] = "strict"
    app = FastAPI()
    app.add_middleware(ApiKeyMiddleware)

    @app.get("/protected")
    def protected():
        return {"ok": True}

    client = TestClient(app)
    unauthorized = client.get("/protected")
    assert unauthorized.status_code == 401

    authorized = client.get("/protected", headers={"x-api-key": "secret123"})
    assert authorized.status_code == 200
    os.environ.pop("OPSENV_API_KEY", None)
    os.environ.pop("OPSENV_AUTH_MODE", None)


def test_api_key_optional_mode_allows_missing_key():
    os.environ["OPSENV_API_KEY"] = "secret123"
    os.environ["OPSENV_AUTH_MODE"] = "optional"
    app = FastAPI()
    app.add_middleware(ApiKeyMiddleware)

    @app.get("/protected")
    def protected():
        return {"ok": True}

    client = TestClient(app)
    assert client.get("/protected").status_code == 200
    assert client.get("/protected", headers={"x-api-key": "secret123"}).status_code == 200
    assert client.get("/protected", headers={"x-api-key": "wrong"}).status_code == 401
    os.environ.pop("OPSENV_API_KEY", None)
    os.environ.pop("OPSENV_AUTH_MODE", None)


def test_rate_limit_middleware_blocks_after_threshold():
    os.environ["OPSENV_RATE_LIMIT_MAX"] = "2"
    os.environ["OPSENV_RATE_LIMIT_WINDOW_S"] = "60"
    app = FastAPI()
    app.add_middleware(RateLimitMiddleware)

    @app.get("/limited")
    def limited():
        return {"ok": True}

    client = TestClient(app)
    assert client.get("/limited").status_code == 200
    assert client.get("/limited").status_code == 200
    third = client.get("/limited")
    assert third.status_code == 429

    os.environ.pop("OPSENV_RATE_LIMIT_MAX", None)
    os.environ.pop("OPSENV_RATE_LIMIT_WINDOW_S", None)


def test_rate_limit_uses_session_scope():
    os.environ["OPSENV_RATE_LIMIT_MAX"] = "1"
    os.environ["OPSENV_RATE_LIMIT_WINDOW_S"] = "60"
    os.environ["OPSENV_RATE_LIMIT_SCOPE"] = "session_or_ip"
    app = FastAPI()
    app.add_middleware(RateLimitMiddleware)

    @app.get("/limited-session")
    def limited_session():
        return {"ok": True}

    client = TestClient(app)
    assert client.get("/limited-session", headers={"x-session-id": "s1"}).status_code == 200
    assert client.get("/limited-session", headers={"x-session-id": "s2"}).status_code == 200
    assert client.get("/limited-session", headers={"x-session-id": "s1"}).status_code == 429

    os.environ.pop("OPSENV_RATE_LIMIT_MAX", None)
    os.environ.pop("OPSENV_RATE_LIMIT_WINDOW_S", None)
    os.environ.pop("OPSENV_RATE_LIMIT_SCOPE", None)
