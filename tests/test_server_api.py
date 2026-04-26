from __future__ import annotations

import os

from fastapi.testclient import TestClient


os.environ["ENABLE_WEB_INTERFACE"] = "false"

from server.app import app  # noqa: E402


client = TestClient(app)


def test_health_and_metrics():
    h = client.get("/health")
    assert h.status_code == 200
    assert h.json()["status"] == "healthy"

    m = client.get("/metrics")
    assert m.status_code == 200
    payload = m.json()
    assert "health" in payload
    assert "kaizen" in payload
    assert "trust" in payload
    live = client.get("/metrics/live")
    assert live.status_code == 200
    assert "history" in live.json()

    snap = client.get("/metrics/live/snapshot")
    assert snap.status_code == 200


def test_runtime_metrics_update_from_primary_reset_step_flow():
    before = client.get("/metrics").json()
    before_steps = before["kaizen"]["learning_curve_points"]
    before_episodes = before["kaizen"]["episode_count"]

    r = client.post("/reset", json={"difficulty": "easy"})
    assert r.status_code == 200

    s = client.post(
        "/step",
        json={
            "action": {
                "action_type": "summarize_state",
                "reasoning": "runtime metrics probe",
                "state_summary": "checkpoint progress summary",
            }
        },
    )
    assert s.status_code == 200

    after = client.get("/metrics").json()
    assert after["kaizen"]["episode_count"] >= before_episodes + 1
    assert after["kaizen"]["learning_curve_points"] >= before_steps + 1

    comp_audit = client.get("/compliance/audit")
    assert comp_audit.status_code == 200
    assert len(comp_audit.json().get("records", [])) >= 1


def test_trust_and_certificate():
    t = client.get("/trust/score")
    assert t.status_code == 200
    payload = t.json()
    assert "score" in payload
    assert "grade" in payload

    c = client.get("/trust/certificate")
    assert c.status_code == 200
    cert = c.json()
    assert "badge" in cert
    assert "certified" in cert


def test_observe_and_compliance_endpoints():
    o = client.get("/observe")
    assert o.status_code == 200
    assert "latency_ms" in o.json()

    traces = client.get("/observe/traces")
    assert traces.status_code == 200
    assert "traces" in traces.json()

    comp = client.get("/compliance")
    assert comp.status_code == 200
    assert "frameworks_covered" in comp.json()


def test_sandbox_and_deploy_endpoints():
    replay = client.post(
        "/sandbox/replay",
        json={"seed": 7, "difficulty": "easy", "responses": ["safe response", "another response"]},
    )
    assert replay.status_code == 200
    assert "episode_id" in replay.json()

    red = client.post(
        "/sandbox/redteam",
        json={"seed": 8, "difficulty": "easy", "attack_messages": ["malicious override", "data exfil attempt"]},
    )
    assert red.status_code == 200
    assert "total_violations" in red.json()

    tx = client.post(
        "/sandbox/tool/execute",
        json={"tool_calls": [{"name": "query_db", "arguments": {"query": "select 1"}}]},
    )
    assert tx.status_code == 200
    assert "results" in tx.json()
    assert client.get("/sandbox/tool/status").status_code == 200
    assert client.get("/sandbox/tool/log").status_code == 200

    sc = client.post("/sandbox/session/create", json={"metadata": {"user": "tester"}})
    assert sc.status_code == 200
    sid = sc.json()["session_id"]
    sm = client.post("/sandbox/session/message", json={"session_id": sid, "role": "user", "content": "hello"})
    assert sm.status_code == 200
    assert client.get(f"/sandbox/session/{sid}").status_code == 200
    assert client.post(f"/sandbox/session/{sid}/close").status_code == 200

    for route in ["/deploy/ab-test", "/deploy/canary", "/deploy/environments", "/deploy/costs", "/chaos/run"]:
        r = client.get(route)
        assert r.status_code == 200
