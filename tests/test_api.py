"""API contract tests.

Drive the FastAPI app through httpx's ASGI transport with the mock model, so the whole
request path is exercised — run launch, SSE stream, approval round-trip — with no API key and no
spend. The SSE test is the load-bearing one: it proves a client sees the run's events, including
the terminal event, and that an approval raised mid-run can be resolved over HTTP.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx
import pytest
from asgi_lifespan import LifespanManager

from aegis.api.app import app
from aegis.config import Settings, get_settings
from aegis.types import ToolPermission
from mcp_server.env.generator import generate


@pytest.fixture
async def client(tmp_path, monkeypatch) -> AsyncIterator[httpx.AsyncClient]:
    """An app wired to a temp DB and a fresh scenario file, mock-only."""
    scenarios, failures = generate(count=8, seed=2024)
    assert failures == []
    scenario_file = tmp_path / "scenarios.jsonl"
    scenario_file.write_text("\n".join(json.dumps(s.as_dict()) for s in scenarios))

    # One settings object for the whole app; get_settings is cached so patching it is global.
    settings = Settings(
        anthropic_api_key="test",
        db_path=tmp_path / "aegis.db",
        scenarios_path=scenario_file,
        tool_policy={"propose_remediation": ToolPermission.ASK},
        approval_timeout_s=10.0,
    )
    get_settings.cache_clear()
    monkeypatch.setattr("aegis.api.app.get_settings", lambda: settings)

    transport = httpx.ASGITransport(app=app)
    async with (
        LifespanManager(app),
        httpx.AsyncClient(transport=transport, base_url="http://test") as http_client,
    ):
        yield http_client
    get_settings.cache_clear()


async def _drain_stream(client: httpx.AsyncClient, run_id: str) -> list[dict]:
    """Collect SSE events for a run until run.finished."""
    events: list[dict] = []
    async with client.stream("GET", f"/api/runs/{run_id}/stream") as response:
        async for line in response.aiter_lines():
            if line.startswith("data:"):
                payload = json.loads(line[len("data:") :].strip())
                events.append(payload)
                if payload.get("type") == "run.finished":
                    break
    return events


# --------------------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------------------


async def test_health(client):
    assert (await client.get("/api/health")).json() == {"status": "ok"}


async def test_scenarios_are_listed_without_ground_truth(client):
    scenarios = (await client.get("/api/scenarios")).json()
    assert len(scenarios) == 8
    blob = json.dumps(scenarios)
    # Difficulty context is allowed; the causal chain and notes are not.
    assert "causal_event_ids" not in blob
    assert "ground_truth" not in blob
    assert all("alert" in s for s in scenarios)


async def test_prompts_come_from_the_mcp_server(client):
    prompts = (await client.get("/api/prompts")).json()
    names = {p["name"] for p in prompts}
    assert names == {"triage_incident", "write_postmortem", "severity_review"}
    # Arguments are described so the dashboard can render a form.
    triage = next(p for p in prompts if p["name"] == "triage_incident")
    assert any(a["name"] == "incident_id" for a in triage["arguments"])


# --------------------------------------------------------------------------------------
# Runs and streaming
# --------------------------------------------------------------------------------------


async def test_start_run_returns_immediately_and_streams_to_completion(client):
    scenarios = (await client.get("/api/scenarios")).json()
    incident = scenarios[0]["id"]

    start = await client.post(
        "/api/runs", json={"incident_id": incident, "topology": "single_agent", "mock": True}
    )
    assert start.status_code == 200
    run_id = start.json()["run_id"]

    events = await _drain_stream(client, run_id)
    types = [e["type"] for e in events]
    assert types[0] == "run.started"
    assert types[-1] == "run.finished"
    assert events[-1]["status"] == "succeeded"

    # The finished run is durably readable, with its RCA and spans.
    detail = (await client.get(f"/api/runs/{run_id}")).json()
    assert detail["run"]["status"] == "succeeded"
    assert detail["run"]["rca"]["root_cause_class"] != ""
    assert detail["spans"]


async def test_unknown_incident_is_rejected(client):
    resp = await client.post("/api/runs", json={"incident_id": "nope", "mock": True})
    assert resp.status_code == 404


async def test_a_late_subscriber_replays_via_the_after_cursor(client):
    """Connect after the run finishes; ?after=0 must replay the whole thing."""
    scenarios = (await client.get("/api/scenarios")).json()
    start = await client.post(
        "/api/runs",
        json={"incident_id": scenarios[1]["id"], "topology": "single_agent", "mock": True},
    )
    run_id = start.json()["run_id"]

    # Drain once to let the run finish.
    await _drain_stream(client, run_id)
    # Now a fresh subscriber from seq 0 still sees the full history.
    replay = await _drain_stream(client, run_id)
    assert replay[0]["type"] == "run.started"
    assert replay[-1]["type"] == "run.finished"


# --------------------------------------------------------------------------------------
# Human-in-the-loop over HTTP
# --------------------------------------------------------------------------------------


async def test_approval_surfaces_on_the_stream_and_can_be_resolved(client):
    """The full HITL round-trip: an ask tool fires, the operator sees it and approves.

    Uses the full topology so the coordinator reaches propose_remediation, which is policy `ask`.
    Polls the approvals endpoint rather than resolving inside an open SSE stream — that is how a
    real client behaves (a separate fetch POST), and it avoids the httpx ASGITransport limitation
    where a concurrent request during an open stream does not interleave.
    """
    import asyncio

    scenarios = (await client.get("/api/scenarios")).json()
    incident = scenarios[0]["id"]
    start = await client.post(
        "/api/runs", json={"incident_id": incident, "topology": "full", "mock": True}
    )
    run_id = start.json()["run_id"]

    # Poll until the coordinator's proposal is awaiting a decision, then approve it.
    approval_id = None
    for _ in range(200):
        pending = (await client.get(f"/api/runs/{run_id}/approvals")).json()
        if pending:
            approval_id = pending[0]["approval_id"]
            assert pending[0]["name"] == "propose_remediation"
            assert pending[0]["blast_radius"]
            break
        await asyncio.sleep(0.02)
    assert approval_id, "no approval was raised"

    approve = await client.post(
        f"/api/approvals/{approval_id}", json={"decision": "approved", "reason": "looks right"}
    )
    assert approve.status_code == 200
    assert approve.json()["decision"] == "approved"

    # The run now completes, and the event log records both the request and its resolution.
    for _ in range(200):
        row = (await client.get(f"/api/runs/{run_id}")).json()["run"]
        if row["status"] in ("succeeded", "failed", "budget_exceeded"):
            break
        await asyncio.sleep(0.02)
    assert row["status"] == "succeeded"

    events = await _drain_stream(client, run_id)
    types = [e["type"] for e in events]
    assert "approval.requested" in types
    assert "approval.resolved" in types
    resolution = next(e for e in events if e["type"] == "approval.resolved")
    assert resolution["decision"] == "approved"


async def test_resolving_an_unknown_approval_is_404(client):
    resp = await client.post("/api/approvals/apr-nope", json={"decision": "approved"})
    assert resp.status_code == 404


async def test_invalid_decision_is_rejected(client):
    # Even a well-formed but wrong decision value is a 400, before any lookup.
    resp = await client.post("/api/approvals/apr-x", json={"decision": "maybe"})
    assert resp.status_code == 400
