"""The FastAPI surface the dashboard talks to.

Endpoints, and what each is for:

    GET  /api/health                 liveness
    GET  /api/scenarios              the incidents you can launch (from the scenario file)
    GET  /api/prompts                the MCP server's workflow catalogue (discovered, not hardcoded)
    POST /api/runs                   start a run; returns its id immediately
    GET  /api/runs                   recent runs
    GET  /api/runs/{id}              one run's persisted state (RCA, spans, tool timeline)
    GET  /api/runs/{id}/stream       Server-Sent Events: the live (and replayed) event stream
    GET  /api/runs/{id}/approvals    approvals awaiting a decision
    POST /api/approvals/{id}         approve or deny a pending action

The run executes as a background task (see ``RunManager``); the stream and approval endpoints
observe and steer it. The stream replays history before going live, so a browser that connects
late — or reconnects after a drop — never misses events. That closes SSE's no-replay gap.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from aegis.api.manager import RunManager
from aegis.config import get_settings
from aegis.logging import configure as configure_logging
from aegis.orchestrator import RunConfig
from aegis.types import ApprovalDecision


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    manager = RunManager(settings=settings)
    _load_scenarios(app, settings.scenarios_path)
    app.state.manager = manager
    try:
        yield
    finally:
        await manager.shutdown()


app = FastAPI(title="Aegis", version="0.1.0", lifespan=lifespan)

# The dashboard is served from a different origin in dev (Vite on 5173). Locked to localhost
# origins; this is a local tool, not a public service.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["*"],
    allow_headers=["*"],
)


def _load_scenarios(app: FastAPI, path: Any) -> None:
    """Register the scenario set with the in-process MCP server, once at startup."""
    from mcp_server.server import load_scenarios

    app.state.scenarios = []
    try:
        from mcp_server.env.generator import load_scenarios as read

        app.state.scenarios = read(path)
        load_scenarios(path)
    except FileNotFoundError:
        pass


def manager(request: Request) -> RunManager:
    return request.app.state.manager  # type: ignore[no-any-return]


# --------------------------------------------------------------------------------------
# Request/response models
# --------------------------------------------------------------------------------------


class StartRunRequest(BaseModel):
    incident_id: str
    topology: str = Field(
        default="full",
        description="full | single_agent | no_verifier — chooses the ablation arm.",
    )
    mock: bool = Field(default=False, description="Use the offline mock model (no spend).")


class ApprovalRequestBody(BaseModel):
    decision: str = Field(description="approved | denied")
    reason: str = ""


# --------------------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------------------


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/scenarios")
async def scenarios(request: Request) -> list[dict[str, Any]]:
    """The incidents available to launch. Ground truth is deliberately not exposed."""
    return [
        {
            "id": s.id,
            "alert": s.alert,
            "fault_class": s.fault_class,  # shown as difficulty context, not a hint to the agents
            "with_injection": s.with_injection,
        }
        for s in request.app.state.scenarios
    ]


@app.get("/api/prompts")
async def prompts(request: Request) -> list[dict[str, Any]]:
    """The MCP server's workflow catalogue, discovered live via ``list_prompts``.

    The dashboard renders this rather than hardcoding a menu — the point of the prompt primitive
    is that the server owns the list of things it can be asked to do.
    """
    from aegis.mcp_client.session import connect_in_memory
    from mcp_server.server import mcp as mcp_server_app

    async with connect_in_memory(mcp_server_app) as client:
        return await client.list_prompts()


@app.post("/api/runs")
async def start_run(request: Request, body: StartRunRequest) -> dict[str, str]:
    mgr = manager(request)
    scenario = next((s for s in request.app.state.scenarios if s.id == body.incident_id), None)
    if scenario is None:
        raise HTTPException(404, f"unknown incident {body.incident_id!r}")

    run_config = {
        "full": RunConfig(),
        "single_agent": RunConfig.single_agent(),
        "no_verifier": RunConfig.no_verifier(),
    }.get(body.topology)
    if run_config is None:
        raise HTTPException(400, f"unknown topology {body.topology!r}")

    model = None
    mcp_server = None
    if body.mock:
        from evals.mock_model import MockModel
        from mcp_server.server import mcp as mcp_server_app
        from mcp_server.server import register_scenario

        register_scenario(scenario)
        model = MockModel(scenario=scenario)
        mcp_server = mcp_server_app

    active = mgr.start(
        scenario.id, scenario.alert, run_config=run_config, model=model, mcp_server=mcp_server
    )
    return {"run_id": active.run_id, "incident_id": scenario.id}


@app.get("/api/runs")
async def list_runs(request: Request, limit: int = 50) -> list[dict[str, Any]]:
    return manager(request).store.list_runs(limit)


@app.get("/api/runs/{run_id}")
async def get_run(request: Request, run_id: str) -> dict[str, Any]:
    mgr = manager(request)
    row = mgr.store.get_run(run_id)
    if row is None:
        raise HTTPException(404, f"unknown run {run_id!r}")
    return {
        "run": {**row, "rca": json.loads(row["rca_json"]) if row["rca_json"] else None},
        "spans": mgr.store.list_spans(run_id),
        "tool_calls": mgr.store.list_tool_calls(run_id),
        "approvals": mgr.pending_approvals(run_id),
    }


@app.get("/api/runs/{run_id}/approvals")
async def run_approvals(request: Request, run_id: str) -> list[dict[str, Any]]:
    return manager(request).pending_approvals(run_id)


@app.post("/api/approvals/{approval_id}")
async def resolve_approval(
    request: Request, approval_id: str, body: ApprovalRequestBody
) -> dict[str, Any]:
    mgr = manager(request)
    try:
        decision = ApprovalDecision(body.decision)
    except ValueError as exc:
        raise HTTPException(400, "decision must be 'approved' or 'denied'") from exc
    if decision not in (ApprovalDecision.APPROVED, ApprovalDecision.DENIED):
        raise HTTPException(400, "decision must be 'approved' or 'denied'")

    # We do not hold a run-id -> approval-id map, so try each active run's broker. Approval ids
    # are globally unique, so at most one resolves.
    for active in list(mgr._runs.values()):
        if mgr.resolve_approval(active.run_id, approval_id, decision, body.reason):
            return {"resolved": True, "run_id": active.run_id, "decision": decision.value}
    raise HTTPException(404, f"no pending approval {approval_id!r}")


@app.get("/api/runs/{run_id}/stream")
async def stream_run(request: Request, run_id: str, after: int = 0) -> EventSourceResponse:
    """SSE stream of a run's events, replaying from ``after`` then going live.

    ``after`` is the last sequence number the client saw; on reconnect the client passes it and
    receives only what it missed. Each SSE event carries its ``seq`` as the event id, so the
    browser's ``EventSource`` sets ``Last-Event-ID`` automatically.
    """
    mgr = manager(request)
    if mgr.store.get_run(run_id) is None:
        raise HTTPException(404, f"unknown run {run_id!r}")

    async def publisher() -> AsyncIterator[dict[str, Any]]:
        async for event in mgr.bus.subscribe(run_id, after_seq=after):
            if await request.is_disconnected():
                break
            yield {
                "id": str(event.get("seq", "")),
                "event": event.get("type", "message"),
                "data": json.dumps(event),
            }
            if event.get("type") == "run.finished":
                # Give the client a beat to render the terminal event before the stream closes.
                await asyncio.sleep(0)
                break

    return EventSourceResponse(publisher())
