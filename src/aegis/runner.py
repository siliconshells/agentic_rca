"""Top-level run assembly.

Wires a run together — MCP connection, store, model client, budget, permission gate,
orchestrator — behind one async entry point that both the CLI and the API call. Everything that
must live for exactly one run and be torn down after it goes here, so neither caller has to know
the assembly order.

The MCP connection is the constraint that shapes this: it must be opened and closed inside a
single task (see ``mcp_client.session``), so the whole run body executes inside its context
manager rather than holding the client as instance state.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from aegis.config import Settings, get_settings
from aegis.harness import telemetry
from aegis.harness.budget import Budget
from aegis.harness.client import AnthropicModelClient, ModelClient
from aegis.harness.events import EventBus
from aegis.harness.loop import Harness
from aegis.harness.permissions import ApprovalBroker, AutoDeny, PermissionGate
from aegis.logging import bind, get_logger
from aegis.mcp_client.session import McpClient, connect, connect_in_memory
from aegis.orchestrator import Orchestrator, RunConfig, RunResult
from aegis.store.db import Store

log = get_logger(__name__)


@dataclass
class RunRequest:
    incident_id: str
    alert: str
    run_id: str = ""
    run_config: RunConfig | None = None
    resume: bool = False

    def resolved_run_id(self) -> str:
        return self.run_id or f"run-{uuid.uuid4().hex[:12]}"


async def execute_run(
    request: RunRequest,
    *,
    settings: Settings | None = None,
    store: Store | None = None,
    bus: EventBus | None = None,
    model: ModelClient | None = None,
    broker: ApprovalBroker | None = None,
    mcp_server: object | None = None,
) -> RunResult:
    """Run one incident end to end.

    Every collaborator is injectable: the CLI passes a real client and an HTTP MCP connection,
    the eval harness passes the mock model and an in-process server, and tests pass both plus a
    scripted broker. The orchestration in between is identical in all three.
    """
    settings = settings or get_settings()
    run_id = request.resolved_run_id()
    owns_store = store is None
    store = store or Store(settings.db_path)
    bus = bus or EventBus(store)
    # Fail closed: with no broker supplied, world-mutating (`ask`) tools are denied rather than
    # silently auto-approved. Callers opt into an approval mechanism explicitly — the CLI a
    # terminal prompt, the API its queue broker, unattended evals AutoApprove.
    broker = broker or AutoDeny()
    model = model or AnthropicModelClient(settings.anthropic_api_key, settings.retry)

    telemetry.configure(settings.otel_endpoint, settings.otel_service_name)

    if store.get_run(run_id) is None:
        store.create_run(run_id, request.incident_id, request.alert)

    budget = Budget(settings.budget)
    gate = PermissionGate(settings.tool_policy, settings.default_tool_permission, broker)

    async def with_client(mcp: McpClient) -> RunResult:
        harness = Harness(
            settings=settings,
            client=model,
            mcp=mcp,
            store=store,
            bus=bus,
            budget=budget,
            permissions=gate,
            run_id=run_id,
            resume=request.resume,
        )
        orchestrator = Orchestrator(
            harness=harness, settings=settings, budget=budget, run_config=request.run_config
        )
        return await orchestrator.run(request.incident_id, request.alert)

    try:
        with bind(run_id=run_id):
            if mcp_server is not None:
                async with connect_in_memory(mcp_server) as mcp:
                    result = await with_client(mcp)
            else:
                async with connect(
                    url=settings.mcp_url,
                    stdio=settings.mcp_stdio,
                    scenarios_path=None,
                ) as mcp:
                    result = await with_client(mcp)
    finally:
        if isinstance(model, AnthropicModelClient):
            await model.aclose()
        if owns_store:
            store.close()

    return result
