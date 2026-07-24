"""Test scaffolding: a real Harness wired to an in-memory MCP server and a scripted model.

Everything except the model and the transport is the production path — the same loop, store,
budget, permission gate, and event bus the CLI uses. That is deliberate: a test that exercises a
parallel implementation proves nothing about the one that ships.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aegis.config import BudgetConfig, RetryConfig, Settings
from aegis.harness.budget import Budget
from aegis.harness.client import DeltaCallback, ModelRequest, ModelResponse
from aegis.harness.events import EventBus
from aegis.harness.loop import Harness
from aegis.harness.permissions import ApprovalBroker, AutoApprove, PermissionGate
from aegis.mcp_client.session import McpClient, connect_in_memory
from aegis.store.db import Store
from aegis.types import ToolPermission, Usage
from mcp_server.env.faults import Scenario, make_scenario
from mcp_server.server import mcp as mcp_server_app
from mcp_server.server import register_scenario


@dataclass
class ScriptedModel:
    """Returns a fixed list of responses, one per call.

    Used where a test needs to force an exact `stop_reason` sequence that a smarter mock would
    never produce — truncation, pauses, runaway loops.
    """

    responses: list[ModelResponse]
    calls: list[ModelRequest] = field(default_factory=list)
    repeat_last: bool = False

    async def complete(
        self,
        request: ModelRequest,
        on_text: DeltaCallback | None = None,
        on_thinking: DeltaCallback | None = None,
    ) -> ModelResponse:
        self.calls.append(request)
        index = len(self.calls) - 1
        if index >= len(self.responses):
            if not self.repeat_last:
                raise AssertionError(
                    f"model called {len(self.calls)} times but only "
                    f"{len(self.responses)} responses were scripted"
                )
            index = len(self.responses) - 1
        response = self.responses[index]
        if on_text and response.text:
            on_text(response.text)
        return response


def response(
    *content: dict[str, Any],
    stop_reason: str = "end_turn",
    usage: Usage | None = None,
) -> ModelResponse:
    return ModelResponse(
        content=list(content),
        stop_reason=stop_reason,
        usage=usage or Usage(input_tokens=100, output_tokens=50),
        model="scripted",
    )


def text_block(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def thinking_block(text: str, signature: str = "sig-abc123") -> dict[str, Any]:
    return {"type": "thinking", "thinking": text, "signature": signature}


def tool_block(name: str, tool_input: dict[str, Any], block_id: str) -> dict[str, Any]:
    return {"type": "tool_use", "id": block_id, "name": name, "input": tool_input}


@dataclass
class Rig:
    """Everything a test may want to assert against."""

    harness: Harness
    store: Store
    bus: EventBus
    budget: Budget
    mcp: McpClient
    scenario: Scenario
    run_id: str

    def events(self, event_type: str | None = None) -> list[dict[str, Any]]:
        payloads = [p for _, p in self.store.list_events(self.run_id)]
        if event_type is None:
            return payloads
        return [p for p in payloads if p.get("type") == event_type]


@asynccontextmanager
async def make_rig(
    tmp_path: Path,
    model: Any,
    *,
    scenario: Scenario | None = None,
    policy: dict[str, ToolPermission] | None = None,
    broker: ApprovalBroker | None = None,
    resume: bool = False,
    max_cost_usd: float = 10.0,
    max_turns: int | None = None,
    resources: dict[str, str] | None = None,
    run_id: str = "run-test",
) -> AsyncIterator[Rig]:
    """Build a Harness backed by the real MCP server over the in-memory transport."""
    scenario = (
        scenario
        or make_scenario(
            "inc-rig", seed=4242, fault_class="bad_deploy", culprit="payments-svc", onset=50
        )[0]
    )
    register_scenario(scenario)

    settings = Settings(
        anthropic_api_key="test",
        budget=BudgetConfig(max_cost_usd=max_cost_usd, max_wall_clock_s=600),
        retry=RetryConfig(max_attempts=3, base_delay_s=0.0, max_delay_s=0.0),
        tool_policy=policy if policy is not None else {},
    )
    if max_turns is not None:
        for cfg in settings.roles.values():
            cfg.max_turns = max_turns

    store = Store(tmp_path / "aegis.db")
    # A resumed run already has its row; creating it again would violate the primary key.
    if store.get_run(run_id) is None:
        store.create_run(run_id, scenario.id, scenario.alert)
    bus = EventBus(store)
    budget = Budget(settings.budget)
    gate = PermissionGate(
        settings.tool_policy, settings.default_tool_permission, broker or AutoApprove()
    )

    async with connect_in_memory(mcp_server_app) as mcp:
        harness = Harness(
            settings=settings,
            client=model,
            mcp=mcp,
            store=store,
            bus=bus,
            budget=budget,
            permissions=gate,
            run_id=run_id,
            resources=resources,
            resume=resume,
        )
        try:
            yield Rig(
                harness=harness,
                store=store,
                bus=bus,
                budget=budget,
                mcp=mcp,
                scenario=scenario,
                run_id=run_id,
            )
        finally:
            store.close()
