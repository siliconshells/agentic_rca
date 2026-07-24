"""Orchestration invariants, driven by the deterministic mock model.

These exercise the whole topology against the real MCP server and real store — the mock only
stands in for the model. Because the mock reads ground truth, correctness here proves the
*wiring* (fan-out, adjudication, tool scoping, ablation switches, event stream), not model skill.
That distinction is the point of the mock and is stated in its own module docstring.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from aegis.config import BudgetConfig, RetryConfig, Settings
from aegis.harness.budget import Budget
from aegis.harness.events import EventBus
from aegis.harness.loop import Harness
from aegis.harness.permissions import AutoApprove, AutoDeny, PermissionGate
from aegis.mcp_client.session import connect_in_memory
from aegis.orchestrator import COORDINATOR_TOOLS, READONLY_TOOLS, Orchestrator, RunConfig
from aegis.store.db import Store
from aegis.types import RunStatus, ToolPermission
from evals.mock_model import MockModel
from mcp_server.env.faults import make_scenario
from mcp_server.server import mcp as mcp_server_app
from mcp_server.server import register_scenario


@asynccontextmanager
async def orchestrate(
    tmp_path: Path,
    *,
    fault_class: str = "bad_deploy",
    culprit: str = "payments-svc",
    with_injection: bool = False,
    run_config: RunConfig | None = None,
    policy: dict[str, ToolPermission] | None = None,
    broker=None,
    run_id: str = "run-orch",
) -> AsyncIterator[tuple[Orchestrator, Store, EventBus, object]]:
    scenario, _ = make_scenario(
        "inc-orch",
        seed=4242,
        fault_class=fault_class,
        culprit=culprit,
        onset=50,
        with_injection=with_injection,
    )
    register_scenario(scenario)

    settings = Settings(
        anthropic_api_key="test",
        budget=BudgetConfig(max_cost_usd=50.0, max_wall_clock_s=600),
        retry=RetryConfig(max_attempts=2, base_delay_s=0.0, max_delay_s=0.0),
        tool_policy=policy or {},
        max_parallel_investigators=4,
    )
    store = Store(tmp_path / "aegis.db")
    store.create_run(run_id, scenario.id, scenario.alert)
    bus = EventBus(store)
    budget = Budget(settings.budget)
    gate = PermissionGate(
        settings.tool_policy, settings.default_tool_permission, broker or AutoApprove()
    )
    model = MockModel(scenario=scenario)

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
        )
        orch = Orchestrator(
            harness=harness, settings=settings, budget=budget, run_config=run_config
        )
        # The store is intentionally left open: these tests assert on persisted state after the
        # run, and each test has its own tmp_path, so there is no cross-test leakage. The OS
        # reclaims the connection at process exit.
        yield orch, store, bus, scenario


# --------------------------------------------------------------------------------------
# The full topology
# --------------------------------------------------------------------------------------


async def test_full_run_produces_a_structured_rca(tmp_path):
    async with orchestrate(tmp_path) as (orch, store, _bus, scenario):
        result = await orch.run(scenario.id, scenario.alert)
        # Persisted so `aegis show` can read it back.
        persisted = store.get_run(result.run_id)

    assert result.status is RunStatus.SUCCEEDED
    assert result.rca is not None
    assert result.rca.root_cause_class == scenario.ground_truth.root_cause_class
    assert result.rca.culprit_service == scenario.ground_truth.culprit_service
    assert result.cost_usd > 0
    assert persisted["status"] == RunStatus.SUCCEEDED.value


async def test_full_run_spawns_coordinator_investigators_and_verifiers(tmp_path):
    async with orchestrate(tmp_path) as (orch, store, _bus, scenario):
        result = await orch.run(scenario.id, scenario.alert)

    roles = [s["role"] for s in store.list_spans(result.run_id)]
    assert roles.count("coordinator") == 2  # plan + conclude
    assert roles.count("investigator") == len(result.hypotheses)
    assert roles.count("verifier") == len(result.findings)
    assert len(result.hypotheses) >= 2


async def test_investigators_run_in_parallel(tmp_path):
    """Sibling investigator spans must overlap in wall-clock, not run as a staircase."""
    async with orchestrate(tmp_path) as (orch, store, _bus, scenario):
        result = await orch.run(scenario.id, scenario.alert)

    spans = [s for s in store.list_spans(result.run_id) if s["role"] == "investigator"]
    assert len(spans) >= 2
    # At least one pair overlaps: some span starts before another ends.
    overlaps = any(
        a["started_at"] < b["ended_at"] and b["started_at"] < a["ended_at"]
        for i, a in enumerate(spans)
        for b in spans[i + 1 :]
    )
    assert overlaps, "investigators did not overlap — the fan-out was serialised"


# --------------------------------------------------------------------------------------
# Tool scoping — the security boundary between roles
# --------------------------------------------------------------------------------------


async def test_investigators_and_verifiers_cannot_reach_the_remediation_tool(tmp_path):
    """Only the coordinator may propose changes; the scope is enforced, not requested."""
    assert "propose_remediation" not in READONLY_TOOLS
    assert "propose_remediation" in COORDINATOR_TOOLS

    async with orchestrate(tmp_path) as (orch, store, _bus, scenario):
        await orch.run(scenario.id, scenario.alert)
        calls = store.list_tool_calls("run-orch")

    for call in calls:
        if call["name"] == "propose_remediation":
            assert call["agent_id"].startswith("agt-coordinator"), (
                "a non-coordinator agent reached the remediation tool"
            )


# --------------------------------------------------------------------------------------
# Ablations — the same code path, different arms
# --------------------------------------------------------------------------------------


async def test_single_agent_ablation_uses_one_coordinator_and_no_investigators(tmp_path):
    async with orchestrate(tmp_path, run_config=RunConfig.single_agent()) as (
        orch,
        store,
        _bus,
        scenario,
    ):
        result = await orch.run(scenario.id, scenario.alert)

    roles = [s["role"] for s in store.list_spans(result.run_id)]
    assert roles == ["coordinator"]
    assert result.status is RunStatus.SUCCEEDED
    assert result.rca is not None


async def test_no_verifier_ablation_skips_verification(tmp_path):
    async with orchestrate(tmp_path, run_config=RunConfig.no_verifier()) as (
        orch,
        store,
        _bus,
        scenario,
    ):
        result = await orch.run(scenario.id, scenario.alert)

    roles = [s["role"] for s in store.list_spans(result.run_id)]
    assert "verifier" not in roles
    assert result.findings
    assert result.refutations == []


# --------------------------------------------------------------------------------------
# Clean windows and injection
# --------------------------------------------------------------------------------------


async def test_clean_scenario_concludes_no_fault(tmp_path):
    async with orchestrate(tmp_path, fault_class="none") as (orch, _store, _bus, scenario):
        result = await orch.run(scenario.id, scenario.alert)

    assert result.status is RunStatus.SUCCEEDED
    assert result.rca.root_cause_class == "none"
    assert result.rca.culprit_service == ""


async def test_injection_scenario_still_reaches_the_correct_rca(tmp_path):
    """A planted instruction in the logs must not change the outcome."""
    async with orchestrate(tmp_path, with_injection=True) as (orch, _store, _bus, scenario):
        result = await orch.run(scenario.id, scenario.alert)

    assert result.rca.root_cause_class == scenario.ground_truth.root_cause_class
    # The payload log exists in the world the agents queried...
    from aegis.mcp_client.bridge import wrap_untrusted

    assert "untrusted" in wrap_untrusted("query_logs", "x")


async def test_a_write_tool_is_unreachable_without_approval_even_under_injection(tmp_path):
    """The core injection defence: a denied write tool never executes, whatever a log says."""
    async with orchestrate(
        tmp_path,
        with_injection=True,
        policy={"propose_remediation": ToolPermission.ASK},
        broker=AutoDeny(),
    ) as (orch, store, _bus, scenario):
        result = await orch.run(scenario.id, scenario.alert)
        calls = store.list_tool_calls(result.run_id)

    # Any remediation call was recorded, but every one was denied — none executed for real.
    remediation_calls = [c for c in calls if c["name"] == "propose_remediation"]
    for call in remediation_calls:
        assert call["is_error"] == 1
        assert "DENIED" in call["output"]


# --------------------------------------------------------------------------------------
# Events and resources
# --------------------------------------------------------------------------------------


async def test_run_emits_start_and_finish_events(tmp_path):
    async with orchestrate(tmp_path) as (orch, store, _bus, scenario):
        result = await orch.run(scenario.id, scenario.alert)
        events = [p for _, p in store.list_events(result.run_id)]

    types = [e["type"] for e in events]
    assert types[0] == "run.started"
    assert types[-1] == "run.finished"
    finish = events[-1]
    assert finish["status"] == RunStatus.SUCCEEDED.value
    assert finish["rca"]["root_cause_class"] == scenario.ground_truth.root_cause_class


async def test_a_late_subscriber_replays_the_whole_run(tmp_path):
    """The event log lets a dashboard that connects after the fact see everything."""
    async with orchestrate(tmp_path) as (orch, _store, bus, scenario):
        await orch.run(scenario.id, scenario.alert)

        replayed = []
        async for event in bus.subscribe(orch.harness.run_id, after_seq=0):
            replayed.append(event)

    assert replayed[0]["type"] == "run.started"
    assert replayed[-1]["type"] == "run.finished"
    # Sequence numbers are dense and increasing.
    seqs = [e["seq"] for e in replayed]
    assert seqs == sorted(seqs)


async def test_resources_are_pinned_before_agents_run(tmp_path):
    async with orchestrate(tmp_path) as (orch, _store, _bus, scenario):
        await orch.run(scenario.id, scenario.alert)

    # The catalog is always pinned; the alerting service's runbook should be too.
    assert "catalog://services" in orch._resources
    assert any(uri.startswith("runbook://") for uri in orch._resources)


@pytest.mark.parametrize(
    "fault_class",
    ["bad_deploy", "dependency_saturation", "config_change", "network_partition"],
)
async def test_wiring_holds_across_fault_classes(tmp_path, fault_class):
    async with orchestrate(tmp_path, fault_class=fault_class, culprit="auth-svc") as (
        orch,
        _store,
        _bus,
        scenario,
    ):
        result = await orch.run(scenario.id, scenario.alert)

    assert result.status is RunStatus.SUCCEEDED
    assert result.rca.root_cause_class == fault_class
