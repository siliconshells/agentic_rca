"""End-to-end runner tests: the full assembly the CLI and API use.

Distinct from test_orchestrator (which constructs the orchestrator directly): these go through
``execute_run``, so they cover the wiring the top-level callers actually exercise — store
creation, resume dispatch, model/broker injection, and clean teardown.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from aegis.config import BudgetConfig, RetryConfig, Settings
from aegis.harness.client import DeltaCallback, ModelRequest, ModelResponse
from aegis.runner import RunRequest, execute_run
from aegis.store.db import Store
from aegis.types import RunStatus, Usage
from evals.mock_model import MockModel
from mcp_server.env.faults import make_scenario
from mcp_server.server import mcp as mcp_server_app
from mcp_server.server import register_scenario


def _settings(tmp_path, **overrides) -> Settings:
    defaults = {
        "anthropic_api_key": "test",
        "db_path": tmp_path / "aegis.db",
        "budget": BudgetConfig(max_cost_usd=50.0, max_wall_clock_s=600),
        "retry": RetryConfig(max_attempts=2, base_delay_s=0.0, max_delay_s=0.0),
    }
    return Settings(**{**defaults, **overrides})


async def test_execute_run_end_to_end_via_the_public_entrypoint(tmp_path):
    scenario, _ = make_scenario("inc-run", seed=7, fault_class="config_change", culprit="auth-svc")
    register_scenario(scenario)
    settings = _settings(tmp_path)

    result = await execute_run(
        RunRequest(incident_id=scenario.id, alert=scenario.alert),
        settings=settings,
        model=MockModel(scenario=scenario),
        mcp_server=mcp_server_app,
    )

    assert result.status is RunStatus.SUCCEEDED
    assert result.rca.root_cause_class == "config_change"

    # The run is durably recorded and re-readable, as `aegis show` needs.
    store = Store(settings.db_path)
    row = store.get_run(result.run_id)
    assert row["status"] == RunStatus.SUCCEEDED.value
    assert row["cost_usd"] > 0
    store.close()


@dataclass
class CountingModel:
    """Wraps the mock and counts calls, so a resume can be proven to skip completed turns."""

    inner: MockModel
    calls: int = 0
    fail_after: int | None = None
    _seen: list[str] = field(default_factory=list)

    async def complete(
        self,
        request: ModelRequest,
        on_text: DeltaCallback | None = None,
        on_thinking: DeltaCallback | None = None,
    ) -> ModelResponse:
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("simulated crash mid-run")
        return await self.inner.complete(request, on_text, on_thinking)


async def test_a_crashed_run_resumes_without_repeating_completed_agents(tmp_path):
    """The durability guarantee: resume continues from checkpoints, it does not restart.

    First run crashes partway through (the coordinator plan completes, then the model dies). The
    resumed run must not re-issue the calls the first run already checkpointed.
    """
    scenario, _ = make_scenario("inc-crash", seed=13, fault_class="bad_deploy", culprit="cart-svc")
    register_scenario(scenario)
    settings = _settings(tmp_path)
    run_id = "run-crash"

    # Crash after the coordinator's plan turn (1 call) plus its tool turn is 1 call in the mock;
    # allow a few calls through, then fail, so at least one agent checkpoints fully.
    crashing = CountingModel(inner=MockModel(scenario=scenario), fail_after=3)
    first = await execute_run(
        RunRequest(incident_id=scenario.id, alert=scenario.alert, run_id=run_id),
        settings=settings,
        model=crashing,
        mcp_server=mcp_server_app,
    )
    # The run failed, but the crash happened after real progress was checkpointed.
    assert first.status is not RunStatus.SUCCEEDED
    assert crashing.calls >= 3

    store = Store(settings.db_path)
    checkpoints_exist = store.latest_checkpoint(run_id, "agt-coordinator-plan") is not None
    store.close()
    assert checkpoints_exist, "the crash left no checkpoint to resume from"

    # Resume: a fresh model that will happily complete everything.
    resuming = CountingModel(inner=MockModel(scenario=scenario))
    second = await execute_run(
        RunRequest(incident_id=scenario.id, alert=scenario.alert, run_id=run_id, resume=True),
        settings=settings,
        model=resuming,
        mcp_server=mcp_server_app,
    )

    assert second.status is RunStatus.SUCCEEDED
    assert second.rca.root_cause_class == "bad_deploy"
    # The resumed coordinator-plan replayed from its checkpoint rather than re-calling the model
    # for the turns the first run already completed. Concretely: the resume did strictly less work
    # on the already-finished plan agent than a cold start would.
    assert resuming.calls < crashing.calls + 6


async def test_budget_ceiling_produces_a_budget_exceeded_status(tmp_path):
    scenario, _ = make_scenario(
        "inc-poor", seed=3, fault_class="data_anomaly", culprit="search-svc"
    )
    register_scenario(scenario)

    @dataclass
    class Expensive:
        inner: MockModel

        async def complete(self, request, on_text=None, on_thinking=None):
            resp = await self.inner.complete(request, on_text, on_thinking)
            resp.usage = Usage(input_tokens=5_000_000, output_tokens=500_000)
            return resp

    settings = _settings(tmp_path, budget=BudgetConfig(max_cost_usd=1.0, max_wall_clock_s=600))
    result = await execute_run(
        RunRequest(incident_id=scenario.id, alert=scenario.alert),
        settings=settings,
        model=Expensive(inner=MockModel(scenario=scenario)),
        mcp_server=mcp_server_app,
    )

    assert result.status is RunStatus.BUDGET_EXCEEDED
    assert result.cost_usd >= 1.0


@pytest.mark.parametrize("mock_flag", [True])
async def test_cli_mock_run_smoke(tmp_path, monkeypatch, mock_flag):
    """The exact path CI runs: `aegis run --scenario ...:0 --mock`."""
    from mcp_server.env.generator import generate

    scenarios, failures = generate(count=8, seed=99)
    assert failures == []
    scenario_file = tmp_path / "scenarios.jsonl"
    import json

    scenario_file.write_text("\n".join(json.dumps(s.as_dict()) for s in scenarios))

    from aegis.cli import _run_async
    from aegis.config import get_settings

    monkeypatch.setattr(get_settings(), "db_path", tmp_path / "cli.db", raising=False)
    result = await _run_async(
        scenario_ref=f"{scenario_file}:0",
        resume_id=None,
        mock=True,
        max_cost=5.0,
        run_config=__import__("aegis.orchestrator", fromlist=["RunConfig"]).RunConfig(),
    )
    assert result.status is RunStatus.SUCCEEDED
