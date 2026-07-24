"""Run orchestration: the multi-agent topology, above the single-agent loop.

The shape, and why each part earns its place:

    coordinator.plan → N investigators (parallel) → verifiers (parallel) → coordinator.conclude

* **Plan / conclude are the same coordinator agent, resumed.** One agent, two phases, so the
  adjudication sees its own hypotheses in context. The plan phase is deliberately forbidden from
  solving the incident — its output is the fan-out list.
* **Investigators run in parallel**, each owning exactly one hypothesis over a read-only tool
  scope. This is where the wall-clock win comes from, and the ablation harness can collapse it to
  a single agent to measure what the fan-out buys.
* **Verifiers are adversarial and run per surviving finding.** They exist to kill plausible-but-
  wrong findings before they reach the RCA. Toggled off in the ``no_verifier`` ablation.
* **Every stage is optional via ``RunConfig``**, so the same orchestrator produces the
  single-agent, no-verifier, and full-topology arms of the eval from one code path.

Concurrency is bounded by a semaphore, not by launching everything at once: a shared rate limit
turns an unbounded fan-out into a retry storm.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field

from aegis.agents import prompts
from aegis.config import Settings
from aegis.harness.budget import Budget
from aegis.harness.loop import AgentOutcome, AgentSpec, Harness
from aegis.logging import get_logger
from aegis.types import (
    Effort,
    Finding,
    HypothesisSet,
    Refutation,
    Role,
    RootCauseAnalysis,
    RunFinished,
    RunStarted,
    RunStatus,
)

log = get_logger(__name__)

# Resources pinned into every agent's cached prefix. Incident-independent standing knowledge.
BASE_RESOURCE_URIS = ["catalog://services"]

# Read-only telemetry, granted to investigators and verifiers. The remediation tool is withheld
# from everyone but the coordinator — a scope boundary, enforced at tool-list construction.
READONLY_TOOLS = {
    "query_metrics",
    "query_logs",
    "list_deploys",
    "sample_traces",
    "get_trace",
    "get_dependency_graph",
    "search_similar_incidents",
}
COORDINATOR_TOOLS = READONLY_TOOLS | {"propose_remediation"}


@dataclass
class RunConfig:
    """Which arms of the topology are active. The ablation surface lives here."""

    use_coordinator: bool = True
    use_investigators: bool = True
    use_verifier: bool = True
    max_hypotheses: int = 6
    investigator_effort: str | None = None  # override for the effort sweep

    @classmethod
    def single_agent(cls) -> RunConfig:
        """One coordinator does everything itself — the baseline to beat."""
        return cls(use_investigators=False, use_verifier=False)

    @classmethod
    def no_verifier(cls) -> RunConfig:
        return cls(use_verifier=False)


@dataclass
class RunResult:
    run_id: str
    status: RunStatus
    rca: RootCauseAnalysis | None
    cost_usd: float
    duration_s: float
    hypotheses: list[dict[str, object]] = field(default_factory=list)
    findings: list[dict[str, object]] = field(default_factory=list)
    refutations: list[dict[str, object]] = field(default_factory=list)
    error: str | None = None
    agent_outcomes: list[AgentOutcome] = field(default_factory=list)


class Orchestrator:
    """Drives one incident to an RCA. One instance per run."""

    def __init__(
        self,
        *,
        harness: Harness,
        settings: Settings,
        budget: Budget,
        run_config: RunConfig | None = None,
    ) -> None:
        self.harness = harness
        self.settings = settings
        self.budget = budget
        self.config = run_config or RunConfig()
        self._sema = asyncio.Semaphore(settings.max_parallel_investigators)
        self._resources: dict[str, str] = {}

    async def run(self, incident_id: str, alert: str) -> RunResult:
        run_id = self.harness.run_id
        started = time.monotonic()
        self.harness.store.update_run(run_id, status=RunStatus.RUNNING)
        await self.harness.bus.emit(RunStarted(run_id=run_id, scenario_id=incident_id, alert=alert))

        outcomes: list[AgentOutcome] = []
        result: RunResult
        try:
            await self._load_resources(incident_id, alert)

            if self.config.use_investigators:
                result = await self._multi_agent(incident_id, alert, outcomes)
            else:
                result = await self._single_agent(incident_id, alert, outcomes)

        except Exception as exc:  # the run must always finish with a status
            log.exception("run failed")
            result = self._failed_result(run_id, started, str(exc))

        result.duration_s = time.monotonic() - started
        result.cost_usd = self.budget.spent_usd
        result.agent_outcomes = outcomes
        self._persist(result)
        await self.harness.bus.emit(
            RunFinished(
                run_id=run_id,
                status=result.status,
                cost_usd=result.cost_usd,
                duration_s=result.duration_s,
                rca=result.rca,
                error=result.error,
            )
        )
        await self.harness.bus.close_run(run_id)
        return result

    # -- resources ----------------------------------------------------------------------

    async def _load_resources(self, incident_id: str, alert: str) -> None:
        """Read the runbooks relevant to this incident once, for the cached prefix.

        The alerting service and its dependencies get their runbooks pinned. This is the concrete
        payoff of resources-over-tools: the model reads them at ~0.1x on every turn instead of
        making a tool call each time it needs one.
        """
        from mcp_server.env.world import SERVICES, downstream_of

        uris = list(BASE_RESOURCE_URIS)
        # Pull the alerting service out of the alert text if we can; fall back to the whole estate.
        mentioned = [name for name in SERVICES if name in alert]
        for svc in mentioned:
            uris.append(f"runbook://{svc}")
            for dep in downstream_of(svc, depth=2):
                uris.append(f"runbook://{dep}")

        # De-dupe while preserving order, then cap so the prefix stays a sensible size.
        seen: dict[str, None] = {}
        for uri in uris:
            seen.setdefault(uri, None)
        self._resources = await self.harness.mcp.read_resources(list(seen)[:8])
        self.harness.resources = self._resources
        log.info("pinned resources", extra={"count": len(self._resources)})

    # -- single-agent baseline ----------------------------------------------------------

    async def _single_agent(
        self, incident_id: str, alert: str, outcomes: list[AgentOutcome]
    ) -> RunResult:
        spec = AgentSpec(
            role=Role.COORDINATOR,
            label="coordinator",
            system_prompt=prompts.COORDINATOR,
            task=prompts.coordinator_conclude_task(
                incident_id,
                "You are working alone; no investigators are available. Investigate the incident "
                "yourself using the tools, then produce the analysis.",
            ),
            allowed_tools=COORDINATOR_TOOLS,
            output_schema=RootCauseAnalysis.api_schema(),
        )
        outcome = await self.harness.run_agent(spec, agent_id="agt-coordinator")
        outcomes.append(outcome)
        return self._result_from_rca(outcome)

    # -- full topology ------------------------------------------------------------------

    async def _multi_agent(
        self, incident_id: str, alert: str, outcomes: list[AgentOutcome]
    ) -> RunResult:
        # Phase 1: PLAN. Same coordinator agent id as the conclude phase, so phase 2 resumes it.
        plan_spec = AgentSpec(
            role=Role.COORDINATOR,
            label="coordinator·plan",
            system_prompt=prompts.COORDINATOR,
            task=prompts.coordinator_plan_task(alert, incident_id),
            allowed_tools=COORDINATOR_TOOLS,
            output_schema=HypothesisSet.api_schema(),
        )
        plan = await self.harness.run_agent(plan_spec, agent_id="agt-coordinator-plan")
        outcomes.append(plan)

        hypotheses = self._extract_hypotheses(plan)
        if not hypotheses:
            # Planning failed; fall back to a single-agent pass rather than aborting the run.
            log.warning("planning produced no hypotheses; falling back to single-agent")
            return await self._single_agent(incident_id, alert, outcomes)

        # Phase 2: investigate + verify each hypothesis independently, in parallel.
        findings, refutations = await self._investigate_all(incident_id, hypotheses, outcomes)

        # Phase 3: CONCLUDE. Resume the coordinator with the assembled briefing.
        briefing = self._briefing(hypotheses, findings, refutations)
        conclude_spec = AgentSpec(
            role=Role.COORDINATOR,
            label="coordinator·conclude",
            system_prompt=prompts.COORDINATOR,
            task=prompts.coordinator_conclude_task(incident_id, briefing),
            allowed_tools=COORDINATOR_TOOLS,
            output_schema=RootCauseAnalysis.api_schema(),
        )
        conclusion = await self.harness.run_agent(
            conclude_spec, agent_id="agt-coordinator-conclude"
        )
        outcomes.append(conclusion)

        result = self._result_from_rca(conclusion)
        result.hypotheses = hypotheses
        result.findings = findings
        result.refutations = refutations
        return result

    async def _investigate_all(
        self, incident_id: str, hypotheses: list[dict[str, object]], outcomes: list[AgentOutcome]
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        """Run one investigate→verify pipeline per hypothesis, all concurrently.

        Each hypothesis flows through its own investigator and (if enabled) verifier without a
        barrier between hypotheses — hypothesis B's investigator does not wait on A's verifier.
        The semaphore bounds how many model calls are in flight at once.
        """

        async def pipeline(
            index: int, hypothesis: dict[str, object]
        ) -> tuple[dict[str, object] | None, dict[str, object] | None]:
            async with self._sema:
                if not self.budget.affordable():
                    return None, None
                finding = await self._investigate(incident_id, index, hypothesis, outcomes)
            if finding is None:
                return None, None
            if not self.config.use_verifier:
                return finding, None
            async with self._sema:
                if not self.budget.affordable():
                    return finding, None
                refutation = await self._verify(incident_id, index, finding, hypothesis, outcomes)
            return finding, refutation

        results = await asyncio.gather(
            *(pipeline(i, h) for i, h in enumerate(hypotheses)), return_exceptions=True
        )

        findings: list[dict[str, object]] = []
        refutations: list[dict[str, object]] = []
        for outcome in results:
            if isinstance(outcome, BaseException):
                log.warning("investigation pipeline failed", extra={"error": str(outcome)})
                continue
            finding, refutation = outcome
            if finding is not None:
                findings.append(finding)
            if refutation is not None:
                refutations.append(refutation)
        return findings, refutations

    async def _investigate(
        self,
        incident_id: str,
        index: int,
        hypothesis: dict[str, object],
        outcomes: list[AgentOutcome],
    ) -> dict[str, object] | None:
        cfg_effort = self.config.investigator_effort
        spec = AgentSpec(
            role=Role.INVESTIGATOR,
            label=f"investigator·{hypothesis.get('id', index + 1)}",
            system_prompt=prompts.INVESTIGATOR,
            task=prompts.investigator_task(hypothesis, incident_id),
            allowed_tools=READONLY_TOOLS,
            output_schema=Finding.api_schema(),
        )
        if cfg_effort:
            # Effort sweep: temporarily override the role's configured effort for this agent.
            self.settings.role(Role.INVESTIGATOR).effort = _effort(cfg_effort)
        outcome = await self.harness.run_agent(
            spec, parent_agent_id="agt-coordinator-plan", agent_id=f"agt-inv-{index}"
        )
        outcomes.append(outcome)
        return outcome.parsed

    async def _verify(
        self,
        incident_id: str,
        index: int,
        finding: dict[str, object],
        hypothesis: dict[str, object],
        outcomes: list[AgentOutcome],
    ) -> dict[str, object] | None:
        spec = AgentSpec(
            role=Role.VERIFIER,
            label=f"verifier·{hypothesis.get('id', index + 1)}",
            system_prompt=prompts.VERIFIER,
            task=prompts.verifier_task(finding, hypothesis, incident_id),
            allowed_tools=READONLY_TOOLS,
            output_schema=Refutation.api_schema(),
        )
        outcome = await self.harness.run_agent(
            spec, parent_agent_id=f"agt-inv-{index}", agent_id=f"agt-ver-{index}"
        )
        outcomes.append(outcome)
        return outcome.parsed

    # -- assembling the briefing --------------------------------------------------------

    def _briefing(
        self,
        hypotheses: list[dict[str, object]],
        findings: list[dict[str, object]],
        refutations: list[dict[str, object]],
    ) -> str:
        """Render investigator + verifier output into the coordinator's phase-2 context."""
        refutation_by_hyp = {r.get("hypothesis_id"): r for r in refutations}
        blocks: list[str] = []
        for hypothesis in hypotheses:
            hid = hypothesis.get("id")
            finding = next((f for f in findings if f.get("hypothesis_id") == hid), None)
            if finding is None:
                blocks.append(f"HYPOTHESIS {hid}: {hypothesis.get('statement')}\n  (no finding)")
                continue

            lines = [
                f"HYPOTHESIS {hid}: {hypothesis.get('statement')}",
                f"  investigator verdict: {finding.get('verdict')} "
                f"(confidence {finding.get('confidence')})",
                f"  evidence: {finding.get('evidence_event_ids')}",
                f"  reasoning: {finding.get('reasoning')}",
            ]
            if finding.get("contradicting_evidence"):
                lines.append(f"  contradicting: {finding.get('contradicting_evidence')}")

            refutation = refutation_by_hyp.get(hid)
            if refutation is not None:
                verdict = "REFUTED" if refutation.get("refuted") else "upheld"
                lines.append(
                    f"  verifier: {verdict} (confidence {refutation.get('confidence')}) — "
                    f"{refutation.get('reasoning')}"
                )
                if refutation.get("unsupported_claims"):
                    lines.append(
                        f"  verifier flagged unsupported: {refutation.get('unsupported_claims')}"
                    )
            blocks.append("\n".join(lines))

        return "\n\n".join(blocks)

    # -- extraction and results ---------------------------------------------------------

    def _extract_hypotheses(self, outcome: AgentOutcome) -> list[dict[str, object]]:
        if not outcome.ok or not outcome.parsed:
            return []
        try:
            validated = HypothesisSet.model_validate(outcome.parsed)
        except Exception as exc:
            log.warning("hypothesis set failed validation", extra={"error": str(exc)})
            return []
        return [h.model_dump() for h in validated.hypotheses[: self.config.max_hypotheses]]

    def _result_from_rca(self, outcome: AgentOutcome) -> RunResult:
        run_id = self.harness.run_id
        if not outcome.ok:
            return RunResult(
                run_id=run_id,
                status=self._status_from_error(outcome.error),
                rca=None,
                cost_usd=0.0,
                duration_s=0.0,
                error=outcome.error,
            )
        rca = self._parse_rca(outcome)
        return RunResult(
            run_id=run_id,
            status=RunStatus.SUCCEEDED if rca else RunStatus.FAILED,
            rca=rca,
            cost_usd=0.0,
            duration_s=0.0,
            error=None if rca else "coordinator produced no valid RCA",
        )

    def _parse_rca(self, outcome: AgentOutcome) -> RootCauseAnalysis | None:
        if not outcome.parsed:
            return None
        try:
            return RootCauseAnalysis.model_validate(outcome.parsed)
        except Exception as exc:
            log.warning("RCA failed validation", extra={"error": str(exc)})
            return None

    def _status_from_error(self, error: str | None) -> RunStatus:
        text = error or ""
        if "budget exceeded" in text or "wall clock" in text:
            return RunStatus.BUDGET_EXCEEDED
        return RunStatus.FAILED

    def _failed_result(self, run_id: str, started: float, error: str) -> RunResult:
        return RunResult(
            run_id=run_id,
            status=self._status_from_error(error),
            rca=None,
            cost_usd=self.budget.spent_usd,
            duration_s=time.monotonic() - started,
            error=error,
        )

    def _persist(self, result: RunResult) -> None:
        self.harness.store.update_run(
            result.run_id,
            status=result.status,
            cost_usd=result.cost_usd,
            duration_s=result.duration_s,
            rca=json.loads(result.rca.model_dump_json()) if result.rca else None,
            error=result.error,
        )


def _effort(value: str) -> Effort:
    return Effort(value)
