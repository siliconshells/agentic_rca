"""A deterministic offline stand-in for the model.

**This is a pipeline test, not a result.** The mock reads the scenario's ground truth and answers
correctly by construction, so any accuracy number produced under ``--mock`` is 100% and means
nothing about model capability. What it *does* verify — hermetically, in CI, with no API key and
no spend — is that the machinery works end to end:

* the loop drives ``stop_reason`` correctly and pairs every ``tool_use`` with a ``tool_result``
* tools actually execute against the MCP server and their output reaches the next turn
* structured output parses into the schema each agent was given
* budget accounting, cache accounting, checkpointing, and the event stream all fire

It implements the same ``ModelClient`` protocol as the real client, so the code under test is the
production path, not a parallel one.

Failure modes can be injected (``fail_first_n``, ``stop_reason_override``) to exercise the retry
and truncation paths that are otherwise only reachable against a misbehaving API.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any

import anthropic
import httpx

from aegis.harness.client import DeltaCallback, ModelRequest, ModelResponse
from aegis.types import Usage
from mcp_server.env.faults import Scenario

# Roughly plausible token counts, so cost and cache maths exercise real magnitudes.
PREFIX_TOKENS = 5200
TURN_INPUT_TOKENS = 400
TURN_OUTPUT_TOKENS = 350


def _schema_kind(schema: dict[str, Any] | None) -> str:
    """Identify which agent is calling by the shape it was asked to return."""
    if not schema:
        return "text"
    properties = set(schema.get("schema", {}).get("properties", {}))
    if "hypotheses" in properties:
        return "hypotheses"
    if "verdict" in properties:
        return "finding"
    if "refuted" in properties:
        return "refutation"
    if "root_cause_class" in properties and "summary" in properties:
        return "rca"
    return "text"


@dataclass
class MockModel:
    """Scripted model. One tool-calling turn, then the structured answer."""

    scenario: Scenario
    # Injected failures, for exercising retry and truncation paths.
    fail_first_n: int = 0
    fail_with: type[Exception] = anthropic.InternalServerError
    stop_reason_override: str | None = None
    emit_deltas: bool = True

    calls: list[ModelRequest] = field(default_factory=list)
    _turns_by_agent: dict[str, int] = field(default_factory=dict)
    _failures: int = 0

    # -- the protocol -------------------------------------------------------------------

    async def complete(
        self,
        request: ModelRequest,
        on_text: DeltaCallback | None = None,
        on_thinking: DeltaCallback | None = None,
    ) -> ModelResponse:
        if self._failures < self.fail_first_n:
            self._failures += 1
            raise self._make_failure()

        self.calls.append(request)
        kind = _schema_kind(request.output_schema)

        # Turn count is per conversation, inferred from history length rather than tracked
        # externally, so the mock stays stateless with respect to agent identity.
        turn = sum(1 for m in request.messages if m["role"] == "assistant") + 1

        if on_thinking and self.emit_deltas:
            on_thinking(f"Considering the {kind} step for {self.scenario.id}. ")

        tool_names = {t["name"] for t in request.tools}
        propose_available = "propose_remediation" in tool_names
        needs_remediation = kind == "rca" and self.scenario.ground_truth.root_cause_class != "none"

        # Turn 1: gather evidence. Exercises the parallel-tool path.
        if turn == 1 and request.tools and kind != "text":
            return self._tool_turn(request, turn)

        # Turn 2 for a coordinator conclusion: actually propose remediation, so the human-in-the-
        # loop approval gate is exercised end to end rather than only described in the RCA text.
        if turn == 2 and needs_remediation and propose_available:
            return self._remediation_turn(turn)

        text = self._answer(kind)
        if on_text and self.emit_deltas:
            on_text(text)
        return self._response(
            [{"type": "text", "text": text}],
            stop_reason=self.stop_reason_override or "end_turn",
            turn=turn,
        )

    # -- turn construction --------------------------------------------------------------

    def _tool_turn(self, request: ModelRequest, turn: int) -> ModelResponse:
        """Ask for two tools at once, so the parallel-execution path is exercised."""
        names = [t["name"] for t in request.tools]
        truth = self.scenario.ground_truth
        target = truth.culprit_service or truth.symptom_service

        wanted: list[dict[str, Any]] = []
        if "query_metrics" in names:
            wanted.append(
                {
                    "name": "query_metrics",
                    "input": {
                        "incident_id": self.scenario.id,
                        "service": target,
                        "metric": "error_rate",
                    },
                }
            )
        if "query_logs" in names:
            wanted.append(
                {
                    "name": "query_logs",
                    "input": {
                        "incident_id": self.scenario.id,
                        "service": target,
                        "level": "ERROR",
                        "limit": 10,
                    },
                }
            )
        if not wanted:
            wanted.append({"name": names[0], "input": {}})

        blocks: list[dict[str, Any]] = [
            {"type": "text", "text": f"Checking {target} before concluding."}
        ]
        blocks += [
            {
                "type": "tool_use",
                "id": f"toolu_{uuid.uuid4().hex[:16]}",
                "name": call["name"],
                "input": call["input"],
            }
            for call in wanted
        ]
        return self._response(blocks, stop_reason="tool_use", turn=turn)

    def _remediation_turn(self, turn: int) -> ModelResponse:
        """A single propose_remediation call — the write action the harness gates on approval."""
        truth = self.scenario.ground_truth
        culprit = truth.culprit_service or truth.symptom_service
        blocks: list[dict[str, Any]] = [
            {"type": "text", "text": "Root cause identified; proposing remediation."},
            {
                "type": "tool_use",
                "id": f"toolu_{uuid.uuid4().hex[:16]}",
                "name": "propose_remediation",
                "input": {
                    "incident_id": self.scenario.id,
                    "action": f"mitigate {truth.root_cause_class} at {culprit}",
                    "target": culprit,
                    "rationale": "Directly addresses the identified root cause.",
                    "blast_radius": f"{culprit} traffic during the change",
                    "reversible": True,
                },
            },
        ]
        return self._response(blocks, stop_reason="tool_use", turn=turn)

    def _response(
        self, content: list[dict[str, Any]], stop_reason: str, turn: int
    ) -> ModelResponse:
        """Attach usage that mimics a working prompt cache.

        Turn 1 writes the prefix; every later turn reads it. Tests assert on exactly this to
        catch a prefix that has silently stopped caching.
        """
        if turn == 1:
            usage = Usage(
                input_tokens=TURN_INPUT_TOKENS,
                output_tokens=TURN_OUTPUT_TOKENS,
                cache_creation_input_tokens=PREFIX_TOKENS,
            )
        else:
            usage = Usage(
                input_tokens=TURN_INPUT_TOKENS,
                output_tokens=TURN_OUTPUT_TOKENS,
                cache_read_input_tokens=PREFIX_TOKENS,
            )
        return ModelResponse(
            content=content,
            stop_reason=stop_reason,
            usage=usage,
            model="mock-opus",
        )

    def _make_failure(self) -> Exception:
        """Build a real SDK exception, so retry classification is exercised for real."""
        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        response = httpx.Response(500, request=request)
        return self.fail_with("simulated upstream failure", response=response, body=None)

    # -- scripted answers ---------------------------------------------------------------

    def _answer(self, kind: str) -> str:
        truth = self.scenario.ground_truth
        culprit = truth.culprit_service or truth.symptom_service
        evidence = truth.causal_event_ids[:4]

        if kind == "hypotheses":
            # One correct hypothesis plus a plausible distractor, so adjudication has work.
            distractor = "bad_deploy" if truth.root_cause_class != "bad_deploy" else "config_change"
            return json.dumps(
                {
                    "hypotheses": [
                        {
                            "id": "h1",
                            "root_cause_class": truth.root_cause_class,
                            "culprit_service": culprit,
                            "statement": f"{culprit} is the origin of the incident.",
                            "investigation_plan": "Check metrics then error logs at the culprit.",
                        },
                        {
                            "id": "h2",
                            "root_cause_class": distractor,
                            "culprit_service": truth.symptom_service,
                            "statement": f"{truth.symptom_service} degraded on its own.",
                            "investigation_plan": "Check deploys at the alerting service.",
                        },
                    ]
                }
            )

        if kind == "finding":
            return json.dumps(
                {
                    "hypothesis_id": "h1",
                    "verdict": "supported",
                    "confidence": 0.86,
                    "evidence_event_ids": evidence,
                    "reasoning": f"Telemetry at {culprit} matches {truth.root_cause_class}.",
                    "contradicting_evidence": "",
                }
            )

        if kind == "refutation":
            return json.dumps(
                {
                    "hypothesis_id": "h1",
                    "refuted": False,
                    "confidence": 0.8,
                    "reasoning": "Every claim is backed by a cited event id.",
                    "unsupported_claims": [],
                }
            )

        if kind == "rca":
            payload: dict[str, Any] = {
                "root_cause_class": truth.root_cause_class,
                "culprit_service": culprit if truth.root_cause_class != "none" else "",
                "summary": (
                    f"{culprit} is the root cause; the alert at {truth.symptom_service} is "
                    f"downstream symptom."
                    if truth.root_cause_class != "none"
                    else "No service is breaching SLO. This page is a false positive."
                ),
                "evidence_event_ids": evidence,
                "confidence": 0.88,
                "rejected_hypotheses": ["h2: no deploy or error signature at the alerting service"],
                "remediation": None,
            }
            if truth.root_cause_class != "none":
                payload["remediation"] = {
                    "action": f"mitigate {truth.root_cause_class} at {culprit}",
                    "target": culprit,
                    "rationale": "Directly addresses the identified root cause.",
                    "blast_radius": f"{culprit} traffic during the change",
                    "reversible": True,
                }
            return json.dumps(payload)

        return f"Investigated {self.scenario.id}; see structured output."
