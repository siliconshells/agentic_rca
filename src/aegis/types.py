"""Core domain types.

Split into three groups:

* **Wire schemas** (``Finding``, ``Refutation``, ``RootCauseAnalysis``) — pydantic models whose
  JSON Schema is handed to the API via ``output_config.format``. They must round-trip exactly, so
  they set ``extra="forbid"`` (emits ``additionalProperties: false``, which strict schemas require).
* **Accounting** (``Usage``, ``CostReport``) — accumulated across every model call.
* **Stream events** — what the harness emits to the dashboard over SSE.
"""

from __future__ import annotations

import time
import uuid
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# --------------------------------------------------------------------------------------
# Enums
# --------------------------------------------------------------------------------------


class Role(StrEnum):
    """Which agent in the topology. Drives model, effort, tool scope, and system prompt."""

    COORDINATOR = "coordinator"
    INVESTIGATOR = "investigator"
    VERIFIER = "verifier"


class Effort(StrEnum):
    """`output_config.effort`. The cost lever — we do not downgrade model tier."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    MAX = "max"


class ToolPermission(StrEnum):
    """Per-tool policy enforced by the harness, never by the model."""

    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


class RootCauseClass(StrEnum):
    """The six injected fault classes, plus NONE for clean windows."""

    BAD_DEPLOY = "bad_deploy"
    DEPENDENCY_SATURATION = "dependency_saturation"
    CONFIG_CHANGE = "config_change"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    DATA_ANOMALY = "data_anomaly"
    NETWORK_PARTITION = "network_partition"
    NONE = "none"


class Verdict(StrEnum):
    SUPPORTED = "supported"
    REFUTED = "refuted"
    INCONCLUSIVE = "inconclusive"


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    BUDGET_EXCEEDED = "budget_exceeded"
    CANCELLED = "cancelled"


class ApprovalDecision(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"


# --------------------------------------------------------------------------------------
# Wire schemas — handed to the API as strict JSON Schema
# --------------------------------------------------------------------------------------


class StrictModel(BaseModel):
    """Base for anything whose schema is sent to the API.

    ``extra="forbid"`` makes pydantic emit ``additionalProperties: false``, which the strict
    structured-output path requires; without it the API rejects the schema.
    """

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def api_schema(cls) -> dict[str, Any]:
        """JSON Schema in the shape ``output_config.format`` expects.

        The constraints stripped by ``_strip_unsupported`` (numeric bounds, string/array lengths)
        are still enforced client-side: pydantic keeps them on the model, so a parsed response is
        validated against them via ``model_validate``. Only the wire schema drops them.
        """
        inlined = _inline_defs(cls.model_json_schema())
        return {"type": "json_schema", "schema": _strip_unsupported(inlined)}


class Hypothesis(StrictModel):
    """One candidate explanation. The coordinator emits a list; each gets its own investigator."""

    id: str = Field(description="Short stable id, e.g. 'h1'.")
    root_cause_class: RootCauseClass = Field(
        description="Which fault class this hypothesis posits."
    )
    culprit_service: str = Field(description="Service suspected of originating the fault.")
    statement: str = Field(description="One sentence: what happened and why it explains the alert.")
    investigation_plan: str = Field(
        description="Which tools to call and what evidence would confirm or kill this."
    )


class HypothesisSet(StrictModel):
    """Coordinator's planning output."""

    hypotheses: list[Hypothesis] = Field(description="Between 2 and 6 distinct hypotheses.")


class Finding(StrictModel):
    """An investigator's structured verdict on the single hypothesis it owns."""

    hypothesis_id: str
    verdict: Verdict
    confidence: float = Field(ge=0.0, le=1.0, description="0 = no signal, 1 = certain.")
    evidence_event_ids: list[str] = Field(
        description="IDs of concrete observed events supporting the verdict. Never invent these."
    )
    reasoning: str = Field(description="How the evidence leads to the verdict.")
    contradicting_evidence: str = Field(
        default="", description="Anything observed that cuts against the verdict. '' if none."
    )


class Refutation(StrictModel):
    """A verifier's adversarial pass over one surviving finding."""

    hypothesis_id: str
    refuted: bool = Field(description="True if the finding does not hold up.")
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="Claims in the finding not backed by a cited event id.",
    )


class RemediationProposal(StrictModel):
    action: str = Field(description="The concrete action, e.g. 'rollback deploy d-1043'.")
    target: str = Field(description="Service or resource the action touches.")
    rationale: str
    blast_radius: str = Field(description="What else this could affect if it goes wrong.")
    reversible: bool


class RootCauseAnalysis(StrictModel):
    """The coordinator's final deliverable."""

    root_cause_class: RootCauseClass
    culprit_service: str
    summary: str = Field(description="Two or three sentences an on-call engineer can act on.")
    evidence_event_ids: list[str] = Field(description="The causal chain, in order.")
    confidence: float = Field(ge=0.0, le=1.0)
    rejected_hypotheses: list[str] = Field(
        default_factory=list, description="Hypothesis ids considered and ruled out, with why."
    )
    remediation: RemediationProposal | None = None


def _inline_defs(schema: dict[str, Any]) -> dict[str, Any]:
    """Resolve ``$ref``/``$defs`` into an inline schema.

    The structured-output path supports ``$ref``/``$defs``, but inlining keeps the schema
    byte-stable across pydantic versions — and schema bytes are part of the cached prefix.
    """
    defs = schema.pop("$defs", {})
    if not defs:
        return schema

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                target = defs[ref.removeprefix("#/$defs/")]
                merged = {**resolve(target), **{k: v for k, v in node.items() if k != "$ref"}}
                return merged
            return {k: resolve(v) for k, v in node.items()}
        if isinstance(node, list):
            return [resolve(item) for item in node]
        return node

    return dict(resolve(schema))


# Keys the structured-output (`output_config.format`) path rejects with a 400. pydantic emits
# these from `Field(ge=…, le=…)`, `max_length`, etc.; we strip them from the wire schema and let
# pydantic enforce them client-side on the parsed response instead. (The SDK's `messages.parse`
# helper does the same; we drive `output_config.format` directly, so we do it ourselves.)
_UNSUPPORTED_SCHEMA_KEYS = frozenset(
    {
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "minProperties",
        "maxProperties",
        "pattern",
    }
)


def _strip_unsupported(node: Any) -> Any:
    """Recursively drop constraint keywords the structured-output schema does not accept."""
    if isinstance(node, dict):
        return {
            k: _strip_unsupported(v) for k, v in node.items() if k not in _UNSUPPORTED_SCHEMA_KEYS
        }
    if isinstance(node, list):
        return [_strip_unsupported(item) for item in node]
    return node


# --------------------------------------------------------------------------------------
# Accounting
# --------------------------------------------------------------------------------------


class Usage(BaseModel):
    """Token counts from one or more model calls.

    Cache fields are tracked separately because they are priced differently and because a
    ``cache_read_input_tokens`` of zero across turns is the canonical symptom of a broken
    cache prefix — we assert on it in tests.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_creation_input_tokens=(
                self.cache_creation_input_tokens + other.cache_creation_input_tokens
            ),
            cache_read_input_tokens=(self.cache_read_input_tokens + other.cache_read_input_tokens),
        )

    @property
    def total_input(self) -> int:
        """Every input token the request actually carried, cached or not."""
        return self.input_tokens + self.cache_creation_input_tokens + self.cache_read_input_tokens

    @property
    def cache_hit_rate(self) -> float:
        """Share of input tokens served from cache. 0.0 when nothing was sent."""
        return self.cache_read_input_tokens / self.total_input if self.total_input else 0.0

    @classmethod
    def from_api(cls, usage: Any) -> Usage:
        """Build from an SDK ``usage`` object, tolerating absent cache fields."""
        return cls(
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            cache_creation_input_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
            cache_read_input_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
        )


class CostReport(BaseModel):
    """Usage attributed to one agent (or rolled up for a run)."""

    usage: Usage = Field(default_factory=Usage)
    cost_usd: float = 0.0
    calls: int = 0

    def __add__(self, other: CostReport) -> CostReport:
        return CostReport(
            usage=self.usage + other.usage,
            cost_usd=self.cost_usd + other.cost_usd,
            calls=self.calls + other.calls,
        )


# --------------------------------------------------------------------------------------
# Stream events — harness -> SSE -> dashboard
# --------------------------------------------------------------------------------------


def _event_id() -> str:
    return uuid.uuid4().hex[:12]


class BaseEvent(BaseModel):
    id: str = Field(default_factory=_event_id)
    ts: float = Field(default_factory=time.time)
    run_id: str
    agent_id: str | None = None


class RunStarted(BaseEvent):
    type: Literal["run.started"] = "run.started"
    scenario_id: str
    alert: str


class RunFinished(BaseEvent):
    type: Literal["run.finished"] = "run.finished"
    status: RunStatus
    cost_usd: float
    duration_s: float
    rca: RootCauseAnalysis | None = None
    error: str | None = None


class AgentStarted(BaseEvent):
    type: Literal["agent.started"] = "agent.started"
    role: Role
    label: str
    parent_agent_id: str | None = None


class AgentFinished(BaseEvent):
    type: Literal["agent.finished"] = "agent.finished"
    role: Role
    label: str
    cost_usd: float
    turns: int
    error: str | None = None


class TextDelta(BaseEvent):
    type: Literal["agent.text_delta"] = "agent.text_delta"
    text: str


class ThinkingDelta(BaseEvent):
    type: Literal["agent.thinking_delta"] = "agent.thinking_delta"
    text: str


class ToolCalled(BaseEvent):
    type: Literal["tool.called"] = "tool.called"
    tool_use_id: str
    name: str
    input: dict[str, Any]


class ToolResulted(BaseEvent):
    type: Literal["tool.resulted"] = "tool.resulted"
    tool_use_id: str
    name: str
    duration_ms: int
    is_error: bool = False
    preview: str = Field(description="Truncated result, for the timeline. Not the full payload.")


class ApprovalRequested(BaseEvent):
    type: Literal["approval.requested"] = "approval.requested"
    approval_id: str
    tool_use_id: str
    name: str
    input: dict[str, Any]
    blast_radius: str = ""


class ApprovalResolved(BaseEvent):
    type: Literal["approval.resolved"] = "approval.resolved"
    approval_id: str
    decision: ApprovalDecision
    reason: str = ""


class BudgetWarning(BaseEvent):
    type: Literal["budget.warning"] = "budget.warning"
    spent_usd: float
    limit_usd: float
    fraction: float


Event = Annotated[
    RunStarted
    | RunFinished
    | AgentStarted
    | AgentFinished
    | TextDelta
    | ThinkingDelta
    | ToolCalled
    | ToolResulted
    | ApprovalRequested
    | ApprovalResolved
    | BudgetWarning,
    Field(discriminator="type"),
]
