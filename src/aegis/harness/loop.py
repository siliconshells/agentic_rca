"""The agent loop.

This is the piece the rest of the project exists to demonstrate. One agent, one loop, driven by
``stop_reason``:

* ``tool_use``   — execute every requested tool concurrently, return **all** results in a single
  user message, continue. Splitting results across messages teaches the model to stop making
  parallel calls, so it is a correctness issue, not a formatting one.
* ``end_turn``   — done.
* ``pause_turn`` — a server-side tool hit its iteration cap. Re-send as-is; do **not** append a
  "continue" message, which would corrupt the turn.
* ``max_tokens`` — output was truncated. Recorded as an error rather than accepted, because a
  half-written RCA that looks complete is worse than a failed one.
* ``refusal``    — raised by the client before content is read.

Invariants worth stating because they are easy to break silently:

1. The **full** ``response.content`` is appended to history — thinking blocks and their
   signatures included. Reconstructing a text-only assistant turn breaks signature validation.
2. Every ``tool_use`` gets exactly one ``tool_result`` with a matching id, including denials and
   errors. A missing pairing is a 400 on the next request.
3. Tool results are wrapped as untrusted data, and permission is checked before execution — not
   after the model has already been told it succeeded.
4. A checkpoint is written after every turn, so a crashed run resumes rather than restarts.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from aegis.config import RoleConfig, Settings
from aegis.harness import telemetry
from aegis.harness.budget import Budget, cost_of
from aegis.harness.cache import PromptPrefix, build_prefix
from aegis.harness.client import ModelClient, ModelRequest, ModelResponse, ToolUse
from aegis.harness.errors import BudgetExceeded, DeadlineExceeded, MaxTurnsExceeded, ModelRefusal
from aegis.harness.events import DeltaBuffer, EventBus
from aegis.harness.permissions import ApprovalRequest, PermissionGate
from aegis.logging import bind, get_logger
from aegis.mcp_client.bridge import to_tool_result_block
from aegis.mcp_client.session import McpClient
from aegis.store.db import Store
from aegis.types import (
    AgentFinished,
    AgentStarted,
    ApprovalDecision,
    ApprovalRequested,
    ApprovalResolved,
    Role,
    TextDelta,
    ThinkingDelta,
    ToolCalled,
    ToolResulted,
    Usage,
)

log = get_logger(__name__)

# Truncated result shown in the dashboard timeline.
PREVIEW_CHARS = 400
# Full tool output goes to the model; SQLite keeps a bounded copy for the trace view.
MAX_STORED_RESULT_CHARS = 8_000


@dataclass
class AgentSpec:
    """Everything that distinguishes one agent from another."""

    role: Role
    label: str
    system_prompt: str
    task: str
    allowed_tools: set[str] | None = None
    output_schema: dict[str, Any] | None = None
    seed_messages: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class AgentOutcome:
    agent_id: str
    role: Role
    label: str
    text: str
    parsed: dict[str, Any] | None
    stop_reason: str | None
    turns: int
    usage: Usage
    cost_usd: float
    messages: list[dict[str, Any]]
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


class Harness:
    """Shared runtime for every agent in one run."""

    def __init__(
        self,
        *,
        settings: Settings,
        client: ModelClient,
        mcp: McpClient,
        store: Store,
        bus: EventBus,
        budget: Budget,
        permissions: PermissionGate,
        run_id: str,
        resources: dict[str, str] | None = None,
        resume: bool = False,
    ) -> None:
        self.settings = settings
        self.client = client
        self.mcp = mcp
        self.store = store
        self.bus = bus
        self.budget = budget
        self.permissions = permissions
        self.run_id = run_id
        self.resources = resources or {}
        self.resume = resume
        self._prefixes: dict[str, PromptPrefix] = {}

    # -- prefix -------------------------------------------------------------------------

    async def prefix_for(self, spec: AgentSpec) -> PromptPrefix:
        """Build (and memoise) the cached prefix for a spec.

        Memoised per system prompt + tool scope: every investigator shares one prefix, so the
        second one onward reads it from cache instead of writing it again.
        """
        tools = await self.mcp.list_tools(allow=spec.allowed_tools)
        key = f"{spec.role}:{sorted(spec.allowed_tools or [])}:{hash(spec.system_prompt)}"
        if key not in self._prefixes:
            prefix = build_prefix(
                system_prompt=spec.system_prompt, resources=self.resources, tools=tools
            )
            if not prefix.meets_minimum():
                log.warning(
                    "prompt prefix is below the cacheable minimum; it will not be cached",
                    extra={"estimated_tokens": prefix.estimated_tokens(), "role": spec.role.value},
                )
            self._prefixes[key] = prefix
        return self._prefixes[key]

    # -- the loop -----------------------------------------------------------------------

    async def run_agent(
        self, spec: AgentSpec, *, parent_agent_id: str | None = None, agent_id: str | None = None
    ) -> AgentOutcome:
        agent_id = agent_id or f"agt-{uuid.uuid4().hex[:10]}"
        cfg = self.settings.role(spec.role)
        prefix = await self.prefix_for(spec)

        messages, start_turn, usage = self._initial_state(agent_id, spec)
        # On resume, the restored usage is spend that already happened. Charge the (fresh,
        # per-process) budget ledger for it and seed cost_total, so both the run's reported cost
        # and the ceiling enforcement account for what prior attempts already spent — otherwise a
        # resumed run silently under-reports cost and its budget resets on every crash.
        cost_total = (
            self.budget.add_prior_spend(agent_id, usage, cfg.model)
            if usage.total_input or usage.output_tokens
            else 0.0
        )
        outcome_error: str | None = None
        response: ModelResponse | None = None
        turn = start_turn

        self.store.start_span(agent_id, self.run_id, spec.role.value, spec.label, parent_agent_id)
        await self.bus.emit(
            AgentStarted(
                run_id=self.run_id,
                agent_id=agent_id,
                role=spec.role,
                label=spec.label,
                parent_agent_id=parent_agent_id,
            )
        )

        with (
            bind(run_id=self.run_id, agent=spec.label),
            telemetry.agent_span(agent_id, spec.role.value, spec.label, cfg.model) as span,
        ):
            try:
                while turn < cfg.max_turns:
                    turn += 1
                    # Reserve budget for this call before making it, so concurrent agents cannot
                    # all start at the ceiling and overshoot. The reservation is released by the
                    # matching charge, or by release() if the call fails before spending.
                    reservation = self.budget.reserve()

                    with telemetry.turn_span(turn):
                        try:
                            response = await self._one_turn(
                                agent_id=agent_id,
                                spec=spec,
                                cfg=cfg,
                                prefix=prefix,
                                messages=messages,
                            )
                        except BaseException:
                            self.budget.release(reservation)
                            raise
                        usage = usage + response.usage
                        cost_total += self.budget.charge(
                            agent_id, response.usage, cfg.model, reservation=reservation
                        )

                        # Append the assistant turn verbatim (thinking + tool_use blocks intact).
                        messages.append({"role": "assistant", "content": response.content})

                        # Execute tools and append their results BEFORE checkpointing. The
                        # checkpoint must capture a *resumable* message list — one that never ends
                        # in an unanswered tool_use, which the API rejects with a 400. So the state
                        # persisted here always ends in either an assistant end/pause turn or a
                        # user tool_results message. This is the durability guarantee: a crash
                        # anywhere resumes from a valid conversation, not a dangling tool call.
                        tool_uses = response.tool_uses
                        has_pending_tools = response.stop_reason == "tool_use" and bool(tool_uses)
                        if has_pending_tools:
                            results = await self._execute_tools(agent_id, tool_uses)
                            messages.append({"role": "user", "content": results})

                        await asyncio.to_thread(
                            self.store.save_checkpoint,
                            self.run_id,
                            agent_id,
                            turn,
                            messages,
                            usage.model_dump(),
                        )

                        if response.stop_reason == "end_turn":
                            break

                        if response.stop_reason == "max_tokens":
                            outcome_error = (
                                f"output truncated at max_tokens ({cfg.max_tokens}); "
                                "the result is incomplete and must not be treated as final"
                            )
                            log.warning("agent hit max_tokens", extra={"turn": turn})
                            break

                        if response.stop_reason == "pause_turn":
                            # Re-send unchanged; the server resumes where it left off.
                            continue

                        if not has_pending_tools:
                            # Not end_turn, not paused, and no tools requested: nothing further
                            # will happen, so stop rather than loop forever.
                            break
                else:
                    raise MaxTurnsExceeded(
                        f"{spec.label} did not finish within {cfg.max_turns} turns"
                    )

            except (BudgetExceeded, DeadlineExceeded) as exc:
                outcome_error = str(exc)
                log.warning("agent stopped by budget", extra={"error": str(exc)})
            except ModelRefusal as exc:
                outcome_error = f"model refused: {exc}"
                telemetry.record_error(span, exc)
            except MaxTurnsExceeded as exc:
                outcome_error = str(exc)
                telemetry.record_error(span, exc)
            except Exception as exc:
                outcome_error = f"{type(exc).__name__}: {exc}"
                log.exception("agent failed")
                telemetry.record_error(span, exc)

            telemetry.set_attributes(
                span,
                **{
                    "aegis.turns": turn,
                    "aegis.cost_usd": round(cost_total, 6),
                    "aegis.cache_hit_rate": round(usage.cache_hit_rate, 4),
                },
            )

        text = response.text if response else ""
        parsed, parse_error = self._parse_structured(spec, text, outcome_error)
        outcome_error = outcome_error or parse_error

        self.store.end_span(
            agent_id,
            cost_usd=cost_total,
            turns=turn,
            usage=usage.model_dump(),
            error=outcome_error,
        )
        await self.bus.emit(
            AgentFinished(
                run_id=self.run_id,
                agent_id=agent_id,
                role=spec.role,
                label=spec.label,
                cost_usd=round(cost_total, 6),
                turns=turn,
                error=outcome_error,
            )
        )

        return AgentOutcome(
            agent_id=agent_id,
            role=spec.role,
            label=spec.label,
            text=text,
            parsed=parsed,
            stop_reason=response.stop_reason if response else None,
            turns=turn,
            usage=usage,
            cost_usd=cost_total,
            messages=messages,
            error=outcome_error,
        )

    # -- one model call -----------------------------------------------------------------

    async def _one_turn(
        self,
        *,
        agent_id: str,
        spec: AgentSpec,
        cfg: RoleConfig,
        prefix: PromptPrefix,
        messages: list[dict[str, Any]],
    ) -> ModelResponse:
        text_buffer = DeltaBuffer()
        thinking_buffer = DeltaBuffer()
        pending: list[asyncio.Task[Any]] = []

        def flush(chunk: str | None, *, thinking: bool) -> None:
            if not chunk:
                return
            event = (
                ThinkingDelta(run_id=self.run_id, agent_id=agent_id, text=chunk)
                if thinking
                else TextDelta(run_id=self.run_id, agent_id=agent_id, text=chunk)
            )
            pending.append(asyncio.create_task(self.bus.emit(event)))

        request = ModelRequest(
            model=cfg.model,
            system_blocks=prefix.system_blocks,
            messages=messages,
            tools=prefix.tools,
            max_tokens=cfg.max_tokens,
            effort=cfg.effort.value,
            task_budget_tokens=cfg.task_budget_tokens,
            output_schema=spec.output_schema,
        )

        with telemetry.model_span(cfg.model, cfg.effort.value) as span:
            response = await self.client.complete(
                request,
                on_text=lambda t: flush(text_buffer.add(t), thinking=False),
                on_thinking=lambda t: flush(thinking_buffer.add(t), thinking=True),
            )
            flush(text_buffer.flush(), thinking=False)
            flush(thinking_buffer.flush(), thinking=True)
            if pending:
                await asyncio.gather(*pending)

            # This call's cost, not the run total — a span attribute that silently reported
            # cumulative spend would make every trace look progressively more expensive.
            telemetry.record_usage(
                span,
                response.usage,
                cost_of(response.usage, cfg.model),
                response.stop_reason,
            )
        return response

    # -- tools --------------------------------------------------------------------------

    async def _execute_tools(self, agent_id: str, tool_uses: list[ToolUse]) -> list[dict[str, Any]]:
        """Run every requested tool concurrently and return the results in request order.

        Order is preserved and every id is answered: the API rejects the next request if any
        ``tool_use`` lacks a matching ``tool_result``.
        """
        results = await asyncio.gather(*(self._execute_one(agent_id, use) for use in tool_uses))
        return list(results)

    async def _execute_one(self, agent_id: str, use: ToolUse) -> dict[str, Any]:
        await self.bus.emit(
            ToolCalled(
                run_id=self.run_id,
                agent_id=agent_id,
                tool_use_id=use.id,
                name=use.name,
                input=use.input,
            )
        )

        started = time.perf_counter()
        with telemetry.tool_span(use.name, use.id) as span:
            decision = await self.permissions.authorize(
                self.run_id, agent_id, use.id, use.name, use.input, on_requested=self._on_approval
            )
            telemetry.set_attributes(span, **{"aegis.tool.allowed": decision.allowed})

            if decision.approval_id is not None:
                # An ask tool was raised; record how it resolved for late viewers and the stream.
                resolved = (
                    ApprovalDecision.APPROVED if decision.allowed else ApprovalDecision.DENIED
                )
                await asyncio.to_thread(
                    self.store.resolve_approval, decision.approval_id, resolved, decision.reason
                )
                await self.bus.emit(
                    ApprovalResolved(
                        run_id=self.run_id,
                        agent_id=agent_id,
                        approval_id=decision.approval_id,
                        decision=resolved,
                        reason=decision.reason,
                    )
                )

            if not decision.allowed:
                text = (
                    f"DENIED by the operator. Reason: {decision.reason}\n"
                    f"The action was not performed. Do not retry it; continue with what you can "
                    f"establish from read-only evidence, and state clearly in your conclusion "
                    f"that this action was proposed but not taken."
                )
                is_error = True
            else:
                outcome = await self.mcp.call_tool(use.name, use.input)
                text, is_error = outcome.text, outcome.is_error

        duration_ms = int((time.perf_counter() - started) * 1000)

        await asyncio.to_thread(
            self.store.record_tool_call,
            f"tc-{uuid.uuid4().hex[:10]}",
            self.run_id,
            agent_id,
            use.id,
            use.name,
            use.input,
            text[:MAX_STORED_RESULT_CHARS],
            is_error,
            duration_ms,
        )
        await self.bus.emit(
            ToolResulted(
                run_id=self.run_id,
                agent_id=agent_id,
                tool_use_id=use.id,
                name=use.name,
                duration_ms=duration_ms,
                is_error=is_error,
                preview=text[:PREVIEW_CHARS],
            )
        )
        return to_tool_result_block(use.id, use.name, text, is_error)

    async def _on_approval(self, req: ApprovalRequest) -> None:
        """Persist and broadcast a pending approval the instant it is raised.

        Runs before the gate blocks on the operator, so a dashboard subscribed to the stream —
        or one reading ``/approvals`` — sees the request while the agent is still waiting.
        """
        await asyncio.to_thread(
            self.store.create_approval,
            req.approval_id,
            req.run_id,
            req.agent_id,
            req.tool_use_id,
            req.name,
            req.input,
            req.blast_radius,
        )
        await self.bus.emit(
            ApprovalRequested(
                run_id=req.run_id,
                agent_id=req.agent_id,
                approval_id=req.approval_id,
                tool_use_id=req.tool_use_id,
                name=req.name,
                input=req.input,
                blast_radius=req.blast_radius,
            )
        )

    # -- helpers ------------------------------------------------------------------------

    def _initial_state(
        self, agent_id: str, spec: AgentSpec
    ) -> tuple[list[dict[str, Any]], int, Usage]:
        """Resume from the last checkpoint when asked, otherwise start fresh."""
        if self.resume:
            checkpoint = self.store.latest_checkpoint(self.run_id, agent_id)
            if checkpoint:
                log.info("resuming agent", extra={"turn": checkpoint["turn"]})
                return (
                    list(checkpoint["messages"]),
                    int(checkpoint["turn"]),
                    Usage(**checkpoint["usage"]),
                )

        messages = list(spec.seed_messages)
        if spec.task:
            messages.append({"role": "user", "content": spec.task})
        return messages, 0, Usage()

    def _parse_structured(
        self, spec: AgentSpec, text: str, existing_error: str | None
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Decode the final JSON answer when the agent was given an output schema."""
        if not spec.output_schema:
            return None, None
        if existing_error:
            return None, None  # already failed; a parse error would just be noise
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            return None, f"structured output did not parse as JSON: {exc}"
        if not isinstance(parsed, dict):
            return None, f"structured output was {type(parsed).__name__}, expected an object"
        return parsed, None
