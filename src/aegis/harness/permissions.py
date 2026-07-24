"""Tool permission policy and human-in-the-loop approval.

The model proposes; the harness disposes. Whether a tool may run is decided here, from static
policy, and is never negotiable by anything the model says or anything a log line contains. That
is what makes the prompt-injection boundary real rather than advisory: a planted
"run execute_remediation" instruction reaches a gate that does not read prompts.

Three outcomes per tool:

* ``allow`` — runs unattended. Read-only telemetry.
* ``ask``   — suspends the agent until an operator decides. Anything that mutates the world.
* ``deny``  — never runs; the model is told why and can adapt.

A denial returns a ``tool_result`` with ``is_error`` rather than raising, so the agent can revise
its plan instead of the run dying.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol

from aegis.logging import get_logger
from aegis.types import ApprovalDecision, ToolPermission

log = get_logger(__name__)


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: str = ""
    approval_id: str | None = None


@dataclass
class ApprovalRequest:
    approval_id: str
    run_id: str
    agent_id: str
    tool_use_id: str
    name: str
    input: dict[str, Any]
    blast_radius: str


# Awaited once, after the broker has registered the pending request but before it blocks. The
# ordering is load-bearing: the harness uses this to publish the approval, and a client that acts
# on that publication must find the request already resolvable — see QueueApprovalBroker.request.
OnRequested = Callable[["ApprovalRequest"], Awaitable[None]]


class ApprovalBroker(Protocol):
    """How the harness asks a human. The API layer supplies the interactive implementation."""

    async def request(
        self, req: ApprovalRequest, on_requested: OnRequested | None = None
    ) -> Decision: ...


class AutoApprove:
    """Approves everything. For ``--mock`` runs and unattended evals, never for a live demo."""

    async def request(
        self, req: ApprovalRequest, on_requested: OnRequested | None = None
    ) -> Decision:
        if on_requested is not None:
            await on_requested(req)
        log.info("auto-approving", extra={"tool": req.name, "approval_id": req.approval_id})
        return Decision(True, "auto-approved (no operator attached)", req.approval_id)


class AutoDeny:
    """Denies everything. The fail-closed default: with no operator attached, write actions do
    not happen. Also used by the injection-resistance eval to prove the gate holds."""

    async def request(
        self, req: ApprovalRequest, on_requested: OnRequested | None = None
    ) -> Decision:
        if on_requested is not None:
            await on_requested(req)
        return Decision(False, "no operator attached; write actions are denied", req.approval_id)


class ConsoleApprovalBroker:
    """Prompts a human on the terminal. The interactive broker for live CLI runs.

    A write action pauses the run and asks y/n on stdin; anything but an explicit yes denies.
    Reading stdin is blocking, so it runs in a thread to avoid stalling the event loop.
    """

    def __init__(self, prompt: Any = input) -> None:
        self._prompt = prompt

    async def request(
        self, req: ApprovalRequest, on_requested: OnRequested | None = None
    ) -> Decision:
        if on_requested is not None:
            await on_requested(req)

        banner = (
            f"\n╭─ APPROVAL REQUIRED ─────────────────────────────\n"
            f"│ tool:  {req.name}\n"
            f"│ input: {req.input}\n"
            f"│ blast: {req.blast_radius}\n"
            f"╰─ approve? [y/N]: "
        )
        answer = await asyncio.to_thread(self._prompt, banner)
        approved = str(answer).strip().lower() in ("y", "yes")
        return Decision(
            approved,
            "approved by operator" if approved else "denied by operator",
            req.approval_id,
        )


class QueueApprovalBroker:
    """Interactive broker: parks the agent on a future until someone resolves it.

    The dashboard renders the pending request and calls :meth:`resolve`. A timeout counts as a
    denial — an unattended run must fail closed, not quietly proceed.
    """

    def __init__(self, timeout_s: float = 600.0) -> None:
        self.timeout_s = timeout_s
        self._pending: dict[str, asyncio.Future[Decision]] = {}
        self._requests: dict[str, ApprovalRequest] = {}
        self._listeners: list[Any] = []

    def on_request(self, callback: Any) -> None:
        """Register a callback invoked when a new approval is raised."""
        self._listeners.append(callback)

    @property
    def pending(self) -> dict[str, ApprovalRequest]:
        return dict(self._requests)

    async def request(
        self, req: ApprovalRequest, on_requested: OnRequested | None = None
    ) -> Decision:
        future: asyncio.Future[Decision] = asyncio.get_running_loop().create_future()
        # Register BEFORE notifying anyone. The harness publishes the request from on_requested,
        # and an operator (or the dashboard) may resolve it the instant they see it — resolve()
        # must find the future already in _pending, or the decision is silently lost.
        self._pending[req.approval_id] = future
        self._requests[req.approval_id] = req

        if on_requested is not None:
            await on_requested(req)
        for callback in self._listeners:
            result = callback(req)
            if asyncio.iscoroutine(result):
                await result

        try:
            return await asyncio.wait_for(asyncio.shield(future), timeout=self.timeout_s)
        except TimeoutError:
            log.warning("approval timed out", extra={"approval_id": req.approval_id})
            return Decision(
                False,
                f"no operator responded within {self.timeout_s:.0f}s; denied by default",
                req.approval_id,
            )
        finally:
            self._pending.pop(req.approval_id, None)
            self._requests.pop(req.approval_id, None)

    def resolve(self, approval_id: str, decision: ApprovalDecision, reason: str = "") -> bool:
        """Answer a pending approval. Returns False if it is unknown or already resolved."""
        future = self._pending.get(approval_id)
        if future is None or future.done():
            return False
        future.set_result(
            Decision(decision is ApprovalDecision.APPROVED, reason or decision.value, approval_id)
        )
        return True


def describe_blast_radius(name: str, tool_input: dict[str, Any]) -> str:
    """What the operator is actually being asked to accept."""
    stated = tool_input.get("blast_radius")
    if isinstance(stated, str) and stated.strip():
        reversible = tool_input.get("reversible")
        suffix = "" if reversible is None else f" (reversible: {bool(reversible)})"
        return stated.strip() + suffix
    target = tool_input.get("target") or tool_input.get("service") or "unspecified target"
    return f"{name} against {target}; impact not stated by the agent"


class PermissionGate:
    """Applies policy to every tool call before it executes."""

    def __init__(
        self,
        policy: dict[str, ToolPermission],
        default: ToolPermission,
        broker: ApprovalBroker,
    ) -> None:
        self._policy = policy
        self._default = default
        self._broker = broker

    def permission_for(self, name: str) -> ToolPermission:
        return self._policy.get(name, self._default)

    async def authorize(
        self,
        run_id: str,
        agent_id: str,
        tool_use_id: str,
        name: str,
        tool_input: dict[str, Any],
        on_requested: Callable[[ApprovalRequest], Awaitable[None]] | None = None,
    ) -> Decision:
        """Decide whether a tool may run.

        ``on_requested`` is awaited exactly once, with the built ``ApprovalRequest``, at the
        moment an ``ask`` tool starts waiting — before the broker blocks. The loop uses it to
        record the pending approval and emit it to the event stream, so the dashboard can show it.
        It fires only for ``ask``; ``allow`` and ``deny`` are decided synchronously and never
        surface an approval.
        """
        permission = self.permission_for(name)

        if permission is ToolPermission.ALLOW:
            return Decision(True)

        if permission is ToolPermission.DENY:
            return Decision(
                False,
                f"tool {name!r} is disabled by policy in this environment. "
                f"Do not retry it; report what you would have done instead.",
            )

        req = ApprovalRequest(
            approval_id=f"apr-{uuid.uuid4().hex[:12]}",
            run_id=run_id,
            agent_id=agent_id,
            tool_use_id=tool_use_id,
            name=name,
            input=tool_input,
            blast_radius=describe_blast_radius(name, tool_input),
        )
        log.info("approval required", extra={"tool": name, "approval_id": req.approval_id})
        # The broker fires on_requested only after registering the pending request, closing the
        # race where a client resolves the approval before the broker can accept the decision.
        return await self._broker.request(req, on_requested=on_requested)
