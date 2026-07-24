"""Run lifecycle management for the API process.

Owns the resources a run needs to be both driven and observed at once: a run executes as a
background task while SSE subscribers stream its events and operators resolve its approval
requests. Both of those need a handle to the run's broker and the shared event bus, which is
what this registry provides.

One store and one ``EventBus`` are shared across all runs in the process (SQLite WAL makes
concurrent reads safe). Each run gets its own ``QueueApprovalBroker`` because approvals are
run-scoped — an operator approving run A must not unblock run B.
"""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from dataclasses import dataclass

from aegis.config import Settings, get_settings
from aegis.harness.client import AnthropicModelClient, ModelClient
from aegis.harness.events import EventBus
from aegis.harness.permissions import QueueApprovalBroker
from aegis.logging import get_logger
from aegis.orchestrator import RunConfig, RunResult
from aegis.runner import RunRequest, execute_run
from aegis.store.db import Store
from aegis.types import ApprovalDecision

log = get_logger(__name__)


@dataclass
class ActiveRun:
    run_id: str
    incident_id: str
    broker: QueueApprovalBroker
    task: asyncio.Task[RunResult]
    result: RunResult | None = None


class RunManager:
    """Process-wide registry of runs. One instance, held on the FastAPI app."""

    def __init__(
        self,
        settings: Settings | None = None,
        store: Store | None = None,
        bus: EventBus | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.store: Store = store or Store(self.settings.db_path)
        self.bus: EventBus = bus or EventBus(self.store)
        self._runs: dict[str, ActiveRun] = {}

    # -- lifecycle ----------------------------------------------------------------------

    def start(
        self,
        incident_id: str,
        alert: str,
        *,
        run_config: RunConfig | None = None,
        model: ModelClient | None = None,
        mcp_server: object | None = None,
    ) -> ActiveRun:
        """Launch a run as a background task and register it. Returns immediately.

        The caller (an API handler) gets a handle it can hand to the SSE and approval endpoints;
        the run itself proceeds independently.
        """
        run_id = f"run-{uuid.uuid4().hex[:12]}"
        broker = QueueApprovalBroker(timeout_s=self.settings.approval_timeout_s)
        model = model or AnthropicModelClient(self.settings.anthropic_api_key, self.settings.retry)

        # Create the run row synchronously, before the task is scheduled: an SSE client that
        # connects the instant `start` returns must find the run, not a 404 from a row the
        # background task has not written yet.
        self.store.create_run(run_id, incident_id, alert)

        request = RunRequest(
            incident_id=incident_id, alert=alert, run_id=run_id, run_config=run_config
        )

        async def drive() -> RunResult:
            result = await execute_run(
                request,
                settings=self.settings,
                store=self.store,
                bus=self.bus,
                model=model,
                broker=broker,
                mcp_server=mcp_server,
            )
            if run_id in self._runs:
                self._runs[run_id].result = result
            return result

        task = asyncio.create_task(drive(), name=f"run:{run_id}")
        active = ActiveRun(run_id=run_id, incident_id=incident_id, broker=broker, task=task)
        self._runs[run_id] = active
        log.info("run started", extra={"run_id": run_id, "incident_id": incident_id})
        return active

    def get(self, run_id: str) -> ActiveRun | None:
        return self._runs.get(run_id)

    def resolve_approval(
        self, run_id: str, approval_id: str, decision: ApprovalDecision, reason: str = ""
    ) -> bool:
        """Route an operator decision to the run's broker. False if run or approval is unknown.

        Only routes to the broker; the harness loop (which is waiting on that broker) owns
        writing the resolution to the store and emitting ``ApprovalResolved``, so this does not
        touch the store and there is a single writer.
        """
        active = self._runs.get(run_id)
        if active is None:
            return False
        return active.broker.resolve(approval_id, decision, reason)

    def pending_approvals(self, run_id: str) -> list[dict[str, object]]:
        active = self._runs.get(run_id)
        if active is None:
            # The run may have finished and been dropped from memory; fall back to the store.
            return self.store.list_pending_approvals(run_id)
        return [
            {
                "approval_id": req.approval_id,
                "tool_use_id": req.tool_use_id,
                "name": req.name,
                "input": req.input,
                "blast_radius": req.blast_radius,
            }
            for req in active.broker.pending.values()
        ]

    async def shutdown(self) -> None:
        """Cancel in-flight runs and close the store. Called on app shutdown."""
        for active in self._runs.values():
            if not active.task.done():
                active.task.cancel()
        for active in self._runs.values():
            with contextlib.suppress(asyncio.CancelledError):
                await active.task
        self.store.close()
