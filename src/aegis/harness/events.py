"""The run event bus.

Every event is persisted before it is fanned out, so a dashboard that connects late — or
reconnects after a drop — can replay from a sequence number instead of silently missing whatever
happened while it was away. SSE has no replay of its own; this is where that gap gets closed.

Subscribers are per-connection queues. A slow or dead subscriber is dropped rather than allowed
to block the harness: losing a viewer must never stall an incident investigation.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections import OrderedDict
from collections.abc import AsyncIterator
from typing import Any

from aegis.logging import get_logger
from aegis.store.db import Store
from aegis.types import BaseEvent

log = get_logger(__name__)

# Per-subscriber backlog. Generous enough to absorb a burst of text deltas, small enough that a
# disconnected browser cannot pin unbounded memory.
SUBSCRIBER_QUEUE_SIZE = 2048

# How many finished run ids to remember for the "already closed" fast-path. A long-lived API
# process can serve thousands of runs; the flag only needs to cover runs a client might still be
# reconnecting to, so a bounded ring is plenty. Older entries fall off — a subscriber to a
# long-finished run simply replays history from the store and then ends when the stream idles.
CLOSED_RUN_MEMORY = 4096


class EventBus:
    """Persist-then-publish, scoped to a run."""

    def __init__(self, store: Store) -> None:
        self._store = store
        self._subscribers: dict[str, list[asyncio.Queue[dict[str, Any] | None]]] = {}
        # A bounded LRU fast-path of recently-finished runs. It lets a subscriber that connected
        # just before close stop promptly. It is only a fast-path — the durable termination signal
        # is the run's status in the store (see `_is_terminal`), so evicting an old id never causes
        # a hang; a subscriber to a long-finished run replays history and then sees the terminal
        # status. Bounded so the process does not accumulate a run id per run forever.
        self._closed: OrderedDict[str, None] = OrderedDict()

    _TERMINAL = frozenset({"succeeded", "failed", "budget_exceeded", "cancelled"})

    def _is_terminal(self, run_id: str) -> bool:
        if run_id in self._closed:
            return True
        row = self._store.get_run(run_id)
        return bool(row and row["status"] in self._TERMINAL)

    async def emit(self, event: BaseEvent) -> int:
        """Persist an event and push it to live subscribers. Returns its sequence number."""
        payload = event.model_dump(mode="json")
        seq = await asyncio.to_thread(self._store.append_event, event.run_id, payload)
        payload["seq"] = seq

        for queue in list(self._subscribers.get(event.run_id, [])):
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:
                log.warning("dropping event for a lagging subscriber", extra={"seq": seq})
        return seq

    async def close_run(self, run_id: str) -> None:
        """Signal end-of-stream to every subscriber on this run."""
        self._closed[run_id] = None
        self._closed.move_to_end(run_id)
        while len(self._closed) > CLOSED_RUN_MEMORY:
            self._closed.popitem(last=False)
        for queue in list(self._subscribers.get(run_id, [])):
            with contextlib.suppress(asyncio.QueueFull):
                queue.put_nowait(None)

    async def subscribe(self, run_id: str, after_seq: int = 0) -> AsyncIterator[dict[str, Any]]:
        """Yield historical events after ``after_seq``, then live ones.

        History is drained *after* the queue is registered, so an event arriving mid-replay is
        buffered rather than lost. Duplicates from the overlap are filtered by sequence number.
        """
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(SUBSCRIBER_QUEUE_SIZE)
        self._subscribers.setdefault(run_id, []).append(queue)
        try:
            highest = after_seq
            for seq, payload in await asyncio.to_thread(self._store.list_events, run_id, after_seq):
                payload["seq"] = seq
                highest = max(highest, seq)
                yield payload

            # Now tail live events. Two ways to stop, covering every ordering:
            #   * a live subscriber gets the None sentinel from close_run when the run ends;
            #   * a subscriber that connected after the run finished (no sentinel coming) sees the
            #     terminal store status and drains whatever is still queued, then stops.
            # The short timeout re-checks terminality so the race where the run finishes while we
            # block — and the sentinel was already delivered to an earlier subscriber — still ends.
            while True:
                terminal = await asyncio.to_thread(self._is_terminal, run_id)
                if terminal:
                    try:
                        live = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        return
                else:
                    try:
                        live = await asyncio.wait_for(queue.get(), timeout=1.0)
                    except TimeoutError:
                        continue
                if live is None:
                    return  # end-of-stream sentinel from close_run
                if live.get("seq", 0) <= highest:
                    continue  # already replayed from history
                highest = live.get("seq", highest)
                yield live
        finally:
            subscribers = self._subscribers.get(run_id, [])
            if queue in subscribers:
                subscribers.remove(queue)
            if not subscribers:
                self._subscribers.pop(run_id, None)


class DeltaBuffer:
    """Coalesces streaming deltas into sensible event sizes.

    One event per token would swamp both SQLite and the SSE connection; one event per block
    would lose the live feel. Flushing on size or sentence boundary keeps both usable.
    """

    def __init__(self, flush_chars: int = 120) -> None:
        self.flush_chars = flush_chars
        self._buffer: list[str] = []
        self._length = 0

    def add(self, text: str) -> str | None:
        self._buffer.append(text)
        self._length += len(text)
        if self._length >= self.flush_chars or text.endswith(("\n", ". ", "? ", "! ")):
            return self.flush()
        return None

    def flush(self) -> str | None:
        if not self._buffer:
            return None
        out = "".join(self._buffer)
        self._buffer.clear()
        self._length = 0
        return out
