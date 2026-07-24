"""SQLite persistence.

Everything a run produces lands here: the event log (so a reconnecting dashboard can replay
instead of missing what it slept through), per-agent message history (so a crashed run can
resume mid-flight), tool calls, approvals, and cost.

SQLite is deliberate — the whole point is that a reviewer can clone the repo and run it. Writes
are sub-millisecond at this volume, so the store is synchronous and guarded by a single lock
rather than dragging in an async driver.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from aegis.types import ApprovalDecision, RunStatus

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id             TEXT PRIMARY KEY,
    scenario_id    TEXT NOT NULL,
    alert          TEXT NOT NULL,
    status         TEXT NOT NULL,
    created_at     REAL NOT NULL,
    updated_at     REAL NOT NULL,
    cost_usd       REAL NOT NULL DEFAULT 0,
    duration_s     REAL,
    rca_json       TEXT,
    error          TEXT,
    config_json    TEXT
);

CREATE TABLE IF NOT EXISTS agent_spans (
    id               TEXT PRIMARY KEY,
    run_id           TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    parent_agent_id  TEXT,
    role             TEXT NOT NULL,
    label            TEXT NOT NULL,
    started_at       REAL NOT NULL,
    ended_at         REAL,
    cost_usd         REAL NOT NULL DEFAULT 0,
    turns            INTEGER NOT NULL DEFAULT 0,
    usage_json       TEXT,
    error            TEXT
);
CREATE INDEX IF NOT EXISTS idx_spans_run ON agent_spans(run_id);

CREATE TABLE IF NOT EXISTS tool_calls (
    id           TEXT PRIMARY KEY,
    run_id       TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    agent_id     TEXT NOT NULL,
    tool_use_id  TEXT NOT NULL,
    name         TEXT NOT NULL,
    input_json   TEXT NOT NULL,
    output       TEXT,
    is_error     INTEGER NOT NULL DEFAULT 0,
    duration_ms  INTEGER,
    created_at   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tools_run ON tool_calls(run_id);

CREATE TABLE IF NOT EXISTS approvals (
    id           TEXT PRIMARY KEY,
    run_id       TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    agent_id     TEXT NOT NULL,
    tool_use_id  TEXT NOT NULL,
    name         TEXT NOT NULL,
    input_json   TEXT NOT NULL,
    blast_radius TEXT NOT NULL DEFAULT '',
    decision     TEXT NOT NULL,
    reason       TEXT NOT NULL DEFAULT '',
    created_at   REAL NOT NULL,
    resolved_at  REAL
);
CREATE INDEX IF NOT EXISTS idx_approvals_run ON approvals(run_id, decision);

-- One row per completed turn. Resume replays these instead of re-calling the model.
CREATE TABLE IF NOT EXISTS checkpoints (
    run_id        TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    agent_id      TEXT NOT NULL,
    turn          INTEGER NOT NULL,
    messages_json TEXT NOT NULL,
    usage_json    TEXT NOT NULL,
    created_at    REAL NOT NULL,
    PRIMARY KEY (run_id, agent_id, turn)
);

-- Append-only event log. `seq` gives the dashboard a cursor to resume from after a drop.
CREATE TABLE IF NOT EXISTS events (
    run_id       TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    seq          INTEGER NOT NULL,
    payload_json TEXT NOT NULL,
    created_at   REAL NOT NULL,
    PRIMARY KEY (run_id, seq)
);
"""


class Store:
    """Thread-safe SQLite wrapper. One instance per process."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # WAL lets the API read a run's events while the harness is still writing them.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            try:
                yield self._conn
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    # -- runs ---------------------------------------------------------------------------

    def create_run(
        self, run_id: str, scenario_id: str, alert: str, config: dict[str, Any] | None = None
    ) -> None:
        now = time.time()
        with self._tx() as c:
            c.execute(
                "INSERT INTO runs (id, scenario_id, alert, status, created_at, updated_at,"
                " config_json) VALUES (?,?,?,?,?,?,?)",
                (
                    run_id,
                    scenario_id,
                    alert,
                    RunStatus.PENDING.value,
                    now,
                    now,
                    json.dumps(config or {}, default=str),
                ),
            )

    def update_run(
        self,
        run_id: str,
        *,
        status: RunStatus | None = None,
        cost_usd: float | None = None,
        duration_s: float | None = None,
        rca: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        sets: list[str] = ["updated_at = ?"]
        args: list[Any] = [time.time()]
        for column, value in (
            ("status", status.value if status else None),
            ("cost_usd", cost_usd),
            ("duration_s", duration_s),
            ("rca_json", json.dumps(rca, default=str) if rca is not None else None),
            ("error", error),
        ):
            if value is not None:
                sets.append(f"{column} = ?")
                args.append(value)
        args.append(run_id)
        with self._tx() as c:
            # `sets` holds only hardcoded column names from the loop above; every value is
            # a bound parameter. Nothing user-supplied reaches the SQL text.
            c.execute(f"UPDATE runs SET {', '.join(sets)} WHERE id = ?", args)  # noqa: S608

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        return dict(row) if row else None

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    # -- agent spans --------------------------------------------------------------------

    def start_span(
        self, agent_id: str, run_id: str, role: str, label: str, parent_agent_id: str | None
    ) -> None:
        with self._tx() as c:
            c.execute(
                "INSERT OR REPLACE INTO agent_spans (id, run_id, parent_agent_id, role, label,"
                " started_at) VALUES (?,?,?,?,?,?)",
                (agent_id, run_id, parent_agent_id, role, label, time.time()),
            )

    def end_span(
        self,
        agent_id: str,
        *,
        cost_usd: float,
        turns: int,
        usage: dict[str, Any],
        error: str | None = None,
    ) -> None:
        with self._tx() as c:
            c.execute(
                "UPDATE agent_spans SET ended_at=?, cost_usd=?, turns=?, usage_json=?, error=?"
                " WHERE id=?",
                (time.time(), cost_usd, turns, json.dumps(usage), error, agent_id),
            )

    def list_spans(self, run_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM agent_spans WHERE run_id = ? ORDER BY started_at", (run_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    # -- tool calls ---------------------------------------------------------------------

    def record_tool_call(
        self,
        call_id: str,
        run_id: str,
        agent_id: str,
        tool_use_id: str,
        name: str,
        tool_input: dict[str, Any],
        output: str,
        is_error: bool,
        duration_ms: int,
    ) -> None:
        with self._tx() as c:
            c.execute(
                "INSERT OR REPLACE INTO tool_calls (id, run_id, agent_id, tool_use_id, name,"
                " input_json, output, is_error, duration_ms, created_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    call_id,
                    run_id,
                    agent_id,
                    tool_use_id,
                    name,
                    json.dumps(tool_input, default=str),
                    output,
                    int(is_error),
                    duration_ms,
                    time.time(),
                ),
            )

    def list_tool_calls(self, run_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM tool_calls WHERE run_id = ? ORDER BY created_at", (run_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    # -- approvals ----------------------------------------------------------------------

    def create_approval(
        self,
        approval_id: str,
        run_id: str,
        agent_id: str,
        tool_use_id: str,
        name: str,
        tool_input: dict[str, Any],
        blast_radius: str = "",
    ) -> None:
        with self._tx() as c:
            c.execute(
                "INSERT INTO approvals (id, run_id, agent_id, tool_use_id, name, input_json,"
                " blast_radius, decision, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    approval_id,
                    run_id,
                    agent_id,
                    tool_use_id,
                    name,
                    json.dumps(tool_input, default=str),
                    blast_radius,
                    ApprovalDecision.PENDING.value,
                    time.time(),
                ),
            )

    def resolve_approval(
        self, approval_id: str, decision: ApprovalDecision, reason: str = ""
    ) -> None:
        with self._tx() as c:
            c.execute(
                "UPDATE approvals SET decision=?, reason=?, resolved_at=? WHERE id=?",
                (decision.value, reason, time.time(), approval_id),
            )

    def get_approval(self, approval_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM approvals WHERE id = ?", (approval_id,)
            ).fetchone()
        return dict(row) if row else None

    def list_pending_approvals(self, run_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM approvals WHERE run_id=? AND decision=? ORDER BY created_at",
                (run_id, ApprovalDecision.PENDING.value),
            ).fetchall()
        return [dict(r) for r in rows]

    # -- checkpoints --------------------------------------------------------------------

    def save_checkpoint(
        self,
        run_id: str,
        agent_id: str,
        turn: int,
        messages: list[dict[str, Any]],
        usage: dict[str, Any],
    ) -> None:
        with self._tx() as c:
            c.execute(
                "INSERT OR REPLACE INTO checkpoints (run_id, agent_id, turn, messages_json,"
                " usage_json, created_at) VALUES (?,?,?,?,?,?)",
                (
                    run_id,
                    agent_id,
                    turn,
                    json.dumps(messages, default=str),
                    json.dumps(usage),
                    time.time(),
                ),
            )

    def latest_checkpoint(self, run_id: str, agent_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM checkpoints WHERE run_id=? AND agent_id=?"
                " ORDER BY turn DESC LIMIT 1",
                (run_id, agent_id),
            ).fetchone()
        if not row:
            return None
        return {
            "turn": row["turn"],
            "messages": json.loads(row["messages_json"]),
            "usage": json.loads(row["usage_json"]),
        }

    # -- events -------------------------------------------------------------------------

    def append_event(self, run_id: str, payload: dict[str, Any]) -> int:
        """Append to the run's event log and return the assigned sequence number."""
        with self._tx() as c:
            row = c.execute(
                "SELECT COALESCE(MAX(seq), 0) AS m FROM events WHERE run_id = ?", (run_id,)
            ).fetchone()
            seq = int(row["m"]) + 1
            c.execute(
                "INSERT INTO events (run_id, seq, payload_json, created_at) VALUES (?,?,?,?)",
                (run_id, seq, json.dumps(payload, default=str), time.time()),
            )
        return seq

    def list_events(self, run_id: str, after_seq: int = 0) -> list[tuple[int, dict[str, Any]]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT seq, payload_json FROM events WHERE run_id=? AND seq>? ORDER BY seq",
                (run_id, after_seq),
            ).fetchall()
        return [(int(r["seq"]), json.loads(r["payload_json"])) for r in rows]
