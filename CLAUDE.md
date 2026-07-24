# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Aegis is a **hand-written agentic harness** for incident triage: a coordinator agent plans
hypotheses, fans out parallel investigators against a custom MCP server, adversarially verifies
findings, and writes a cited root-cause analysis — pausing for human approval before any write
action. Runs execute against a **deterministic seeded incident world with ground truth**, so the
system is *measured*, not just demoed. Model is `claude-opus-4-8` throughout; **effort** (per role,
in `config.py`), not model tier, is the cost lever.

## Commands

```bash
uv venv && uv pip install -e ".[dev,charts]"   # setup (uv, not pip/poetry)

make check          # everything CI runs: ruff (lint) + mypy --strict + pytest
make test           # uv run pytest  (hermetic — no API key, no network, no spend)
make lint fmt typecheck   # individually
make scenarios      # (re)generate + validate the incident benchmark → evals/scenarios.jsonl
make demo           # one offline run against the mock model
make up             # docker compose: api + mcp + web + jaeger
```

Run a single test / file / pattern:

```bash
uv run pytest tests/test_loop.py                      # one file
uv run pytest tests/test_loop.py::test_pause_turn_resends_without_injecting_a_message
uv run pytest -k "approval and not deny"              # by keyword
```

CLI (installed as `aegis`):

```bash
uv run aegis run --scenario evals/scenarios.jsonl:3 --mock   # offline; the CI end-to-end gate
uv run aegis run --scenario evals/scenarios.jsonl:3 --max-cost 1.00   # live (needs ANTHROPIC_API_KEY)
uv run aegis run --single-agent   # or --no-verifier / --effort low  → ablation arms
uv run aegis run --resume <run_id>                           # continue a crashed run
uv run aegis show <run_id> ; uv run aegis list
```

Eval + charts:

```bash
uv run python evals/run_eval.py --scenarios evals/scenarios.jsonl --mock   # pipeline check (see below)
make eval-live      # real measurement; costs money, --max-cost is the ceiling
make charts         # render docs/charts/*.png from evals/results/summary.json
```

`ANTHROPIC_API_KEY` is read from `.env` (via pydantic-settings) — nothing is exported globally.
The whole test suite and any `--mock` path run with no key.

## Architecture — the parts that span multiple files

**The agent loop is the project.** `src/aegis/harness/loop.py` (`Harness.run_agent`) is a
hand-written `while` over `stop_reason` — not an SDK tool-runner. Read it first. Everything else in
`harness/` is a concern the loop delegates to: `client.py` (streaming + own retry loop, SDK
auto-retry off), `budget.py` (USD accounting + a reservation that stops concurrent overshoot),
`cache.py` (byte-stable prompt prefix + fingerprint), `permissions.py` (the approval gate),
`events.py` (persist-then-publish bus), `telemetry.py` (OTel), `errors.py` (typed retry
classification).

**Two layers, connected by structured output.** `orchestrator.py` sits *above* the single-agent
loop and drives the topology: coordinator **plan** → parallel investigate→verify pipelines
(`asyncio.gather`, semaphore-bounded) → coordinator **conclude**. The two coordinator phases are the
*same agent id resumed*, so conclude sees its own hypotheses. `RunConfig` (in `orchestrator.py`)
holds the ablation switches; `run_eval.py`'s `ARMS` maps names to `RunConfig`s. `runner.py`
(`execute_run`) is the one assembly point both the CLI and the API call — inject `model`, `broker`,
and `mcp_server` here.

**The MCP server is a separate process with three real primitives.** `mcp_server/server.py` exposes
tools (telemetry, model-controlled), resources (runbooks/SLOs, pinned into the cached prefix by the
harness — that's *why* they're resources not tools), and prompts (workflow templates the dashboard
discovers via `list_prompts`). `mcp_client/bridge.py` converts MCP↔Anthropic shapes and contains
the **prompt-injection fence** (`wrap_untrusted`, non-forgeable). Transports: in-memory for
tests/mock, Streamable HTTP (`localhost:8765`) for live.

**The seeded world's determinism is load-bearing.** `mcp_server/env/`: `world.py` builds a healthy
14-service estate from a seed; `faults.py` injects one root cause and records ground truth;
`generator.py` produces a validated set. Critical invariants:
- `faults._materialise` is the **single** function both `make_scenario` (authoring) and
  `build_world` (replay) route through — never let them diverge, or the MCP server serves telemetry
  the scenario was never validated against.
- Replay must be **byte-identical across machines**: no `datetime.now`, no `random.gauss` (uses libm
  transcendentals) — use `world.normal_jitter` instead.
- `validate_scenario` re-derives the fault from observable data and is the guard rail; the generator
  exits non-zero if any scenario fails. A scenario unsolvable from telemetry is *broken*, not hard.
- **Ground truth must never leak** through any tool/resource/prompt (asserted in
  `tests/test_mcp_server.py`).

## Gotchas that will bite you

- **`--mock` accuracy is 100% by construction, not a measurement.** `evals/mock_model.py` reads
  ground truth to answer correctly. It exists to test the *machinery* (loop, tool pairing,
  budget/cache accounting, structured output) hermetically. Real numbers require a live run.
- **Structured-output schemas cannot carry numeric/length constraints.** `output_config.format`
  rejects `minimum`/`maximum`/`multipleOf`/length bounds with a 400. `StrictModel.api_schema` strips
  them; pydantic still validates client-side. `Field(ge=…, le=…)` on a new wire model needs this.
  This was a live-only bug the mock never caught — the mock doesn't exercise the API's schema
  validator.
- **The MCP server's scenario registry is module-level global state** (`server._scenarios`). Tests
  and the API's `load_scenarios` clear it; `tests/test_mcp_server.py` re-registers per-test to stay
  independent of suite ordering. `register_scenario` before any run.
- **The permission gate fails closed by default.** `execute_run`'s default broker is `AutoDeny`, so
  callers must opt into an approval mechanism: CLI live → `ConsoleApprovalBroker` (terminal prompt),
  API → `QueueApprovalBroker`, evals/mock → `AutoApprove` / `AutoDeny` explicitly. Never default a
  live path to `AutoApprove` — that's the human gate failing *open*.
- **The live eval connects to an external MCP server over HTTP** (`AEGIS_MCP_URL`, default
  `localhost:8765`); only `--mock` uses the in-process server. Start `make mcp` (with the scenarios
  loaded) before a live eval, or every tool call returns "unknown incident_id".
- **Checkpoints are written after tool results are appended, never mid-turn** — so resume never
  replays a message list ending in an unanswered `tool_use` (a guaranteed API 400). Preserve this
  ordering in `loop.py`.

## Testing philosophy

Tests exercise the **production path**, not a parallel one: a real `Harness`/`Orchestrator` wired to
the real MCP server over the in-memory transport, with only the model swapped (a `ScriptedModel` in
`tests/helpers.py` for exact `stop_reason` sequences, or `MockModel` for full-topology wiring).
`tests/test_review_fixes.py` holds regressions for eight defects an adversarial review surfaced —
keep additions there when fixing a subtle harness bug. Do not use pytest async *fixtures* for MCP
sessions (anyio cancel scopes must be exited in the entering task) — use an explicit `async with`
per test, as `tests/test_mcp_server.py` does.
