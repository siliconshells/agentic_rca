"""The evaluation harness.

Runs the triage system over a validated scenario set, across ablation arms, and grades every run
against ground truth. This is what turns the project from a demo into an experiment: it produces
the accuracy / evidence-precision / injection-resistance / cost numbers, per arm, that the README
reports.

    # offline pipeline check — no API key, no spend, result is 100% by construction (see mock)
    python evals/run_eval.py --scenarios evals/scenarios.jsonl --mock

    # a real measurement — costs money; --max-cost is a hard ceiling across the whole sweep
    python evals/run_eval.py --arms full,single_agent,no_verifier --limit 30 --max-cost 25

Every run uses the fail-closed ``AutoDeny`` broker: unattended, world-mutating actions are never
taken, which both matches a real headless deployment and makes injection resistance a clean
measurement (a manipulated agent cannot execute the planted action, so the metric isolates whether
its *reasoning* was corrupted).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from aegis.config import BudgetConfig, RetryConfig, Settings, get_settings
from aegis.harness.permissions import AutoDeny
from aegis.orchestrator import RunConfig
from aegis.runner import RunRequest, execute_run
from aegis.store.db import Store
from aegis.types import Effort, Role, Usage
from evals.metrics import Grade, aggregate, grade_rca
from mcp_server.env.faults import Scenario
from mcp_server.env.generator import load_scenarios

ARMS: dict[str, RunConfig] = {
    "full": RunConfig(),
    "no_verifier": RunConfig.no_verifier(),
    "single_agent": RunConfig.single_agent(),
    # Effort sweep on the investigators, full topology otherwise.
    "effort_low": RunConfig(investigator_effort="low"),
    "effort_medium": RunConfig(investigator_effort="medium"),
    "effort_high": RunConfig(investigator_effort="high"),
}


def _run_usage(store: Store, run_id: str) -> tuple[Usage, int]:
    """Aggregate token usage and turn count across a run's agent spans."""
    total = Usage()
    turns = 0
    for span in store.list_spans(run_id):
        turns += int(span["turns"] or 0)
        if span["usage_json"]:
            total = total + Usage(**json.loads(span["usage_json"]))
    return total, turns


def _executed_writes(store: Store, run_id: str) -> int:
    """Count remediation tools that actually executed (were approved). Zero under AutoDeny."""
    return sum(
        1
        for c in store.list_tool_calls(run_id)
        if c["name"] in ("propose_remediation", "execute_remediation") and not c["is_error"]
    )


async def _one_run(
    scenario: Scenario,
    arm: str,
    run_config: RunConfig,
    settings: Settings,
    store: Store,
    *,
    mock: bool,
) -> Grade:
    from mcp_server.server import register_scenario

    register_scenario(scenario)

    model = None
    mcp_server = None
    if mock:
        from evals.mock_model import MockModel
        from mcp_server.server import mcp as mcp_server_app

        model = MockModel(scenario=scenario)
        mcp_server = mcp_server_app

    started = time.monotonic()
    result = await execute_run(
        RunRequest(
            incident_id=scenario.id,
            alert=scenario.alert,
            run_id=f"eval-{arm}-{scenario.id}",
            run_config=run_config,
        ),
        settings=settings,
        store=store,
        model=model,
        broker=AutoDeny(),  # fail-closed: no unattended writes, ever
        mcp_server=mcp_server,
    )
    duration = time.monotonic() - started

    usage, turns = _run_usage(store, result.run_id)
    rca = json.loads(result.rca.model_dump_json()) if result.rca else None
    return grade_rca(
        scenario,
        rca,
        cost_usd=result.cost_usd,
        duration_s=duration,
        turns=turns,
        cache_hit_rate=usage.cache_hit_rate,
        executed_writes=_executed_writes(store, result.run_id),
    )


async def run_eval(
    scenarios: list[Scenario],
    arms: list[str],
    *,
    mock: bool,
    max_cost: float,
    settings: Settings,
    store: Store,
    on_progress=None,
    already_spent: float = 0.0,
) -> tuple[dict[str, list[Grade]], float]:
    """Run every arm over every scenario under a hard cost ceiling.

    ``already_spent`` seeds the running total so a caller invoking this once per arm still gets a
    *cumulative* ceiling — the ceiling is the whole sweep's budget, not each arm's. Returns the
    grades and the new running total.
    """
    grades: dict[str, list[Grade]] = {arm: [] for arm in arms}
    spent = already_spent

    for arm in arms:
        run_config = ARMS[arm]
        for scenario in scenarios:
            if not mock and spent >= max_cost:
                print(
                    f"[eval] cumulative cost ceiling ${max_cost:.2f} reached; "
                    f"stopping (some arms/scenarios not run)",
                    file=sys.stderr,
                )
                return grades, spent
            grade = await _one_run(scenario, arm, run_config, settings, store, mock=mock)
            spent += grade.cost_usd
            grades[arm].append(grade)
            if on_progress:
                on_progress(arm, scenario, grade, spent)

    return grades, spent


def _apply_effort_arm(settings: Settings, arm: str) -> None:
    """The effort-sweep arms override the investigator effort globally for their pass."""
    if arm.startswith("effort_"):
        settings.role(Role.INVESTIGATOR).effort = Effort(arm.removeprefix("effort_"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the incident-triage harness.")
    parser.add_argument("--scenarios", type=Path, default=Path("evals/scenarios.jsonl"))
    parser.add_argument("--arms", default="full,single_agent,no_verifier")
    parser.add_argument("--limit", type=int, default=0, help="Cap scenarios per arm (0 = all).")
    parser.add_argument("--mock", action="store_true", help="Offline pipeline check (no spend).")
    parser.add_argument("--max-cost", type=float, default=10.0, help="Hard USD ceiling.")
    parser.add_argument("--out", type=Path, default=Path("evals/results/summary.json"))
    args = parser.parse_args(argv)

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        print(f"unknown arms: {unknown}. Valid: {sorted(ARMS)}", file=sys.stderr)
        return 1

    scenarios = load_scenarios(args.scenarios)
    if args.limit:
        scenarios = scenarios[: args.limit]

    settings = get_settings()
    if args.mock:
        # Deterministic and fast; make retries instant.
        settings = Settings(
            anthropic_api_key="test",
            db_path=Path("evals/results/eval.db"),
            budget=BudgetConfig(max_cost_usd=100.0, max_wall_clock_s=600),
            retry=RetryConfig(max_attempts=2, base_delay_s=0.0, max_delay_s=0.0),
        )
    Path("evals/results").mkdir(parents=True, exist_ok=True)
    store = Store(settings.db_path)

    print(
        f"[eval] {len(scenarios)} scenarios x {len(arms)} arms "
        f"({'mock' if args.mock else 'LIVE'}), ceiling ${args.max_cost:.2f}"
    )

    def progress(arm, scenario, grade, spent):
        mark = "ok" if grade.class_correct else "XX"
        print(
            f"  [{mark}] {arm:<14} {scenario.id} "
            f"pred={_short(grade)} cost=${grade.cost_usd:.4f} spent=${spent:.2f}"
        )

    # Effort arms need the setting applied before their runs; run them one arm at a time so the
    # global override is scoped correctly. The cost ceiling is CUMULATIVE across arms — threaded
    # via `already_spent` — so `--max-cost` caps the whole sweep, not each arm.
    all_grades: dict[str, list[Grade]] = {}
    spent = 0.0
    for arm in arms:
        _apply_effort_arm(settings, arm)
        grades, spent = asyncio.run(
            run_eval(
                scenarios,
                [arm],
                mock=args.mock,
                max_cost=args.max_cost,
                settings=settings,
                store=store,
                on_progress=progress,
                already_spent=spent,
            )
        )
        all_grades.update(grades)

    summary = {
        "meta": {
            "scenarios": len(scenarios),
            "arms": arms,
            "mock": args.mock,
            "generated_at_epoch": int(time.time()),
        },
        "arms": {arm: aggregate(arm, gs).as_dict() for arm, gs in all_grades.items()},
        "grades": {arm: [g.__dict__ for g in gs] for arm, gs in all_grades.items()},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    store.close()

    _print_table(summary["arms"])
    print(f"\n[eval] wrote {args.out}")
    return 0


def _short(grade: Grade) -> str:
    return f"{grade.fault_class[:4]}{'*' if grade.class_correct else '!'}"


def _print_table(arms: dict[str, dict]) -> None:
    cols = [
        ("arm", "arm", 14),
        ("class_accuracy", "class", 7),
        ("culprit_accuracy", "culprit", 7),
        ("evidence_precision", "ev-prec", 7),
        ("evidence_recall", "ev-rec", 7),
        ("false_positive_rate", "fp", 6),
        ("injection_resistance", "inj", 6),
        ("mean_cost_usd", "cost", 8),
        ("p95_duration_s", "p95_s", 7),
    ]
    print("\n" + "  ".join(f"{h:<{w}}" for _, h, w in cols))
    print("  ".join("-" * w for _, _, w in cols))
    for arm in arms.values():
        cells = []
        for key, _, w in cols:
            v = arm.get(key)
            if v is None:
                s = "—"
            elif key == "arm":
                s = str(v)
            elif key == "mean_cost_usd":
                s = f"${v:.4f}"
            elif "duration" in key:
                s = f"{v:.1f}"
            else:
                s = f"{v:.2f}"
            cells.append(f"{s:<{w}}")
        print("  ".join(cells))


if __name__ == "__main__":
    raise SystemExit(main())
