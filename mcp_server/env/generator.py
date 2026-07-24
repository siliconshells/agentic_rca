"""Scenario generation CLI.

Builds a balanced, validated scenario set. Every scenario is checked against
``faults.validate_scenario`` before it is written; the command exits non-zero if any scenario
fails, because an unsolvable scenario silently deflates every downstream accuracy number.

    python -m mcp_server.env.generator --scenarios 60 --out evals/scenarios.jsonl

Scenarios store only a seed and a fault spec — never the telemetry. The world is rebuilt on
demand by ``faults.build_world``, which keeps the file small and guarantees the MCP server and
the eval harness are looking at identical data.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

from mcp_server.env.faults import (
    CANDIDATE_CULPRITS,
    INJECTORS,
    Scenario,
    build_world,
    make_scenario,
    validate_scenario,
)
from mcp_server.env.world import WINDOW_MINUTES

FAULT_CLASSES = tuple(INJECTORS)


def generate(
    count: int,
    seed: int = 1337,
    clean_fraction: float = 0.12,
    injection_fraction: float = 0.12,
) -> tuple[list[Scenario], list[str]]:
    """Return (scenarios, validation_failures).

    Classes are assigned round-robin rather than sampled, so counts differ by at most one and
    per-class accuracy is comparable without reweighting.
    """
    rng = random.Random(seed)
    scenarios: list[Scenario] = []
    failures: list[str] = []

    n_clean = max(1, round(count * clean_fraction))
    n_faulty = count - n_clean

    plan: list[str] = [FAULT_CLASSES[i % len(FAULT_CLASSES)] for i in range(n_faulty)]
    plan += ["none"] * n_clean
    rng.shuffle(plan)

    n_injection = round(count * injection_fraction)
    injection_slots = set(rng.sample(range(count), k=min(n_injection, count)))

    for i, fault_class in enumerate(plan):
        scenario_seed = seed + i * 101
        culprit = rng.choice(CANDIDATE_CULPRITS) if fault_class != "none" else None
        # Vary onset so agents cannot learn a fixed "look at minute 50" heuristic.
        onset = rng.randint(35, WINDOW_MINUTES - 25)

        scenario, world = make_scenario(
            scenario_id=f"inc-{i + 1:04d}",
            seed=scenario_seed,
            fault_class=fault_class,
            culprit=culprit,
            onset=onset,
            with_injection=i in injection_slots,
        )
        failures.extend(validate_scenario(scenario, world))
        scenarios.append(scenario)

    return scenarios, failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a validated incident scenario set.")
    parser.add_argument("--scenarios", type=int, default=60)
    parser.add_argument("--out", type=Path, default=Path("evals/scenarios.jsonl"))
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--clean-fraction", type=float, default=0.12)
    parser.add_argument("--injection-fraction", type=float, default=0.12)
    parser.add_argument(
        "--strict",
        action="store_true",
        default=True,
        help="Exit non-zero if any scenario fails validation (default).",
    )
    parser.add_argument("--no-strict", dest="strict", action="store_false")
    args = parser.parse_args(argv)

    scenarios, failures = generate(
        args.scenarios, args.seed, args.clean_fraction, args.injection_fraction
    )

    by_class = Counter(s.fault_class for s in scenarios)
    n_injection = sum(1 for s in scenarios if s.with_injection)

    print(f"generated {len(scenarios)} scenarios (seed {args.seed}, window {WINDOW_MINUTES}min)")
    for name, n in sorted(by_class.items()):
        print(f"  {name:<24} {n:>3}")
    print(f"  {'(with prompt injection)':<24} {n_injection:>3}")
    print(f"validation failures: {len(failures)}")
    for failure in failures[:10]:
        print(f"  ! {failure}", file=sys.stderr)
    if len(failures) > 10:
        print(f"  ... and {len(failures) - 10} more", file=sys.stderr)

    if failures and args.strict:
        print("refusing to write an invalid scenario set", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for scenario in scenarios:
            fh.write(json.dumps(scenario.as_dict()) + "\n")
    print(f"wrote {args.out}")
    return 0


def load_scenarios(path: Path | str) -> list[Scenario]:
    """Read a scenario file written by this module."""
    return [
        Scenario.from_dict(json.loads(line))
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]


def load_scenario_ref(ref: str) -> Scenario:
    """Resolve a ``path/to/scenarios.jsonl:INDEX`` or ``path:SCENARIO_ID`` reference."""
    path_str, _, selector = ref.rpartition(":")
    if not path_str:
        raise ValueError(f"expected 'path:index' or 'path:id', got {ref!r}")
    scenarios = load_scenarios(path_str)
    if selector.isdigit():
        return scenarios[int(selector)]
    for scenario in scenarios:
        if scenario.id == selector:
            return scenario
    raise KeyError(f"no scenario {selector!r} in {path_str}")


__all__ = ["build_world", "generate", "load_scenario_ref", "load_scenarios", "main"]

if __name__ == "__main__":
    raise SystemExit(main())
