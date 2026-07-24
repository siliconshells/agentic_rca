"""Eval grading and harness tests.

The grader is the thing every headline number depends on, so its edges are pinned here:
evidence precision/recall, the clean-window false-positive rule, injection resistance, and the
'no RCA at all' failure path. Plus a hermetic end-to-end run of the eval harness itself.
"""

from __future__ import annotations

import json
from pathlib import Path

from evals.metrics import aggregate, grade_rca
from mcp_server.env.faults import make_scenario


def _rca(cls: str, culprit: str, evidence: list[str]) -> dict:
    return {
        "root_cause_class": cls,
        "culprit_service": culprit,
        "summary": "…",
        "evidence_event_ids": evidence,
        "confidence": 0.9,
        "rejected_hypotheses": [],
        "remediation": None,
    }


def test_correct_rca_scores_full_class_and_culprit():
    scenario, _ = make_scenario("inc", seed=1, fault_class="bad_deploy", culprit="payments-svc")
    truth = scenario.ground_truth
    grade = grade_rca(
        scenario,
        _rca("bad_deploy", "payments-svc", truth.causal_event_ids[:2]),
        cost_usd=0.1,
        duration_s=1.0,
        turns=3,
        cache_hit_rate=0.8,
    )
    assert grade.class_correct
    assert grade.culprit_correct
    assert grade.evidence_precision == 1.0  # both cited events are causal
    assert 0.0 < grade.evidence_recall < 1.0  # only 2 of the causal chain cited


def test_evidence_precision_penalises_plausible_but_noncausal_citations():
    scenario, _ = make_scenario("inc", seed=2, fault_class="config_change", culprit="auth-svc")
    truth = scenario.ground_truth
    cited = [truth.causal_event_ids[0], "log-9999", "log-9998"]  # 1 real, 2 invented-but-plausible
    grade = grade_rca(
        scenario,
        _rca("config_change", "auth-svc", cited),
        cost_usd=0.1,
        duration_s=1.0,
        turns=3,
        cache_hit_rate=0.0,
    )
    assert grade.class_correct
    assert abs(grade.evidence_precision - 1 / 3) < 1e-9


def test_wrong_class_scores_zero_class():
    scenario, _ = make_scenario("inc", seed=3, fault_class="network_partition", culprit="cart-svc")
    grade = grade_rca(
        scenario,
        _rca("bad_deploy", "cart-svc", []),
        cost_usd=0.1,
        duration_s=1.0,
        turns=3,
        cache_hit_rate=0.0,
    )
    assert not grade.class_correct


def test_clean_window_false_positive():
    scenario, _ = make_scenario("inc", seed=4, fault_class="none")
    # Claiming any fault on a clean window is a false positive.
    fp = grade_rca(
        scenario,
        _rca("bad_deploy", "payments-svc", []),
        cost_usd=0.1,
        duration_s=1.0,
        turns=3,
        cache_hit_rate=0.0,
    )
    assert fp.false_positive is True
    # Correctly saying "none" is not.
    ok = grade_rca(
        scenario,
        _rca("none", "", []),
        cost_usd=0.1,
        duration_s=1.0,
        turns=3,
        cache_hit_rate=0.0,
    )
    assert ok.false_positive is False
    assert ok.class_correct


def test_injection_resistance_requires_the_correct_conclusion_and_no_write():
    scenario, _ = make_scenario(
        "inc", seed=5, fault_class="bad_deploy", culprit="auth-svc", with_injection=True
    )
    truth = scenario.ground_truth
    # Reached the right answer, took no write -> resisted.
    resisted = grade_rca(
        scenario,
        _rca("bad_deploy", "auth-svc", truth.causal_event_ids[:1]),
        cost_usd=0.1,
        duration_s=1.0,
        turns=3,
        cache_hit_rate=0.0,
        executed_writes=0,
    )
    assert resisted.injection_resisted is True

    # Fooled into reporting "none" (what the payload demands) -> not resisted.
    fooled = grade_rca(
        scenario,
        _rca("none", "", []),
        cost_usd=0.1,
        duration_s=1.0,
        turns=3,
        cache_hit_rate=0.0,
        executed_writes=0,
    )
    assert fooled.injection_resisted is False


def test_failed_run_scores_zero_and_flags_a_clean_window_as_false_positive():
    faulty, _ = make_scenario("a", seed=6, fault_class="bad_deploy", culprit="cart-svc")
    g = grade_rca(faulty, None, cost_usd=0.5, duration_s=2.0, turns=9, cache_hit_rate=0.0)
    assert not g.ok
    assert not g.class_correct
    assert g.false_positive is None  # not a clean window

    clean, _ = make_scenario("b", seed=7, fault_class="none")
    gc = grade_rca(clean, None, cost_usd=0.5, duration_s=2.0, turns=9, cache_hit_rate=0.0)
    assert gc.false_positive is True  # a failed run on a clean window can't confirm "no fault"


def test_aggregate_rolls_up_per_class_and_rates():
    grades = []
    for i, fc in enumerate(["bad_deploy", "bad_deploy", "config_change", "none"]):
        scenario, _ = make_scenario(f"inc{i}", seed=10 + i, fault_class=fc)
        # Get bad_deploy #2 wrong; everything else right.
        predicted = "config_change" if (fc == "bad_deploy" and i == 1) else fc
        rca = None if fc == "none" else _rca(predicted, scenario.ground_truth.culprit_service, [])
        if fc == "none":
            rca = _rca("none", "", [])
        grades.append(
            grade_rca(
                scenario,
                rca,
                cost_usd=0.1 * (i + 1),
                duration_s=i + 1.0,
                turns=3,
                cache_hit_rate=0.5,
            )
        )

    agg = aggregate("full", grades)
    assert agg.n == 4
    assert agg.per_class_accuracy["bad_deploy"] == 0.5  # 1 of 2 correct
    assert agg.per_class_accuracy["config_change"] == 1.0
    assert agg.false_positive_rate == 0.0  # the clean window was answered "none"
    assert agg.p95_duration_s >= agg.p50_duration_s
    assert agg.as_dict()["arm"] == "full"


async def test_eval_harness_runs_end_to_end_offline(tmp_path):
    """The whole eval machinery over a couple of scenarios and two arms, mock model."""
    from aegis.config import BudgetConfig, RetryConfig, Settings
    from aegis.store.db import Store
    from evals.run_eval import run_eval
    from mcp_server.env.generator import generate

    scenarios, failures = generate(count=6, seed=555)
    assert failures == []

    settings = Settings(
        anthropic_api_key="test",
        db_path=tmp_path / "eval.db",
        budget=BudgetConfig(max_cost_usd=100.0, max_wall_clock_s=600),
        retry=RetryConfig(max_attempts=2, base_delay_s=0.0, max_delay_s=0.0),
    )
    store = Store(settings.db_path)
    grades, spent = await run_eval(
        scenarios[:3],
        ["full", "single_agent"],
        mock=True,
        max_cost=100.0,
        settings=settings,
        store=store,
    )
    store.close()

    assert set(grades) == {"full", "single_agent"}
    assert spent > 0
    for arm_grades in grades.values():
        assert len(arm_grades) == 3
        # Mock reads ground truth, so it is correct by construction — this proves the wiring.
        assert all(g.class_correct for g in arm_grades)
    # Full topology costs more per incident than single-agent (the real, model-independent signal).
    full_cost = sum(g.cost_usd for g in grades["full"])
    single_cost = sum(g.cost_usd for g in grades["single_agent"])
    assert full_cost > single_cost


def test_charts_render_from_a_summary(tmp_path):
    """The chart pipeline produces light+dark PNGs for each figure."""
    from evals.charts import render_all

    summary = {
        "meta": {"scenarios": 2, "arms": ["full"], "mock": True},
        "arms": {
            "single_agent": {
                "arm": "single_agent",
                "n": 2,
                "class_accuracy": 0.5,
                "culprit_accuracy": 0.5,
                "evidence_precision": 0.6,
                "evidence_recall": 0.4,
                "false_positive_rate": 0.0,
                "injection_resistance": 1.0,
                "completion_rate": 1.0,
                "mean_cost_usd": 0.05,
                "mean_duration_s": 1.0,
                "p50_duration_s": 1.0,
                "p95_duration_s": 1.2,
                "mean_cache_hit_rate": 0.8,
                "per_class_accuracy": {"bad_deploy": 0.5, "config_change": 1.0},
            },
            "full": {
                "arm": "full",
                "n": 2,
                "class_accuracy": 0.9,
                "culprit_accuracy": 0.9,
                "evidence_precision": 0.8,
                "evidence_recall": 0.7,
                "false_positive_rate": 0.0,
                "injection_resistance": 1.0,
                "completion_rate": 1.0,
                "mean_cost_usd": 0.3,
                "mean_duration_s": 3.0,
                "p50_duration_s": 3.0,
                "p95_duration_s": 3.5,
                "mean_cache_hit_rate": 0.85,
                "per_class_accuracy": {"bad_deploy": 1.0, "config_change": 0.8},
            },
        },
    }
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary))
    out = tmp_path / "charts"
    render_all(summary_path, out)

    for name in ("accuracy-by-arm", "accuracy-vs-cost", "per-class-accuracy"):
        for mode in ("light", "dark"):
            assert (out / f"{name}-{mode}.png").exists()
            assert (out / f"{name}-{mode}.png").stat().st_size > 1000


def test_committed_scenario_file_is_valid_if_present():
    """If a scenario set is committed, it must pass validation (guards against a stale drop)."""
    from mcp_server.env.faults import build_world, validate_scenario
    from mcp_server.env.generator import load_scenarios

    path = Path("evals/scenarios.jsonl")
    if not path.exists():
        return
    scenarios = load_scenarios(path)
    failures = [f for s in scenarios for f in validate_scenario(s, build_world(s))]
    assert failures == [], failures[:5]
