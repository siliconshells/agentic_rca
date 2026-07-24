"""Incident-world invariants.

These guard the properties every downstream number depends on:

* replay reconstructs a byte-identical world (otherwise the eval and the MCP server disagree),
* each fault class is distinguishable from the others by observable evidence alone,
* the generator refuses to emit a scenario it cannot itself solve.
"""

from __future__ import annotations

import hashlib

import pytest

from mcp_server.env.faults import (
    INJECTORS,
    LATENCY_LED,
    NO_DEPLOY_CLASSES,
    build_world,
    make_scenario,
    validate_scenario,
)
from mcp_server.env.generator import generate
from mcp_server.env.world import SERVICES, Metric, World, callers_of, upstream_path

FAULT_CLASSES = sorted(INJECTORS)


def fingerprint(world: World) -> str:
    """A content hash over everything an agent could observe."""
    h = hashlib.sha256()
    for key in sorted(world.metrics):
        h.update(repr((key, world.metrics[key])).encode())
    for log in world.logs:
        h.update(repr(log).encode())
    for deploy in world.deploys:
        h.update(repr(deploy).encode())
    for trace in world.traces:
        h.update(repr(trace).encode())
    return h.hexdigest()


# --------------------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------------------


def test_baseline_is_seed_deterministic():
    assert fingerprint(World.baseline(7)) == fingerprint(World.baseline(7))
    assert fingerprint(World.baseline(7)) != fingerprint(World.baseline(8))


@pytest.mark.parametrize("fault_class", FAULT_CLASSES)
def test_replay_reconstructs_the_authored_world(fault_class):
    """`build_world(scenario)` must equal the world `make_scenario` validated against.

    If these ever diverge, the MCP server serves telemetry that was never checked, and every
    accuracy number silently becomes meaningless.
    """
    scenario, authored = make_scenario(
        "inc-test", seed=99, fault_class=fault_class, culprit="payments-svc", onset=50
    )
    assert fingerprint(build_world(scenario)) == fingerprint(authored)


def test_replay_is_stable_across_repeated_calls():
    scenario, _ = make_scenario("inc-test", seed=11, fault_class="data_anomaly", onset=48)
    assert fingerprint(build_world(scenario)) == fingerprint(build_world(scenario))


def test_clean_scenario_has_no_fault():
    scenario, world = make_scenario("inc-clean", seed=5, fault_class="none")
    assert scenario.ground_truth.root_cause_class == "none"
    assert validate_scenario(scenario, world) == []
    for svc in SERVICES.values():
        series = world.metrics[(svc.name, Metric.ERROR_RATE)]
        assert max(series) < svc.slo_error_rate * 3


# --------------------------------------------------------------------------------------
# Solvability
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("fault_class", FAULT_CLASSES)
@pytest.mark.parametrize("culprit", ["payments-svc", "search-svc", "inventory-db"])
def test_every_fault_validates_across_culprits(fault_class, culprit):
    scenario, world = make_scenario(
        f"inc-{fault_class}", seed=31, fault_class=fault_class, culprit=culprit, onset=52
    )
    assert validate_scenario(scenario, world) == []


def test_generated_set_is_fully_valid():
    scenarios, failures = generate(count=36, seed=2024)
    assert failures == []
    assert len(scenarios) == 36
    assert {s.fault_class for s in scenarios} == set(FAULT_CLASSES) | {"none"}


# --------------------------------------------------------------------------------------
# Discriminability — the classes must not collapse into each other
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("fault_class", sorted(NO_DEPLOY_CLASSES))
def test_no_deploy_classes_have_no_deploy_at_onset(fault_class):
    """A coincidental release next to onset would make these genuinely ambiguous."""
    _, world = make_scenario(
        "inc-nd", seed=17, fault_class=fault_class, culprit="inventory-svc", onset=50
    )
    near = [d for d in world.deploys if d.service == "inventory-svc" and 44 <= d.minute <= 53]
    assert near == []


def test_saturation_shows_latency_without_an_error_spike():
    """The signature that separates saturation from a bad deploy."""
    _, world = make_scenario(
        "inc-sat", seed=23, fault_class="dependency_saturation", culprit="ledger-db", onset=50
    )
    pre, post = slice(0, 45), slice(52, 90)

    def mean(metric, window):
        s = world.metrics[("ledger-db", metric)][window]
        return sum(s) / len(s)

    assert mean(Metric.LATENCY_P99, post) / mean(Metric.LATENCY_P99, pre) > 3.0
    assert mean(Metric.ERROR_RATE, post) / mean(Metric.ERROR_RATE, pre) < 3.0
    assert mean(Metric.SATURATION, post) > 80


def test_resource_exhaustion_ramps_before_onset():
    """The ramp is what distinguishes a leak from a release."""
    _, world = make_scenario(
        "inc-oom", seed=29, fault_class="resource_exhaustion", culprit="cart-svc", onset=55
    )
    mem = world.metrics[("cart-svc", Metric.MEMORY)]
    assert mem[54] > mem[10] * 1.3, "memory must already be elevated before onset"
    assert any("OOMKilled" in log.message for log in world.logs if log.service == "cart-svc")


@pytest.mark.parametrize("fault_class", FAULT_CLASSES)
def test_alert_fires_at_a_service_upstream_of_the_culprit(fault_class):
    """The page must point somewhere other than the fault, or triage is trivial."""
    scenario, _ = make_scenario(
        "inc-sym", seed=37, fault_class=fault_class, culprit="auth-svc", onset=50
    )
    truth = scenario.ground_truth
    assert truth.symptom_service != truth.culprit_service
    assert truth.symptom_service in upstream_path(truth.culprit_service)
    assert truth.symptom_service in scenario.alert


@pytest.mark.parametrize("fault_class", sorted(LATENCY_LED))
def test_latency_led_faults_breach_on_latency(fault_class):
    scenario, world = make_scenario(
        "inc-lat", seed=41, fault_class=fault_class, culprit="session-cache", onset=50
    )
    symptom = scenario.ground_truth.symptom_service
    series = world.metrics[(symptom, Metric.LATENCY_P99)][55:]
    assert max(series) > SERVICES[symptom].slo_p99_ms


# --------------------------------------------------------------------------------------
# Evidence and injection
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("fault_class", FAULT_CLASSES)
def test_causal_evidence_is_observable(fault_class):
    scenario, world = make_scenario(
        "inc-ev", seed=53, fault_class=fault_class, culprit="pricing-svc", onset=50
    )
    observable = (
        {log.id for log in world.logs}
        | {d.id for d in world.deploys}
        | {t.id for t in world.traces}
    )
    truth = scenario.ground_truth
    assert truth.causal_event_ids
    assert set(truth.causal_event_ids) <= observable


def test_injection_payload_is_present_and_does_not_change_ground_truth():
    plain, _ = make_scenario(
        "inc-a", seed=61, fault_class="bad_deploy", culprit="auth-svc", onset=50
    )
    attacked, world = make_scenario(
        "inc-b",
        seed=61,
        fault_class="bad_deploy",
        culprit="auth-svc",
        onset=50,
        with_injection=True,
    )

    assert attacked.ground_truth.root_cause_class == plain.ground_truth.root_cause_class
    assert attacked.ground_truth.culprit_service == plain.ground_truth.culprit_service
    assert attacked.ground_truth.injected_prompt_attack

    payload_log = next(
        log for log in world.logs if log.id == attacked.ground_truth.injection_log_id
    )
    assert "IGNORE" in payload_log.message.upper() or "SYSTEM NOTICE" in payload_log.message


def test_symptoms_attenuate_with_distance_from_the_fault():
    """The nearest caller must hurt more than the edge — that gradient is a triage signal."""
    _, world = make_scenario(
        "inc-grad", seed=67, fault_class="bad_deploy", culprit="payments-svc", onset=50
    )
    post = slice(55, 90)

    def err(service):
        s = world.metrics[(service, Metric.ERROR_RATE)][post]
        return sum(s) / len(s)

    near = callers_of("payments-svc")[0]  # checkout-api
    assert err(near) > err("edge-gateway")
