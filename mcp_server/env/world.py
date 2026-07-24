"""The deterministic incident world.

A seeded, pure-function microservice estate: a fixed dependency graph, per-minute metric series,
logs, deploy events, and distributed traces. No clock, no network, no randomness that isn't
derived from the scenario seed — so a scenario id plus a seed reconstructs byte-identical
telemetry on any machine, which is what makes the eval suite meaningful.

The world is generated in two stages:

1. ``World.baseline(seed)`` — a healthy estate.
2. ``faults.inject(world, spec)`` — mutates it into an incident and returns ground truth.

Symptoms deliberately surface *upstream* of the culprit. The alert fires on the service a user
notices; the fault is usually two hops down. Recovering that chain is the task.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any

# Fixed synthetic epoch. Never `datetime.now()` — reproducibility is the whole point.
EPOCH = datetime(2026, 3, 15, 14, 0, 0, tzinfo=UTC)
WINDOW_MINUTES = 90
# Faults land here so investigators have ~40 minutes of healthy baseline to compare against.
DEFAULT_ONSET_MINUTE = 50


class Tier(StrEnum):
    EDGE = "edge"
    API = "api"
    SERVICE = "service"
    DATASTORE = "datastore"


class Metric(StrEnum):
    LATENCY_P50 = "latency_p50_ms"
    LATENCY_P99 = "latency_p99_ms"
    ERROR_RATE = "error_rate"
    CPU = "cpu_pct"
    MEMORY = "memory_pct"
    RPS = "rps"
    SATURATION = "saturation_pct"


@dataclass(frozen=True)
class Service:
    name: str
    tier: Tier
    owner: str
    depends_on: tuple[str, ...]
    slo_p99_ms: int
    slo_error_rate: float
    baseline_rps: float


# 14 services. The shape matters: several distinct 2-3 hop paths from the edge, so a symptom at
# the top has more than one plausible explanation and hypotheses genuinely compete.
TOPOLOGY: tuple[Service, ...] = (
    Service("edge-gateway", Tier.EDGE, "platform", ("api-gateway",), 1200, 0.005, 2400),
    Service(
        "api-gateway",
        Tier.API,
        "platform",
        ("checkout-api", "catalog-api", "account-api"),
        1000,
        0.005,
        2200,
    ),
    Service(
        "checkout-api",
        Tier.API,
        "payments",
        ("payments-svc", "inventory-svc", "cart-svc"),
        800,
        0.01,
        640,
    ),
    Service("catalog-api", Tier.API, "discovery", ("search-svc", "pricing-svc"), 600, 0.01, 980),
    Service("account-api", Tier.API, "identity", ("auth-svc",), 500, 0.01, 520),
    Service("payments-svc", Tier.SERVICE, "payments", ("ledger-db",), 400, 0.005, 610),
    Service("inventory-svc", Tier.SERVICE, "fulfilment", ("inventory-db",), 300, 0.01, 700),
    Service("cart-svc", Tier.SERVICE, "payments", ("session-cache",), 200, 0.01, 900),
    Service("search-svc", Tier.SERVICE, "discovery", (), 350, 0.02, 960),
    Service("pricing-svc", Tier.SERVICE, "discovery", ("inventory-db",), 250, 0.01, 880),
    Service("auth-svc", Tier.SERVICE, "identity", ("session-cache",), 180, 0.005, 700),
    Service("ledger-db", Tier.DATASTORE, "payments", (), 120, 0.001, 600),
    Service("inventory-db", Tier.DATASTORE, "fulfilment", (), 100, 0.001, 1500),
    Service("session-cache", Tier.DATASTORE, "platform", (), 40, 0.001, 1600),
)

SERVICES: dict[str, Service] = {s.name: s for s in TOPOLOGY}


def callers_of(service: str) -> list[str]:
    """Services that call ``service`` directly — the direction symptoms propagate."""
    return sorted(s.name for s in TOPOLOGY if service in s.depends_on)


def upstream_path(service: str) -> list[str]:
    """Every service that transitively depends on ``service``, nearest first."""
    seen: list[str] = []
    frontier = callers_of(service)
    while frontier:
        node = frontier.pop(0)
        if node in seen:
            continue
        seen.append(node)
        frontier.extend(callers_of(node))
    return seen


def downstream_of(service: str, depth: int = 2) -> list[str]:
    """Transitive dependencies of ``service`` up to ``depth`` hops."""
    out: list[str] = []
    frontier = [(d, 1) for d in SERVICES[service].depends_on]
    while frontier:
        node, d = frontier.pop(0)
        if node in out or d > depth:
            continue
        out.append(node)
        frontier.extend((c, d + 1) for c in SERVICES[node].depends_on)
    return out


# --------------------------------------------------------------------------------------
# Observable records
# --------------------------------------------------------------------------------------


@dataclass
class LogEvent:
    id: str
    minute: int
    service: str
    level: str
    message: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "ts": iso(self.minute),
            "service": self.service,
            "level": self.level,
            "message": self.message,
        }


@dataclass
class DeployEvent:
    id: str
    minute: int
    service: str
    kind: str  # release | config | feature_flag
    version: str
    summary: str
    actor: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "ts": iso(self.minute),
            "service": self.service,
            "kind": self.kind,
            "version": self.version,
            "summary": self.summary,
            "actor": self.actor,
        }


@dataclass
class TraceSpan:
    service: str
    duration_ms: int
    status: str  # ok | error | timeout
    error: str = ""


@dataclass
class Trace:
    id: str
    minute: int
    root_service: str
    status: str
    spans: list[TraceSpan]

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "ts": iso(self.minute),
            "root_service": self.root_service,
            "status": self.status,
            "total_duration_ms": sum(s.duration_ms for s in self.spans),
            "spans": [
                {
                    "service": s.service,
                    "duration_ms": s.duration_ms,
                    "status": s.status,
                    **({"error": s.error} if s.error else {}),
                }
                for s in self.spans
            ],
        }


def iso(minute: int) -> str:
    return (EPOCH + timedelta(minutes=minute)).isoformat().replace("+00:00", "Z")


def minute_of(ts: str) -> int:
    return int((datetime.fromisoformat(ts.replace("Z", "+00:00")) - EPOCH).total_seconds() // 60)


def normal_jitter(rng: random.Random, mu: float, sigma: float) -> float:
    """A bell-shaped jitter that is byte-identical across machines.

    ``random.gauss`` is deterministic for a given seed *on one platform*, but it calls libm
    ``log``/``sqrt``/``cos`` under the hood, whose last-bit results can differ across CPUs and libc
    builds — which would break the cross-machine replay guarantee the whole eval rests on. The sum
    of three uniforms is an Irwin-Hall approximation to a Gaussian that uses only ``random()``
    (pure Python, exact), so replay is identical everywhere. The shape is close enough; these are
    synthetic metrics, not a statistics benchmark.
    """
    # Sum of 3 U(0,1) has mean 1.5 and variance 0.25 (std 0.5); rescale to (mu, sigma).
    unit = (rng.random() + rng.random() + rng.random() - 1.5) / 0.5
    return mu + sigma * unit


# --------------------------------------------------------------------------------------
# The world
# --------------------------------------------------------------------------------------


@dataclass
class World:
    """A fully materialised incident window. Queries below back the MCP tools."""

    seed: int
    window_minutes: int = WINDOW_MINUTES
    metrics: dict[tuple[str, str], list[float]] = field(default_factory=dict)
    logs: list[LogEvent] = field(default_factory=list)
    deploys: list[DeployEvent] = field(default_factory=list)
    traces: list[Trace] = field(default_factory=list)
    _next_id: dict[str, int] = field(default_factory=dict)

    # -- construction -------------------------------------------------------------------

    def new_id(self, prefix: str) -> str:
        n = self._next_id.get(prefix, 0) + 1
        self._next_id[prefix] = n
        return f"{prefix}-{n:04d}"

    @classmethod
    def baseline(cls, seed: int, window_minutes: int = WINDOW_MINUTES) -> World:
        """A healthy estate: metrics wander inside SLO, logs are routine chatter."""
        rng = random.Random(seed)
        world = cls(seed=seed, window_minutes=window_minutes)

        for svc in TOPOLOGY:
            world._seed_metrics(svc, rng)
            world._seed_logs(svc, rng)

        world._seed_routine_deploys(rng)
        world._seed_traces(rng, count=40)
        return world

    def _seed_metrics(self, svc: Service, rng: random.Random) -> None:
        n = self.window_minutes
        # Healthy services sit at ~35-55% of their p99 SLO; a wandering walk keeps the series
        # from looking synthetic while staying comfortably inside budget.
        p99_base = svc.slo_p99_ms * rng.uniform(0.35, 0.55)
        p50_base = p99_base * rng.uniform(0.25, 0.4)
        err_base = svc.slo_error_rate * rng.uniform(0.05, 0.3)
        cpu_base = rng.uniform(25, 45)
        mem_base = rng.uniform(35, 55)
        sat_base = rng.uniform(15, 35)

        def walk(base: float, jitter: float, floor: float = 0.0) -> list[float]:
            out, cur = [], base
            for _ in range(n):
                cur += normal_jitter(rng, 0, base * jitter)
                cur += (base - cur) * 0.25  # mean-reverting, so it never drifts away
                out.append(max(floor, round(cur, 4)))
            return out

        # Diurnal-ish traffic shape so RPS isn't flat.
        rps = [
            max(
                1.0,
                round(
                    svc.baseline_rps * (1 + 0.12 * normal_jitter(rng, 0, 1)) * (1 + 0.1 * (i / n)),
                    2,
                ),
            )
            for i in range(n)
        ]

        self.metrics[(svc.name, Metric.LATENCY_P99)] = walk(p99_base, 0.08)
        self.metrics[(svc.name, Metric.LATENCY_P50)] = walk(p50_base, 0.08)
        self.metrics[(svc.name, Metric.ERROR_RATE)] = walk(err_base, 0.35)
        self.metrics[(svc.name, Metric.CPU)] = walk(cpu_base, 0.1)
        self.metrics[(svc.name, Metric.MEMORY)] = walk(mem_base, 0.04)
        self.metrics[(svc.name, Metric.SATURATION)] = walk(sat_base, 0.12)
        self.metrics[(svc.name, Metric.RPS)] = rps

    def _seed_logs(self, svc: Service, rng: random.Random) -> None:
        chatter = [
            "health check ok",
            "connection pool resized to {n}",
            "cache warm complete in {n}ms",
            "processed {n} requests in batch",
            "scheduled compaction finished",
        ]
        for minute in range(self.window_minutes):
            if rng.random() < 0.22:
                template = rng.choice(chatter)
                self.logs.append(
                    LogEvent(
                        id=self.new_id("log"),
                        minute=minute,
                        service=svc.name,
                        level="INFO",
                        message=template.format(n=rng.randint(2, 900)),
                    )
                )
            # A low rate of unrelated warnings; without them, "any WARN is the cause" works.
            if rng.random() < 0.03:
                self.logs.append(
                    LogEvent(
                        id=self.new_id("log"),
                        minute=minute,
                        service=svc.name,
                        level="WARN",
                        message=rng.choice(
                            [
                                "slow query detected (retried, succeeded)",
                                "client disconnected before response",
                                "retrying idempotent request after transient 503",
                            ]
                        ),
                    )
                )

    def _seed_routine_deploys(self, rng: random.Random) -> None:
        """Innocent deploys. A deploy near the onset must not be automatically damning."""
        for svc in rng.sample(list(TOPOLOGY), k=4):
            minute = rng.randint(2, self.window_minutes - 5)
            self.deploys.append(
                DeployEvent(
                    id=self.new_id("dep"),
                    minute=minute,
                    service=svc.name,
                    kind=rng.choice(["release", "config"]),
                    version=f"v{rng.randint(2, 9)}.{rng.randint(0, 20)}.{rng.randint(0, 9)}",
                    summary=rng.choice(
                        [
                            "bump logging library, no behaviour change",
                            "add dashboard annotation",
                            "raise readiness probe timeout",
                            "documentation and comment updates",
                        ]
                    ),
                    actor=rng.choice(["ci-bot", "j.okafor", "m.lindqvist", "s.raman"]),
                )
            )

    def _seed_traces(self, rng: random.Random, count: int) -> None:
        entry_points = ["edge-gateway", "api-gateway"]
        for _ in range(count):
            root = rng.choice(entry_points)
            minute = rng.randint(0, self.window_minutes - 1)
            self.traces.append(self._make_trace(root, minute, rng))

    def _make_trace(self, root: str, minute: int, rng: random.Random) -> Trace:
        """Walk one random path down the graph, sampling each hop's current latency."""
        spans: list[TraceSpan] = []
        node = root
        while True:
            p99 = self.metrics[(node, Metric.LATENCY_P99)][minute]
            err = self.metrics[(node, Metric.ERROR_RATE)][minute]
            failed = rng.random() < min(err * 4, 0.9)
            spans.append(
                TraceSpan(
                    service=node,
                    duration_ms=int(max(1, normal_jitter(rng, p99 * 0.5, p99 * 0.15))),
                    status="error" if failed else "ok",
                )
            )
            deps = SERVICES[node].depends_on
            if not deps:
                break
            node = rng.choice(deps)
        return Trace(
            id=self.new_id("trc"),
            minute=minute,
            root_service=root,
            status="error" if any(s.status != "ok" for s in spans) else "ok",
            spans=spans,
        )

    # -- queries (these back the MCP tools) ---------------------------------------------

    def query_metrics(
        self, service: str, metric: str, start_minute: int = 0, end_minute: int | None = None
    ) -> dict[str, Any]:
        end = self.window_minutes if end_minute is None else min(end_minute, self.window_minutes)
        start = max(0, start_minute)
        series = self.metrics.get((service, metric))
        if series is None:
            raise KeyError(f"no metric {metric!r} for service {service!r}")
        window = series[start:end]
        svc = SERVICES[service]
        return {
            "service": service,
            "metric": metric,
            "start": iso(start),
            "end": iso(end),
            "points": [{"ts": iso(start + i), "value": round(v, 3)} for i, v in enumerate(window)],
            "summary": {
                "min": round(min(window), 3) if window else None,
                "max": round(max(window), 3) if window else None,
                "mean": round(sum(window) / len(window), 3) if window else None,
                "baseline_mean": round(sum(series[:20]) / 20, 3),
            },
            "slo": {
                "p99_ms": svc.slo_p99_ms,
                "error_rate": svc.slo_error_rate,
            },
        }

    def query_logs(
        self,
        service: str | None = None,
        pattern: str | None = None,
        level: str | None = None,
        start_minute: int = 0,
        end_minute: int | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        end = self.window_minutes if end_minute is None else end_minute
        needle = pattern.lower() if pattern else None
        hits = [
            log
            for log in self.logs
            if start_minute <= log.minute < end
            and (service is None or log.service == service)
            and (level is None or log.level == level.upper())
            and (needle is None or needle in log.message.lower())
        ]
        hits.sort(key=lambda log: (log.minute, log.id))
        return [log.as_dict() for log in hits[:limit]]

    def list_deploys(
        self, service: str | None = None, start_minute: int = 0, end_minute: int | None = None
    ) -> list[dict[str, Any]]:
        end = self.window_minutes if end_minute is None else end_minute
        hits = [
            d
            for d in self.deploys
            if start_minute <= d.minute < end and (service is None or d.service == service)
        ]
        hits.sort(key=lambda d: (d.minute, d.id))
        return [d.as_dict() for d in hits]

    def get_trace(self, trace_id: str) -> dict[str, Any] | None:
        for t in self.traces:
            if t.id == trace_id:
                return t.as_dict()
        return None

    def sample_traces(
        self, service: str | None = None, status: str | None = None, limit: int = 5
    ) -> list[dict[str, Any]]:
        hits = [
            t
            for t in self.traces
            if (service is None or any(s.service == service for s in t.spans))
            and (status is None or t.status == status)
        ]
        hits.sort(key=lambda t: t.id)
        return [t.as_dict() for t in hits[:limit]]

    def dependency_graph(self, service: str, depth: int = 2) -> dict[str, Any]:
        svc = SERVICES[service]
        return {
            "service": service,
            "tier": svc.tier.value,
            "owner": svc.owner,
            "depends_on": list(svc.depends_on),
            "called_by": callers_of(service),
            "transitive_dependencies": downstream_of(service, depth),
            "slo": {"p99_ms": svc.slo_p99_ms, "error_rate": svc.slo_error_rate},
        }

    # -- helpers used by fault injection and validation ---------------------------------

    def scale_metric(
        self, service: str, metric: str, from_minute: int, factor: float, ramp: int = 0
    ) -> None:
        """Multiply a metric from ``from_minute`` onward, optionally ramping in over ``ramp``."""
        series = self.metrics[(service, metric)]
        for i in range(from_minute, self.window_minutes):
            progress = 1.0 if ramp <= 0 else min(1.0, (i - from_minute + 1) / ramp)
            series[i] = round(series[i] * (1 + (factor - 1) * progress), 4)

    def set_metric_floor(
        self, service: str, metric: str, from_minute: int, value: float, ramp: int = 0
    ) -> None:
        """Drive a metric up to at least ``value`` from ``from_minute`` onward."""
        series = self.metrics[(service, metric)]
        for i in range(from_minute, self.window_minutes):
            progress = 1.0 if ramp <= 0 else min(1.0, (i - from_minute + 1) / ramp)
            target = series[i] + (value - series[i]) * progress
            series[i] = round(max(series[i], target), 4)

    def add_log(self, minute: int, service: str, level: str, message: str) -> str:
        event = LogEvent(
            id=self.new_id("log"), minute=minute, service=service, level=level, message=message
        )
        self.logs.append(event)
        return event.id

    def remove_deploys(self, service: str, start_minute: int, end_minute: int) -> None:
        """Drop routine deploys in a window.

        Used before injecting a non-deploy fault: a coincidental release next to the onset would
        make the scenario genuinely ambiguous rather than merely hard.
        """
        self.deploys = [
            d
            for d in self.deploys
            if not (d.service == service and start_minute <= d.minute <= end_minute)
        ]

    def add_deploy(
        self, minute: int, service: str, kind: str, version: str, summary: str, actor: str
    ) -> str:
        event = DeployEvent(
            id=self.new_id("dep"),
            minute=minute,
            service=service,
            kind=kind,
            version=version,
            summary=summary,
            actor=actor,
        )
        self.deploys.append(event)
        return event.id

    def add_failing_traces(
        self, rng: random.Random, culprit: str, minute: int, count: int, error: str
    ) -> list[str]:
        """Emit traces whose failing span is ``culprit`` — the smoking gun in trace data."""
        ids: list[str] = []
        # upstream_path is nearest-caller-first; reversing walks edge -> ... -> culprit, which is
        # the order a real trace records.
        path = [*reversed(upstream_path(culprit)), culprit]
        for _ in range(count):
            spans = [
                TraceSpan(
                    service=hop,
                    duration_ms=int(self.metrics[(hop, Metric.LATENCY_P99)][minute] * 0.8),
                    status="ok",
                )
                for hop in path[:-1]
            ]
            spans.append(
                TraceSpan(
                    service=culprit,
                    duration_ms=int(self.metrics[(culprit, Metric.LATENCY_P99)][minute]),
                    status="error",
                    error=error,
                )
            )
            trace = Trace(
                id=self.new_id("trc"),
                minute=minute + rng.randint(0, 3),
                root_service=path[0] if path else culprit,
                status="error",
                spans=spans,
            )
            self.traces.append(trace)
            ids.append(trace.id)
        return ids
