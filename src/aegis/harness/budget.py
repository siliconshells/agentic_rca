"""Token and cost accounting with a hard ceiling.

A multi-agent run can spend real money in a way a single call cannot: a coordinator that loops,
six investigators that each loop, and a verifier per surviving finding. The ceiling is what makes
that safe to run unattended, and per-agent attribution is what makes it debuggable afterwards.

The ledger is shared by every agent in a run and is the only place cost is computed.
"""

from __future__ import annotations

import threading
import time

from aegis.config import BudgetConfig, pricing_for
from aegis.harness.errors import BudgetExceeded, DeadlineExceeded
from aegis.types import CostReport, Usage


def cost_of(usage: Usage, model: str) -> float:
    """USD for one call's usage.

    Cache writes bill above base input and reads far below it, so folding them into a single
    "input tokens" number would misreport a cache-heavy run by a wide margin.
    """
    price = pricing_for(model)
    input_rate = price.input_per_mtok / 1_000_000
    return (
        usage.input_tokens * input_rate
        + usage.cache_creation_input_tokens * input_rate * price.CACHE_WRITE_MULTIPLIER
        + usage.cache_read_input_tokens * input_rate * price.CACHE_READ_MULTIPLIER
        + usage.output_tokens * price.output_per_mtok / 1_000_000
    )


class Budget:
    """Run-scoped ledger. Thread- and task-safe; investigators charge it concurrently."""

    # A pessimistic seed for the first few reservations, before any real call cost is observed.
    _SEED_ESTIMATE_USD = 0.05

    def __init__(self, config: BudgetConfig, started_at: float | None = None) -> None:
        self.config = config
        self.started_at = started_at if started_at is not None else time.monotonic()
        self._lock = threading.Lock()
        self._total = CostReport()
        self._by_agent: dict[str, CostReport] = {}
        self._warned = False
        # Reservation accounting. `_reserved` is budget claimed by in-flight calls that have not
        # charged yet; `_max_observed` is the largest single call cost seen, used as the estimate
        # for the next reservation. Together they close the check-then-charge race: a turn cannot
        # start if committed + reserved would exceed the ceiling, so concurrent agents cannot each
        # pass the check at the limit and collectively overshoot.
        self._reserved = 0.0
        self._max_observed = self._SEED_ESTIMATE_USD

    # -- recording ----------------------------------------------------------------------

    def charge(self, agent_id: str, usage: Usage, model: str, reservation: float = 0.0) -> float:
        """Record one model call, releasing any reservation held for it. Returns its cost."""
        amount = cost_of(usage, model)
        entry = CostReport(usage=usage, cost_usd=amount, calls=1)
        with self._lock:
            self._reserved = max(0.0, self._reserved - reservation)
            self._max_observed = max(self._max_observed, amount)
            self._total = self._total + entry
            self._by_agent[agent_id] = self._by_agent.get(agent_id, CostReport()) + entry
        return amount

    def add_prior_spend(self, agent_id: str, usage: Usage, model: str) -> float:
        """Fold restored, already-incurred spend into the ledger on resume.

        Unlike :meth:`charge`, this does not touch the per-call estimate — restored ``usage`` is
        cumulative across many prior turns, so treating it as one call's cost would blow up the
        reservation estimate and falsely starve subsequent calls.
        """
        amount = cost_of(usage, model)
        entry = CostReport(usage=usage, cost_usd=amount, calls=0)
        with self._lock:
            self._total = self._total + entry
            self._by_agent[agent_id] = self._by_agent.get(agent_id, CostReport()) + entry
        return amount

    # -- reporting ----------------------------------------------------------------------

    @property
    def spent_usd(self) -> float:
        with self._lock:
            return self._total.cost_usd

    @property
    def total(self) -> CostReport:
        with self._lock:
            return self._total.model_copy(deep=True)

    def by_agent(self) -> dict[str, CostReport]:
        with self._lock:
            return {k: v.model_copy(deep=True) for k, v in self._by_agent.items()}

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.config.max_cost_usd - self.spent_usd)

    @property
    def fraction_used(self) -> float:
        limit = self.config.max_cost_usd
        return self.spent_usd / limit if limit > 0 else 0.0

    @property
    def elapsed_s(self) -> float:
        return time.monotonic() - self.started_at

    def should_warn(self) -> bool:
        """True exactly once, the first time spend crosses the warning threshold."""
        with self._lock:
            if self._warned or self.config.warn_fraction <= 0:
                return False
            limit = self.config.max_cost_usd
            if limit > 0 and self._total.cost_usd / limit >= self.config.warn_fraction:
                self._warned = True
                return True
            return False

    # -- enforcement --------------------------------------------------------------------

    def check(self) -> None:
        """Raise if the run is out of money or time. Non-reserving.

        Used for quick affordability probes. The turn loop uses :meth:`reserve` instead, which
        additionally holds budget for the in-flight call.
        """
        with self._lock:
            self._raise_if_exhausted_locked()

    def reserve(self) -> float:
        """Claim budget for an about-to-start call. Raises if the run is out of money or time.

        Returns the reserved amount, which the matching :meth:`charge` releases. A call is not
        allowed to start if committed spend plus outstanding reservations plus this call's estimate
        would exceed the ceiling — so N concurrent agents cannot all start at the limit.
        """
        with self._lock:
            self._raise_if_exhausted_locked()
            estimate = self._max_observed
            # Starting this call could push the run over. One call is always allowed to proceed
            # when nothing is reserved (so a single large call can't deadlock the run); otherwise
            # hold the line so concurrent agents can't all start at the limit.
            would_overshoot = (
                self._total.cost_usd + self._reserved + estimate > self.config.max_cost_usd
            )
            if would_overshoot and self._reserved > 0.0:
                raise BudgetExceeded(
                    self._total.cost_usd + self._reserved, self.config.max_cost_usd
                )
            self._reserved += estimate
            return estimate

    def release(self, reservation: float) -> None:
        """Return a reservation without charging, e.g. when a call fails before spending."""
        with self._lock:
            self._reserved = max(0.0, self._reserved - reservation)

    def _raise_if_exhausted_locked(self) -> None:
        if self._total.cost_usd >= self.config.max_cost_usd:
            raise BudgetExceeded(self._total.cost_usd, self.config.max_cost_usd)
        if self.elapsed_s >= self.config.max_wall_clock_s:
            raise DeadlineExceeded(
                f"wall clock exceeded: {self.elapsed_s:.0f}s of "
                f"{self.config.max_wall_clock_s:.0f}s allowed"
            )

    def affordable(self) -> bool:
        """Non-raising variant, for deciding whether to start optional work."""
        try:
            self.check()
        except (BudgetExceeded, DeadlineExceeded):
            return False
        return True

    def summary(self) -> dict[str, object]:
        total = self.total
        return {
            "cost_usd": round(total.cost_usd, 6),
            "limit_usd": self.config.max_cost_usd,
            "calls": total.calls,
            "input_tokens": total.usage.input_tokens,
            "output_tokens": total.usage.output_tokens,
            "cache_read_tokens": total.usage.cache_read_input_tokens,
            "cache_creation_tokens": total.usage.cache_creation_input_tokens,
            "cache_hit_rate": round(total.usage.cache_hit_rate, 4),
            "elapsed_s": round(self.elapsed_s, 2),
        }
