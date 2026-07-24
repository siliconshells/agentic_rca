"""Error taxonomy and retry classification.

The SDK retries 429s and 5xx on its own. We turn that off and do it here instead, because a
harness needs to make the decision *visibly*: which failures are retryable, how long to wait, and
when to stop is exactly the behaviour a reviewer should be able to read and a test should be able
to pin down.

Classification is by exception type, never by string-matching a message.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import anthropic


class AegisError(Exception):
    """Base for every error the harness raises deliberately."""


class BudgetExceeded(AegisError):
    """The run hit its cost or token ceiling. Carries partial results, never a bare failure."""

    def __init__(self, spent_usd: float, limit_usd: float) -> None:
        super().__init__(
            f"budget exceeded: ${spent_usd:.4f} spent against a ${limit_usd:.2f} limit"
        )
        self.spent_usd = spent_usd
        self.limit_usd = limit_usd


class DeadlineExceeded(AegisError):
    """The run exceeded its wall-clock allowance."""


class ApprovalTimeout(AegisError):
    """Nobody answered an approval request in time."""


class ToolDenied(AegisError):
    """An operator denied a tool call, or policy forbade it outright."""

    def __init__(self, tool: str, reason: str) -> None:
        super().__init__(f"tool {tool!r} denied: {reason}")
        self.tool = tool
        self.reason = reason


class ModelRefusal(AegisError):
    """The model declined the request (``stop_reason == "refusal"``)."""

    def __init__(self, category: str | None, explanation: str | None) -> None:
        detail = f"model refused ({category or 'unspecified'})"
        super().__init__(f"{detail}: {explanation}" if explanation else detail)
        self.category = category
        self.explanation = explanation


class MaxTurnsExceeded(AegisError):
    """An agent hit its turn ceiling without finishing. Almost always a loop, not bad luck."""


@dataclass(frozen=True)
class RetryDecision:
    retry: bool
    delay_s: float = 0.0
    reason: str = ""


# Retryable: transient server-side or transport conditions. Everything else is a bug in our
# request and retrying it just burns budget and latency to fail identically.
_RETRYABLE: tuple[type[Exception], ...] = (
    anthropic.RateLimitError,
    anthropic.InternalServerError,
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
)

# Never retryable: the request itself is wrong, or we are not allowed to make it.
_FATAL: tuple[type[Exception], ...] = (
    anthropic.BadRequestError,
    anthropic.AuthenticationError,
    anthropic.PermissionDeniedError,
    anthropic.NotFoundError,
    anthropic.UnprocessableEntityError,
)


def retry_after_seconds(exc: BaseException) -> float | None:
    """Honour a server-supplied ``retry-after`` when there is one."""
    response = getattr(exc, "response", None)
    header = getattr(response, "headers", {}) or {}
    raw = header.get("retry-after") if hasattr(header, "get") else None
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return None


def classify(
    exc: BaseException,
    attempt: int,
    max_attempts: int,
    base_delay_s: float,
    max_delay_s: float,
    rng: random.Random | None = None,
) -> RetryDecision:
    """Decide whether to retry, and how long to wait first.

    Exponential backoff with full jitter — jitter matters because the coordinator fans out
    investigators simultaneously, so a shared rate limit would otherwise make them all retry in
    lockstep and collide again.
    """
    if isinstance(exc, AegisError):
        return RetryDecision(False, reason="harness error, not transient")
    if isinstance(exc, _FATAL):
        return RetryDecision(False, reason=f"{type(exc).__name__} is a client-side error")
    if not isinstance(exc, _RETRYABLE):
        # Includes anthropic.APIStatusError subclasses we have not named. Retry 5xx, not 4xx.
        status = getattr(exc, "status_code", None)
        if not (isinstance(status, int) and status >= 500):
            return RetryDecision(False, reason=f"{type(exc).__name__} is not retryable")

    if attempt >= max_attempts:
        return RetryDecision(False, reason=f"exhausted {max_attempts} attempts")

    server_delay = retry_after_seconds(exc)
    if server_delay is not None:
        return RetryDecision(True, min(server_delay, max_delay_s), "honouring retry-after")

    rng = rng or random.Random()
    ceiling = min(base_delay_s * (2 ** (attempt - 1)), max_delay_s)
    return RetryDecision(True, rng.uniform(0, ceiling), f"backoff attempt {attempt}")
