"""The model client.

Wraps the Messages API with the behaviour a harness has to own:

* **Always streams.** Agents run with large ``max_tokens``; a non-streaming request at that size
  risks an HTTP timeout, and streaming is also what lets the dashboard show progress live.
* **Own retry loop.** The SDK's automatic retries are switched off (``max_retries=0``) so the
  policy is explicit, logged, and testable — see ``harness.errors.classify``.
* **Normalises content to plain dicts.** Assistant turns are appended back to the message list
  verbatim (thinking blocks, signatures and all) *and* checkpointed to SQLite, so a single
  JSON-safe representation is used for both.

``ModelClient`` is a Protocol. The offline mock in ``evals/mock_model.py`` implements the same
interface, which is what makes the whole pipeline testable with no API key and no spend.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import anthropic

from aegis.config import RetryConfig
from aegis.harness.errors import ModelRefusal, classify
from aegis.logging import get_logger
from aegis.types import Usage

log = get_logger(__name__)

# The API rejects a task budget below this.
MIN_TASK_BUDGET_TOKENS = 20_000


@dataclass
class ToolUse:
    """A tool call the model asked for."""

    id: str
    name: str
    input: dict[str, Any]


@dataclass
class ModelRequest:
    model: str
    system_blocks: list[dict[str, Any]]
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] = field(default_factory=list)
    max_tokens: int = 16_000
    effort: str = "high"
    task_budget_tokens: int | None = None
    output_schema: dict[str, Any] | None = None


@dataclass
class ModelResponse:
    content: list[dict[str, Any]]
    stop_reason: str | None
    usage: Usage
    model: str
    stop_details: dict[str, Any] | None = None

    @property
    def text(self) -> str:
        return "".join(b.get("text", "") for b in self.content if b.get("type") == "text")

    @property
    def tool_uses(self) -> list[ToolUse]:
        return [
            ToolUse(id=b["id"], name=b["name"], input=b.get("input") or {})
            for b in self.content
            if b.get("type") == "tool_use"
        ]


DeltaCallback = Callable[[str], None]


class ModelClient(Protocol):
    """What the loop needs from a model. Implemented by the real client and by the mock."""

    async def complete(
        self,
        request: ModelRequest,
        on_text: DeltaCallback | None = None,
        on_thinking: DeltaCallback | None = None,
    ) -> ModelResponse: ...


def _block_to_dict(block: Any) -> dict[str, Any]:
    """Normalise an SDK content block to a JSON-safe dict.

    Assistant content is echoed back on the next request, so this must be lossless — in
    particular a thinking block's ``signature`` is validated server-side and cannot be dropped
    or altered.
    """
    if isinstance(block, dict):
        return block
    dumped = block.model_dump(mode="json", exclude_none=True)
    return dict(dumped)


class AnthropicModelClient:
    """The real client."""

    def __init__(self, api_key: str, retry: RetryConfig, rng: random.Random | None = None) -> None:
        self._client = anthropic.AsyncAnthropic(
            api_key=api_key or None,
            timeout=retry.request_timeout_s,
            # We own retries; see harness.errors.classify.
            max_retries=0,
        )
        self._retry = retry
        self._rng = rng or random.Random()

    async def aclose(self) -> None:
        await self._client.close()

    # -- request construction -----------------------------------------------------------

    def _build_kwargs(self, request: ModelRequest) -> tuple[dict[str, Any], bool]:
        """Return (kwargs, use_beta_endpoint)."""
        output_config: dict[str, Any] = {"effort": request.effort}

        use_beta = False
        if request.task_budget_tokens:
            # A model-visible allowance it paces itself against — distinct from max_tokens,
            # which is an enforced ceiling the model never sees.
            output_config["task_budget"] = {
                "type": "tokens",
                "total": max(MIN_TASK_BUDGET_TOKENS, request.task_budget_tokens),
            }
            use_beta = True

        if request.output_schema:
            output_config["format"] = request.output_schema

        kwargs: dict[str, Any] = {
            "model": request.model,
            "max_tokens": request.max_tokens,
            "system": request.system_blocks,
            "messages": request.messages,
            # Must be set explicitly: on Opus 4.8 an omitted `thinking` field means no thinking
            # at all. `summarized` is what makes reasoning renderable in the dashboard.
            "thinking": {"type": "adaptive", "display": "summarized"},
            "output_config": output_config,
        }
        if request.tools:
            kwargs["tools"] = request.tools
        if use_beta:
            kwargs["betas"] = ["task-budgets-2026-03-13"]
        return kwargs, use_beta

    # -- the call -----------------------------------------------------------------------

    async def complete(
        self,
        request: ModelRequest,
        on_text: DeltaCallback | None = None,
        on_thinking: DeltaCallback | None = None,
    ) -> ModelResponse:
        kwargs, use_beta = self._build_kwargs(request)
        attempt = 0
        last_exc: BaseException | None = None

        while True:
            attempt += 1
            try:
                return await self._stream_once(kwargs, use_beta, on_text, on_thinking)
            except Exception as exc:
                last_exc = exc
                decision = classify(
                    exc,
                    attempt=attempt,
                    max_attempts=self._retry.max_attempts,
                    base_delay_s=self._retry.base_delay_s,
                    max_delay_s=self._retry.max_delay_s,
                    rng=self._rng,
                )
                if not decision.retry:
                    log.warning(
                        "model call failed permanently",
                        extra={
                            "error": type(exc).__name__,
                            "attempt": attempt,
                            "reason": decision.reason,
                        },
                    )
                    raise
                log.info(
                    "retrying model call",
                    extra={
                        "error": type(exc).__name__,
                        "attempt": attempt,
                        "delay_s": round(decision.delay_s, 2),
                        "reason": decision.reason,
                    },
                )
                await asyncio.sleep(decision.delay_s)

        raise last_exc  # pragma: no cover - loop only exits via return or raise

    async def _stream_once(
        self,
        kwargs: dict[str, Any],
        use_beta: bool,
        on_text: DeltaCallback | None,
        on_thinking: DeltaCallback | None,
    ) -> ModelResponse:
        endpoint = self._client.beta.messages if use_beta else self._client.messages

        async with endpoint.stream(**kwargs) as stream:
            async for event in stream:
                if event.type != "content_block_delta":
                    continue
                delta = event.delta
                if delta.type == "text_delta" and on_text:
                    on_text(delta.text)
                elif delta.type == "thinking_delta" and on_thinking:
                    on_thinking(delta.thinking)
            message = await stream.get_final_message()

        stop_details = getattr(message, "stop_details", None)
        details = _block_to_dict(stop_details) if stop_details is not None else None

        # Check the stop reason before touching content: on a refusal, content may be empty.
        if message.stop_reason == "refusal":
            raise ModelRefusal(
                category=(details or {}).get("category"),
                explanation=(details or {}).get("explanation"),
            )

        return ModelResponse(
            content=[_block_to_dict(b) for b in message.content],
            stop_reason=message.stop_reason,
            usage=Usage.from_api(message.usage),
            model=message.model,
            stop_details=details,
        )
