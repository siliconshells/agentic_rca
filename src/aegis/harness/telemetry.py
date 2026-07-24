"""OpenTelemetry tracing.

Produces the span tree ``run -> agent -> turn -> {model_request, tool_call}``, so a fan-out is
legible as a shape: parallel investigators appear as overlapping sibling spans, and a serialised
"parallel" section is immediately visible as a staircase.

Attributes follow the ``gen_ai.*`` semantic conventions where they exist, plus ``aegis.*`` for
what they do not cover (USD cost, cache token counts).

Export is optional. With no collector configured the tracer is a no-op, so the harness runs
identically in CI, in tests, and on a laptop with nothing else installed.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Span, StatusCode

from aegis.types import Usage

_configured = False


def configure(endpoint: str = "", service_name: str = "aegis") -> None:
    """Install the tracer provider. Idempotent; a blank endpoint disables export."""
    global _configured
    if _configured:
        return
    _configured = True

    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    if endpoint:
        # Imported lazily so the exporter's transitive deps are not required to run offline.
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    trace.set_tracer_provider(provider)


def tracer() -> trace.Tracer:
    return trace.get_tracer("aegis")


def shutdown() -> None:
    """Flush pending spans. Worth calling before a short-lived CLI process exits."""
    provider = trace.get_tracer_provider()
    if isinstance(provider, TracerProvider):
        provider.shutdown()


@contextmanager
def run_span(run_id: str, scenario_id: str) -> Iterator[Span]:
    with tracer().start_as_current_span("aegis.run") as span:
        span.set_attribute("aegis.run_id", run_id)
        span.set_attribute("aegis.scenario_id", scenario_id)
        yield span


@contextmanager
def agent_span(agent_id: str, role: str, label: str, model: str) -> Iterator[Span]:
    with tracer().start_as_current_span(f"agent.{role}") as span:
        span.set_attribute("aegis.agent_id", agent_id)
        span.set_attribute("aegis.agent_role", role)
        span.set_attribute("aegis.agent_label", label)
        span.set_attribute("gen_ai.request.model", model)
        yield span


@contextmanager
def turn_span(turn: int) -> Iterator[Span]:
    with tracer().start_as_current_span("agent.turn") as span:
        span.set_attribute("aegis.turn", turn)
        yield span


@contextmanager
def model_span(model: str, effort: str) -> Iterator[Span]:
    with tracer().start_as_current_span("gen_ai.chat") as span:
        span.set_attribute("gen_ai.system", "anthropic")
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.request.model", model)
        span.set_attribute("aegis.effort", effort)
        yield span


@contextmanager
def tool_span(name: str, tool_use_id: str) -> Iterator[Span]:
    with tracer().start_as_current_span(f"tool.{name}") as span:
        span.set_attribute("gen_ai.tool.name", name)
        span.set_attribute("aegis.tool_use_id", tool_use_id)
        yield span


def record_usage(span: Span, usage: Usage, cost_usd: float, stop_reason: str | None) -> None:
    """Attach token, cost, and stop-reason attributes to a model span."""
    span.set_attribute("gen_ai.usage.input_tokens", usage.input_tokens)
    span.set_attribute("gen_ai.usage.output_tokens", usage.output_tokens)
    span.set_attribute("aegis.usage.cache_read_tokens", usage.cache_read_input_tokens)
    span.set_attribute("aegis.usage.cache_creation_tokens", usage.cache_creation_input_tokens)
    span.set_attribute("aegis.usage.cache_hit_rate", round(usage.cache_hit_rate, 4))
    span.set_attribute("aegis.cost_usd", round(cost_usd, 6))
    if stop_reason:
        span.set_attribute("gen_ai.response.finish_reason", stop_reason)


def record_error(span: Span, exc: BaseException) -> None:
    span.set_status(StatusCode.ERROR, str(exc))
    span.record_exception(exc)


def set_attributes(span: Span, **attributes: Any) -> None:
    for key, value in attributes.items():
        if value is not None:
            span.set_attribute(key, value)
