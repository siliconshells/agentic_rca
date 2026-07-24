"""MCP server contract tests.

Run over the in-memory transport, so they exercise the real protocol path — schema generation,
serialisation, error mapping — without a subprocess, a port, or an API key.

The leak tests matter most: ground truth must never be reachable through any tool, resource, or
prompt. If it were, every accuracy number the eval suite reports would be meaningless.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest
from mcp import ClientSession
from mcp.shared.memory import create_connected_server_and_client_session

from mcp_server.env.faults import make_scenario
from mcp_server.server import mcp, register_scenario

SCENARIO_ID = "test-inc-0001"

SCENARIO, _WORLD = make_scenario(
    SCENARIO_ID, seed=4242, fault_class="bad_deploy", culprit="payments-svc", onset=50
)


@pytest.fixture(autouse=True)
def _register_scenario():
    """Re-register before every test.

    The MCP server holds its scenario registry as module state, and other test modules (the API
    tests via the app lifespan) call ``load_scenarios``, which clears it. Registering per-test
    keeps these tests independent of suite ordering.
    """
    register_scenario(SCENARIO)


@asynccontextmanager
async def mcp_session() -> AsyncIterator[ClientSession]:
    """One connected session, entered and exited inside a single task.

    Deliberately not a pytest fixture: pytest-asyncio sets up and tears down async-generator
    fixtures in *different* tasks, and anyio cancel scopes must be exited in the task that
    entered them. An explicit `async with` per test sidesteps that entirely and keeps each
    test independent.
    """
    async with create_connected_server_and_client_session(mcp) as s:
        yield s


def payload(result: Any) -> Any:
    """Unwrap a CallToolResult into plain data."""
    assert not result.isError, f"tool errored: {result.content}"
    if getattr(result, "structuredContent", None):
        return result.structuredContent
    text = result.content[0].text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


# --------------------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------------------


async def test_all_three_primitives_are_advertised():
    async with mcp_session() as session:
        tools = {t.name for t in (await session.list_tools()).tools}
        prompts = {p.name for p in (await session.list_prompts()).prompts}
        templates = {
            str(t.uriTemplate) for t in (await session.list_resource_templates()).resourceTemplates
        }
        resources = {str(r.uri) for r in (await session.list_resources()).resources}

        assert tools == {
            "query_metrics",
            "query_logs",
            "list_deploys",
            "sample_traces",
            "get_trace",
            "get_dependency_graph",
            "search_similar_incidents",
            "propose_remediation",
        }
        assert prompts == {"triage_incident", "write_postmortem", "severity_review"}
        assert "catalog://services" in resources
        assert {
            "runbook://{service}",
            "slo://{service}",
            "postmortem://{postmortem_id}",
        } <= templates


async def test_tool_descriptions_say_when_to_call():
    """Descriptions drive tool selection, so each must state a trigger condition."""
    async with mcp_session() as session:
        for tool in (await session.list_tools()).tools:
            assert tool.description, f"{tool.name} has no description"
            assert "Call this" in tool.description or "Use after" in tool.description, (
                f"{tool.name} description does not say when to call it"
            )


# --------------------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------------------


async def test_query_metrics_returns_series_with_slo_and_baseline():
    async with mcp_session() as session:
        data = payload(
            await session.call_tool(
                "query_metrics",
                {"incident_id": SCENARIO_ID, "service": "payments-svc", "metric": "error_rate"},
            )
        )
        assert data["service"] == "payments-svc"
        assert len(data["points"]) == 90
        assert data["slo"]["error_rate"] == 0.005
        # The injected fault must be visible against the pre-incident baseline.
        assert data["summary"]["max"] > data["summary"]["baseline_mean"] * 5


async def test_query_logs_returns_citable_ids():
    async with mcp_session() as session:
        data = payload(
            await session.call_tool(
                "query_logs",
                {"incident_id": SCENARIO_ID, "service": "payments-svc", "level": "ERROR"},
            )
        )
        assert data["count"] > 0
        assert all(e["id"].startswith("log-") for e in data["entries"])
        assert any("NullPointerException" in e["message"] for e in data["entries"])


async def test_list_deploys_surfaces_the_release_at_onset():
    async with mcp_session() as session:
        data = payload(
            await session.call_tool(
                "list_deploys", {"incident_id": SCENARIO_ID, "service": "payments-svc"}
            )
        )
        releases = [d for d in data["deploys"] if d["kind"] == "release"]
        assert releases, "bad_deploy scenario must expose its release"


async def test_traces_expose_the_failing_span():
    async with mcp_session() as session:
        sampled = payload(
            await session.call_tool(
                "sample_traces", {"incident_id": SCENARIO_ID, "status": "error", "limit": 5}
            )
        )
        assert sampled["count"] > 0
        trace = payload(
            await session.call_tool(
                "get_trace", {"incident_id": SCENARIO_ID, "trace_id": sampled["traces"][0]["id"]}
            )
        )
        assert any(s["status"] != "ok" for s in trace["spans"])


async def test_dependency_graph_is_incident_independent():
    async with mcp_session() as session:
        data = payload(await session.call_tool("get_dependency_graph", {"service": "checkout-api"}))
        assert set(data["depends_on"]) == {"payments-svc", "inventory-svc", "cart-svc"}
        assert data["called_by"] == ["api-gateway"]


async def test_similar_incident_search_ranks_by_relevance():
    async with mcp_session() as session:
        data = payload(
            await session.call_tool(
                "search_similar_incidents",
                {"query": "payments-svc connection pool saturation latency"},
            )
        )
        assert data["count"] > 0
        scores = [r["relevance"] for r in data["results"]]
        assert scores == sorted(scores, reverse=True)


async def test_propose_remediation_applies_nothing():
    """The tool records intent only; the harness — not the server — gates execution."""
    async with mcp_session() as session:
        data = payload(
            await session.call_tool(
                "propose_remediation",
                {
                    "incident_id": SCENARIO_ID,
                    "action": "rollback release",
                    "target": "payments-svc",
                    "rationale": "release introduced an unhandled exception",
                    "blast_radius": "payments traffic for ~2 minutes",
                    "reversible": True,
                },
            )
        )
        assert data["status"] == "recorded"
        assert "No change has been applied" in data["note"]


# --------------------------------------------------------------------------------------
# Error handling — agents must be able to recover from a bad call
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tool", "args", "expect"),
    [
        (
            "query_metrics",
            {"incident_id": SCENARIO_ID, "service": "nope-svc", "metric": "error_rate"},
            "Valid services",
        ),
        (
            "query_metrics",
            {"incident_id": SCENARIO_ID, "service": "payments-svc", "metric": "vibes"},
            "Valid metrics",
        ),
        (
            "query_logs",
            {"incident_id": "inc-does-not-exist"},
            "unknown incident_id",
        ),
        (
            "get_trace",
            {"incident_id": SCENARIO_ID, "trace_id": "trc-9999"},
            "no trace",
        ),
    ],
)
async def test_errors_are_actionable(tool, args, expect):
    async with mcp_session() as session:
        result = await session.call_tool(tool, args)
        assert result.isError
        assert expect in result.content[0].text


async def test_bad_timestamps_explain_the_expected_format():
    async with mcp_session() as session:
        result = await session.call_tool(
            "query_metrics",
            {
                "incident_id": SCENARIO_ID,
                "service": "payments-svc",
                "metric": "error_rate",
                "start_ts": "last tuesday",
            },
        )
        assert result.isError
        assert "ISO-8601" in result.content[0].text


# --------------------------------------------------------------------------------------
# Resources
# --------------------------------------------------------------------------------------


async def test_catalog_lists_every_service():
    async with mcp_session() as session:
        text = (await session.read_resource("catalog://services")).contents[0].text
        for name in ("edge-gateway", "payments-svc", "session-cache"):
            assert name in text


async def test_runbook_is_service_specific_and_ordered():
    async with mcp_session() as session:
        text = (await session.read_resource("runbook://payments-svc")).contents[0].text
        assert "# Runbook: payments-svc" in text
        assert "1." in text and "2." in text
        assert "untrusted data" in text


async def test_slo_resource_is_json():
    async with mcp_session() as session:
        data = json.loads((await session.read_resource("slo://checkout-api")).contents[0].text)
        assert data["p99_latency_ms"] == 800
        assert "error_budget_policy" in data


async def test_postmortem_resource_renders_markdown():
    async with mcp_session() as session:
        text = (await session.read_resource("postmortem://pm-2025-114")).contents[0].text
        assert text.startswith("# pm-2025-114")
        assert "## Lesson" in text


# --------------------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------------------


async def test_triage_prompt_embeds_the_alert():
    async with mcp_session() as session:
        result = await session.get_prompt("triage_incident", {"incident_id": SCENARIO_ID})
        text = result.messages[0].content.text
        assert SCENARIO.alert in text
        assert SCENARIO_ID in text


@pytest.mark.parametrize("name", ["triage_incident", "write_postmortem", "severity_review"])
async def test_every_prompt_renders(name):
    async with mcp_session() as session:
        result = await session.get_prompt(name, {"incident_id": SCENARIO_ID})
        assert result.messages
        assert result.messages[0].content.text.strip()


# --------------------------------------------------------------------------------------
# Ground-truth containment — the load-bearing invariant
# --------------------------------------------------------------------------------------


async def test_ground_truth_never_leaks_through_the_server():
    """No tool, resource, or prompt may reveal the answer.

    The scenario's culprit is `payments-svc` and its class is `bad_deploy`; both legitimately
    appear in telemetry. What must never appear is the ground-truth *record* — the notes field,
    the causal event list, or the class label presented as fact.
    """
    truth = SCENARIO.ground_truth
    forbidden = [truth.notes, "causal_event_ids", "ground_truth", "symptom_service"]

    async with mcp_session() as session:
        surfaces: list[str] = []
        for name in ("triage_incident", "write_postmortem", "severity_review"):
            result = await session.get_prompt(name, {"incident_id": SCENARIO_ID})
            surfaces.extend(m.content.text for m in result.messages)

        for uri in ("catalog://services", "runbook://payments-svc", "slo://payments-svc"):
            surfaces.append((await session.read_resource(uri)).contents[0].text)

        surfaces.append(
            json.dumps(
                payload(
                    await session.call_tool(
                        "query_logs", {"incident_id": SCENARIO_ID, "limit": 200}
                    )
                )
            )
        )

    for surface in surfaces:
        for needle in forbidden:
            assert needle not in surface, f"ground truth leaked: {needle!r}"
