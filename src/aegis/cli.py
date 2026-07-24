"""The Aegis command line.

    aegis run --scenario evals/scenarios.jsonl:3            # against the real API
    aegis run --scenario evals/scenarios.jsonl:3 --mock     # offline, deterministic, no spend
    aegis run --resume run-abc123                           # continue a crashed run
    aegis scenarios --count 60                              # (re)generate the scenario set
    aegis show run-abc123                                   # inspect a finished run

The ``--mock`` path uses an in-process MCP server and the deterministic mock model, so it needs
no API key, no running server, and no network — which is exactly what makes it the CI smoke test.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from aegis.config import Settings, get_settings
from aegis.logging import configure as configure_logging
from aegis.orchestrator import RunConfig, RunResult
from aegis.runner import RunRequest, execute_run
from aegis.store.db import Store
from aegis.types import RunStatus

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="A production agentic harness for incident triage and root-cause analysis.",
)
console = Console()


def _config_from_flags(single_agent: bool, no_verifier: bool, effort: str | None) -> RunConfig:
    if single_agent:
        cfg = RunConfig.single_agent()
    elif no_verifier:
        cfg = RunConfig.no_verifier()
    else:
        cfg = RunConfig()
    cfg.investigator_effort = effort
    return cfg


@app.command()
def run(
    scenario: Annotated[
        str | None,
        typer.Option(help="Scenario reference 'path.jsonl:INDEX' or 'path.jsonl:incident_id'."),
    ] = None,
    resume: Annotated[str | None, typer.Option(help="Resume a run by id.")] = None,
    mock: Annotated[
        bool, typer.Option(help="Use the offline mock model (no API key, no spend).")
    ] = False,
    max_cost: Annotated[
        float | None, typer.Option(help="Override the per-run USD ceiling.")
    ] = None,
    single_agent: Annotated[
        bool, typer.Option(help="Ablation: one coordinator, no fan-out.")
    ] = False,
    no_verifier: Annotated[
        bool, typer.Option(help="Ablation: skip the adversarial verifier.")
    ] = False,
    effort: Annotated[str | None, typer.Option(help="Override investigator effort.")] = None,
) -> None:
    """Triage one incident."""
    configure_logging(get_settings().log_level, get_settings().log_json)
    if not scenario and not resume:
        console.print("[red]provide --scenario or --resume[/red]")
        raise typer.Exit(1)

    result = asyncio.run(
        _run_async(
            scenario_ref=scenario,
            resume_id=resume,
            mock=mock,
            max_cost=max_cost,
            run_config=_config_from_flags(single_agent, no_verifier, effort),
        )
    )
    _render_result(result)
    raise typer.Exit(0 if result.status is RunStatus.SUCCEEDED else 1)


async def _run_async(
    *,
    scenario_ref: str | None,
    resume_id: str | None,
    mock: bool,
    max_cost: float | None,
    run_config: RunConfig,
) -> RunResult:
    from mcp_server.env.generator import load_scenario_ref

    settings = get_settings()
    if max_cost is not None:
        settings.budget.max_cost_usd = max_cost

    store = Store(settings.db_path)

    if resume_id:
        row = store.get_run(resume_id)
        if row is None:
            console.print(f"[red]no run {resume_id!r}[/red]")
            raise typer.Exit(1)
        # Rebuild the scenario from what was recorded, so resume needs no extra arguments.
        request = RunRequest(
            incident_id=row["scenario_id"],
            alert=row["alert"],
            run_id=resume_id,
            run_config=run_config,
            resume=True,
        )
    else:
        assert scenario_ref is not None
        scenario = load_scenario_ref(scenario_ref)
        request = RunRequest(incident_id=scenario.id, alert=scenario.alert, run_config=run_config)

    if mock:
        return await _run_mock(request, settings, store, scenario_ref)

    console.print(
        Panel.fit(
            f"[bold]{request.incident_id}[/bold]\n{request.alert}",
            title="incident",
            border_style="cyan",
        )
    )
    # A live run is interactive: any world-mutating action prompts the operator on the terminal.
    from aegis.harness.permissions import ConsoleApprovalBroker

    return await execute_run(
        request, settings=settings, store=store, broker=ConsoleApprovalBroker()
    )


async def _run_mock(
    request: RunRequest, settings: Settings, store: Store, scenario_ref: str | None
) -> RunResult:
    """Offline path: in-process MCP server + deterministic mock model.

    Resolves the scenario by id from the standard scenario file — the mock needs the full
    scenario (it reads ground truth to answer correctly), so a resume with ``--mock`` requires
    that file to still be present.
    """
    from evals.mock_model import MockModel
    from mcp_server.env.generator import load_scenario_ref, load_scenarios
    from mcp_server.server import mcp as mcp_server_app
    from mcp_server.server import register_scenario

    if scenario_ref:
        scenario = load_scenario_ref(scenario_ref)
    else:
        default = Path("evals/scenarios.jsonl")
        if not await asyncio.to_thread(default.exists):
            console.print("[red]--mock resume needs evals/scenarios.jsonl[/red]")
            raise typer.Exit(1)
        scenario = next(s for s in load_scenarios(default) if s.id == request.incident_id)
    register_scenario(scenario)

    console.print(
        Panel.fit(
            f"[bold]{scenario.id}[/bold]  [dim](mock model — result is a pipeline check, "
            f"not a measurement)[/dim]\n{scenario.alert}",
            title="incident",
            border_style="yellow",
        )
    )
    # The mock runs against the simulated world, unattended — auto-approve so the demo flows.
    from aegis.harness.permissions import AutoApprove

    return await execute_run(
        request,
        settings=settings,
        store=store,
        model=MockModel(scenario=scenario),
        mcp_server=mcp_server_app,
        broker=AutoApprove(),
    )


@app.command()
def scenarios(
    count: Annotated[int, typer.Option(help="How many scenarios to generate.")] = 60,
    out: Annotated[Path, typer.Option(help="Output path.")] = Path("evals/scenarios.jsonl"),
    seed: Annotated[int, typer.Option()] = 1337,
) -> None:
    """Generate a validated incident scenario set."""
    from mcp_server.env.generator import main as generate_main

    code = generate_main(["--scenarios", str(count), "--out", str(out), "--seed", str(seed)])
    raise typer.Exit(code)


@app.command()
def show(run_id: Annotated[str, typer.Argument(help="Run id to inspect.")]) -> None:
    """Print a finished run: RCA, per-agent cost, and the tool timeline."""
    configure_logging("WARNING")
    store = Store(get_settings().db_path)
    row = store.get_run(run_id)
    if row is None:
        console.print(f"[red]no run {run_id!r}[/red]")
        raise typer.Exit(1)

    console.print(
        Panel.fit(
            f"status: [bold]{row['status']}[/bold]   cost: ${row['cost_usd']:.4f}   "
            f"duration: {row['duration_s'] or 0:.1f}s",
            title=f"run {run_id}",
        )
    )

    if row["rca_json"]:
        rca = json.loads(row["rca_json"])
        console.print(
            Panel(
                f"[bold]{rca['root_cause_class']}[/bold] @ {rca['culprit_service'] or '(none)'}  "
                f"(confidence {rca['confidence']})\n\n{rca['summary']}\n\n"
                f"evidence: {', '.join(rca['evidence_event_ids'])}",
                title="root cause",
                border_style="green",
            )
        )

    spans = store.list_spans(run_id)
    if spans:
        table = Table(title="agents")
        table.add_column("agent")
        table.add_column("role")
        table.add_column("turns", justify="right")
        table.add_column("cost", justify="right")
        table.add_column("status")
        for span in spans:
            table.add_row(
                span["label"],
                span["role"],
                str(span["turns"]),
                f"${span['cost_usd']:.4f}",
                "[red]error[/red]" if span["error"] else "[green]ok[/green]",
            )
        console.print(table)
    store.close()


@app.command(name="list")
def list_runs(limit: Annotated[int, typer.Option()] = 20) -> None:
    """List recent runs."""
    configure_logging("WARNING")
    store = Store(get_settings().db_path)
    table = Table(title="recent runs")
    for column in ("run", "incident", "status", "cost", "when"):
        table.add_column(column)
    for row in store.list_runs(limit):
        import datetime

        when = datetime.datetime.fromtimestamp(row["created_at"], tz=datetime.UTC)
        table.add_row(
            row["id"],
            row["scenario_id"],
            row["status"],
            f"${row['cost_usd']:.4f}",
            when.strftime("%Y-%m-%d %H:%M"),
        )
    console.print(table)
    store.close()


def _render_result(result: RunResult) -> None:
    color = {
        RunStatus.SUCCEEDED: "green",
        RunStatus.BUDGET_EXCEEDED: "yellow",
    }.get(result.status, "red")
    console.print(
        Panel.fit(
            f"status: [bold]{result.status}[/bold]   cost: ${result.cost_usd:.4f}   "
            f"duration: {result.duration_s:.1f}s   run: {result.run_id}",
            border_style=color,
        )
    )
    if result.rca:
        remediation = ""
        if result.rca.remediation:
            rem = result.rca.remediation
            remediation = f"\n\nproposed: {rem.action} → {rem.target}"
        console.print(
            Panel(
                f"[bold]{result.rca.root_cause_class}[/bold] @ "
                f"{result.rca.culprit_service or '(none)'}  "
                f"(confidence {result.rca.confidence})\n\n{result.rca.summary}\n\n"
                f"evidence: {', '.join(result.rca.evidence_event_ids)}{remediation}",
                title="root cause",
                border_style="green",
            )
        )
    elif result.error:
        console.print(f"[red]{result.error}[/red]")


if __name__ == "__main__":
    app()
