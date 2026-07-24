"""Render the eval summary into README charts.

Three figures, each in a light and a dark variant (the README picks per theme via <picture>):

  1. accuracy-by-arm      — grouped bars: how topology (single-agent → +verifier → full) moves
                            class accuracy, culprit accuracy, and injection resistance.
  2. accuracy-vs-cost     — the effort/topology frontier: accuracy against mean $/run, so the
                            cost of each accuracy gain is legible.
  3. per-class-accuracy   — a heatmap of accuracy per fault class per arm; shows which failure
                            modes each arm actually recovers.

Colors come from the dataviz reference palette (pre-validated for CVD and contrast in both
modes); marks follow its specs — thin bars with a surface gap, recessive grid, direct labels,
a legend for multiple series.

    python evals/charts.py --summary evals/results/summary.json --out docs/charts
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Fixed categorical order (never cycled). From references/palette.md, both modes pre-validated.
CATEGORICAL_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
CATEGORICAL_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300"]
SEQUENTIAL_BLUE = ["#eef5fe", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]


@dataclass
class Theme:
    name: str
    surface: str
    ink: str
    secondary: str
    grid: str
    categorical: list[str]

    @classmethod
    def light(cls) -> Theme:
        return cls("light", "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", CATEGORICAL_LIGHT)

    @classmethod
    def dark(cls) -> Theme:
        return cls("dark", "#1a1a19", "#ffffff", "#c3c2b7", "#33332f", CATEGORICAL_DARK)


ARM_LABELS = {
    "single_agent": "single agent",
    "no_verifier": "coordinator\n+ investigators",
    "full": "full\n(+ verifier)",
    "effort_low": "low",
    "effort_medium": "medium",
    "effort_high": "high",
}

CLASS_LABELS = {
    "bad_deploy": "bad deploy",
    "dependency_saturation": "dep. saturation",
    "config_change": "config change",
    "resource_exhaustion": "resource exhaust.",
    "data_anomaly": "data anomaly",
    "network_partition": "net. partition",
}


def _style(ax: Any, theme: Theme) -> None:
    ax.set_facecolor(theme.surface)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(theme.grid)
    ax.tick_params(colors=theme.secondary, labelsize=9, length=0)
    ax.yaxis.label.set_color(theme.secondary)
    ax.xaxis.label.set_color(theme.secondary)
    ax.title.set_color(theme.ink)
    ax.grid(axis="y", color=theme.grid, linewidth=0.8, alpha=0.6)
    ax.set_axisbelow(True)


def _fig(theme: Theme, w: float = 8.0, h: float = 4.6) -> tuple[Any, Any]:
    fig, ax = plt.subplots(figsize=(w, h), dpi=160)
    fig.patch.set_facecolor(theme.surface)
    _style(ax, theme)
    return fig, ax


def _save(fig: Any, out: Path, name: str, theme: Theme) -> None:
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}-{theme.name}.png"
    fig.savefig(path, facecolor=theme.surface, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Chart 1: accuracy by arm
# --------------------------------------------------------------------------------------


def chart_accuracy_by_arm(arms: dict[str, dict], out: Path, theme: Theme) -> None:
    order = [a for a in ("single_agent", "no_verifier", "full") if a in arms]
    if not order:
        return
    metrics = [
        ("class_accuracy", "root-cause\nclass"),
        ("culprit_accuracy", "culprit\nservice"),
        ("injection_resistance", "injection\nresistance"),
    ]

    fig, ax = _fig(theme)
    n_arms = len(order)
    group_w = 0.8
    bar_w = group_w / n_arms
    x = range(len(metrics))

    for i, arm in enumerate(order):
        vals = [arms[arm].get(m[0]) or 0.0 for m in metrics]
        offsets = [xi - group_w / 2 + bar_w * (i + 0.5) for xi in x]
        bars = ax.bar(
            offsets, vals, bar_w * 0.86, color=theme.categorical[i], label=ARM_LABELS[arm]
        )
        for rect, v in zip(bars, vals, strict=True):
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                v + 0.02,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=theme.secondary,
            )

    ax.set_xticks(list(x))
    ax.set_xticklabels([m[1] for m in metrics])
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("accuracy")
    ax.set_title(
        "Accuracy by topology",
        fontsize=13,
        fontweight="bold",
        loc="left",
        pad=12,
        color=theme.ink,
    )
    leg = ax.legend(
        frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=n_arms
    )
    for text in leg.get_texts():
        text.set_color(theme.secondary)
    _save(fig, out, "accuracy-by-arm", theme)


# --------------------------------------------------------------------------------------
# Chart 2: accuracy vs cost frontier
# --------------------------------------------------------------------------------------


def chart_accuracy_vs_cost(arms: dict[str, dict], out: Path, theme: Theme) -> None:
    points = [
        (data["mean_cost_usd"], data["class_accuracy"], arm)
        for arm, data in arms.items()
        if data["n"] > 0
    ]
    if not points:
        return
    fig, ax = _fig(theme)

    # Connect the effort sweep, if present, to draw the frontier.
    sweep = sorted(
        [(c, a, arm) for c, a, arm in points if arm.startswith("effort_")],
        key=lambda p: p[0],
    )
    if len(sweep) >= 2:
        ax.plot(
            [c for c, _, _ in sweep],
            [a for _, a, _ in sweep],
            color=theme.categorical[0],
            linewidth=2,
            alpha=0.5,
            zorder=1,
        )

    for i, (cost, acc, arm) in enumerate(points):
        ax.scatter(
            cost,
            acc,
            s=120,
            color=theme.categorical[i % len(theme.categorical)],
            zorder=3,
            edgecolors=theme.surface,
            linewidths=1.5,
        )
        ax.annotate(
            ARM_LABELS.get(arm, arm).replace("\n", " "),
            (cost, acc),
            textcoords="offset points",
            xytext=(8, 6),
            fontsize=9,
            color=theme.ink,
        )

    ax.set_xlabel("mean cost per incident (USD)")
    ax.set_ylabel("root-cause class accuracy")
    ax.set_ylim(0, 1.08)
    ax.set_title(
        "Accuracy vs. cost frontier",
        fontsize=13,
        fontweight="bold",
        loc="left",
        pad=12,
        color=theme.ink,
    )
    _save(fig, out, "accuracy-vs-cost", theme)


# --------------------------------------------------------------------------------------
# Chart 3: per-class accuracy heatmap
# --------------------------------------------------------------------------------------


def chart_per_class(arms: dict[str, dict], out: Path, theme: Theme) -> None:
    order = [a for a in ("single_agent", "no_verifier", "full") if a in arms]
    classes = list(CLASS_LABELS)
    order = [a for a in order if arms[a].get("per_class_accuracy")]
    if not order:
        return

    grid = [[arms[a]["per_class_accuracy"].get(c) for c in classes] for a in order]

    fig, ax = _fig(theme, w=8.5, h=0.7 * len(order) + 2.2)
    cmap = LinearSegmentedColormap.from_list("blues", SEQUENTIAL_BLUE)
    data = [[v if v is not None else float("nan") for v in row] for row in grid]
    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels([CLASS_LABELS[c] for c in classes], rotation=30, ha="right")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([ARM_LABELS[a].replace("\n", " ") for a in order])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for r, row in enumerate(data):
        for c, v in enumerate(row):
            if v != v:  # nan
                continue
            # Label ink flips to stay legible against the cell.
            ink = "#ffffff" if v > 0.55 else theme.ink
            ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=8.5, color=ink)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(colors=theme.secondary, labelsize=8, length=0)
    ax.set_title(
        "Accuracy per fault class",
        fontsize=13,
        fontweight="bold",
        loc="left",
        pad=12,
        color=theme.ink,
    )
    _save(fig, out, "per-class-accuracy", theme)


def render_all(summary_path: Path, out: Path) -> None:
    summary = json.loads(summary_path.read_text())
    arms = summary["arms"]
    for theme in (Theme.light(), Theme.dark()):
        chart_accuracy_by_arm(arms, out, theme)
        chart_accuracy_vs_cost(arms, out, theme)
        chart_per_class(arms, out, theme)
    print(f"[charts] wrote light+dark PNGs to {out}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render eval charts for the README.")
    parser.add_argument("--summary", type=Path, default=Path("evals/results/summary.json"))
    parser.add_argument("--out", type=Path, default=Path("docs/charts"))
    args = parser.parse_args(argv)
    if not args.summary.exists():
        print(f"no summary at {args.summary}; run evals/run_eval.py first")
        return 1
    render_all(args.summary, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
