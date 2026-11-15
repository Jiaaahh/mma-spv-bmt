"""Create publication-ready figures from the frozen multi-budget release.

This is a deterministic post-processing step.  It reads only the released
summary tables and never re-runs, filters, or modifies the formal experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MultipleLocator


EXPECTED_BUDGETS = (32, 48, 64, 96, 128)
EXPECTED_METHODS = ("ideal_only", "ideal_plus_multi4")
DISPLAY_NAMES = {
    "ideal_only": "Ideal only",
    "ideal_plus_multi4": "Ideal + Multi-4",
}
COLORS = {
    "ideal_only": "#4C78A8",
    "ideal_plus_multi4": "#F58518",
}
MARKERS = {
    "ideal_only": "o",
    "ideal_plus_multi4": "s",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_columns(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")


def _validate_release(
    accuracy: pd.DataFrame,
    delta: pd.DataFrame,
    naubc: pd.DataFrame,
    main: dict,
) -> None:
    _require_columns(
        accuracy,
        {
            "method",
            "budget",
            "n_samples",
            "n_correct",
            "accuracy",
            "ci_lower",
            "ci_upper",
        },
        "accuracy_budget.csv",
    )
    _require_columns(
        delta,
        {
            "budget",
            "n_pairs",
            "reference_accuracy",
            "treatment_accuracy",
            "delta_accuracy",
            "ci_lower",
            "ci_upper",
        },
        "delta_budget.csv",
    )
    _require_columns(
        naubc,
        {"method", "naubc", "ci_lower", "ci_upper"},
        "naubc.csv",
    )

    expected_pairs = {
        (method, budget)
        for method in EXPECTED_METHODS
        for budget in EXPECTED_BUDGETS
    }
    actual_pairs = set(zip(accuracy["method"], accuracy["budget"], strict=True))
    if actual_pairs != expected_pairs or len(accuracy) != len(expected_pairs):
        raise ValueError("accuracy table does not contain the exact method-budget grid")
    if tuple(sorted(delta["budget"].tolist())) != EXPECTED_BUDGETS:
        raise ValueError("delta table does not contain each expected budget exactly once")
    if set(naubc["method"]) != set(EXPECTED_METHODS) or len(naubc) != 2:
        raise ValueError("nAUBC table does not contain exactly the two expected methods")

    for frame, point_column in (
        (accuracy, "accuracy"),
        (delta, "delta_accuracy"),
        (naubc, "naubc"),
    ):
        if not (
            (frame["ci_lower"] <= frame[point_column])
            & (frame[point_column] <= frame["ci_upper"])
        ).all():
            raise ValueError(f"confidence interval does not bracket {point_column}")

    pivot = accuracy.pivot(index="budget", columns="method", values="accuracy")
    expected_delta = (
        pivot["ideal_plus_multi4"] - pivot["ideal_only"]
    ).reindex(delta["budget"]).to_numpy()
    if not np.allclose(expected_delta, delta["delta_accuracy"].to_numpy(), atol=1e-12):
        raise ValueError("delta values do not match the released accuracy curves")
    if not np.allclose(
        pivot["ideal_only"].reindex(delta["budget"]).to_numpy(),
        delta["reference_accuracy"].to_numpy(),
        atol=1e-12,
    ) or not np.allclose(
        pivot["ideal_plus_multi4"].reindex(delta["budget"]).to_numpy(),
        delta["treatment_accuracy"].to_numpy(),
        atol=1e-12,
    ):
        raise ValueError("delta component accuracies do not match the released curves")

    naubc_lookup = dict(zip(naubc["method"], naubc["naubc"], strict=True))
    computed_naubc = {
        method: float(
            np.trapz(
                pivot[method].reindex(EXPECTED_BUDGETS).to_numpy(),
                np.asarray(EXPECTED_BUDGETS, dtype=float),
            )
            / (EXPECTED_BUDGETS[-1] - EXPECTED_BUDGETS[0])
        )
        for method in EXPECTED_METHODS
    }
    for method in EXPECTED_METHODS:
        if not np.isclose(computed_naubc[method], naubc_lookup[method], atol=1e-12):
            raise ValueError(f"nAUBC does not match trapezoidal integration for {method}")

    bootstrap = main.get("bootstrap", {})
    delta_naubc = bootstrap.get("delta_naubc", {})
    computed_delta_naubc = (
        computed_naubc["ideal_plus_multi4"] - computed_naubc["ideal_only"]
    )
    if not (
        main.get("analysis_release") is True
        and main.get("status") == "ANALYSIS_RELEASED"
        and main.get("primary_success") is True
        and float(delta_naubc["lower"]) > 0.0
        and np.isclose(
            computed_delta_naubc, float(main["delta_naubc"]), atol=1e-12
        )
        and np.isclose(
            computed_delta_naubc, float(delta_naubc["point"]), atol=1e-12
        )
    ):
        raise ValueError("released primary endpoint does not satisfy its frozen rule")

    target = float(pivot.loc[64, "ideal_only"])
    attained_budgets = [
        budget
        for budget in EXPECTED_BUDGETS
        if float(pivot.loc[budget, "ideal_plus_multi4"]) >= target
    ]
    computed_beq64 = min(attained_budgets) if attained_budgets else None
    released_beq64 = main.get("b_eq64", {})
    if (
        released_beq64.get("budget") != computed_beq64
        or not np.isclose(
            float(released_beq64["target_accuracy"]), target, atol=1e-12
        )
    ):
        raise ValueError("B_eq64 does not match the released accuracy curves")


def _percent(value: float, _position: float) -> str:
    return f"{value * 100:.0f}"


def _make_figure(
    accuracy: pd.DataFrame,
    delta: pd.DataFrame,
    main: dict,
) -> plt.Figure:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    figure, (axis_curve, axis_delta) = plt.subplots(
        2,
        1,
        figsize=(7.2, 6.8),
        sharex=True,
        gridspec_kw={"height_ratios": (1.65, 1.0), "hspace": 0.16},
    )

    for method in EXPECTED_METHODS:
        points = accuracy.loc[accuracy["method"] == method].sort_values("budget")
        yerr = np.vstack(
            (
                points["accuracy"].to_numpy() - points["ci_lower"].to_numpy(),
                points["ci_upper"].to_numpy() - points["accuracy"].to_numpy(),
            )
        )
        axis_curve.errorbar(
            points["budget"],
            points["accuracy"],
            yerr=yerr,
            color=COLORS[method],
            marker=MARKERS[method],
            markersize=6.2,
            markeredgecolor="white",
            markeredgewidth=0.8,
            linewidth=2.0,
            capsize=3.5,
            elinewidth=1.2,
            label=DISPLAY_NAMES[method],
            zorder=3,
        )

    axis_curve.set_title("Accuracy-budget performance", loc="left", pad=9)
    axis_curve.set_ylabel("Accuracy (%)")
    axis_curve.set_ylim(0.32, 0.64)
    axis_curve.yaxis.set_major_locator(MultipleLocator(0.05))
    axis_curve.yaxis.set_major_formatter(FuncFormatter(_percent))
    axis_curve.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.8)
    axis_curve.legend(loc="upper left", frameon=False, ncol=2)
    axis_curve.text(
        -0.095,
        1.035,
        "A",
        transform=axis_curve.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
    )

    delta = delta.sort_values("budget")
    delta_yerr = np.vstack(
        (
            delta["delta_accuracy"].to_numpy() - delta["ci_lower"].to_numpy(),
            delta["ci_upper"].to_numpy() - delta["delta_accuracy"].to_numpy(),
        )
    )
    axis_delta.axhline(0.0, color="#777777", linewidth=0.9, zorder=1)
    axis_delta.errorbar(
        delta["budget"],
        delta["delta_accuracy"],
        yerr=delta_yerr,
        color="#2A9D8F",
        marker="D",
        markersize=5.8,
        markeredgecolor="white",
        markeredgewidth=0.8,
        linewidth=1.8,
        capsize=3.5,
        elinewidth=1.2,
        zorder=3,
    )
    axis_delta.fill_between(
        delta["budget"],
        delta["ci_lower"],
        delta["ci_upper"],
        color="#2A9D8F",
        alpha=0.10,
        linewidth=0,
        zorder=2,
    )
    axis_delta.set_title("Paired gain: Ideal + Multi-4 minus Ideal only", loc="left", pad=8)
    axis_delta.set_xlabel("Budget")
    axis_delta.set_ylabel("Accuracy gain\n(percentage points)")
    axis_delta.set_ylim(0.0, 0.215)
    axis_delta.yaxis.set_major_locator(MultipleLocator(0.05))
    axis_delta.yaxis.set_major_formatter(FuncFormatter(_percent))
    axis_delta.set_xticks(EXPECTED_BUDGETS)
    axis_delta.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.8)
    axis_delta.text(
        -0.095,
        1.05,
        "B",
        transform=axis_delta.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
    )

    interval = main["bootstrap"]["delta_naubc"]
    annotation = (
        "Primary endpoint: ΔnAUBC = "
        f"{float(interval['point']) * 100:.2f} pp\n"
        f"95% CI {float(interval['lower']) * 100:.2f}-"
        f"{float(interval['upper']) * 100:.2f} pp; lower bound > 0"
    )
    axis_delta.text(
        0.01,
        0.95,
        annotation,
        transform=axis_delta.transAxes,
        ha="left",
        va="top",
        fontsize=8.8,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#B7B7B7",
            "linewidth": 0.7,
            "alpha": 0.95,
        },
    )

    replicates = int(main["bootstrap"]["replicates"])
    cluster = str(main["bootstrap"]["cluster"])
    n_per_point = int(accuracy["n_samples"].iloc[0])
    figure.text(
        0.125,
        0.015,
        f"Points pool 5 seeds (n={n_per_point:,} per method-budget point).\n"
        f"Error bars: 95% paired {cluster}-cluster bootstrap CIs "
        f"({replicates:,} replicates).",
        ha="left",
        va="bottom",
        fontsize=8.1,
        color="#4F4F4F",
    )
    figure.subplots_adjust(left=0.14, right=0.985, top=0.96, bottom=0.15)
    return figure


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release-root",
        type=Path,
        default=Path("outputs/multibudget_analysis_release"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/multibudget_paper_assets"),
    )
    arguments = parser.parse_args()

    release_root = arguments.release_root.resolve()
    output_root = arguments.output_root.resolve()
    input_paths = {
        "accuracy_budget": release_root / "accuracy_budget.csv",
        "delta_budget": release_root / "delta_budget.csv",
        "naubc": release_root / "naubc.csv",
        "main_results": release_root / "main_results.json",
        "release_manifest": release_root / "release_manifest.json",
    }
    for name, path in input_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing {name}: {path}")

    accuracy = pd.read_csv(input_paths["accuracy_budget"])
    delta = pd.read_csv(input_paths["delta_budget"])
    naubc = pd.read_csv(input_paths["naubc"])
    with input_paths["main_results"].open("r", encoding="utf-8") as handle:
        main_results = json.load(handle)
    _validate_release(accuracy, delta, naubc, main_results)

    output_root.mkdir(parents=True, exist_ok=True)
    figure = _make_figure(accuracy, delta, main_results)
    output_paths = {
        "png": output_root / "accuracy_budget_curve.png",
        "pdf": output_root / "accuracy_budget_curve.pdf",
        "svg": output_root / "accuracy_budget_curve.svg",
    }
    figure.savefig(output_paths["png"], dpi=450, facecolor="white")
    figure.savefig(output_paths["pdf"], facecolor="white")
    figure.savefig(output_paths["svg"], facecolor="white")
    plt.close(figure)

    interval = main_results["bootstrap"]["delta_naubc"]
    beq = main_results["b_eq64"]
    manifest = {
        "schema": "mma-multibudget-v1-paper-figure",
        "schema_version": 1,
        "status": "FIGURE_GENERATED_AND_NUMERICALLY_VALIDATED",
        "source_release_manifest_sha256": _sha256(
            input_paths["release_manifest"]
        ),
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in input_paths.items()
        },
        "outputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in output_paths.items()
        },
        "validated_grid": {
            "methods": list(EXPECTED_METHODS),
            "budgets": list(EXPECTED_BUDGETS),
            "accuracy_points": int(len(accuracy)),
            "delta_points": int(len(delta)),
        },
        "primary_endpoint": {
            "name": main_results["primary_endpoint"],
            "point": float(interval["point"]),
            "ci_lower": float(interval["lower"]),
            "ci_upper": float(interval["upper"]),
            "success": bool(main_results["primary_success"]),
        },
        "b_eq64": beq,
    }
    manifest_path = output_root / "figure_manifest.json"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    print(
        json.dumps(
            {
                "status": manifest["status"],
                "output_root": str(output_root),
                "figure_outputs": {
                    name: str(path) for name, path in output_paths.items()
                },
                "manifest": str(manifest_path),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
