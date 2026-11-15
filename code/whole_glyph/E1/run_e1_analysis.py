"""Analyze a fully committed E1 prediction archive."""

from __future__ import annotations

import argparse

from src.e1.experiment import analyze_confirmation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/e1_confirmation.yaml")
    args = parser.parse_args()
    report = analyze_confirmation(args.config)
    primary = report["primary_comparison"]
    bootstrap = primary["paired_bootstrap"]
    print(report["status"])
    print(
        "Primary difference: "
        f"{primary['mean_accuracy_difference_pp']:.3f} pp; "
        f"95% CI [{bootstrap['ci95_pp'][0]:.3f}, "
        f"{bootstrap['ci95_pp'][1]:.3f}] pp"
    )


if __name__ == "__main__":
    main()
