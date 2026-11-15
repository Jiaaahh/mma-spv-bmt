"""Command-line entry point for CA-SkelCov experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.v2.experiment import (
    ALLOWED_BUDGETS,
    ALLOWED_METHODS,
    ExperimentError,
    execute_plan,
    load_experiment_config,
    select_conditions,
    validate_plan_files,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run auditable calibration-aware Hanzi SPV conditions."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--method", choices=ALLOWED_METHODS)
    parser.add_argument("--budget", type=int, choices=ALLOWED_BUDGETS)
    parser.add_argument("--mapping-seed", type=int)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="deliberately replace matching result directories",
    )
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    try:
        config = load_experiment_config(arguments.config)
        selected = select_conditions(
            config,
            run_id=arguments.run_id,
            method=arguments.method,
            budget=arguments.budget,
            mapping_seed=arguments.mapping_seed,
        )
        if arguments.dry_run:
            selected_ids = {
                condition.run_id(config.experiment_id) for condition in selected
            }
            plan = [
                item
                for item in validate_plan_files(config)
                if item["run_id"] in selected_ids
            ]
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return 0
        results = execute_plan(config, selected, overwrite=arguments.overwrite)
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return 0
    except ExperimentError as error:
        raise SystemExit(f"ERROR: {error}") from error


if __name__ == "__main__":
    raise SystemExit(main())
