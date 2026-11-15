"""Build a descriptive summary after every planned condition is complete."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.v2.audit import summarize_plan, write_summary_files
from src.v2.experiment import load_experiment_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--csv-output", type=Path)
    parser.add_argument(
        "--allow-source-drift",
        action="store_true",
        help="summarize one uniform historical source version after code moved on",
    )
    arguments = parser.parse_args()
    config = load_experiment_config(arguments.config)
    json_output = arguments.json_output or config.output_root / "summary.json"
    csv_output = arguments.csv_output or config.output_root / "summary.csv"
    summary = summarize_plan(
        config,
        allow_source_drift=arguments.allow_source_drift,
    )
    write_summary_files(summary, json_path=json_output, csv_path=csv_output)
    print(
        json.dumps(
            {
                "status": "PASS",
                "json_output": str(json_output),
                "csv_output": str(csv_output),
                "condition_count": len(summary["conditions"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
