"""Read-only integrity audit for a configured experiment matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.v2.audit import audit_plan
from src.v2.experiment import load_experiment_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--allow-source-drift",
        action="store_true",
        help="audit historical outputs even if current source has changed",
    )
    arguments = parser.parse_args()
    config = load_experiment_config(arguments.config)
    result = audit_plan(
        config,
        require_current_source=not arguments.allow_source_drift,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
