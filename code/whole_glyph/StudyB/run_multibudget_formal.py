"""Prepare, validate, or resume the exact-prefix Study B formal session."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from src.multibudget.formal import (
    FormalExecutionError,
    prepare_formal_session,
    resume_formal_session,
    validate_formal_lock,
    validate_formal_session,
)
from src.multibudget.protocol import MultiBudgetProtocolError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/multibudget_protocol.yaml"),
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--prepare-session",
        action="store_true",
        help="create the zero-of-fifty session without running a prediction",
    )
    action.add_argument(
        "--resume",
        action="store_true",
        help="irreversibly run and commit the next frozen condition(s)",
    )
    action.add_argument(
        "--validate",
        action="store_true",
        help="read-only verify the prepared session and completed prefix",
    )
    action.add_argument(
        "--dry-run",
        action="store_true",
        help="read-only verify the formal lock without preparing or predicting",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=1,
        help="maximum conditions to commit with --resume (default: 1)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.max_runs != 1 and not arguments.resume:
        print(
            json.dumps(
                {"status": "FAIL", "error": "--max-runs requires --resume"},
                ensure_ascii=False,
            )
        )
        return 2
    try:
        if arguments.prepare_session:
            result = prepare_formal_session(arguments.config)
        elif arguments.resume:
            result = resume_formal_session(arguments.config, max_runs=arguments.max_runs)
        elif arguments.validate:
            result = validate_formal_session(arguments.config)
        else:
            # Default is intentionally performance-blind.
            lock = validate_formal_lock(arguments.config)
            result = {
                **lock,
                "status": "PASS_FORMAL_RUNNER_DRY_RUN_LOCK_ONLY",
                "formal_prediction_executed": False,
                "hint": "use --prepare-session; --resume is required to execute",
            }
    except (FormalExecutionError, MultiBudgetProtocolError, OSError, ValueError) as error:
        print(json.dumps({"status": "FAIL", "error": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
