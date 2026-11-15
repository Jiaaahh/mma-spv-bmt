"""Freeze or read-only validate the Study B formal-execution lock."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from src.multibudget.formal import FormalExecutionError, freeze_formal_lock
from src.multibudget.protocol import MultiBudgetProtocolError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/multibudget_protocol.yaml"),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--validate",
        action="store_true",
        help="validate an existing lock without writing any file",
    )
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="validate freeze readiness without writing the lock",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        result = freeze_formal_lock(
            arguments.config,
            validate_only=arguments.validate,
            dry_run=arguments.dry_run,
        )
    except (FormalExecutionError, MultiBudgetProtocolError, OSError, ValueError) as error:
        print(json.dumps({"status": "FAIL", "error": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
