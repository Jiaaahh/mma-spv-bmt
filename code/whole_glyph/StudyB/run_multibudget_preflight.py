"""Freeze or read-only validate the MMA multi-budget 0/50 preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from src.multibudget.protocol import MultiBudgetProtocolError, prepare_preflight


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/multibudget_protocol.yaml"),
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="validate existing artifacts without writing any file",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        result = prepare_preflight(
            arguments.config,
            validate_only=arguments.validate,
        )
    except (MultiBudgetProtocolError, OSError, ValueError) as error:
        print(json.dumps({"status": "FAIL", "error": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
