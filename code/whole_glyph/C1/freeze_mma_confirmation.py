"""Freeze the C1 execution/config binding before one-shot scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.v2.mma_confirmation import (
    MmaConfirmationError,
    freeze_confirmation_lock,
    load_confirmation_config,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze MMA confirmation source and protocol; no scoring."
    )
    parser.add_argument("--config", required=True, type=Path)
    arguments = parser.parse_args()
    try:
        result = freeze_confirmation_lock(load_confirmation_config(arguments.config))
    except (MmaConfirmationError, OSError, ValueError) as error:
        raise SystemExit(f"ERROR: {error}") from error
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
