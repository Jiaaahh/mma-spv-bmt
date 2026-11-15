"""CLI for the frozen one-shot MMA confirmation experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.v2.mma_confirmation import (
    MmaConfirmationError,
    execute_confirmation,
    load_confirmation_config,
    validate_confirmation_inputs,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run frozen independent MMA confirmation.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    try:
        config = load_confirmation_config(arguments.config)
        result = (
            validate_confirmation_inputs(config)
            if arguments.dry_run
            else execute_confirmation(config)
        )
    except (MmaConfirmationError, OSError, ValueError) as error:
        raise SystemExit(f"ERROR: {error}") from error
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
