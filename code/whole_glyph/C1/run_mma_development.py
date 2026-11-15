"""Command-line entry point for the D2 MMA development cross-validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.v2.mma_development import (
    MmaDevelopmentError,
    execute_mma_development,
    load_mma_development_config,
    validate_mma_plan_files,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run development-only 6-font x 5-mapping cross-validation for "
            "multi-mapping cosine prototypes."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate frozen paths, hashes, environment, and the 30-fold plan",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="deliberately replace the complete D2 output directory",
    )
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    try:
        config = load_mma_development_config(arguments.config)
        if arguments.dry_run:
            result = validate_mma_plan_files(config)
        else:
            result = execute_mma_development(
                config,
                overwrite=arguments.overwrite,
            )
    except (MmaDevelopmentError, OSError) as error:
        raise SystemExit(f"ERROR: {error}") from error
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

