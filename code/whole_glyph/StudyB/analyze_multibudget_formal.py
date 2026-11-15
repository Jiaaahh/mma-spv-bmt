"""Inspect or explicitly release the completed Study B Accuracy--Budget analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from src.multibudget.formal import FormalExecutionError
from src.multibudget.protocol import MultiBudgetProtocolError
from src.multibudget.release import (
    AnalysisReleaseError,
    inspect_release_gate,
    release_multibudget_analysis,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/multibudget_protocol.yaml"),
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--release",
        action="store_true",
        help=(
            "after an exact verified 50/50 session, read prediction tokens and "
            "truth and atomically emit the preregistered analysis"
        ),
    )
    action.add_argument(
        "--validate",
        action="store_true",
        help="inspect administrative completion metadata without prediction/truth I/O",
    )
    action.add_argument(
        "--dry-run",
        action="store_true",
        help="same prediction- and truth-blind administrative inspection as default",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="project-contained release directory; valid only with --release",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.output_dir is not None and not arguments.release:
        print(
            json.dumps(
                {"status": "FAIL", "error": "--output-dir requires --release"},
                ensure_ascii=False,
            )
        )
        return 2
    try:
        if arguments.release:
            result = release_multibudget_analysis(
                arguments.config, output_dir=arguments.output_dir
            )
        else:
            # Default, --dry-run, and --validate are intentionally identical:
            # no prediction CSV and no glyph truth array is opened.
            result = inspect_release_gate(arguments.config).report()
    except (
        AnalysisReleaseError,
        FormalExecutionError,
        MultiBudgetProtocolError,
        OSError,
        ValueError,
    ) as error:
        print(json.dumps({"status": "FAIL", "error": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
