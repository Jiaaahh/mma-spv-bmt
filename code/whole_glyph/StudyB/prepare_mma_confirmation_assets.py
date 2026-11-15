"""CLI for freezing independent MMA confirmation inputs without scoring them."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.v2.mma_assets import (
    ConfirmationFont,
    MmaAssetError,
    prepare_confirmation_assets,
)


def _font_spec(value: str) -> ConfirmationFont:
    try:
        font_id, path, face = value.split("|", 2)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "font must be FONT_ID|ABSOLUTE_PATH|FACE_INDEX"
        ) from exc
    try:
        face_index = int(face)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("font face index must be an integer") from exc
    try:
        return ConfirmationFont(font_id, Path(path), face_index)
    except MmaAssetError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze new-font/new-mapping confirmation assets; no accuracy run."
    )
    parser.add_argument("--source-dataset", required=True, type=Path)
    parser.add_argument("--font", required=True, action="append", type=_font_spec)
    parser.add_argument(
        "--mapping-seeds",
        nargs="+",
        type=int,
        default=[2031, 2032, 2033, 2034, 2035],
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--replace", action="store_true")
    arguments = parser.parse_args()
    try:
        report = prepare_confirmation_assets(
            source_dataset=arguments.source_dataset,
            fonts=arguments.font,
            mapping_seeds=arguments.mapping_seeds,
            output_dir=arguments.output_dir,
            replace=arguments.replace,
        )
    except (MmaAssetError, OSError, ValueError) as error:
        raise SystemExit(f"ERROR: {error}") from error
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
