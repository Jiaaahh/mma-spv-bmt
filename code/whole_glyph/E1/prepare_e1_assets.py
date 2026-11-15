"""Command-line entry point for frozen E1 input-asset preparation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.e1.assets import DEFAULT_FONT_PATHS, prepare_e1_assets


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = Path(r"/historical/deepseek\hanzi-spv-benchmark\data\processed\benchmark\glyphs.npz")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare E1 glyph/mapping inputs without prediction or scoring."
    )
    parser.add_argument("--source-dataset", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "data" / "e1_assets"
    )
    parser.add_argument(
        "--font",
        action="append",
        type=Path,
        dest="fonts",
        help="Exactly four font paths; omitted uses the frozen Windows font list.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace only an unexposed bundle whose manifest still says no predictions.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    fonts = tuple(args.fonts) if args.fonts is not None else DEFAULT_FONT_PATHS
    result = prepare_e1_assets(
        source_dataset=args.source_dataset,
        output_dir=args.output_dir,
        font_paths=fonts,
        force=args.force,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "manifest_path": result["manifest_path"],
                "manifest_sha256": result["manifest_sha256"],
                "glyph_sha256": result["glyph_archive"]["file_sha256"],
                "mapping_count": len(result["mappings"]),
                "prediction_executed": result["prediction_executed"],
                "accuracy_read": result["accuracy_read"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
