"""CLI to freeze or read-only validate Study B inputs without scoring."""

from __future__ import annotations

import argparse
import json

from src.multibudget.assets import (
    MultiBudgetAssetError,
    prepare_multibudget_assets,
    validate_multibudget_assets,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze four blinded fonts and five unseen moderate mappings; "
            "never run predictions or read accuracy."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--font-spec", required=True)
    parser.add_argument("--exclusions", required=True)
    parser.add_argument(
        "--historical-manifest",
        required=True,
        action="append",
        help="Historical mapping manifest; repeat for every prior asset family.",
    )
    parser.add_argument("--contact-sheets", action="store_true")
    parser.add_argument("--validate", action="store_true")
    arguments = parser.parse_args()
    try:
        if arguments.validate:
            report = validate_multibudget_assets(
                config_path=arguments.config,
                font_spec_path=arguments.font_spec,
                exclusion_path=arguments.exclusions,
                historical_manifest_paths=arguments.historical_manifest,
            )
        else:
            report = prepare_multibudget_assets(
                config_path=arguments.config,
                font_spec_path=arguments.font_spec,
                exclusion_path=arguments.exclusions,
                historical_manifest_paths=arguments.historical_manifest,
                include_contact_sheets=arguments.contact_sheets,
            )
    except (MultiBudgetAssetError, OSError, ValueError) as error:
        raise SystemExit(f"ERROR: {error}") from error
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
