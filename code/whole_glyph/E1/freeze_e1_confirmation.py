"""Freeze or validate the one-shot E1 confirmation integrity lock."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from src.e1.integrity import E1IntegrityError, freeze, validate_lock


def _input_mapping(values: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name.strip() or not path.strip():
            raise E1IntegrityError("--input must use NAME=PATH")
        if name in result:
            raise E1IntegrityError(f"duplicate --input name: {name}")
        result[name] = path
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--asset-manifest", type=Path)
    parser.add_argument("--input", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--requirements", type=Path)
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--validate", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.validate:
            if arguments.config is not None:
                root = arguments.config.resolve().parent.parent
                lock = arguments.lock or root / "locks/e1_confirmation.lock.json"
            else:
                root = None
                lock = arguments.lock or Path("locks/e1_confirmation.lock.json")
            result = validate_lock(lock, project_root=root)
        else:
            if arguments.config is None:
                raise E1IntegrityError("--config is required when freezing")
            explicit_inputs = _input_mapping(arguments.input)
            result = freeze(
                arguments.config,
                arguments.asset_manifest,
                explicit_inputs or None,
                lock_path=arguments.lock,
                requirements_path=arguments.requirements,
                force=arguments.force,
            )
    except E1IntegrityError as error:
        print(json.dumps({"status": "FAIL", "error": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
