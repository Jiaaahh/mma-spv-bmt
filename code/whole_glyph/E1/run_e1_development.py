"""Run D3 LinearSVC development selection without reading E1 test assets."""

from __future__ import annotations

import argparse

from src.e1.experiment import run_development


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/d3_svc.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    summary = run_development(args.config, force=args.force)
    print(f"D3 complete; frozen LinearSVC C={summary['selected_c']}")
    print("E1 test assets read: no; E1 predictions executed: no")


if __name__ == "__main__":
    main()
