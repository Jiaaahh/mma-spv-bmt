"""Run the frozen one-shot E1 prediction stage without performance aggregation."""

from __future__ import annotations

import argparse

from src.e1.experiment import run_confirmation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/e1_confirmation.yaml")
    parser.add_argument("--lock", default=None)
    args = parser.parse_args()
    completion = run_confirmation(args.config, lock_path=args.lock)
    print(completion["status"])
    print(f"prediction rows: {completion['prediction_row_count']}")
    print("No aggregate accuracy or bootstrap was read during this stage.")


if __name__ == "__main__":
    main()
