#!/usr/bin/env python3
"""Report top-1 accuracy for AE baseline images."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from common import (
    DEFAULT_RESULTS,
    MODELS,
    correct_field,
    default_analysis_dir,
    int_sort_key,
    is_available_correct,
    read_jsonl,
    write_csv,
)


def is_ae_row(row: dict[str, Any]) -> bool:
    return str(row.get("param_id", "")) == "0" or row.get("exposure_mode") == "auto"


def accuracy_row(model: str, items: list[dict[str, Any]], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    correct_count = sum(1 for item in items if item.get(correct_field(model)) is True)
    n_images = len(items)
    row: dict[str, Any] = {
        "model": model,
        "n_images": n_images,
        "correct_count": correct_count,
        "accuracy": correct_count / n_images if n_images else None,
    }
    if extra:
        row = {**extra, **row}
    return row


def by_lighting(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for model in MODELS:
            if is_available_correct(row.get(correct_field(model))):
                grouped[(model, str(row.get("light_id", "")))].append(row)

    output: list[dict[str, Any]] = []
    for (model, light_id), items in sorted(grouped.items(), key=lambda item: (item[0][0], int_sort_key(item[0][1]))):
        first = items[0]
        output.append(
            accuracy_row(
                model,
                items,
                {
                    "scope": "lighting",
                    "light_id": light_id,
                    "position": first.get("position", ""),
                    "intensity": first.get("intensity"),
                },
            )
        )
    return output


def overall(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for model in MODELS:
        items = [row for row in rows if is_available_correct(row.get(correct_field(model)))]
        if items:
            output.append(
                accuracy_row(
                    model,
                    items,
                    {
                        "scope": "overall",
                        "light_id": "all",
                        "position": "all",
                        "intensity": "all",
                    },
                )
            )
    return output


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Report AE baseline downstream accuracy.")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--output-file", type=Path)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    results_path = args.results.expanduser()
    output_dir = args.output_dir.expanduser() if args.output_dir else default_analysis_dir(results_path, "ae_baseline")
    output_file = args.output_file.expanduser() if args.output_file else output_dir / "ae_baseline.csv"
    rows = [row for row in read_jsonl(results_path) if is_ae_row(row)]
    output_rows = by_lighting(rows) + overall(rows)
    write_csv(
        output_file,
        ["scope", "model", "light_id", "position", "intensity", "n_images", "correct_count", "accuracy"],
        output_rows,
    )
    print(f"Output: {output_file}")


if __name__ == "__main__":
    main()
