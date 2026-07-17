#!/usr/bin/env python3
"""Aggregate top-1 downstream correctness by capture parameter."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from common import (
    DEFAULT_RESULTS,
    MODELS,
    correct_field,
    default_analysis_dir,
    int_sort_key,
    is_available_correct,
    maybe_exclude_auto,
    object_sort_key,
    read_jsonl,
    write_csv,
)


def bool_accuracy(correct_count: int, n: int) -> float | None:
    return correct_count / n if n else None


def first_meta(rows: list[dict[str, Any]]) -> dict[str, Any]:
    first = rows[0]
    return {
        "param_file": first.get("param_file", ""),
        "aperture": first.get("aperture", ""),
        "iso": first.get("iso", ""),
        "shutter_speed": first.get("shutter_speed", ""),
        "exposure_mode": first.get("exposure_mode", ""),
    }


def aggregate_by_param(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for model in MODELS:
            if is_available_correct(row.get(correct_field(model))):
                grouped[(model, str(row.get("param_id", "")))].append(row)

    output: list[dict[str, Any]] = []
    for (model, param_id), items in grouped.items():
        correct_count = sum(1 for item in items if item.get(correct_field(model)) is True)
        output.append(
            {
                "model": model,
                "param_id": param_id,
                **first_meta(items),
                "n_images": len(items),
                "correct_count": correct_count,
                "accuracy": bool_accuracy(correct_count, len(items)),
            }
        )
    return sorted(output, key=lambda row: (row["model"], int_sort_key(row["param_id"])))


def aggregate_by_object_param(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for model in MODELS:
            if is_available_correct(row.get(correct_field(model))):
                grouped[(model, str(row.get("object_key", "")), str(row.get("param_id", "")))].append(row)

    output: list[dict[str, Any]] = []
    for (model, object_key, param_id), items in grouped.items():
        first = items[0]
        correct_count = sum(1 for item in items if item.get(correct_field(model)) is True)
        output.append(
            {
                "model": model,
                "object_key": object_key,
                "class_id": first.get("class_id", ""),
                "sample_id": first.get("sample_id", ""),
                "view_id": first.get("view_id", ""),
                "param_id": param_id,
                "n_lighting": len(items),
                "correct_count": correct_count,
                "accuracy": bool_accuracy(correct_count, len(items)),
            }
        )
    return sorted(
        output,
        key=lambda row: (row["model"], object_sort_key(row), int_sort_key(row["param_id"])),
    )


def distribution(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for model in MODELS:
        model_rows = [row for row in rows if row["model"] == model]
        if not model_rows:
            continue
        max_count = max(int(row["n_images"]) for row in model_rows)
        counts = Counter(int(row["correct_count"]) for row in model_rows)
        for correct_count in range(max_count + 1):
            output.append(
                {
                    "model": model,
                    "correct_count": correct_count,
                    "n_params": counts.get(correct_count, 0),
                }
            )
    return output


def distribution_by_object(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["object_key"])].append(row)

    output: list[dict[str, Any]] = []
    for (model, object_key), items in sorted(grouped.items()):
        max_count = max(int(row["n_lighting"]) for row in items)
        counts = Counter(int(row["correct_count"]) for row in items)
        for correct_count in range(max_count + 1):
            output.append(
                {
                    "model": model,
                    "object_key": object_key,
                    "correct_count": correct_count,
                    "n_params": counts.get(correct_count, 0),
                }
            )
    return output


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate top-1 correctness by parameter.")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--exclude-auto-param", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    results_path = args.results.expanduser()
    output_dir = args.output_dir.expanduser() if args.output_dir else default_analysis_dir(results_path, "param_correct_counts")
    rows = maybe_exclude_auto(read_jsonl(results_path), args.exclude_auto_param)
    by_param = aggregate_by_param(rows)
    by_object_param = aggregate_by_object_param(rows)

    write_csv(
        output_dir / "param_correct_counts.csv",
        ["model", "param_id", "param_file", "aperture", "iso", "shutter_speed", "exposure_mode", "n_images", "correct_count", "accuracy"],
        by_param,
    )
    write_csv(
        output_dir / "param_correct_counts_by_object.csv",
        ["model", "object_key", "class_id", "sample_id", "view_id", "param_id", "n_lighting", "correct_count", "accuracy"],
        by_object_param,
    )
    write_csv(
        output_dir / "correct_count_distribution.csv",
        ["model", "correct_count", "n_params"],
        distribution(by_param),
    )
    write_csv(
        output_dir / "correct_count_distribution_by_object.csv",
        ["model", "object_key", "correct_count", "n_params"],
        distribution_by_object(by_object_param),
    )
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
