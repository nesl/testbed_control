#!/usr/bin/env python3
"""Measure overlap of successful parameters across lighting conditions."""

from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import combinations
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
    read_jsonl,
    write_csv,
)


def light_sort_key(light_id: str) -> tuple[int, str]:
    return int_sort_key(light_id)


def build_light_sets(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        object_key = str(row.get("object_key", ""))
        light_id = str(row.get("light_id", ""))
        param_id = str(row.get("param_id", ""))
        for model in MODELS:
            value = row.get(correct_field(model))
            if not is_available_correct(value):
                continue
            key = (model, object_key, light_id)
            item = grouped.setdefault(
                key,
                {
                    "model": model,
                    "object_key": object_key,
                    "light_id": light_id,
                    "position": row.get("position", ""),
                    "intensity": row.get("intensity"),
                    "available_params": set(),
                    "correct_params": set(),
                },
            )
            item["available_params"].add(param_id)
            if value is True:
                item["correct_params"].add(param_id)
    return grouped


def overlap_by_object(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    light_sets = build_light_sets(rows)
    object_model_keys = sorted({(model, object_key) for model, object_key, _ in light_sets})
    output: list[dict[str, Any]] = []

    for model, object_key in object_model_keys:
        light_ids = sorted(
            [light_id for key_model, key_object, light_id in light_sets if key_model == model and key_object == object_key],
            key=light_sort_key,
        )
        for left_light_id, right_light_id in combinations(light_ids, 2):
            left = light_sets[(model, object_key, left_light_id)]
            right = light_sets[(model, object_key, right_light_id)]
            shared_params = left["available_params"] & right["available_params"]
            left_correct = left["correct_params"] & shared_params
            right_correct = right["correct_params"] & shared_params
            both_correct = left_correct & right_correct
            n_params = len(shared_params)
            output.append(
                {
                    "model": model,
                    "object_key": object_key,
                    "left_light_id": left_light_id,
                    "left_position": left["position"],
                    "left_intensity": left["intensity"],
                    "right_light_id": right_light_id,
                    "right_position": right["position"],
                    "right_intensity": right["intensity"],
                    "n_params": n_params,
                    "left_correct_count": len(left_correct),
                    "right_correct_count": len(right_correct),
                    "both_correct_count": len(both_correct),
                    "both_correct_accuracy": len(both_correct) / n_params if n_params else None,
                }
            )
    return output


def overlap_average(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["left_light_id"], row["right_light_id"])].append(row)

    output: list[dict[str, Any]] = []
    for (model, left_light_id, right_light_id), items in sorted(grouped.items()):
        n_objects = len(items)
        output.append(
            {
                "model": model,
                "left_light_id": left_light_id,
                "right_light_id": right_light_id,
                "n_objects": n_objects,
                "mean_left_correct_count": sum(float(item["left_correct_count"]) for item in items) / n_objects,
                "mean_right_correct_count": sum(float(item["right_correct_count"]) for item in items) / n_objects,
                "mean_both_correct_count": sum(float(item["both_correct_count"]) for item in items) / n_objects,
                "mean_both_correct_accuracy": sum(float(item["both_correct_accuracy"]) for item in items if item["both_correct_accuracy"] is not None) / n_objects,
            }
        )
    return output


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure lighting overlap of successful parameters.")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--exclude-auto-param", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    results_path = args.results.expanduser()
    output_dir = args.output_dir.expanduser() if args.output_dir else default_analysis_dir(results_path, "lighting_overlap")
    rows = maybe_exclude_auto(read_jsonl(results_path), args.exclude_auto_param)
    by_object = overlap_by_object(rows)
    average = overlap_average(by_object)
    write_csv(
        output_dir / "lighting_overlap_by_object.csv",
        [
            "model",
            "object_key",
            "left_light_id",
            "left_position",
            "left_intensity",
            "right_light_id",
            "right_position",
            "right_intensity",
            "n_params",
            "left_correct_count",
            "right_correct_count",
            "both_correct_count",
            "both_correct_accuracy",
        ],
        by_object,
    )
    write_csv(
        output_dir / "lighting_overlap_average.csv",
        [
            "model",
            "left_light_id",
            "right_light_id",
            "n_objects",
            "mean_left_correct_count",
            "mean_right_correct_count",
            "mean_both_correct_count",
            "mean_both_correct_accuracy",
        ],
        average,
    )
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
