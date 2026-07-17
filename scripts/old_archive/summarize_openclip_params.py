#!/usr/bin/env python3
"""Summarize per-image OpenCLIP downstream results by lighting and parameter.

Input is the sample_results.jsonl produced by evaluate_lighting_dog_accuracy.py.
The script does not rerun models; it aggregates existing per-image predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_DATASET_ROOT = Path("dataset/test/cropped")
DEFAULT_SAMPLE_RESULTS = DEFAULT_DATASET_ROOT / "dog_diagnostics" / "sample_results.jsonl"
DEFAULT_OUTPUT_DIR = DEFAULT_DATASET_ROOT / "dog_diagnostics" / "openclip_param_summary"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required CSV: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required JSONL: {path}")
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean(values: Iterable[float]) -> float | None:
    items = list(values)
    if not items:
        return None
    return sum(items) / len(items)


def as_int(value: str | int | None) -> int | None:
    if value in {None, ""}:
        return None
    return int(value)


def as_float(value: str | float | None) -> float | None:
    if value in {None, ""}:
        return None
    return float(value)


def load_param_meta(dataset_root: Path) -> dict[str, dict[str, str]]:
    return {
        row["param_id"]: row
        for row in read_csv_rows(dataset_root / "maps" / "params.csv")
    }


def load_light_meta(dataset_root: Path) -> dict[str, dict[str, str]]:
    return {
        row["light_id"]: row
        for row in read_csv_rows(dataset_root / "maps" / "lights.csv")
    }


def openclip_rows(
    sample_results_path: Path,
    param_meta: dict[str, dict[str, str]],
    light_meta: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in read_jsonl(sample_results_path):
        prediction = record.get("openclip")
        if not record.get("exists") or prediction is None:
            continue

        param_id = str(record.get("param_id", ""))
        light_id = str(record.get("light_id", ""))
        param = param_meta.get(param_id, {})
        light = light_meta.get(light_id, {})
        rows.append(
            {
                "image_path": record.get("image_path", ""),
                "class_id": record.get("class_id", ""),
                "sample_id": record.get("sample_id", ""),
                "light_id": light_id,
                "light_folder": light.get("light_folder", ""),
                "intensity": as_int(record.get("intensity")),
                "light_percent": as_float(record.get("light_percent")),
                "cct": as_int(record.get("cct")),
                "view_id": record.get("view_id", ""),
                "param_id": param_id,
                "param_file": param.get("param_file", ""),
                "aperture": as_float(param.get("aperture")),
                "iso": as_int(param.get("iso")),
                "shutter_speed": param.get("shutter_speed", ""),
                "top1_is_dog": bool(prediction.get("top1_is_dog")),
                "top5_has_dog": bool(prediction.get("top5_has_dog")),
                "top1_label": prediction.get("top1_label", ""),
                "top1_similarity": as_float(prediction.get("top1_similarity")),
                "top5_labels": "|".join(prediction.get("top5_labels", [])),
            }
        )
    return rows


def aggregate_by_param(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["param_id"]].append(row)

    output: list[dict[str, Any]] = []
    for param_id, items in grouped.items():
        first = items[0]
        top1_count = sum(1 for item in items if item["top1_is_dog"])
        top5_count = sum(1 for item in items if item["top5_has_dog"])
        output.append(
            {
                "param_id": param_id,
                "param_file": first["param_file"],
                "aperture": first["aperture"],
                "iso": first["iso"],
                "shutter_speed": first["shutter_speed"],
                "n": len(items),
                "top1_count": top1_count,
                "top1_accuracy": top1_count / len(items),
                "top5_count": top5_count,
                "top5_accuracy": top5_count / len(items),
                "mean_top1_similarity": mean(
                    item["top1_similarity"]
                    for item in items
                    if item["top1_similarity"] is not None
                ),
                "dog_top1_intensities": "|".join(
                    str(item["intensity"]) for item in items if item["top1_is_dog"]
                ),
                "dog_top5_intensities": "|".join(
                    str(item["intensity"]) for item in items if item["top5_has_dog"]
                ),
            }
        )

    return sorted(
        output,
        key=lambda item: (
            -item["top5_accuracy"],
            -item["top1_accuracy"],
            -(item["mean_top1_similarity"] or -999.0),
            float(item["aperture"]),
            int(item["iso"]),
            item["shutter_speed"],
        ),
    )


def aggregate_by_exposure(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["aperture"], row["iso"], row["shutter_speed"])].append(row)

    output: list[dict[str, Any]] = []
    for (aperture, iso, shutter_speed), items in grouped.items():
        param_ids = sorted({int(item["param_id"]) for item in items})
        top1_count = sum(1 for item in items if item["top1_is_dog"])
        top5_count = sum(1 for item in items if item["top5_has_dog"])
        output.append(
            {
                "aperture": aperture,
                "iso": iso,
                "shutter_speed": shutter_speed,
                "param_ids": "|".join(str(value) for value in param_ids),
                "n": len(items),
                "top1_count": top1_count,
                "top1_accuracy": top1_count / len(items),
                "top5_count": top5_count,
                "top5_accuracy": top5_count / len(items),
                "mean_top1_similarity": mean(
                    item["top1_similarity"]
                    for item in items
                    if item["top1_similarity"] is not None
                ),
            }
        )

    return sorted(
        output,
        key=lambda item: (
            -item["top5_accuracy"],
            -item["top1_accuracy"],
            -(item["mean_top1_similarity"] or -999.0),
            float(item["aperture"]),
            int(item["iso"]),
            item["shutter_speed"],
        ),
    )


def summarize_payload(rows: list[dict[str, Any]], by_param: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n_openclip_images": len(rows),
        "n_params": len(by_param),
        "top1_accuracy": mean(1.0 if row["top1_is_dog"] else 0.0 for row in rows),
        "top5_accuracy": mean(1.0 if row["top5_has_dog"] else 0.0 for row in rows),
        "params_with_zero_top5": sum(1 for row in by_param if row["top5_count"] == 0),
        "params_with_zero_top1": sum(1 for row in by_param if row["top1_count"] == 0),
        "best_params_by_top5": by_param[:20],
        "worst_params_by_top5": sorted(
            by_param,
            key=lambda item: (
                item["top5_accuracy"],
                item["top1_accuracy"],
                item["mean_top1_similarity"] or -999.0,
            ),
        )[:20],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate existing OpenCLIP results by lighting and capture parameter."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--sample-results", type=Path, default=DEFAULT_SAMPLE_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    param_meta = load_param_meta(args.dataset_root)
    light_meta = load_light_meta(args.dataset_root)
    rows = openclip_rows(args.sample_results, param_meta, light_meta)
    by_param = aggregate_by_param(rows)
    by_exposure = aggregate_by_exposure(rows)

    light_param_fields = [
        "image_path",
        "class_id",
        "sample_id",
        "light_id",
        "light_folder",
        "intensity",
        "light_percent",
        "cct",
        "view_id",
        "param_id",
        "param_file",
        "aperture",
        "iso",
        "shutter_speed",
        "top1_is_dog",
        "top5_has_dog",
        "top1_label",
        "top1_similarity",
        "top5_labels",
    ]
    param_fields = [
        "param_id",
        "param_file",
        "aperture",
        "iso",
        "shutter_speed",
        "n",
        "top1_count",
        "top1_accuracy",
        "top5_count",
        "top5_accuracy",
        "mean_top1_similarity",
        "dog_top1_intensities",
        "dog_top5_intensities",
    ]
    exposure_fields = [
        "aperture",
        "iso",
        "shutter_speed",
        "param_ids",
        "n",
        "top1_count",
        "top1_accuracy",
        "top5_count",
        "top5_accuracy",
        "mean_top1_similarity",
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "openclip_by_light_param.csv", light_param_fields, rows)
    write_csv(args.output_dir / "openclip_by_param.csv", param_fields, by_param)
    write_csv(args.output_dir / "openclip_by_exposure.csv", exposure_fields, by_exposure)
    summary = summarize_payload(rows, by_param)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
