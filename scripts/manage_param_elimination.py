#!/usr/bin/env python3
"""Maintain an iterative parameter-elimination view of a cropped dataset.

Images stay in the parent cropped dataset, and clear/maps/images.jsonl points
to them with ../ relative paths. Elimination state, maps, and stats are JSONL.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


DEFAULT_DATASET_ROOT = Path("dataset/test/cropped")
DEFAULT_CLEAR_ROOT = DEFAULT_DATASET_ROOT / "clear"
PARAMETER_NAMES = ["aperture", "iso", "shutter_speed"]
DEFAULT_ISO_SIMILARITY_THRESHOLD = 0.98


def read_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required CSV: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def write_csv_rows(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalize_value(value: Any) -> Any:
    if value == "":
        return None
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if not isinstance(value, str):
        return value
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = {key: normalize_value(value) for key, value in row.items()}
    for key in ["dog_top1_intensities", "dog_top5_intensities", "param_ids"]:
        value = row.get(key)
        if isinstance(value, str):
            normalized[key] = [normalize_value(item) for item in value.split("|") if item]
    return normalized


def sort_value(value: Any) -> tuple[int, float | str]:
    if isinstance(value, (int, float)):
        return (0, float(value))
    try:
        return (0, float(value))
    except (TypeError, ValueError):
        return (1, str(value))


def pair_sort_key(values: tuple[Any, ...]) -> tuple[tuple[int, float | str], ...]:
    return tuple(sort_value(value) for value in values)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(normalize_row(row), sort_keys=True) + "\n")


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required JSONL: {path}")
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def parse_param_ids(value: str | None) -> set[str]:
    if not value:
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


def now_timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S%z")


def sample_results_default(dataset_root: Path) -> Path:
    return dataset_root / "dog_diagnostics" / "sample_results.jsonl"


def status_path(clear_root: Path) -> Path:
    return clear_root / "param_status.jsonl"


def history_path(clear_root: Path) -> Path:
    return clear_root / "elimination_history.jsonl"


def mean(values: Iterable[float]) -> float | None:
    items = list(values)
    if not items:
        return None
    return sum(items) / len(items)


def copy_static_maps(dataset_root: Path, clear_root: Path) -> None:
    input_maps = dataset_root / "maps"
    output_maps = clear_root / "maps"
    output_maps.mkdir(parents=True, exist_ok=True)
    for name in ["classes.csv", "lights.csv", "samples.csv", "views.csv"]:
        source = input_maps / name
        if source.exists():
            rows, _ = read_csv_rows(source)
            write_jsonl(output_maps / f"{source.stem}.jsonl", rows)


def load_param_meta(dataset_root: Path) -> dict[str, dict[str, str]]:
    rows, _ = read_csv_rows(dataset_root / "maps" / "params.csv")
    return {row["param_id"]: row for row in rows}


def normalize_relative_image_path(value: Any) -> str:
    path = str(value)
    while path.startswith("../"):
        path = path[3:]
    return path


def build_initial_status_rows(
    dataset_root: Path,
    sample_results_path: Path,
) -> list[dict[str, Any]]:
    param_meta = load_param_meta(dataset_root)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for record in read_jsonl(sample_results_path):
        prediction = record.get("openclip")
        if not record.get("exists") or prediction is None:
            continue
        grouped[str(record["param_id"])].append(record)

    rows: list[dict[str, Any]] = []
    for param_id in sorted(grouped, key=lambda value: int(value)):
        records = grouped[param_id]
        meta = param_meta.get(param_id)
        if meta is None:
            raise KeyError(f"param_id {param_id} is missing from maps/params.csv")

        top1_records = [
            record for record in records if record["openclip"].get("top1_is_dog")
        ]
        top5_records = [
            record for record in records if record["openclip"].get("top5_has_dog")
        ]
        similarities = [
            float(record["openclip"]["top1_similarity"])
            for record in records
            if record["openclip"].get("top1_similarity") is not None
        ]
        rows.append(
            {
                "param_id": param_id,
                "param_file": meta.get("param_file", ""),
                "aperture": meta.get("aperture", ""),
                "iso": meta.get("iso", ""),
                "shutter_speed": meta.get("shutter_speed", ""),
                "status": "kept",
                "n": len(records),
                "top1_correct_count": len(top1_records),
                "top1_accuracy": len(top1_records) / len(records),
                "top5_correct_count": len(top5_records),
                "top5_accuracy": len(top5_records) / len(records),
                "mean_top1_similarity": mean(similarities),
                "dog_top1_intensities": "|".join(
                    str(record.get("intensity")) for record in top1_records
                ),
                "dog_top5_intensities": "|".join(
                    str(record.get("intensity")) for record in top5_records
                ),
                "deleted_round": "",
                "deleted_reason": "",
                "deleted_at": "",
            }
        )
    return rows


def read_status_rows(clear_root: Path) -> list[dict[str, Any]]:
    return list(read_jsonl(status_path(clear_root)))


def kept_param_ids(status_rows: Iterable[dict[str, Any]]) -> set[str]:
    return {str(row["param_id"]) for row in status_rows if row.get("status") == "kept"}


def rewrite_clear_maps(
    dataset_root: Path,
    clear_root: Path,
    status_rows: list[dict[str, Any]],
) -> dict[str, int]:
    kept_ids = kept_param_ids(status_rows)

    image_rows, _ = read_csv_rows(dataset_root / "maps" / "images.csv")
    kept_image_rows: list[dict[str, str]] = []
    missing_image_rows: list[dict[str, str]] = []
    for row in image_rows:
        if str(row.get("param_id")) not in kept_ids:
            continue
        output_row = dict(row)
        output_row["image_path"] = str(Path("..") / row["image_path"])
        if not (clear_root / output_row["image_path"]).exists():
            missing_image_rows.append(output_row)
            continue
        kept_image_rows.append(output_row)

    param_rows, _ = read_csv_rows(dataset_root / "maps" / "params.csv")
    kept_param_rows = [row for row in param_rows if str(row.get("param_id")) in kept_ids]

    write_jsonl(clear_root / "maps" / "images.jsonl", kept_image_rows)
    write_jsonl(clear_root / "maps" / "params.jsonl", kept_param_rows)
    copy_static_maps(dataset_root, clear_root)
    return {
        "kept_params": len(kept_ids),
        "kept_images": len(kept_image_rows),
        "missing_images_omitted": len(missing_image_rows),
    }


def current_image_keys(clear_root: Path) -> set[tuple[str, str]]:
    path = clear_root / "maps" / "images.jsonl"
    if not path.exists():
        return set()
    return {
        (str(row["param_id"]), normalize_relative_image_path(row["image_path"]))
        for row in read_jsonl(path)
    }


def write_status_files(clear_root: Path, status_rows: list[dict[str, Any]]) -> None:
    sorted_rows = sorted(status_rows, key=lambda row: int(row["param_id"]))
    write_jsonl(status_path(clear_root), sorted_rows)


def current_openclip_rows(
    dataset_root: Path,
    clear_root: Path,
    sample_results_path: Path,
) -> list[dict[str, Any]]:
    param_meta = load_param_meta(dataset_root)
    keys = current_image_keys(clear_root)
    rows: list[dict[str, Any]] = []

    for record in read_jsonl(sample_results_path):
        prediction = record.get("openclip")
        key = (str(record.get("param_id")), normalize_relative_image_path(record.get("image_path")))
        if not record.get("exists") or prediction is None or key not in keys:
            continue

        param = param_meta[str(record["param_id"])]
        rows.append(
            {
                "image_path": record["image_path"],
                "param_id": int(record["param_id"]),
                "param_file": param.get("param_file", ""),
                "aperture": normalize_value(param.get("aperture", "")),
                "iso": normalize_value(param.get("iso", "")),
                "shutter_speed": param.get("shutter_speed", ""),
                "light_id": normalize_value(record.get("light_id")),
                "intensity": normalize_value(record.get("intensity")),
                "light_percent": normalize_value(record.get("light_percent")),
                "cct": normalize_value(record.get("cct")),
                "top1_correct": bool(prediction.get("top1_is_dog")),
                "top5_correct": bool(prediction.get("top5_has_dog")),
                "top1_label": prediction.get("top1_label", ""),
                "top1_similarity": normalize_value(prediction.get("top1_similarity")),
            }
        )
    return rows


def aggregate_correctness(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    top1 = sum(1 for row in rows if row["top1_correct"])
    top5 = sum(1 for row in rows if row["top5_correct"])
    return {
        "n": n,
        "top1_correct_count": top1,
        "top1_accuracy": top1 / n if n else None,
        "top5_correct_count": top5,
        "top5_accuracy": top5 / n if n else None,
    }


def write_combo_lighting_stats(stats_dir: Path, rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["param_id"]), int(row["light_id"]))].append(row)

    output_rows: list[dict[str, Any]] = []
    for (param_id, light_id), items in grouped.items():
        first = items[0]
        output_rows.append(
            {
                "param_id": param_id,
                "param_file": first["param_file"],
                "aperture": first["aperture"],
                "iso": first["iso"],
                "shutter_speed": first["shutter_speed"],
                "light_id": light_id,
                "intensity": first["intensity"],
                "light_percent": first["light_percent"],
                "cct": first["cct"],
                **aggregate_correctness(items),
            }
        )

    write_jsonl(
        stats_dir / "by_combo_lighting_correct_counts.jsonl",
        sorted(
            output_rows,
            key=lambda row: (
                int(row["param_id"]),
                int(row["light_id"]),
            ),
        ),
    )


def combo_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["param_id"])].append(row)

    output_rows: list[dict[str, Any]] = []
    for param_id, items in grouped.items():
        first = items[0]
        lighting = sorted(
            (
                {
                    "light_id": item["light_id"],
                    "intensity": item["intensity"],
                    "top1_correct": item["top1_correct"],
                    "top5_correct": item["top5_correct"],
                }
                for item in items
            ),
            key=lambda item: int(item["light_id"]),
        )
        counts = aggregate_correctness(items)
        output_rows.append(
            {
                "param_id": param_id,
                "param_file": first["param_file"],
                "aperture": first["aperture"],
                "iso": first["iso"],
                "shutter_speed": first["shutter_speed"],
                "n_lighting": counts["n"],
                "top1_correct_count": counts["top1_correct_count"],
                "top1_correct_over": f"{counts['top1_correct_count']}/{counts['n']}",
                "top1_accuracy": counts["top1_accuracy"],
                "top5_correct_count": counts["top5_correct_count"],
                "top5_correct_over": f"{counts['top5_correct_count']}/{counts['n']}",
                "top5_accuracy": counts["top5_accuracy"],
                "lighting": lighting,
            }
        )
    return sorted(output_rows, key=lambda row: int(row["param_id"]))


def build_single_param_matrices(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    combos = combo_summaries(rows)
    output_rows: list[dict[str, Any]] = []
    parameter_order = {name: index for index, name in enumerate(PARAMETER_NAMES)}

    for parameter_name in PARAMETER_NAMES:
        other_names = [name for name in PARAMETER_NAMES if name != parameter_name]
        other_value_tuples = sorted(
            {
                tuple(combo[name] for name in other_names)
                for combo in combos
            },
            key=pair_sort_key,
        )
        other_combo_index = {
            values: index
            for index, values in enumerate(other_value_tuples)
        }

        grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
        for combo in combos:
            grouped[combo[parameter_name]].append(combo)

        for parameter_value, value_combos in grouped.items():
            cells: list[dict[str, Any]] = []
            by_other_values = {
                tuple(combo[name] for name in other_names): combo
                for combo in value_combos
            }
            for values in other_value_tuples:
                combo = by_other_values.get(values)
                if combo is None:
                    cells.append(
                        {
                            "other_combo_index": other_combo_index[values],
                            "paired_with": dict(zip(other_names, values)),
                            "param_id": None,
                            "missing": True,
                            "top1_correct_count": 0,
                            "top5_correct_count": 0,
                        }
                    )
                    continue

                cells.append(
                    {
                        "other_combo_index": other_combo_index[values],
                        "paired_with": dict(zip(other_names, values)),
                        "param_id": combo["param_id"],
                        "param_file": combo["param_file"],
                        "missing": False,
                        "n_lighting": combo["n_lighting"],
                        "top1_correct_count": combo["top1_correct_count"],
                        "top1_correct_over": combo["top1_correct_over"],
                        "top5_correct_count": combo["top5_correct_count"],
                        "top5_correct_over": combo["top5_correct_over"],
                        "lighting": combo["lighting"],
                    }
                )

            top1_vector = [int(cell["top1_correct_count"]) for cell in cells]
            top5_vector = [int(cell["top5_correct_count"]) for cell in cells]
            output_rows.append(
                {
                    "parameter_name": parameter_name,
                    "parameter_value": parameter_value,
                    "other_parameter_names": other_names,
                    "other_combo_index": [
                        {
                            "other_combo_index": index,
                            "paired_with": dict(zip(other_names, values)),
                        }
                        for values, index in sorted(
                            other_combo_index.items(),
                            key=lambda item: item[1],
                        )
                    ],
                    "cell_count": len(cells),
                    "present_cell_count": sum(1 for cell in cells if not cell["missing"]),
                    "total_top1_correct_count": sum(top1_vector),
                    "max_top1_correct_count": max(top1_vector, default=0),
                    "top1_vector": top1_vector,
                    "total_top5_correct_count": sum(top5_vector),
                    "max_top5_correct_count": max(top5_vector, default=0),
                    "top5_vector": top5_vector,
                    "cells": cells,
                }
            )

    return sorted(
        output_rows,
        key=lambda row: (
            parameter_order[row["parameter_name"]],
            sort_value(row["parameter_value"]),
        ),
    )


def cosine_similarity(left: list[int | float], right: list[int | float]) -> float:
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = sum(float(value) ** 2 for value in left) ** 0.5
    right_norm = sum(float(value) ** 2 for value in right) ** 0.5
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def build_single_param_similarity(
    matrices: list[dict[str, Any]],
    parameter_name: str = "iso",
    threshold: float = DEFAULT_ISO_SIMILARITY_THRESHOLD,
) -> list[dict[str, Any]]:
    selected = [
        row
        for row in matrices
        if row["parameter_name"] == parameter_name
    ]
    output_rows: list[dict[str, Any]] = []
    for left_index, left in enumerate(selected):
        for right in selected[left_index + 1 :]:
            score = cosine_similarity(left["top1_vector"], right["top1_vector"])
            left_total = int(left["total_top1_correct_count"])
            right_total = int(right["total_top1_correct_count"])
            if left_total < right_total:
                delete_value = left["parameter_value"]
                keep_value = right["parameter_value"]
            elif right_total < left_total:
                delete_value = right["parameter_value"]
                keep_value = left["parameter_value"]
            else:
                values = sorted(
                    [left["parameter_value"], right["parameter_value"]],
                    key=sort_value,
                )
                keep_value = values[0]
                delete_value = values[-1]

            output_rows.append(
                {
                    "parameter_name": parameter_name,
                    "left_value": left["parameter_value"],
                    "right_value": right["parameter_value"],
                    "cosine_similarity_top1": score,
                    "threshold": threshold,
                    "is_similar": score >= threshold,
                    "left_total_top1_correct_count": left_total,
                    "right_total_top1_correct_count": right_total,
                    "recommended_keep_value": keep_value,
                    "recommended_delete_value": delete_value,
                }
            )
    return sorted(
        output_rows,
        key=lambda row: (
            not row["is_similar"],
            -row["cosine_similarity_top1"],
            sort_value(row["left_value"]),
            sort_value(row["right_value"]),
        ),
    )


def build_single_param_status(
    status_rows: list[dict[str, Any]],
    matrices: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    matrix_by_key = {
        (row["parameter_name"], str(row["parameter_value"])): row
        for row in matrices
    }
    output_rows: list[dict[str, Any]] = []
    for parameter_name in PARAMETER_NAMES:
        values = sorted(
            {
                row[parameter_name]
                for row in status_rows
            },
            key=sort_value,
        )
        for value in values:
            related = [
                row
                for row in status_rows
                if row[parameter_name] == value
            ]
            kept = [row for row in related if row["status"] == "kept"]
            matrix = matrix_by_key.get((parameter_name, str(value)))
            deleted_rows = [row for row in related if row["status"] == "deleted"]
            output_rows.append(
                {
                    "parameter_name": parameter_name,
                    "parameter_value": value,
                    "status": "kept" if kept else "deleted",
                    "total_combo_count": len(related),
                    "kept_combo_count": len(kept),
                    "deleted_combo_count": len(deleted_rows),
                    "current_cell_count": matrix["cell_count"] if matrix else 0,
                    "current_present_cell_count": matrix["present_cell_count"] if matrix else 0,
                    "total_top1_correct_count": matrix["total_top1_correct_count"] if matrix else 0,
                    "max_top1_correct_count": matrix["max_top1_correct_count"] if matrix else 0,
                    "never_win_top1": bool(matrix and matrix["max_top1_correct_count"] == 0),
                    "deleted_rounds": sorted(
                        {
                            int(row["deleted_round"])
                            for row in deleted_rows
                            if row.get("deleted_round") not in {None, ""}
                        }
                    ),
                    "deleted_reasons": sorted(
                        {
                            row["deleted_reason"]
                            for row in deleted_rows
                            if row.get("deleted_reason")
                        }
                    ),
                }
            )
    parameter_order = {name: index for index, name in enumerate(PARAMETER_NAMES)}
    return sorted(
        output_rows,
        key=lambda row: (
            parameter_order[row["parameter_name"]],
            sort_value(row["parameter_value"]),
        ),
    )


def write_stats(
    dataset_root: Path,
    clear_root: Path,
    status_rows: list[dict[str, Any]],
    sample_results_path: Path,
) -> None:
    stats_dir = clear_root / "stats"
    current_rows = current_openclip_rows(dataset_root, clear_root, sample_results_path)
    write_combo_lighting_stats(stats_dir, current_rows)
    matrices = build_single_param_matrices(current_rows)
    write_jsonl(stats_dir / "single_param_matrices.jsonl", matrices)
    write_jsonl(
        stats_dir / "single_param_similarity.jsonl",
        build_single_param_similarity(matrices),
    )
    write_jsonl(
        clear_root / "single_param_status.jsonl",
        build_single_param_status(status_rows, matrices),
    )


def current_round(clear_root: Path) -> int:
    path = history_path(clear_root)
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def append_history(clear_root: Path, payload: dict[str, Any]) -> None:
    path = history_path(clear_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def init_command(args: argparse.Namespace) -> None:
    dataset_root = args.dataset_root.expanduser()
    clear_root = args.clear_root.expanduser()
    sample_results = (
        args.sample_results.expanduser()
        if args.sample_results is not None
        else sample_results_default(dataset_root)
    )

    if clear_root.exists() and any(clear_root.iterdir()):
        if not args.force:
            raise FileExistsError(
                f"{clear_root} already exists and is not empty. Use --force to rebuild it."
            )
        shutil.rmtree(clear_root)
    clear_root.mkdir(parents=True, exist_ok=True)

    status_rows = build_initial_status_rows(dataset_root, sample_results)
    write_status_files(clear_root, status_rows)
    map_counts = rewrite_clear_maps(dataset_root, clear_root, status_rows)
    write_stats(dataset_root, clear_root, status_rows, sample_results)

    summary = {
        "command": "init",
        "dataset_root": str(dataset_root),
        "sample_results": str(sample_results),
        "clear_root": str(clear_root),
        "params": len(status_rows),
        **map_counts,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


def status_command(args: argparse.Namespace) -> None:
    clear_root = args.clear_root.expanduser()
    rows = read_status_rows(clear_root)
    kept = [row for row in rows if row.get("status") == "kept"]
    deleted = [row for row in rows if row.get("status") == "deleted"]
    image_rows = list(read_jsonl(clear_root / "maps" / "images.jsonl"))
    missing = sum(1 for row in image_rows if not (clear_root / row["image_path"]).exists())
    payload = {
        "clear_root": str(clear_root),
        "total_params": len(rows),
        "kept_params": len(kept),
        "deleted_params": len(deleted),
        "kept_images": len(image_rows),
        "missing_images_in_current_maps": missing,
        "rounds": current_round(clear_root),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def should_delete(
    row: dict[str, Any],
    explicit_ids: set[str],
    max_top1: int | None,
    max_top5: int | None,
) -> bool:
    if row.get("status") != "kept":
        return False
    if str(row["param_id"]) in explicit_ids:
        return True
    if max_top1 is not None and int(row["top1_correct_count"]) <= max_top1:
        return True
    if max_top5 is not None and int(row["top5_correct_count"]) <= max_top5:
        return True
    return False


def delete_matching_single_param_values(
    status_rows: list[dict[str, Any]],
    values_to_delete: set[tuple[str, str]],
    round_id: int,
    reason: str,
    deleted_at: str,
) -> list[dict[str, Any]]:
    deleted: list[dict[str, Any]] = []
    for row in status_rows:
        if row.get("status") != "kept":
            continue
        for parameter_name, parameter_value in values_to_delete:
            if str(row[parameter_name]) == parameter_value:
                row["status"] = "deleted"
                row["deleted_round"] = str(round_id)
                row["deleted_reason"] = reason
                row["deleted_at"] = deleted_at
                deleted.append(row)
                break
    return deleted


def finalize_elimination_round(
    dataset_root: Path,
    clear_root: Path,
    sample_results: Path,
    status_rows: list[dict[str, Any]],
    round_id: int,
    deleted_at: str,
    reason: str,
    deleted: list[dict[str, Any]],
    extra_history: dict[str, Any],
) -> dict[str, Any]:
    write_status_files(clear_root, status_rows)
    map_counts = rewrite_clear_maps(dataset_root, clear_root, status_rows)
    write_stats(dataset_root, clear_root, status_rows, sample_results)

    history = {
        "round": round_id,
        "deleted_at": deleted_at,
        "reason": reason,
        "sample_results": str(sample_results),
        "deleted_count": len(deleted),
        "deleted_param_ids": sorted({int(row["param_id"]) for row in deleted}),
        **map_counts,
        **extra_history,
    }
    append_history(clear_root, history)
    print(json.dumps(history, indent=2, sort_keys=True))
    return history


def eliminate_command(args: argparse.Namespace) -> None:
    clear_root = args.clear_root.expanduser()
    dataset_root = args.dataset_root.expanduser()
    sample_results = (
        args.sample_results.expanduser()
        if args.sample_results is not None
        else sample_results_default(dataset_root)
    )
    rows = read_status_rows(clear_root)
    explicit_ids = parse_param_ids(args.param_ids)
    if not explicit_ids and args.max_top1_correct_count is None and args.max_top5_correct_count is None:
        raise ValueError(
            "Provide --param-ids, --max-top1-correct-count, or --max-top5-correct-count."
        )

    round_id = current_round(clear_root) + 1
    deleted_at = now_timestamp()
    deleted: list[dict[str, Any]] = []
    for row in rows:
        if should_delete(
            row,
            explicit_ids,
            args.max_top1_correct_count,
            args.max_top5_correct_count,
        ):
            row["status"] = "deleted"
            row["deleted_round"] = str(round_id)
            row["deleted_reason"] = args.reason
            row["deleted_at"] = deleted_at
            deleted.append(row)

    finalize_elimination_round(
        dataset_root=dataset_root,
        clear_root=clear_root,
        sample_results=sample_results,
        status_rows=rows,
        round_id=round_id,
        deleted_at=deleted_at,
        reason=args.reason,
        deleted=deleted,
        extra_history={
            "param_ids_argument": sorted(explicit_ids, key=lambda value: int(value)),
            "max_top1_correct_count": args.max_top1_correct_count,
            "max_top5_correct_count": args.max_top5_correct_count,
        },
    )


def eliminate_never_win_command(args: argparse.Namespace) -> None:
    clear_root = args.clear_root.expanduser()
    dataset_root = args.dataset_root.expanduser()
    sample_results = (
        args.sample_results.expanduser()
        if args.sample_results is not None
        else sample_results_default(dataset_root)
    )
    rows = read_status_rows(clear_root)
    current_rows = current_openclip_rows(dataset_root, clear_root, sample_results)
    matrices = build_single_param_matrices(current_rows)
    candidates = [
        row
        for row in matrices
        if row["max_top1_correct_count"] == 0
    ]
    values_to_delete = {
        (row["parameter_name"], str(row["parameter_value"]))
        for row in candidates
    }

    round_id = current_round(clear_root) + 1
    deleted_at = now_timestamp()
    reason = args.reason or "single_param_never_top1_correct"
    deleted = delete_matching_single_param_values(
        rows,
        values_to_delete,
        round_id,
        reason,
        deleted_at,
    )
    finalize_elimination_round(
        dataset_root=dataset_root,
        clear_root=clear_root,
        sample_results=sample_results,
        status_rows=rows,
        round_id=round_id,
        deleted_at=deleted_at,
        reason=reason,
        deleted=deleted,
        extra_history={
            "elimination_type": "single_param_never_win",
            "correct_metric": "top1",
            "candidate_single_params": [
                {
                    "parameter_name": row["parameter_name"],
                    "parameter_value": row["parameter_value"],
                    "cell_count": row["cell_count"],
                }
                for row in candidates
            ],
            "deleted_single_params": [
                {
                    "parameter_name": name,
                    "parameter_value": normalize_value(value),
                }
                for name, value in sorted(values_to_delete)
            ],
        },
    )


def eliminate_similar_iso_command(args: argparse.Namespace) -> None:
    clear_root = args.clear_root.expanduser()
    dataset_root = args.dataset_root.expanduser()
    sample_results = (
        args.sample_results.expanduser()
        if args.sample_results is not None
        else sample_results_default(dataset_root)
    )
    rows = read_status_rows(clear_root)
    current_rows = current_openclip_rows(dataset_root, clear_root, sample_results)
    matrices = build_single_param_matrices(current_rows)
    similarities = build_single_param_similarity(
        matrices,
        parameter_name="iso",
        threshold=args.cosine_threshold,
    )
    similar_pairs = [row for row in similarities if row["is_similar"]]

    values_to_delete: set[tuple[str, str]] = set()
    selected_pairs: list[dict[str, Any]] = []
    for pair in similar_pairs:
        left_value = str(pair["left_value"])
        right_value = str(pair["right_value"])
        delete_value = str(pair["recommended_delete_value"])
        if ("iso", left_value) in values_to_delete or ("iso", right_value) in values_to_delete:
            continue
        values_to_delete.add(("iso", delete_value))
        selected_pairs.append(pair)

    round_id = current_round(clear_root) + 1
    deleted_at = now_timestamp()
    reason = args.reason or f"similar_iso_cosine_top1_gte_{args.cosine_threshold:g}"
    deleted = delete_matching_single_param_values(
        rows,
        values_to_delete,
        round_id,
        reason,
        deleted_at,
    )
    finalize_elimination_round(
        dataset_root=dataset_root,
        clear_root=clear_root,
        sample_results=sample_results,
        status_rows=rows,
        round_id=round_id,
        deleted_at=deleted_at,
        reason=reason,
        deleted=deleted,
        extra_history={
            "elimination_type": "similar_iso",
            "correct_metric": "top1",
            "similarity_metric": "cosine",
            "cosine_threshold": args.cosine_threshold,
            "similar_pairs": selected_pairs,
            "deleted_single_params": [
                {
                    "parameter_name": name,
                    "parameter_value": normalize_value(value),
                }
                for name, value in sorted(values_to_delete, key=lambda item: sort_value(item[1]))
            ],
        },
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Maintain clear/ param elimination state for cropped captures."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Initialize clear state.")
    init_parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    init_parser.add_argument("--clear-root", type=Path, default=DEFAULT_CLEAR_ROOT)
    init_parser.add_argument("--sample-results", type=Path)
    init_parser.add_argument("--force", action="store_true", help="Rebuild an existing clear root.")
    init_parser.set_defaults(func=init_command)

    status_parser = subparsers.add_parser("status", help="Show current clear state.")
    status_parser.add_argument("--clear-root", type=Path, default=DEFAULT_CLEAR_ROOT)
    status_parser.set_defaults(func=status_command)

    eliminate_parser = subparsers.add_parser("eliminate", help="Delete params from the kept set.")
    eliminate_parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    eliminate_parser.add_argument("--clear-root", type=Path, default=DEFAULT_CLEAR_ROOT)
    eliminate_parser.add_argument("--sample-results", type=Path)
    eliminate_parser.add_argument("--param-ids", help="Comma-separated param IDs to delete.")
    eliminate_parser.add_argument("--max-top1-correct-count", type=int)
    eliminate_parser.add_argument("--max-top5-correct-count", type=int)
    eliminate_parser.add_argument("--reason", required=True)
    eliminate_parser.set_defaults(func=eliminate_command)

    never_win_parser = subparsers.add_parser(
        "eliminate-never-win",
        help="Delete single parameter values that never have top1 correct in any current combo.",
    )
    never_win_parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    never_win_parser.add_argument("--clear-root", type=Path, default=DEFAULT_CLEAR_ROOT)
    never_win_parser.add_argument("--sample-results", type=Path)
    never_win_parser.add_argument("--reason")
    never_win_parser.set_defaults(func=eliminate_never_win_command)

    similar_iso_parser = subparsers.add_parser(
        "eliminate-similar-iso",
        help="Delete weaker iso values whose single-param matrices are highly similar.",
    )
    similar_iso_parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    similar_iso_parser.add_argument("--clear-root", type=Path, default=DEFAULT_CLEAR_ROOT)
    similar_iso_parser.add_argument("--sample-results", type=Path)
    similar_iso_parser.add_argument(
        "--cosine-threshold",
        type=float,
        default=DEFAULT_ISO_SIMILARITY_THRESHOLD,
    )
    similar_iso_parser.add_argument("--reason")
    similar_iso_parser.set_defaults(func=eliminate_similar_iso_command)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
