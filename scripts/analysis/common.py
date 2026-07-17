"""Shared helpers for downstream analysis scripts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable


DEFAULT_RESULTS = Path("dataset/test_ver3_cropped/downstream_eval/image_results.jsonl")
MODELS = ("resnet50", "openclip")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing results file: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}: {exc}") from exc
    return rows


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def default_analysis_dir(results_path: Path, analysis_name: str) -> Path:
    return results_path.expanduser().resolve().parent.parent / "analysis" / analysis_name


def int_sort_key(value: Any) -> tuple[int, str]:
    text = "" if value is None else str(value)
    return (0, f"{int(text):012d}") if text.isdigit() else (1, text)


def object_sort_key(row: dict[str, Any]) -> tuple[tuple[int, str], tuple[int, str], tuple[int, str]]:
    return (
        int_sort_key(row.get("class_id", "")),
        int_sort_key(row.get("sample_id", "")),
        int_sort_key(row.get("view_id", "")),
    )


def correct_field(model: str) -> str:
    return f"{model}_top1_correct"


def is_available_correct(value: Any) -> bool:
    return value is True or value is False


def maybe_exclude_auto(rows: list[dict[str, Any]], exclude_auto_param: bool) -> list[dict[str, Any]]:
    if not exclude_auto_param:
        return rows
    return [
        row
        for row in rows
        if str(row.get("param_id", "")) != "0" and row.get("exposure_mode") != "auto"
    ]
