"""Remove captured images and matching JSON/JSONL map records.

Dry-run by default. Add --apply to write changes.
No backup files are generated.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import time
from pathlib import Path
from typing import Any


PAD_WIDTH = 3


def format_id(prefix: str, value: int) -> str:
    return f"{prefix}{value:0{PAD_WIDTH}d}"


def timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove dataset images and matching maps/parameters.json + maps/captures.jsonl records."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--class-id", type=int)
    parser.add_argument("--sample-id", type=int)
    parser.add_argument("--light-id", type=int)
    parser.add_argument("--light-position")
    parser.add_argument("--intensity", type=int)
    parser.add_argument("--cct", type=int)
    parser.add_argument("--view-id", type=int)
    parser.add_argument("--param-id", type=int)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete files and rewrite maps. Without this, only prints a preview.",
    )
    return parser.parse_args()


def require_filter(args: argparse.Namespace) -> None:
    filters = [
        args.class_id,
        args.sample_id,
        args.light_id,
        args.light_position,
        args.intensity,
        args.cct,
        args.view_id,
        args.param_id,
    ]
    if all(value is None for value in filters):
        raise SystemExit("Refusing to remove everything. Pass at least one filter.")
    if args.sample_id is not None and args.class_id is None:
        raise SystemExit("--sample-id requires --class-id.")


def load_parameters(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Missing parameter map: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_captures(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return records


def record_matches(record: dict[str, Any], args: argparse.Namespace) -> bool:
    checks = [
        ("class_id", args.class_id),
        ("sample_id", args.sample_id),
        ("light_id", args.light_id),
        ("position", args.light_position),
        ("intensity", args.intensity),
        ("cct", args.cct),
        ("view_id", args.view_id),
        ("param_id", args.param_id),
    ]
    return all(expected is None or record.get(key) == expected for key, expected in checks)


def image_paths_from_records(
    output_dir: Path,
    records: list[dict[str, Any]],
) -> set[Path]:
    paths = set()
    for record in records:
        image_path = record.get("image_path")
        if isinstance(image_path, str) and image_path:
            paths.add(output_dir / image_path)
    return paths


def directory_paths_from_filters(output_dir: Path, args: argparse.Namespace) -> set[Path]:
    paths: set[Path] = set()
    light_dirs = (
        [output_dir / format_id("l", args.light_id)]
        if args.light_id is not None
        else sorted(output_dir.glob("l[0-9][0-9][0-9]"))
    )
    for light_dir in light_dirs:
        class_dirs = (
            [light_dir / format_id("c", args.class_id)]
            if args.class_id is not None
            else sorted(light_dir.glob("c[0-9][0-9][0-9]"))
        )
        for class_dir in class_dirs:
            if args.sample_id is not None:
                sample_dirs = [class_dir / format_id("s", args.sample_id)]
            else:
                sample_dirs = sorted(class_dir.glob("s[0-9][0-9][0-9]"))

            for sample_dir in sample_dirs:
                if args.view_id is None and args.param_id is None:
                    if args.class_id is not None and args.sample_id is None:
                        paths.add(class_dir)
                    elif args.sample_id is not None:
                        paths.add(sample_dir)
                    elif args.light_id is not None:
                        paths.add(light_dir)
                    continue

                view_dirs = (
                    [sample_dir / format_id("v", args.view_id)]
                    if args.view_id is not None
                    else sorted(sample_dir.glob("v[0-9][0-9][0-9]"))
                )
                for view_dir in view_dirs:
                    if args.param_id is None:
                        paths.add(view_dir)
                    else:
                        paths.add(view_dir / f"{format_id('p', args.param_id)}.jpg")
    return paths


def collapse_paths(paths: set[Path]) -> set[Path]:
    collapsed: set[Path] = set()
    for path in sorted(paths, key=lambda item: len(item.parts)):
        if any(parent in collapsed for parent in path.parents):
            continue
        collapsed.add(path)
    return collapsed


def item_id(item: dict[str, Any], key: str) -> int | None:
    value = item.get(key)
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def sample_light_ids(sample: dict[str, Any]) -> list[int]:
    light_ids = sample.get("light_ids")
    if isinstance(light_ids, list):
        return [int(value) for value in light_ids if isinstance(value, int) or str(value).isdigit()]
    light_id = sample.get("light_id")
    if isinstance(light_id, int):
        return [light_id]
    if isinstance(light_id, str) and light_id.isdigit():
        return [int(light_id)]
    return []


def capture_sample_keys(captures: list[dict[str, Any]]) -> set[tuple[int, int, int]]:
    return {
        (int(record["light_id"]), int(record["class_id"]), int(record["sample_id"]))
        for record in captures
        if "light_id" in record and "class_id" in record and "sample_id" in record
    }


def prune_parameters(
    parameters: dict[str, Any],
    remaining_captures: list[dict[str, Any]],
    removed_captures: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    used_classes = {int(record["class_id"]) for record in remaining_captures if "class_id" in record}
    used_samples = capture_sample_keys(remaining_captures)
    used_lights = {int(record["light_id"]) for record in remaining_captures if "light_id" in record}
    used_views = {int(record["view_id"]) for record in remaining_captures if "view_id" in record}
    used_params = {int(record["param_id"]) for record in remaining_captures if "param_id" in record}
    removed_classes = {int(record["class_id"]) for record in removed_captures if "class_id" in record}
    removed_samples = capture_sample_keys(removed_captures)
    removed_lights = {int(record["light_id"]) for record in removed_captures if "light_id" in record}
    removed_views = {int(record["view_id"]) for record in removed_captures if "view_id" in record}
    removed_params = {int(record["param_id"]) for record in removed_captures if "param_id" in record}
    if args.class_id is not None:
        removed_classes.add(args.class_id)
    if args.class_id is not None and args.sample_id is not None:
        if args.light_id is not None:
            removed_samples.add((args.light_id, args.class_id, args.sample_id))
    if args.light_id is not None:
        removed_lights.add(args.light_id)
    if args.view_id is not None:
        removed_views.add(args.view_id)
    if args.param_id is not None:
        removed_params.add(args.param_id)

    parameters["classes"] = [
        item
        for item in parameters.get("classes", [])
        if should_keep_item(item, "class_id", used_classes, removed_classes)
    ]
    parameters["samples"] = [
        updated_sample(item, remaining_captures)
        for item in parameters.get("samples", [])
        if should_keep_sample(item, used_samples, removed_samples)
    ]
    parameters["lights"] = [
        item
        for item in parameters.get("lights", [])
        if should_keep_item(item, "light_id", used_lights, removed_lights)
    ]
    parameters["views"] = [
        item
        for item in parameters.get("views", [])
        if should_keep_item(item, "view_id", used_views, removed_views)
    ]
    parameters["params"] = [
        item
        for item in parameters.get("params", [])
        if should_keep_item(item, "param_id", used_params, removed_params)
    ]
    parameters["updated_at"] = timestamp()
    return parameters


def should_keep_item(
    item: dict[str, Any],
    id_key: str,
    used_ids: set[int],
    removed_ids: set[int],
) -> bool:
    value = item_id(item, id_key)
    return value in used_ids or value not in removed_ids


def should_keep_sample(
    sample: dict[str, Any],
    used_samples: set[tuple[int, int, int]],
    removed_samples: set[tuple[int, int, int]],
) -> bool:
    class_id = item_id(sample, "class_id")
    sample_id = item_id(sample, "sample_id")
    light_ids = sample_light_ids(sample)
    if not light_ids:
        return True
    keys = {(light_id, class_id, sample_id) for light_id in light_ids}
    return bool(keys & used_samples) or not bool(keys & removed_samples)


def updated_sample(
    sample: dict[str, Any],
    remaining_captures: list[dict[str, Any]],
) -> dict[str, Any]:
    class_id = item_id(sample, "class_id")
    sample_id = item_id(sample, "sample_id")
    light_ids = set(sample_light_ids(sample))
    total_captures = sum(
        1
        for record in remaining_captures
        if record.get("class_id") == class_id and record.get("sample_id") == sample_id
        and (not light_ids or record.get("light_id") in light_ids)
    )
    sample = dict(sample)
    sample["total_captures"] = total_captures
    return sample


def write_parameters(path: Path, parameters: dict[str, Any]) -> None:
    path.write_text(json.dumps(parameters, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_captures(path: Path, captures: list[dict[str, Any]]) -> None:
    lines = [json.dumps(record, ensure_ascii=True, sort_keys=True) for record in captures]
    path.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")


def map_change_summary(
    before: dict[str, Any],
    after: dict[str, Any],
) -> list[str]:
    lines = []
    for key in ["classes", "samples", "lights", "views", "params"]:
        before_count = len(before.get(key, []))
        after_count = len(after.get(key, []))
        removed = before_count - after_count
        lines.append(f"{key}: {before_count} -> {after_count} ({removed} removed)")
    return lines


def remove_paths(paths: set[Path], output_dir: Path) -> tuple[int, int]:
    files_removed = 0
    dirs_removed = 0
    for path in sorted(paths, key=lambda item: len(item.parts), reverse=True):
        if not path.exists():
            continue
        if output_dir not in path.parents and path != output_dir:
            raise RuntimeError(f"Refusing to delete outside output-dir: {path}")
        if path.is_dir():
            shutil.rmtree(path)
            dirs_removed += 1
        else:
            path.unlink()
            files_removed += 1
    return files_removed, dirs_removed


def prune_empty_dirs(output_dir: Path) -> int:
    removed = 0
    candidates = sorted(
        [
            path
            for pattern in ("l*/**",)
            for path in output_dir.glob(pattern)
            if path.is_dir()
        ],
        key=lambda item: len(item.parts),
        reverse=True,
    )
    for path in candidates:
        if path.exists() and not any(path.iterdir()):
            path.rmdir()
            removed += 1
    return removed


def main() -> None:
    args = parse_args()
    require_filter(args)

    output_dir = args.output_dir.expanduser().resolve()
    map_dir = output_dir / "maps"
    parameters_path = map_dir / "parameters.json"
    captures_path = map_dir / "captures.jsonl"

    parameters = load_parameters(parameters_path)
    original_parameters = copy.deepcopy(parameters)
    captures = load_captures(captures_path)
    removed_captures = [record for record in captures if record_matches(record, args)]
    remaining_captures = [record for record in captures if not record_matches(record, args)]
    paths = image_paths_from_records(output_dir, removed_captures)
    paths.update(directory_paths_from_filters(output_dir, args))
    paths = collapse_paths(paths)
    updated_parameters = prune_parameters(parameters, remaining_captures, removed_captures, args)

    print(f"Output dir: {output_dir}")
    print(f"Matched capture records: {len(removed_captures)}")
    print(f"Remaining capture records: {len(remaining_captures)}")
    print("Map changes:")
    for line in map_change_summary(original_parameters, updated_parameters):
        print(f"  {line}")
    print(f"Paths to remove: {sum(1 for path in paths if path.exists())}")
    for path in sorted(paths):
        if path.exists():
            print(f"  remove {path.relative_to(output_dir)}")

    if not args.apply:
        print("\nDry run only. Re-run with --apply to delete files and rewrite maps.")
        return

    files_removed, dirs_removed = remove_paths(paths, output_dir)
    empty_dirs_removed = prune_empty_dirs(output_dir)
    write_parameters(parameters_path, updated_parameters)
    if captures_path.exists():
        write_captures(captures_path, remaining_captures)

    print("\nApplied cleanup.")
    print(f"Files removed: {files_removed}")
    print(f"Directories removed: {dirs_removed + empty_dirs_removed}")


if __name__ == "__main__":
    main()
