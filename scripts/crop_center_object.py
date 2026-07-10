#!/usr/bin/env python3
"""Crop captured dataset images to square crops centered on the object.

The capture rig has a fixed camera and turntable, so the default crop is a
parameterized square around the expected object position rather than a heavy
detector. The original dataset is left untouched; outputs are written as a new
dataset root with copied map files and a filtered images.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageStat


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_DATASET_ROOT = Path("dataset/test")
DEFAULT_OUTPUT_ROOT = Path("dataset/test_cropped")
DEFAULT_CROP_SIDE_RATIO = 0.42
DEFAULT_CENTER_X_RATIO = 0.50
DEFAULT_CENTER_Y_RATIO = 0.64
DEFAULT_MIN_MEAN_BRIGHTNESS = 5.0
DEFAULT_MIN_LUMA_STD = 3.0
DEFAULT_PROGRESS_EVERY = 50


@dataclass(frozen=True)
class CropBox:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    def as_list(self) -> list[int]:
        return [self.left, self.top, self.right, self.bottom]


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def ratio(value: str) -> float:
    parsed = float(value)
    if parsed <= 0 or parsed >= 1:
        raise argparse.ArgumentTypeError("ratio must be greater than 0 and less than 1")
    return parsed


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def read_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required CSV: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def write_csv_rows(
    path: Path,
    fieldnames: list[str],
    rows: Iterable[dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def copy_non_image_maps(input_maps_dir: Path, output_maps_dir: Path) -> None:
    output_maps_dir.mkdir(parents=True, exist_ok=True)
    for path in input_maps_dir.iterdir():
        if path.name == "images.csv" or not path.is_file():
            continue
        shutil.copy2(path, output_maps_dir / path.name)


def compute_square_crop_box(
    width: int,
    height: int,
    crop_side: int | None,
    crop_side_ratio: float,
    center_x_ratio: float,
    center_y_ratio: float,
) -> CropBox:
    side = crop_side if crop_side is not None else round(min(width, height) * crop_side_ratio)
    side = max(1, min(side, width, height))

    center_x = round(width * center_x_ratio)
    center_y = round(height * center_y_ratio)
    left = center_x - side // 2
    top = center_y - side // 2
    left = max(0, min(left, width - side))
    top = max(0, min(top, height - side))
    return CropBox(left=left, top=top, right=left + side, bottom=top + side)


def image_quality(image: Image.Image) -> dict[str, float]:
    luma = image.convert("L")
    stat = ImageStat.Stat(luma)
    return {
        "mean_brightness": float(stat.mean[0]),
        "luma_std": float(stat.stddev[0]),
    }


def is_valid_image(
    quality: dict[str, float],
    min_mean_brightness: float,
    min_luma_std: float,
) -> tuple[bool, str | None]:
    if quality["mean_brightness"] < min_mean_brightness:
        return False, "mean_brightness_below_threshold"
    if quality["luma_std"] < min_luma_std:
        return False, "luma_std_below_threshold"
    return True, None


def write_report(handle, payload: dict[str, Any]) -> None:
    handle.write(json.dumps(payload, sort_keys=True) + "\n")


def should_include_row(row: dict[str, str], class_id: str | None) -> bool:
    return class_id is None or row.get("class_id", "") == class_id


def progress_target(rows: list[dict[str, str]], class_id: str | None) -> int:
    return sum(1 for row in rows if should_include_row(row, class_id))


def print_progress(
    counts: dict[str, int],
    target: int,
    force: bool = False,
) -> None:
    if target <= 0:
        return
    processed = counts["cropped"] + counts["missing"] + counts["invalid"] + counts["unsupported"] + counts["errors"]
    if not force and processed % counts["progress_every"] != 0:
        return

    fraction = min(processed / target, 1.0)
    bar_width = 28
    filled = round(bar_width * fraction)
    bar = "#" * filled + "-" * (bar_width - filled)
    print(
        "\r"
        f"[{bar}] {fraction * 100:6.2f}% "
        f"selected_processed={processed}/{target} "
        f"cropped={counts['cropped']} "
        f"invalid={counts['invalid']} "
        f"missing={counts['missing']} "
        f"errors={counts['errors']}",
        end="",
        flush=True,
    )


def process_rows(args: argparse.Namespace) -> dict[str, int]:
    dataset_root = args.dataset_root.expanduser()
    output_root = args.output_root.expanduser()
    maps_dir = dataset_root / "maps"
    output_maps_dir = output_root / "maps"
    image_rows, image_fieldnames = read_csv_rows(maps_dir / "images.csv")
    if "image_path" not in image_fieldnames:
        raise ValueError(f"{maps_dir / 'images.csv'} must contain an image_path column")

    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)
        copy_non_image_maps(maps_dir, output_maps_dir)

    report_path = output_root / "crop_report.jsonl"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    output_rows: list[dict[str, str]] = []
    counts = {
        "total_rows": 0,
        "selected_rows": 0,
        "cropped": 0,
        "missing": 0,
        "invalid": 0,
        "unsupported": 0,
        "errors": 0,
        "progress_every": args.progress_every,
    }
    target = progress_target(image_rows, args.class_id)

    with report_path.open("w", encoding="utf-8") as report:
        for row in image_rows:
            if args.limit is not None and counts["cropped"] >= args.limit:
                break

            counts["total_rows"] += 1
            if not should_include_row(row, args.class_id):
                continue
            counts["selected_rows"] += 1

            relative_path = Path(row["image_path"])
            input_path = dataset_root / relative_path
            output_path = output_root / relative_path
            base_payload: dict[str, Any] = {
                "image_path": row["image_path"],
                "input_path": str(input_path),
                "output_path": str(output_path),
            }

            if relative_path.suffix.lower() not in IMAGE_EXTENSIONS:
                counts["unsupported"] += 1
                write_report(report, {**base_payload, "status": "unsupported_extension"})
                print_progress(counts, target)
                continue
            if not input_path.exists():
                counts["missing"] += 1
                write_report(report, {**base_payload, "status": "missing"})
                print_progress(counts, target)
                continue

            try:
                with Image.open(input_path) as raw_image:
                    image = raw_image.convert("RGB")
                    quality = image_quality(image)
                    valid, invalid_reason = is_valid_image(
                        quality,
                        args.min_mean_brightness,
                        args.min_luma_std,
                    )
                    crop_box = compute_square_crop_box(
                        width=image.width,
                        height=image.height,
                        crop_side=args.crop_side,
                        crop_side_ratio=args.crop_side_ratio,
                        center_x_ratio=args.center_x_ratio,
                        center_y_ratio=args.center_y_ratio,
                    )
                    report_payload = {
                        **base_payload,
                        "original_size": [image.width, image.height],
                        "crop_box": crop_box.as_list(),
                        "crop_size": [crop_box.width, crop_box.height],
                        **quality,
                    }

                    if not valid and not args.include_invalid:
                        counts["invalid"] += 1
                        write_report(
                            report,
                            {
                                **report_payload,
                                "status": "invalid",
                                "reason": invalid_reason,
                            },
                        )
                        print_progress(counts, target)
                        continue

                    if not args.dry_run:
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        cropped = image.crop(crop_box.as_list())
                        save_kwargs: dict[str, Any] = {}
                        if output_path.suffix.lower() in {".jpg", ".jpeg"}:
                            save_kwargs.update({"quality": args.jpeg_quality, "subsampling": 0})
                        cropped.save(output_path, **save_kwargs)

                    output_rows.append(row)
                    counts["cropped"] += 1
                    write_report(report, {**report_payload, "status": "cropped"})
                    print_progress(counts, target)
            except Exception as exc:  # noqa: BLE001 - keep batch processing resilient.
                counts["errors"] += 1
                write_report(
                    report,
                    {
                        **base_payload,
                        "status": "error",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
                print_progress(counts, target)

    print_progress(counts, target, force=True)
    if target > 0:
        print()

    if not args.dry_run:
        write_csv_rows(output_maps_dir / "images.csv", image_fieldnames, output_rows)

    counts["output_rows"] = len(output_rows)
    counts.pop("progress_every")
    print(json.dumps(counts, indent=2, sort_keys=True))
    print(f"Report: {report_path}")
    if not args.dry_run:
        print(f"Output dataset: {output_root}")
    return counts


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a new dataset with square crops centered on the object in "
            "fixed-rig capture images."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--class-id", help="Only process rows with this class_id.")
    parser.add_argument("--limit", type=positive_int, help="Stop after this many crops.")
    parser.add_argument("--dry-run", action="store_true", help="Write only the report.")
    parser.add_argument(
        "--crop-side-ratio",
        type=ratio,
        default=DEFAULT_CROP_SIDE_RATIO,
        help="Square side as a fraction of min(image width, image height).",
    )
    parser.add_argument(
        "--crop-side",
        type=positive_int,
        help="Explicit square side in pixels; overrides --crop-side-ratio.",
    )
    parser.add_argument(
        "--center-x-ratio",
        type=ratio,
        default=DEFAULT_CENTER_X_RATIO,
        help="Crop center x position as a fraction of image width.",
    )
    parser.add_argument(
        "--center-y-ratio",
        type=ratio,
        default=DEFAULT_CENTER_Y_RATIO,
        help="Crop center y position as a fraction of image height.",
    )
    parser.add_argument(
        "--min-mean-brightness",
        type=positive_float,
        default=DEFAULT_MIN_MEAN_BRIGHTNESS,
        help="Skip images darker than this mean luma unless --include-invalid is set.",
    )
    parser.add_argument(
        "--min-luma-std",
        type=positive_float,
        default=DEFAULT_MIN_LUMA_STD,
        help="Skip near-flat images below this luma stddev unless --include-invalid is set.",
    )
    parser.add_argument(
        "--include-invalid",
        action="store_true",
        help="Crop even images that fail brightness/contrast checks.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=positive_int,
        default=95,
        help="JPEG quality for cropped .jpg/.jpeg outputs.",
    )
    parser.add_argument(
        "--progress-every",
        type=positive_int,
        default=DEFAULT_PROGRESS_EVERY,
        help="Refresh progress after this many processed selected rows.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.jpeg_quality > 100:
        raise ValueError("--jpeg-quality must be <= 100")
    process_rows(args)


if __name__ == "__main__":
    main()
