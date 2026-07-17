#!/usr/bin/env python3
"""Crop dim_test images with one fixed square crop per object folder."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_INPUT_ROOT = Path("dataset/dim_test")
DEFAULT_OUTPUT_ROOT = Path("dataset/dim_test_cropped")
DEFAULT_JPEG_QUALITY = 95

CROP_CONFIG = {
    "c001": {"center": [1970, 2078], "side": 720},
    "c002": {"center": [1992, 1940], "side": 3000},
}


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


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def iter_image_paths(input_root: Path) -> Iterable[Path]:
    for path in sorted(input_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def compute_clamped_square_box(
    image_width: int,
    image_height: int,
    center_x: int,
    center_y: int,
    side: int,
) -> CropBox:
    side = max(1, min(side, image_width, image_height))
    left = round(center_x - side / 2)
    top = round(center_y - side / 2)
    left = max(0, min(left, image_width - side))
    top = max(0, min(top, image_height - side))
    return CropBox(left=left, top=top, right=left + side, bottom=top + side)


def write_jsonl(handle, payload: dict[str, Any]) -> None:
    handle.write(json.dumps(payload, sort_keys=True) + "\n")


def write_crop_config(output_root: Path, input_root: Path, dry_run: bool) -> None:
    payload = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "dry_run": dry_run,
        "crop_config": CROP_CONFIG,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "crop_config.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def process_images(args: argparse.Namespace) -> dict[str, int]:
    input_root = args.input_root.expanduser()
    output_root = args.output_root.expanduser()
    if not input_root.exists():
        raise FileNotFoundError(f"Missing input root: {input_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    write_crop_config(output_root, input_root, args.dry_run)
    report_path = output_root / "crop_report.jsonl"

    counts = {
        "total_images": 0,
        "cropped": 0,
        "dry_run": 0,
        "skipped_existing": 0,
        "unknown_folder": 0,
        "errors": 0,
    }
    per_folder: dict[str, int] = {folder: 0 for folder in CROP_CONFIG}

    with report_path.open("w", encoding="utf-8") as report:
        for input_path in iter_image_paths(input_root):
            counts["total_images"] += 1
            relative_path = input_path.relative_to(input_root)
            folder = relative_path.parts[0] if relative_path.parts else ""
            output_path = output_root / relative_path
            base_payload: dict[str, Any] = {
                "input_path": str(input_path),
                "output_path": str(output_path),
                "relative_path": str(relative_path),
                "folder": folder,
            }

            config = CROP_CONFIG.get(folder)
            if config is None:
                counts["unknown_folder"] += 1
                write_jsonl(report, {**base_payload, "status": "unknown_folder"})
                continue

            if output_path.exists() and not args.overwrite and not args.dry_run:
                counts["skipped_existing"] += 1
                write_jsonl(report, {**base_payload, "status": "skipped_existing"})
                continue

            try:
                with Image.open(input_path) as raw_image:
                    image = raw_image.convert("RGB")
                    center_x, center_y = config["center"]
                    crop_box = compute_clamped_square_box(
                        image_width=image.width,
                        image_height=image.height,
                        center_x=center_x,
                        center_y=center_y,
                        side=config["side"],
                    )
                    payload = {
                        **base_payload,
                        "status": "dry_run" if args.dry_run else "cropped",
                        "original_size": [image.width, image.height],
                        "configured_center": [center_x, center_y],
                        "configured_side": config["side"],
                        "crop_box": crop_box.as_list(),
                        "output_size": [crop_box.width, crop_box.height],
                    }

                    if args.dry_run:
                        counts["dry_run"] += 1
                    else:
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        cropped = image.crop(crop_box.as_list())
                        save_kwargs: dict[str, Any] = {}
                        if output_path.suffix.lower() in {".jpg", ".jpeg"}:
                            save_kwargs.update({"quality": args.jpeg_quality, "subsampling": 0})
                        cropped.save(output_path, **save_kwargs)
                        counts["cropped"] += 1

                    per_folder[folder] += 1
                    write_jsonl(report, payload)
            except Exception as exc:  # noqa: BLE001 - keep batch processing resilient.
                counts["errors"] += 1
                write_jsonl(
                    report,
                    {
                        **base_payload,
                        "status": "error",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )

    counts["per_folder_crops"] = per_folder
    print(json.dumps(counts, indent=2, sort_keys=True))
    print(f"Report: {report_path}")
    if args.dry_run:
        print("Dry run: no images were written.")
    else:
        print(f"Output dataset: {output_root}")
    return counts


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Crop dataset/dim_test with one fixed square crop per top-level "
            "object folder."
        )
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dry-run", action="store_true", help="Write only config/report metadata.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output images instead of skipping them.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=positive_int,
        default=DEFAULT_JPEG_QUALITY,
        help="JPEG quality for cropped .jpg/.jpeg outputs.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.jpeg_quality > 100:
        raise ValueError("--jpeg-quality must be <= 100")
    process_images(args)


if __name__ == "__main__":
    main()
