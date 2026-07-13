#!/usr/bin/env python3
"""Evaluate ResNet50/OpenCLIP top-1 correctness on cropped dim_test images."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from PIL import Image
import torch
from torchvision.models import ResNet50_Weights, resnet50


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_DATASET_ROOT = Path("dataset/dim_test_cropped")
DEFAULT_OUTPUT_DIR = DEFAULT_DATASET_ROOT / "downstream_eval"
VALID_MODELS = {"resnet50", "openclip"}
DOG_INDEX_START = 151
DOG_INDEX_END = 268
CAR_LABELS = {
    "ambulance",
    "beach wagon",
    "cab",
    "convertible",
    "jeep",
    "limousine",
    "minivan",
    "moving van",
    "passenger car",
    "pickup",
    "police van",
    "racer",
    "recreational vehicle",
    "sports car",
    "tow truck",
    "trailer truck",
}


@dataclass(frozen=True)
class ImageRecord:
    record_index: int
    image_path: Path
    image_path_relative: str
    class_folder: str


def parse_models(value: str) -> list[str]:
    models = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not models:
        raise argparse.ArgumentTypeError("at least one model is required")
    unknown = sorted(set(models) - VALID_MODELS)
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown model(s): {', '.join(unknown)}; valid choices are "
            f"{', '.join(sorted(VALID_MODELS))}"
        )
    return models


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def choose_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return device


def require_openclip_if_requested(models: list[str]) -> None:
    if "openclip" not in models:
        return
    try:
        import open_clip  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "OpenCLIP evaluation requested, but open_clip is not installed. "
            "Install it with: pip install open_clip_torch"
        ) from exc


def imagenet_correctness(categories: list[str]) -> dict[str, Callable[[int], bool]]:
    labels_to_index = {label: index for index, label in enumerate(categories)}
    missing_car_labels = sorted(CAR_LABELS - set(labels_to_index))
    if missing_car_labels:
        raise RuntimeError(f"Missing expected ImageNet car labels: {missing_car_labels}")

    car_indices = {labels_to_index[label] for label in CAR_LABELS}
    return {
        "c001": lambda index: DOG_INDEX_START <= index <= DOG_INDEX_END,
        "c002": lambda index: index in car_indices,
    }


def is_under(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def load_records(dataset_root: Path, output_dir: Path, limit: int | None) -> list[ImageRecord]:
    dataset_root = dataset_root.resolve()
    output_dir = output_dir.resolve()
    records: list[ImageRecord] = []

    for image_path in sorted(dataset_root.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        resolved_path = image_path.resolve()
        if is_under(resolved_path, output_dir):
            continue

        relative_path = resolved_path.relative_to(dataset_root)
        if not relative_path.parts:
            continue
        class_folder = relative_path.parts[0]
        if class_folder not in {"c001", "c002"}:
            continue

        records.append(
            ImageRecord(
                record_index=len(records),
                image_path=resolved_path,
                image_path_relative=str(relative_path),
                class_folder=class_folder,
            )
        )
        if limit is not None and len(records) >= limit:
            break

    return records


def open_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def iter_batches(
    records: Iterable[ImageRecord],
    transform,
    batch_size: int,
    device: torch.device,
) -> Iterable[tuple[list[ImageRecord], torch.Tensor]]:
    batch_records: list[ImageRecord] = []
    batch_images: list[torch.Tensor] = []

    for record in records:
        batch_records.append(record)
        batch_images.append(transform(open_image(record.image_path)))
        if len(batch_images) == batch_size:
            yield batch_records, torch.stack(batch_images).to(device)
            batch_records = []
            batch_images = []

    if batch_images:
        yield batch_records, torch.stack(batch_images).to(device)


def build_imagenet_prompts(categories: list[str]) -> list[str]:
    return [f"a photo of a {category}." for category in categories]


def evaluate_resnet50(
    records: list[ImageRecord],
    batch_size: int,
    device: torch.device,
    is_correct_by_class: dict[str, Callable[[int], bool]],
) -> dict[int, bool]:
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights).to(device)
    model.eval()
    transform = weights.transforms()

    results: dict[int, bool] = {}
    with torch.inference_mode():
        for batch_records, images in iter_batches(records, transform, batch_size, device):
            top1_indices = model(images).argmax(dim=1).detach().cpu().tolist()
            for record, top1_index in zip(batch_records, top1_indices):
                results[record.record_index] = is_correct_by_class[record.class_folder](
                    int(top1_index)
                )
    return results


def evaluate_openclip(
    records: list[ImageRecord],
    batch_size: int,
    device: torch.device,
    categories: list[str],
    is_correct_by_class: dict[str, Callable[[int], bool]],
    model_name: str,
    pretrained: str,
) -> dict[int, bool]:
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        device=device,
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()

    prompts = build_imagenet_prompts(categories)
    with torch.inference_mode():
        text_tokens = tokenizer(prompts).to(device)
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    results: dict[int, bool] = {}
    with torch.inference_mode():
        for batch_records, images in iter_batches(records, preprocess, batch_size, device):
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            top1_indices = (image_features @ text_features.T).argmax(dim=1).detach().cpu().tolist()
            for record, top1_index in zip(batch_records, top1_indices):
                results[record.record_index] = is_correct_by_class[record.class_folder](
                    int(top1_index)
                )
    return results


def write_image_results(
    path: Path,
    records: list[ImageRecord],
    predictions: dict[str, dict[int, bool]],
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            payload: dict[str, object] = {
                "image_path": record.image_path_relative,
                "class_folder": record.class_folder,
            }
            for model_name, model_predictions in predictions.items():
                payload[f"{model_name}_correct"] = bool(
                    model_predictions.get(record.record_index, False)
                )
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def summarize_by_class(
    records: list[ImageRecord],
    predictions: dict[str, dict[int, bool]],
) -> list[dict[str, object]]:
    grouped: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        grouped[record.class_folder].append(record)

    rows: list[dict[str, object]] = []
    for class_folder in sorted(grouped):
        class_records = grouped[class_folder]
        row: dict[str, object] = {
            "class_folder": class_folder,
            "n_images": len(class_records),
        }
        for model_name, model_predictions in predictions.items():
            correct_count = sum(
                1
                for record in class_records
                if model_predictions.get(record.record_index, False)
            )
            row[f"{model_name}_correct"] = correct_count
            row[f"{model_name}_accuracy"] = correct_count / len(class_records)
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], models: list[str]) -> None:
    fieldnames = ["class_folder", "n_images"]
    for model_name in models:
        fieldnames.extend([f"{model_name}_correct", f"{model_name}_accuracy"])

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate top-1 downstream accuracy on cropped dim_test images."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--models", type=parse_models, default=parse_models("resnet50,openclip"))
    parser.add_argument("--batch-size", type=positive_int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=positive_int, help="Evaluate only this many images.")
    parser.add_argument("--openclip-model", default="ViT-B-32")
    parser.add_argument("--openclip-pretrained", default="laion2b_s34b_b79k")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    device = choose_device(args.device)
    require_openclip_if_requested(args.models)

    categories = ResNet50_Weights.IMAGENET1K_V2.meta["categories"]
    if len(categories) != 1000:
        raise RuntimeError(f"Expected 1000 ImageNet categories, got {len(categories)}")
    is_correct_by_class = imagenet_correctness(categories)

    records = load_records(args.dataset_root, args.output_dir, args.limit)
    if not records:
        raise RuntimeError(f"No c001/c002 images found under {args.dataset_root}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions: dict[str, dict[int, bool]] = {}
    if "resnet50" in args.models:
        predictions["resnet50"] = evaluate_resnet50(
            records,
            args.batch_size,
            device,
            is_correct_by_class,
        )
    if "openclip" in args.models:
        predictions["openclip"] = evaluate_openclip(
            records,
            args.batch_size,
            device,
            categories,
            is_correct_by_class,
            args.openclip_model,
            args.openclip_pretrained,
        )

    image_results_path = args.output_dir / "image_results.jsonl"
    per_class_path = args.output_dir / "per_class_accuracy.csv"
    summary_rows = summarize_by_class(records, predictions)
    write_image_results(image_results_path, records, predictions)
    write_csv(per_class_path, summary_rows, args.models)

    print(f"Evaluated {len(records)} images on {device}.")
    print(f"Wrote {image_results_path}")
    print(f"Wrote {per_class_path}")
    for row in summary_rows:
        parts = [f"class_folder={row['class_folder']}", f"n={row['n_images']}"]
        for model_name in args.models:
            parts.append(f"{model_name}_accuracy={float(row[f'{model_name}_accuracy']):.4f}")
        print("  ".join(parts))


if __name__ == "__main__":
    main()
