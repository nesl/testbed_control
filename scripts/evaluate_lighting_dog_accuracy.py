#!/usr/bin/env python3
"""Evaluate dog classification accuracy grouped by lighting condition.

The captured dataset currently contains one semantic class: dog. This script
therefore treats an ImageNet-1K prediction as correct when its class index is
one of the ImageNet dog breed classes, 151 through 268 inclusive.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image
import torch
from torchvision.models import ResNet50_Weights, resnet50


DOG_INDEX_START = 151
DOG_INDEX_END = 268
VALID_MODELS = {"resnet50", "openclip"}


@dataclass(frozen=True)
class ImageRecord:
    record_index: int
    image_path: Path
    image_path_relative: str
    class_id: str
    sample_id: str
    light_id: str
    view_id: str
    param_id: str
    session_id: str
    sequence: str
    captured_at: str
    light_meta: dict[str, str]
    exists: bool


def is_dog_index(index: int) -> bool:
    return DOG_INDEX_START <= index <= DOG_INDEX_END


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required CSV: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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
            "Install it in the control env with: pip install open_clip_torch"
        ) from exc


def load_records(dataset_root: Path, allow_missing_images: bool) -> tuple[list[ImageRecord], int]:
    maps_dir = dataset_root / "maps"
    image_rows = read_csv_rows(maps_dir / "images.csv")
    light_rows = read_csv_rows(maps_dir / "lights.csv")
    lights_by_id = {row["light_id"]: row for row in light_rows}

    records: list[ImageRecord] = []
    missing_count = 0
    missing_examples: list[Path] = []

    for index, row in enumerate(image_rows):
        image_path_relative = row["image_path"]
        image_path = dataset_root / image_path_relative
        exists = image_path.exists()
        if not exists:
            missing_count += 1
            if len(missing_examples) < 5:
                missing_examples.append(image_path)

        records.append(
            ImageRecord(
                record_index=index,
                image_path=image_path,
                image_path_relative=image_path_relative,
                class_id=row.get("class_id", ""),
                sample_id=row.get("sample_id", ""),
                light_id=row.get("light_id", ""),
                view_id=row.get("view_id", ""),
                param_id=row.get("param_id", ""),
                session_id=row.get("session_id", ""),
                sequence=row.get("sequence", ""),
                captured_at=row.get("captured_at", ""),
                light_meta=lights_by_id.get(row.get("light_id", ""), {}),
                exists=exists,
            )
        )

    if missing_count and not allow_missing_images:
        examples = "\n".join(f"  {path}" for path in missing_examples)
        raise FileNotFoundError(
            f"{missing_count} image(s) listed in images.csv are missing. "
            f"Use --allow-missing-images to skip them.\n{examples}"
        )
    return records, missing_count


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
        image = open_image(record.image_path)
        batch_records.append(record)
        batch_images.append(transform(image))
        if len(batch_images) == batch_size:
            yield batch_records, torch.stack(batch_images).to(device)
            batch_records = []
            batch_images = []

    if batch_images:
        yield batch_records, torch.stack(batch_images).to(device)


def topk_payload(
    indices: torch.Tensor,
    scores: torch.Tensor,
    categories: list[str],
) -> dict[str, Any]:
    index_values = [int(value) for value in indices.detach().cpu().tolist()]
    score_values = [float(value) for value in scores.detach().cpu().tolist()]
    labels = [categories[index] for index in index_values]
    return {
        "top1_index": index_values[0],
        "top1_label": labels[0],
        "top1_score": score_values[0],
        "top1_is_dog": is_dog_index(index_values[0]),
        "top5_indices": index_values,
        "top5_labels": labels,
        "top5_scores": score_values,
        "top5_has_dog": any(is_dog_index(index) for index in index_values),
    }


def evaluate_resnet50(
    records: list[ImageRecord],
    batch_size: int,
    device: torch.device,
    categories: list[str],
) -> dict[int, dict[str, Any]]:
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights).to(device)
    model.eval()
    transform = weights.transforms()

    results: dict[int, dict[str, Any]] = {}
    with torch.inference_mode():
        for batch_records, images in iter_batches(records, transform, batch_size, device):
            probabilities = model(images).softmax(dim=1)
            scores, indices = probabilities.topk(5, dim=1)
            for record, row_indices, row_scores in zip(batch_records, indices, scores):
                payload = topk_payload(row_indices, row_scores, categories)
                payload["top1_confidence"] = payload.pop("top1_score")
                payload["top5_confidences"] = payload.pop("top5_scores")
                results[record.record_index] = payload
    return results


def build_imagenet_prompts(categories: list[str]) -> list[str]:
    return [f"a photo of a {category}." for category in categories]


def evaluate_openclip(
    records: list[ImageRecord],
    batch_size: int,
    device: torch.device,
    categories: list[str],
    model_name: str,
    pretrained: str,
) -> dict[int, dict[str, Any]]:
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

    results: dict[int, dict[str, Any]] = {}
    with torch.inference_mode():
        for batch_records, images in iter_batches(records, preprocess, batch_size, device):
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            similarities = image_features @ text_features.T
            scores, indices = similarities.topk(5, dim=1)
            for record, row_indices, row_scores in zip(batch_records, indices, scores):
                payload = topk_payload(row_indices, row_scores, categories)
                payload["top1_similarity"] = payload.pop("top1_score")
                payload["top5_similarities"] = payload.pop("top5_scores")
                results[record.record_index] = payload
    return results


def record_base_payload(record: ImageRecord) -> dict[str, Any]:
    return {
        "record_index": record.record_index,
        "image_path": record.image_path_relative,
        "image_path_absolute": str(record.image_path),
        "class_id": record.class_id,
        "sample_id": record.sample_id,
        "session_id": record.session_id,
        "sequence": record.sequence,
        "captured_at": record.captured_at,
        "light_id": record.light_id,
        "intensity": parse_optional_int(record.light_meta.get("intensity")),
        "light_percent": parse_optional_float(record.light_meta.get("light_percent")),
        "cct": parse_optional_int(record.light_meta.get("cct")),
        "view_id": record.view_id,
        "param_id": record.param_id,
        "exists": record.exists,
    }


def parse_optional_int(value: str | None) -> int | None:
    if value in {None, ""}:
        return None
    return int(value)


def parse_optional_float(value: str | None) -> float | None:
    if value in {None, ""}:
        return None
    return float(value)


def write_jsonl(
    path: Path,
    records: list[ImageRecord],
    predictions: dict[str, dict[int, dict[str, Any]]],
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            payload = record_base_payload(record)
            for model_name, model_predictions in predictions.items():
                payload[model_name] = model_predictions.get(record.record_index)
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def summarize_by_lighting(
    records: list[ImageRecord],
    predictions: dict[str, dict[int, dict[str, Any]]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        grouped[record.light_id].append(record)

    summaries: list[dict[str, Any]] = []
    for light_id in sorted(grouped, key=lambda value: int(value) if value.isdigit() else value):
        light_records = grouped[light_id]
        valid_records = [record for record in light_records if record.exists]
        missing_count = len(light_records) - len(valid_records)
        first = light_records[0]
        summary: dict[str, Any] = {
            "light_id": light_id,
            "intensity": parse_optional_int(first.light_meta.get("intensity")),
            "light_percent": parse_optional_float(first.light_meta.get("light_percent")),
            "cct": parse_optional_int(first.light_meta.get("cct")),
            "n_total": len(light_records),
            "n_images": len(valid_records),
            "n_missing": missing_count,
        }

        for model_name, model_predictions in predictions.items():
            available = [
                model_predictions[record.record_index]
                for record in valid_records
                if record.record_index in model_predictions
            ]
            summary[f"{model_name}_n"] = len(available)
            summary[f"{model_name}_top1_accuracy"] = mean_bool(
                item["top1_is_dog"] for item in available
            )
            summary[f"{model_name}_top5_accuracy"] = mean_bool(
                item["top5_has_dog"] for item in available
            )
        summaries.append(summary)
    return summaries


def mean_bool(values: Iterable[bool]) -> float | None:
    items = list(values)
    if not items:
        return None
    return sum(1 for item in items if item) / len(items)


def format_accuracy(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.2f}%"


def find_best_worst(
    summaries: list[dict[str, Any]],
    models: list[str],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for model_name in models:
        metric = f"{model_name}_top1_accuracy"
        valid = [summary for summary in summaries if summary.get(metric) is not None]
        if not valid:
            continue
        best = max(valid, key=lambda item: item[metric])
        worst = min(valid, key=lambda item: item[metric])
        output[model_name] = {
            "metric": metric,
            "best_light_id": best["light_id"],
            "best_intensity": best["intensity"],
            "best_accuracy": best[metric],
            "worst_light_id": worst["light_id"],
            "worst_intensity": worst["intensity"],
            "worst_accuracy": worst[metric],
            "gap": best[metric] - worst[metric],
        }
    return output


def write_text_summary(
    path: Path,
    summaries: list[dict[str, Any]],
    best_worst: dict[str, dict[str, Any]],
    models: list[str],
) -> None:
    lines = [
        "Lighting dog accuracy",
        "Correctness: ImageNet-1K top prediction is dog if class index is 151..268.",
        "",
    ]

    for summary in summaries:
        parts = [
            f"light_id={summary['light_id']}",
            f"intensity={summary['intensity']}",
            f"light_percent={summary['light_percent']}",
            f"cct={summary['cct']}",
            f"n_images={summary['n_images']}",
            f"n_missing={summary['n_missing']}",
        ]
        for model_name in models:
            parts.extend(
                [
                    f"{model_name}_top1={format_accuracy(summary.get(f'{model_name}_top1_accuracy'))}",
                    f"{model_name}_top5={format_accuracy(summary.get(f'{model_name}_top5_accuracy'))}",
                ]
            )
        lines.append("  ".join(parts))

    if best_worst:
        lines.extend(["", "Best/worst by top-1 accuracy:"])
        for model_name, item in best_worst.items():
            lines.append(
                "  ".join(
                    [
                        f"model={model_name}",
                        f"best_light_id={item['best_light_id']}",
                        f"best_intensity={item['best_intensity']}",
                        f"best={format_accuracy(item['best_accuracy'])}",
                        f"worst_light_id={item['worst_light_id']}",
                        f"worst_intensity={item['worst_intensity']}",
                        f"worst={format_accuracy(item['worst_accuracy'])}",
                        f"gap={format_accuracy(item['gap'])}",
                    ]
                )
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_json(
    path: Path,
    args: argparse.Namespace,
    device: torch.device,
    total_records: int,
    missing_count: int,
    summaries: list[dict[str, Any]],
    best_worst: dict[str, dict[str, Any]],
) -> None:
    payload = {
        "dataset_root": str(args.dataset_root),
        "models": args.models,
        "device": str(device),
        "batch_size": args.batch_size,
        "openclip_model": args.openclip_model,
        "openclip_pretrained": args.openclip_pretrained,
        "dog_index_start": DOG_INDEX_START,
        "dog_index_end": DOG_INDEX_END,
        "total_records": total_records,
        "missing_images": missing_count,
        "lighting": summaries,
        "best_worst": best_worst,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate dog classification accuracy by lighting condition."
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset/test"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/lighting_dog_accuracy"))
    parser.add_argument("--models", type=parse_models, default=parse_models("resnet50,openclip"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--openclip-model", default="ViT-B-32")
    parser.add_argument("--openclip-pretrained", default="laion2b_s34b_b79k")
    parser.add_argument("--allow-missing-images", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    device = choose_device(args.device)
    require_openclip_if_requested(args.models)
    records, missing_count = load_records(args.dataset_root, args.allow_missing_images)
    valid_records = [record for record in records if record.exists]
    categories = ResNet50_Weights.IMAGENET1K_V2.meta["categories"]

    if len(categories) != 1000:
        raise RuntimeError(f"Expected 1000 ImageNet categories, got {len(categories)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    predictions: dict[str, dict[int, dict[str, Any]]] = {}
    if "resnet50" in args.models:
        predictions["resnet50"] = evaluate_resnet50(
            valid_records,
            args.batch_size,
            device,
            categories,
        )
    if "openclip" in args.models:
        predictions["openclip"] = evaluate_openclip(
            valid_records,
            args.batch_size,
            device,
            categories,
            args.openclip_model,
            args.openclip_pretrained,
        )

    summaries = summarize_by_lighting(records, predictions)
    best_worst = find_best_worst(summaries, args.models)

    write_jsonl(args.output_dir / "sample_results.jsonl", records, predictions)
    write_text_summary(args.output_dir / "lighting_accuracy.txt", summaries, best_worst, args.models)
    write_summary_json(
        args.output_dir / "summary.json",
        args,
        device,
        len(records),
        missing_count,
        summaries,
        best_worst,
    )

    print(f"Wrote {args.output_dir / 'lighting_accuracy.txt'}")
    print(f"Wrote {args.output_dir / 'sample_results.jsonl'}")
    print(f"Wrote {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
