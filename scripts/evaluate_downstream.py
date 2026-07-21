#!/usr/bin/env python3
"""Run top-1 downstream evaluation and write per-image results.

This script only performs model inference. Analysis scripts under
scripts/analysis/ consume the generated image_results.jsonl.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image
import torch
from torchvision.models import ResNet50_Weights, resnet50


DEFAULT_DATASET_ROOT = Path("dataset/test_ver3_cropped")
DEFAULT_OUTPUT_DIR = DEFAULT_DATASET_ROOT / "downstream_eval"
VALID_MODELS = {"resnet50", "openclip"}
OPENCLIP_CLASS_SPACES = {"configured", "imagenet"}


@dataclass(frozen=True)
class ImageRecord:
    record_index: int
    image_path: Path
    image_path_relative: str
    class_id: str
    sample_id: str
    view_id: str
    object_key: str
    light_id: str
    position: str
    intensity: int | None
    param_id: str
    param_file: str
    aperture: str
    iso: str
    shutter_speed: str
    exposure_mode: str


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


def openclip_class_space(value: str) -> str:
    parsed = value.strip().lower()
    if parsed not in OPENCLIP_CLASS_SPACES:
        raise argparse.ArgumentTypeError(
            f"unknown OpenCLIP class space: {value}; valid choices are "
            f"{', '.join(sorted(OPENCLIP_CLASS_SPACES))}"
        )
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSONL file: {path}")
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


def load_class_labels(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing class labels: {path}. Copy class_labels.template.json and edit it first."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    labels: dict[str, dict[str, Any]] = {}
    for class_id, item in payload.items():
        prompts = item.get("openclip_prompts", [])
        imagenet_labels = item.get("imagenet_labels", [])
        if not isinstance(prompts, list) or not prompts:
            raise ValueError(f"class {class_id} must define a non-empty openclip_prompts list")
        if not isinstance(imagenet_labels, list):
            raise ValueError(f"class {class_id} imagenet_labels must be a list")
        labels[str(class_id)] = {
            "name": str(item.get("name", class_id)),
            "openclip_prompts": [str(value) for value in prompts],
            "imagenet_labels": [str(value) for value in imagenet_labels],
        }
    return labels


def normalize_scalar(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def parse_optional_int(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    return int(value)


def load_records(
    dataset_root: Path,
    allow_missing_images: bool,
    limit: int | None,
) -> list[ImageRecord]:
    rows = read_jsonl(dataset_root / "maps" / "captures.jsonl")
    records: list[ImageRecord] = []
    missing: list[Path] = []

    for row in rows:
        image_path_relative = str(row["image_path"])
        image_path = dataset_root / image_path_relative
        if not image_path.exists():
            missing.append(image_path)
            if not allow_missing_images:
                continue

        class_id = normalize_scalar(row.get("class_id"))
        sample_id = normalize_scalar(row.get("sample_id"))
        view_id = normalize_scalar(row.get("view_id"))
        param_file = Path(image_path_relative).name
        records.append(
            ImageRecord(
                record_index=len(records),
                image_path=image_path,
                image_path_relative=image_path_relative,
                class_id=class_id,
                sample_id=sample_id,
                view_id=view_id,
                object_key=f"c{class_id}/s{sample_id}/v{view_id}",
                light_id=normalize_scalar(row.get("light_id")),
                position=normalize_scalar(row.get("position")),
                intensity=parse_optional_int(row.get("intensity")),
                param_id=normalize_scalar(row.get("param_id")),
                param_file=param_file,
                aperture=normalize_scalar(row.get("aperture")),
                iso=normalize_scalar(row.get("iso")),
                shutter_speed=normalize_scalar(row.get("shutter_speed")),
                exposure_mode=normalize_scalar(row.get("exposure_mode")),
            )
        )
        if limit is not None and len(records) >= limit:
            break

    if missing and not allow_missing_images:
        examples = "\n".join(f"  {path}" for path in missing[:5])
        raise FileNotFoundError(
            f"{len(missing)} image(s) from captures.jsonl are missing. "
            f"Use --allow-missing-images to skip them.\n{examples}"
        )
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


def print_progress(label: str, processed: int, total: int, force: bool = False) -> None:
    if total <= 0:
        return
    if not force and processed >= total:
        return
    fraction = min(processed / total, 1.0)
    bar_width = 28
    filled = round(bar_width * fraction)
    bar = "#" * filled + "-" * (bar_width - filled)
    print(
        "\r"
        f"{label}: [{bar}] {fraction * 100:6.2f}% "
        f"processed={processed}/{total}",
        end="",
        flush=True,
    )


def finish_progress(label: str, processed: int, total: int) -> None:
    print_progress(label, processed, total, force=True)
    print()


def validate_imagenet_labels(
    class_labels: dict[str, dict[str, Any]],
    categories: list[str],
) -> dict[str, set[str]]:
    category_set = set(categories)
    labels_by_class: dict[str, set[str]] = {}
    missing: dict[str, list[str]] = {}
    for class_id, item in class_labels.items():
        labels = set(item["imagenet_labels"])
        not_found = sorted(labels - category_set)
        if not_found:
            missing[class_id] = not_found
        labels_by_class[class_id] = labels
    if missing:
        raise ValueError(f"ImageNet label(s) not found in ResNet50 categories: {missing}")
    return labels_by_class


def evaluate_resnet50(
    records: list[ImageRecord],
    class_labels: dict[str, dict[str, Any]],
    batch_size: int,
    device: torch.device,
) -> dict[int, dict[str, Any]]:
    weights = ResNet50_Weights.IMAGENET1K_V2
    categories = weights.meta["categories"]
    labels_by_class = validate_imagenet_labels(class_labels, categories)
    model = resnet50(weights=weights).to(device)
    model.eval()
    transform = weights.transforms()

    results: dict[int, dict[str, Any]] = {}
    processed = 0
    total = len(records)
    with torch.inference_mode():
        for batch_records, images in iter_batches(records, transform, batch_size, device):
            top1_indices = model(images).argmax(dim=1).detach().cpu().tolist()
            for record, top1_index in zip(batch_records, top1_indices):
                label = categories[int(top1_index)]
                allowed = labels_by_class.get(record.class_id, set())
                results[record.record_index] = {
                    "top1_label": label,
                    "top1_correct": bool(label in allowed) if allowed else None,
                }
            processed += len(batch_records)
            print_progress("resnet50", processed, total)
    finish_progress("resnet50", processed, total)
    return results


def openclip_candidates(
    class_labels: dict[str, dict[str, Any]],
    class_space: str,
) -> list[dict[str, Any]]:
    class_ids = sorted(class_labels, key=lambda value: int(value) if value.isdigit() else value)

    if class_space == "configured":
        return [
            {
                "candidate_id": class_id,
                "name": class_labels[class_id]["name"],
                "target_class_id": class_id,
                "prompts": class_labels[class_id]["openclip_prompts"],
            }
            for class_id in class_ids
        ]

    categories = list(ResNet50_Weights.DEFAULT.meta["categories"])
    labels_by_class = validate_imagenet_labels(class_labels, categories)
    target_by_label: dict[str, str] = {}
    for class_id in class_ids:
        for label in labels_by_class[class_id]:
            previous = target_by_label.get(label)
            if previous is not None:
                raise ValueError(
                    f"ImageNet label {label!r} is assigned to both class {previous} and {class_id}"
                )
            target_by_label[label] = class_id

    candidates = [
        {
            "candidate_id": f"imagenet:{index}",
            "name": label,
            "target_class_id": target_by_label.get(label),
            "prompts": [f"a photo of a {label}"],
        }
        for index, label in enumerate(categories)
    ]
    for class_id in class_ids:
        if not labels_by_class[class_id]:
            candidates.append(
                {
                    "candidate_id": f"custom:{class_id}",
                    "name": class_labels[class_id]["name"],
                    "target_class_id": class_id,
                    "prompts": class_labels[class_id]["openclip_prompts"],
                }
            )
    return candidates


def openclip_class_features(
    class_labels: dict[str, dict[str, Any]],
    model,
    tokenizer,
    device: torch.device,
    class_space: str,
    text_batch_size: int,
) -> tuple[list[dict[str, Any]], torch.Tensor]:
    candidates = openclip_candidates(class_labels, class_space)
    prompts: list[str] = []
    prompt_owners: list[int] = []
    for candidate_index, candidate in enumerate(candidates):
        candidate_prompts = candidate["prompts"]
        prompts.extend(candidate_prompts)
        prompt_owners.extend([candidate_index] * len(candidate_prompts))

    prompt_feature_batches: list[torch.Tensor] = []

    with torch.inference_mode():
        for start in range(0, len(prompts), text_batch_size):
            tokens = tokenizer(prompts[start : start + text_batch_size]).to(device)
            prompt_features = model.encode_text(tokens)
            prompt_features = prompt_features / prompt_features.norm(dim=-1, keepdim=True)
            prompt_feature_batches.append(prompt_features)

        all_prompt_features = torch.cat(prompt_feature_batches)
        owner_indices = torch.tensor(prompt_owners, dtype=torch.long, device=device)
        class_features = torch.zeros(
            (len(candidates), all_prompt_features.shape[1]),
            dtype=all_prompt_features.dtype,
            device=device,
        )
        class_features.index_add_(0, owner_indices, all_prompt_features)
        counts = torch.bincount(owner_indices, minlength=len(candidates)).to(class_features.dtype)
        class_features = class_features / counts.unsqueeze(1)
        class_features = class_features / class_features.norm(dim=-1, keepdim=True)
    return candidates, class_features


def evaluate_openclip(
    records: list[ImageRecord],
    class_labels: dict[str, dict[str, Any]],
    batch_size: int,
    device: torch.device,
    model_name: str,
    pretrained: str,
    class_space: str,
    text_batch_size: int,
) -> dict[int, dict[str, Any]]:
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        device=device,
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    candidates, text_features = openclip_class_features(
        class_labels,
        model,
        tokenizer,
        device,
        class_space,
        text_batch_size,
    )

    results: dict[int, dict[str, Any]] = {}
    processed = 0
    total = len(records)
    with torch.inference_mode():
        for batch_records, images in iter_batches(records, preprocess, batch_size, device):
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            similarities = image_features @ text_features.T
            scores, indices = similarities.max(dim=1)
            for record, index, score in zip(batch_records, indices.detach().cpu().tolist(), scores.detach().cpu().tolist()):
                candidate = candidates[int(index)]
                predicted_target_class_id = candidate["target_class_id"]
                results[record.record_index] = {
                    "top1_candidate_id": candidate["candidate_id"],
                    "top1_class_id": predicted_target_class_id or candidate["candidate_id"],
                    "top1_target_class_id": predicted_target_class_id,
                    "top1_label": candidate["name"],
                    "top1_score": float(score),
                    "top1_correct": predicted_target_class_id == record.class_id,
                }
            processed += len(batch_records)
            print_progress("openclip", processed, total)
    finish_progress("openclip", processed, total)
    return results


def base_payload(record: ImageRecord) -> dict[str, Any]:
    return {
        "image_path": record.image_path_relative,
        "class_id": record.class_id,
        "sample_id": record.sample_id,
        "view_id": record.view_id,
        "object_key": record.object_key,
        "light_id": record.light_id,
        "position": record.position,
        "intensity": record.intensity,
        "param_id": record.param_id,
        "param_file": record.param_file,
        "aperture": record.aperture,
        "iso": record.iso,
        "shutter_speed": record.shutter_speed,
        "exposure_mode": record.exposure_mode,
    }


def write_results(
    path: Path,
    records: list[ImageRecord],
    predictions: dict[str, dict[int, dict[str, Any]]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            payload = base_payload(record)
            resnet = predictions.get("resnet50", {}).get(record.record_index, {})
            openclip = predictions.get("openclip", {}).get(record.record_index, {})
            payload.update(
                {
                    "resnet50_top1_label": resnet.get("top1_label"),
                    "resnet50_top1_correct": resnet.get("top1_correct"),
                    "openclip_top1_class_id": openclip.get("top1_class_id"),
                    "openclip_top1_candidate_id": openclip.get("top1_candidate_id"),
                    "openclip_top1_target_class_id": openclip.get("top1_target_class_id"),
                    "openclip_top1_label": openclip.get("top1_label"),
                    "openclip_top1_correct": openclip.get("top1_correct"),
                }
            )
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run top-1 downstream evaluation.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--class-labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--models", type=parse_models, default=parse_models("resnet50,openclip"))
    parser.add_argument("--batch-size", type=positive_int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=positive_int)
    parser.add_argument("--allow-missing-images", action="store_true")
    parser.add_argument("--openclip-model", default="ViT-B-32")
    parser.add_argument("--openclip-pretrained", default="laion2b_s34b_b79k")
    parser.add_argument(
        "--openclip-class-space",
        type=openclip_class_space,
        default="configured",
        help="Use configured semantic classes or the full ImageNet-1K label space.",
    )
    parser.add_argument("--openclip-text-batch-size", type=positive_int, default=256)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    dataset_root = args.dataset_root.expanduser()
    output_dir = args.output_dir.expanduser() if args.output_dir else dataset_root / "downstream_eval"
    device = choose_device(args.device)
    require_openclip_if_requested(args.models)
    class_labels = load_class_labels(args.class_labels.expanduser())
    records = load_records(dataset_root, args.allow_missing_images, args.limit)
    if not records:
        raise RuntimeError(f"No image records found under {dataset_root}")

    predictions: dict[str, dict[int, dict[str, Any]]] = {}
    if "resnet50" in args.models:
        predictions["resnet50"] = evaluate_resnet50(records, class_labels, args.batch_size, device)
    if "openclip" in args.models:
        predictions["openclip"] = evaluate_openclip(
            records,
            class_labels,
            args.batch_size,
            device,
            args.openclip_model,
            args.openclip_pretrained,
            args.openclip_class_space,
            args.openclip_text_batch_size,
        )

    output_path = output_dir / "image_results.jsonl"
    write_results(output_path, records, predictions)
    print(f"Evaluated {len(records)} images on {device}.")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
