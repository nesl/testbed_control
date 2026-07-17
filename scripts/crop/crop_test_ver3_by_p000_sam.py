#!/usr/bin/env python3
"""Crop test_ver3 with SAM candidates and optional manual mask selection.

The intended workflow is:
1. prepare-candidates: generate candidate masks from l001/<c>/<s>/<v>/p000.jpg.
2. crop: read a manual selection file, union selected bboxes per <c>/<s>/<v>,
   then apply that crop to all light folders for the group.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_DATASET_ROOT = Path("dataset/test_ver3")
DEFAULT_OUTPUT_ROOT = Path("dataset/test_ver3_cropped")
DEFAULT_SAM_MODEL_TYPE = "vit_b"
DEFAULT_SAM_METHOD = "hybrid"
DEFAULT_SAM_MAX_SIDE = 2048
DEFAULT_TARGET_BBOX_AREA_RATIO = 0.50
DEFAULT_MIN_CROP_SIDE = 512
DEFAULT_BBOX_SIDE_FRACTION = 0.85
DEFAULT_ANCHOR_X = 0.50
DEFAULT_ANCHOR_Y = 0.66
DEFAULT_JPEG_QUALITY = 95
DEFAULT_PROGRESS_EVERY = 100


@dataclass(frozen=True)
class BBox:
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

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.left + self.right) / 2, (self.top + self.bottom) / 2)

    def as_list(self) -> list[int]:
        return [self.left, self.top, self.right, self.bottom]


@dataclass(frozen=True)
class ViewCrop:
    view_key: str
    p000_path: Path
    original_size: tuple[int, int]
    resized_size: tuple[int, int]
    scale_to_original: float
    mask_score: float
    mask_area: int
    mask_area_ratio: float
    bbox: BBox
    crop_box: BBox


@dataclass(frozen=True)
class Candidate:
    candidate_id: int
    source: str
    bbox: BBox
    area: int
    score: float
    mask_area_ratio: float
    bbox_area_ratio: float
    center: tuple[float, float]


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}: {exc}") from exc
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_jsonl(handle, payload: dict[str, Any]) -> None:
    handle.write(json.dumps(payload, sort_keys=True) + "\n")


def iter_view_dirs(dataset_root: Path) -> list[Path]:
    view_dirs = [
        path
        for path in sorted(dataset_root.glob("l*/c*/s*/v*"))
        if path.is_dir()
    ]
    if not view_dirs:
        raise RuntimeError(f"No view directories found under {dataset_root}")
    return view_dirs


def iter_group_keys(dataset_root: Path) -> list[str]:
    groups: set[str] = set()
    for view_dir in iter_view_dirs(dataset_root):
        rel = view_dir.relative_to(dataset_root)
        if len(rel.parts) != 4:
            continue
        groups.add(str(Path(*rel.parts[1:])))
    if not groups:
        raise RuntimeError(f"No c*/s*/v* groups found under {dataset_root}")
    return sorted(groups)


def group_key_from_image_path(relative_path: Path) -> str:
    if len(relative_path.parts) < 5:
        return ""
    return str(Path(*relative_path.parts[1:4]))


def view_key_from_image_path(relative_path: Path) -> str:
    if len(relative_path.parts) < 5:
        return ""
    return str(Path(*relative_path.parts[:4]))


def iter_image_paths(view_dir: Path) -> Iterable[Path]:
    for path in sorted(view_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def resolve_device(requested: str) -> str:
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return requested


def check_sam_dependencies(args: argparse.Namespace) -> None:
    try:
        import segment_anything  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'segment_anything'. Install it in the target environment, "
            "for example: pip install git+https://github.com/facebookresearch/segment-anything.git"
        ) from exc
    if args.sam_method in {"automatic", "hybrid"} and args.sam_min_mask_region_area > 0:
        try:
            import cv2  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependency 'opencv-python'. It is required when "
                "--sam-min-mask-region-area is greater than 0."
            ) from exc


def load_sam_runner(args: argparse.Namespace, device: str):
    check_sam_dependencies(args)
    from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry

    checkpoint = args.sam_checkpoint.expanduser()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing SAM checkpoint: {checkpoint}")
    if args.sam_model_type not in sam_model_registry:
        available = ", ".join(sorted(sam_model_registry))
        raise ValueError(f"Unknown SAM model type {args.sam_model_type!r}; available: {available}")

    sam = sam_model_registry[args.sam_model_type](checkpoint=str(checkpoint))
    sam.to(device=device)
    if args.sam_method == "point":
        return SamPredictor(sam)
    if args.sam_method == "hybrid":
        return {
            "predictor": SamPredictor(sam),
            "generator": SamAutomaticMaskGenerator(
                sam,
                points_per_side=args.sam_points_per_side,
                pred_iou_thresh=args.sam_pred_iou_thresh,
                stability_score_thresh=args.sam_stability_score_thresh,
                crop_n_layers=args.sam_crop_n_layers,
                min_mask_region_area=args.sam_min_mask_region_area,
            ),
        }
    return SamAutomaticMaskGenerator(
        sam,
        points_per_side=args.sam_points_per_side,
        pred_iou_thresh=args.sam_pred_iou_thresh,
        stability_score_thresh=args.sam_stability_score_thresh,
        crop_n_layers=args.sam_crop_n_layers,
        min_mask_region_area=args.sam_min_mask_region_area,
    )


def resize_for_sam(image: Image.Image, max_side: int) -> tuple[Image.Image, float]:
    width, height = image.size
    longest = max(width, height)
    if longest <= max_side:
        return image, 1.0
    scale = max_side / longest
    resized_size = (round(width * scale), round(height * scale))
    return image.resize(resized_size, Image.Resampling.LANCZOS), 1 / scale


def annotation_bbox_to_original(annotation: dict[str, Any], scale_to_original: float, width: int, height: int) -> BBox:
    x, y, box_width, box_height = annotation["bbox"]
    left = max(0, min(width - 1, math.floor(x * scale_to_original)))
    top = max(0, min(height - 1, math.floor(y * scale_to_original)))
    right = max(left + 1, min(width, math.ceil((x + box_width) * scale_to_original)))
    bottom = max(top + 1, min(height, math.ceil((y + box_height) * scale_to_original)))
    return BBox(left=left, top=top, right=right, bottom=bottom)


def bbox_union(boxes: Iterable[BBox]) -> BBox:
    boxes = list(boxes)
    if not boxes:
        raise ValueError("Cannot union an empty bbox list")
    return BBox(
        left=min(box.left for box in boxes),
        top=min(box.top for box in boxes),
        right=max(box.right for box in boxes),
        bottom=max(box.bottom for box in boxes),
    )


def mask_to_annotation(mask: np.ndarray, score: float) -> dict[str, Any] | None:
    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        return None
    left = int(xs.min())
    top = int(ys.min())
    right = int(xs.max()) + 1
    bottom = int(ys.max()) + 1
    return {
        "bbox": [left, top, right - left, bottom - top],
        "area": int(mask.sum()),
        "predicted_iou": float(score),
        "stability_score": float(score),
    }


def point_prompt_annotations(predictor: Any, image_array: np.ndarray, args: argparse.Namespace) -> list[dict[str, Any]]:
    height, width = image_array.shape[:2]
    predictor.set_image(image_array)
    offsets = [
        (0.0, -0.14),
        (0.0, -0.08),
        (0.0, 0.0),
        (0.0, 0.06),
        (-0.05, 0.02),
        (0.05, 0.02),
    ]
    prompt_sets: list[list[tuple[float, float]]] = []
    prompt_sets.extend([[(args.anchor_x + dx, args.anchor_y + dy)] for dx, dy in offsets])
    prompt_sets.append([(args.anchor_x + dx, args.anchor_y + dy) for dx, dy in offsets])

    annotations: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for prompt_set in prompt_sets:
        points = np.array(
            [
                [
                    min(max(x_ratio, 0.0), 1.0) * width,
                    min(max(y_ratio, 0.0), 1.0) * height,
                ]
                for x_ratio, y_ratio in prompt_set
            ],
            dtype=np.float32,
        )
        labels = np.ones(len(prompt_set), dtype=np.int32)
        masks, scores, _ = predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=True,
        )
        for mask, score in zip(masks, scores):
            annotation = mask_to_annotation(mask, float(score))
            if annotation is None:
                continue
            bbox_key = tuple(int(v) for v in annotation["bbox"])
            if bbox_key in seen:
                continue
            seen.add(bbox_key)
            annotations.append(annotation)
    return annotations


def mask_score(
    annotation: dict[str, Any],
    image_width: int,
    image_height: int,
    anchor_x: float,
    anchor_y: float,
    min_mask_area_ratio: float,
    max_mask_area_ratio: float,
    max_bbox_area_ratio: float,
    edge_margin_ratio: float,
) -> tuple[float | None, dict[str, float]]:
    x, y, box_width, box_height = annotation["bbox"]
    image_area = image_width * image_height
    mask_area_ratio = float(annotation["area"]) / image_area
    bbox_area_ratio = float(box_width * box_height) / image_area
    edge_margin = min(image_width, image_height) * edge_margin_ratio

    metrics = {
        "mask_area_ratio": mask_area_ratio,
        "bbox_area_ratio": bbox_area_ratio,
        "center_x_ratio": (x + box_width / 2) / image_width,
        "center_y_ratio": (y + box_height / 2) / image_height,
    }
    if mask_area_ratio < min_mask_area_ratio or mask_area_ratio > max_mask_area_ratio:
        return None, metrics
    if bbox_area_ratio > max_bbox_area_ratio:
        return None, metrics
    bbox_aspect = max(box_width / max(box_height, 1), box_height / max(box_width, 1))
    if bbox_aspect > 5.0:
        return None, metrics
    if metrics["center_y_ratio"] < 0.50:
        return None, metrics
    if x <= edge_margin or y <= edge_margin:
        return None, metrics
    if x + box_width >= image_width - edge_margin or y + box_height >= image_height - edge_margin:
        return None, metrics

    dx = metrics["center_x_ratio"] - anchor_x
    dy = metrics["center_y_ratio"] - anchor_y
    distance = math.sqrt(dx * dx + dy * dy)
    compactness_penalty = max(0.0, bbox_area_ratio - mask_area_ratio)
    bbox_area_preference = abs(math.log(max(bbox_area_ratio, 1e-8) / 0.03))
    score = distance + 0.05 * compactness_penalty + 0.08 * bbox_area_preference
    return score, metrics


def select_mask_annotation(
    annotations: list[dict[str, Any]],
    image_width: int,
    image_height: int,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], float, dict[str, float]]:
    best: tuple[float, dict[str, Any], dict[str, float]] | None = None
    rejected = 0
    for annotation in annotations:
        score, metrics = mask_score(
            annotation=annotation,
            image_width=image_width,
            image_height=image_height,
            anchor_x=args.anchor_x,
            anchor_y=args.anchor_y,
            min_mask_area_ratio=args.min_mask_area_ratio,
            max_mask_area_ratio=args.max_mask_area_ratio,
            max_bbox_area_ratio=args.max_bbox_area_ratio,
            edge_margin_ratio=args.edge_margin_ratio,
        )
        if score is None:
            rejected += 1
            continue
        if best is None or score < best[0]:
            best = (score, annotation, metrics)

    if best is None:
        raise RuntimeError(
            f"SAM produced {len(annotations)} masks, but none passed filters "
            f"(rejected={rejected}). Try relaxing mask area or edge filters."
        )
    return best[1], best[0], best[2]


def candidate_from_annotation(
    candidate_id: int,
    annotation: dict[str, Any],
    source: str,
    image_width: int,
    image_height: int,
    scale_to_original: float,
    original_width: int,
    original_height: int,
) -> Candidate:
    x, y, box_width, box_height = annotation["bbox"]
    image_area = image_width * image_height
    bbox = annotation_bbox_to_original(
        annotation,
        scale_to_original=scale_to_original,
        width=original_width,
        height=original_height,
    )
    score = float(annotation.get("predicted_iou", annotation.get("stability_score", 0.0)))
    return Candidate(
        candidate_id=candidate_id,
        source=source,
        bbox=bbox,
        area=int(round(float(annotation["area"]) * scale_to_original * scale_to_original)),
        score=score,
        mask_area_ratio=float(annotation["area"]) / image_area,
        bbox_area_ratio=float(box_width * box_height) / image_area,
        center=((x + box_width / 2) / image_width, (y + box_height / 2) / image_height),
    )


def candidate_payload(candidate: Candidate) -> dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "source": candidate.source,
        "bbox": candidate.bbox.as_list(),
        "bbox_size": [candidate.bbox.width, candidate.bbox.height],
        "area": candidate.area,
        "score": candidate.score,
        "mask_area_ratio": candidate.mask_area_ratio,
        "bbox_area_ratio": candidate.bbox_area_ratio,
        "center": list(candidate.center),
    }


def candidate_sort_key(candidate: Candidate) -> tuple[float, float, int]:
    anchor_distance = math.sqrt((candidate.center[0] - DEFAULT_ANCHOR_X) ** 2 + (candidate.center[1] - DEFAULT_ANCHOR_Y) ** 2)
    return (anchor_distance, -candidate.bbox.area, candidate.candidate_id)


def collect_group_candidates(
    group_key: str,
    p000_path: Path,
    runner: Any,
    args: argparse.Namespace,
) -> tuple[list[Candidate], tuple[int, int], tuple[int, int], float]:
    with Image.open(p000_path) as raw_image:
        image = raw_image.convert("RGB")
        original_size = image.size
        sam_image, scale_to_original = resize_for_sam(image, args.sam_max_side)
        sam_array = np.asarray(sam_image)

    annotations_with_source: list[tuple[str, dict[str, Any]]] = []
    if args.sam_method in {"point", "hybrid"}:
        predictor = runner["predictor"] if args.sam_method == "hybrid" else runner
        annotations_with_source.extend(
            ("point", annotation)
            for annotation in point_prompt_annotations(predictor, sam_array, args)
        )
    if args.sam_method in {"automatic", "hybrid"}:
        generator = runner["generator"] if args.sam_method == "hybrid" else runner
        annotations_with_source.extend(
            ("automatic", annotation)
            for annotation in generator.generate(sam_array)
        )

    candidates: list[Candidate] = []
    seen: set[tuple[int, int, int, int]] = set()
    for source, annotation in annotations_with_source:
        bbox = annotation_bbox_to_original(
            annotation,
            scale_to_original=scale_to_original,
            width=original_size[0],
            height=original_size[1],
        )
        bbox_key = tuple(bbox.as_list())
        if bbox_key in seen:
            continue
        seen.add(bbox_key)
        candidate = candidate_from_annotation(
            candidate_id=len(candidates),
            annotation=annotation,
            source=source,
            image_width=sam_image.width,
            image_height=sam_image.height,
            scale_to_original=scale_to_original,
            original_width=original_size[0],
            original_height=original_size[1],
        )
        if candidate.bbox_area_ratio > args.max_bbox_area_ratio:
            continue
        if candidate.mask_area_ratio < args.min_mask_area_ratio:
            continue
        candidates.append(candidate)

    candidates.sort(key=candidate_sort_key)
    candidates = candidates[: args.max_candidates]
    candidates = [
        Candidate(
            candidate_id=index,
            source=candidate.source,
            bbox=candidate.bbox,
            area=candidate.area,
            score=candidate.score,
            mask_area_ratio=candidate.mask_area_ratio,
            bbox_area_ratio=candidate.bbox_area_ratio,
            center=candidate.center,
        )
        for index, candidate in enumerate(candidates)
    ]
    if not candidates:
        raise RuntimeError(f"No candidates generated for {group_key} from {p000_path}")
    return candidates, original_size, sam_image.size, scale_to_original


def draw_candidate_contact_sheet(
    p000_path: Path,
    candidates: list[Candidate],
    output_path: Path,
    thumb_size: int,
    columns: int,
) -> None:
    with Image.open(p000_path) as raw_image:
        image = raw_image.convert("RGB")
    rows = math.ceil(len(candidates) / columns)
    label_height = 42
    sheet = Image.new("RGB", (columns * thumb_size, rows * (thumb_size + label_height)), "white")

    for candidate in candidates:
        panel = image.copy()
        draw = ImageDraw.Draw(panel)
        draw.rectangle(candidate.bbox.as_list(), outline=(255, 80, 0), width=max(4, image.width // 650))
        label = (
            f"id={candidate.candidate_id} {candidate.source} "
            f"box={candidate.bbox.width}x{candidate.bbox.height} "
            f"area={candidate.bbox_area_ratio:.3f}"
        )
        draw.rectangle([12, 12, min(image.width - 12, 12 + len(label) * 18), 70], fill=(0, 0, 0))
        draw.text((24, 28), label, fill=(255, 255, 255))
        panel.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)

        x_index = candidate.candidate_id % columns
        y_index = candidate.candidate_id // columns
        x = x_index * thumb_size
        y = y_index * (thumb_size + label_height)
        sheet.paste(panel, (x, y))
        sheet_draw = ImageDraw.Draw(sheet)
        sheet_draw.text(
            (x + 8, y + thumb_size + 8),
            f"id {candidate.candidate_id}: {candidate.bbox.as_list()}",
            fill=(0, 0, 0),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92, subsampling=0)


def compute_clamped_square_box(
    image_width: int,
    image_height: int,
    center_x: float,
    center_y: float,
    side: int,
) -> BBox:
    side = max(1, min(side, image_width, image_height))
    left = round(center_x - side / 2)
    top = round(center_y - side / 2)
    left = max(0, min(left, image_width - side))
    top = max(0, min(top, image_height - side))
    return BBox(left=left, top=top, right=left + side, bottom=top + side)


def compute_crop_box(bbox: BBox, image_width: int, image_height: int, args: argparse.Namespace) -> BBox:
    area_side = math.sqrt(bbox.area / args.target_bbox_area_ratio)
    dimension_side = max(bbox.width, bbox.height) / args.bbox_side_fraction
    side = math.ceil(max(area_side, dimension_side, args.min_crop_side))
    center_x, center_y = bbox.center
    return compute_clamped_square_box(
        image_width=image_width,
        image_height=image_height,
        center_x=center_x,
        center_y=center_y,
        side=side,
    )


def infer_view_crop(view_dir: Path, dataset_root: Path, generator: Any, args: argparse.Namespace) -> ViewCrop:
    p000_path = view_dir / "p000.jpg"
    if not p000_path.exists():
        raise FileNotFoundError(f"Missing p000.jpg for view: {view_dir}")

    with Image.open(p000_path) as raw_image:
        image = raw_image.convert("RGB")
        original_size = image.size
        sam_image, scale_to_original = resize_for_sam(image, args.sam_max_side)
        sam_array = np.asarray(sam_image)

    if args.sam_method == "point":
        annotations = point_prompt_annotations(generator, sam_array, args)
    elif args.sam_method == "hybrid":
        annotations = point_prompt_annotations(generator["predictor"], sam_array, args)
        annotations.extend(generator["generator"].generate(sam_array))
    else:
        annotations = generator.generate(sam_array)
    selected, score, metrics = select_mask_annotation(
        annotations=annotations,
        image_width=sam_image.width,
        image_height=sam_image.height,
        args=args,
    )
    bbox = annotation_bbox_to_original(
        selected,
        scale_to_original=scale_to_original,
        width=original_size[0],
        height=original_size[1],
    )
    crop_box = compute_crop_box(
        bbox=bbox,
        image_width=original_size[0],
        image_height=original_size[1],
        args=args,
    )
    return ViewCrop(
        view_key=str(view_dir.relative_to(dataset_root)),
        p000_path=p000_path,
        original_size=original_size,
        resized_size=sam_image.size,
        scale_to_original=scale_to_original,
        mask_score=score,
        mask_area=int(round(float(selected["area"]) * scale_to_original * scale_to_original)),
        mask_area_ratio=metrics["mask_area_ratio"],
        bbox=bbox,
        crop_box=crop_box,
    )


def draw_qc_overlay(view_crop: ViewCrop, output_path: Path, final_size: int) -> None:
    with Image.open(view_crop.p000_path) as raw_image:
        image = raw_image.convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle(view_crop.crop_box.as_list(), outline=(0, 180, 255), width=8)
    draw.rectangle(view_crop.bbox.as_list(), outline=(255, 80, 0), width=8)
    label = (
        f"{view_crop.view_key} bbox={view_crop.bbox.as_list()} "
        f"crop={view_crop.crop_box.as_list()} final={final_size}"
    )
    draw.rectangle([16, 16, 16 + min(len(label) * 16, image.width - 32), 64], fill=(0, 0, 0))
    draw.text((24, 28), label, fill=(255, 255, 255))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=90, subsampling=0)


def save_image(image: Image.Image, output_path: Path, jpeg_quality: int) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs: dict[str, Any] = {}
    if output_path.suffix.lower() in {".jpg", ".jpeg"}:
        save_kwargs.update({"quality": jpeg_quality, "subsampling": 0})
    image.save(output_path, **save_kwargs)
    return output_path.stat().st_size


def copy_static_maps(input_maps_dir: Path, output_maps_dir: Path, dry_run: bool) -> None:
    if dry_run or not input_maps_dir.exists():
        return
    output_maps_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(input_maps_dir.iterdir()):
        if path.name == "captures.jsonl" or not path.is_file():
            continue
        shutil.copy2(path, output_maps_dir / path.name)


def print_progress(processed: int, total: int, progress_every: int, force: bool = False) -> None:
    if total <= 0:
        return
    if not force and processed % progress_every != 0:
        return
    fraction = min(processed / total, 1.0)
    bar_width = 28
    filled = round(bar_width * fraction)
    bar = "#" * filled + "-" * (bar_width - filled)
    print(f"\r[{bar}] {fraction * 100:6.2f}% processed={processed}/{total}", end="", flush=True)


def view_crop_payload(view_crop: ViewCrop) -> dict[str, Any]:
    return {
        "p000_path": str(view_crop.p000_path),
        "original_size": list(view_crop.original_size),
        "sam_resized_size": list(view_crop.resized_size),
        "scale_to_original": view_crop.scale_to_original,
        "mask_score": view_crop.mask_score,
        "mask_area": view_crop.mask_area,
        "mask_area_ratio_on_sam_image": view_crop.mask_area_ratio,
        "bbox": view_crop.bbox.as_list(),
        "bbox_size": [view_crop.bbox.width, view_crop.bbox.height],
        "crop_box": view_crop.crop_box.as_list(),
        "crop_size": [view_crop.crop_box.width, view_crop.crop_box.height],
    }


def build_view_crops(args: argparse.Namespace, dataset_root: Path, generator: Any) -> dict[str, ViewCrop]:
    view_crops: dict[str, ViewCrop] = {}
    view_dirs = iter_view_dirs(dataset_root)
    print(f"Found {len(view_dirs)} view directories.")
    for index, view_dir in enumerate(view_dirs, start=1):
        view_crop = infer_view_crop(view_dir, dataset_root, generator, args)
        view_crops[view_crop.view_key] = view_crop
        print(
            f"[{index}/{len(view_dirs)}] {view_crop.view_key} "
            f"bbox={view_crop.bbox.as_list()} crop_side={view_crop.crop_box.width}"
        )
    return view_crops


def prepare_candidates(args: argparse.Namespace, dataset_root: Path, output_root: Path, runner: Any) -> dict[str, Any]:
    candidate_root = output_root / "candidate_qc"
    groups = iter_group_keys(dataset_root)
    candidate_data: dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "candidate_source_light": args.candidate_source_light,
        "groups": {},
    }
    selection_template: dict[str, dict[str, list[int]]] = {}
    print(f"Preparing candidates for {len(groups)} groups from {args.candidate_source_light}.")
    for index, group_key in enumerate(groups, start=1):
        p000_path = dataset_root / args.candidate_source_light / group_key / "p000.jpg"
        if not p000_path.exists():
            raise FileNotFoundError(f"Missing candidate source image: {p000_path}")
        candidates, original_size, sam_size, scale_to_original = collect_group_candidates(
            group_key=group_key,
            p000_path=p000_path,
            runner=runner,
            args=args,
        )
        qc_name = group_key.replace("/", "__") + ".jpg"
        draw_candidate_contact_sheet(
            p000_path=p000_path,
            candidates=candidates,
            output_path=candidate_root / qc_name,
            thumb_size=args.candidate_thumb_size,
            columns=args.candidate_columns,
        )
        candidate_data["groups"][group_key] = {
            "p000_path": str(p000_path),
            "qc_path": str(candidate_root / qc_name),
            "original_size": list(original_size),
            "sam_resized_size": list(sam_size),
            "scale_to_original": scale_to_original,
            "candidates": [candidate_payload(candidate) for candidate in candidates],
        }
        selection_template[group_key] = {"candidate_ids": []}
        print(f"[{index}/{len(groups)}] {group_key}: {len(candidates)} candidates -> {candidate_root / qc_name}")

    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "candidate_masks.json", candidate_data)
    write_json(output_root / "manual_selections.template.json", selection_template)
    print(f"Candidates: {output_root / 'candidate_masks.json'}")
    print(f"Selection template: {output_root / 'manual_selections.template.json'}")
    print(f"QC contact sheets: {candidate_root}")
    return candidate_data


def load_manual_selections(path: Path) -> dict[str, list[int]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing manual selections file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    selections: dict[str, list[int]] = {}
    for group_key, value in payload.items():
        if isinstance(value, dict):
            ids = value.get("candidate_ids", [])
        else:
            ids = value
        if not isinstance(ids, list) or not ids:
            raise ValueError(f"{path}: {group_key} must contain a non-empty candidate_ids list")
        selections[group_key] = [int(candidate_id) for candidate_id in ids]
    return selections


def build_group_crops_from_manual(args: argparse.Namespace, dataset_root: Path) -> dict[str, ViewCrop]:
    candidate_path = args.candidate_masks.expanduser()
    candidate_payload_root = json.loads(candidate_path.read_text(encoding="utf-8"))
    selections = load_manual_selections(args.manual_selections.expanduser())
    groups = candidate_payload_root.get("groups", {})
    group_crops: dict[str, ViewCrop] = {}
    for group_key in iter_group_keys(dataset_root):
        if group_key not in selections:
            raise ValueError(f"Missing manual selection for group {group_key}")
        if group_key not in groups:
            raise ValueError(f"Missing candidates for group {group_key} in {candidate_path}")

        candidates_by_id = {
            int(candidate["candidate_id"]): candidate
            for candidate in groups[group_key].get("candidates", [])
        }
        selected_boxes: list[BBox] = []
        for candidate_id in selections[group_key]:
            if candidate_id not in candidates_by_id:
                raise ValueError(f"Unknown candidate id {candidate_id} for group {group_key}")
            bbox_values = candidates_by_id[candidate_id]["bbox"]
            selected_boxes.append(BBox(*[int(value) for value in bbox_values]))

        bbox = bbox_union(selected_boxes)
        original_size = tuple(int(value) for value in groups[group_key]["original_size"])
        crop_box = compute_crop_box(
            bbox=bbox,
            image_width=original_size[0],
            image_height=original_size[1],
            args=args,
        )
        p000_path = Path(groups[group_key]["p000_path"])
        group_crops[group_key] = ViewCrop(
            view_key=group_key,
            p000_path=p000_path,
            original_size=original_size,
            resized_size=tuple(int(value) for value in groups[group_key]["sam_resized_size"]),
            scale_to_original=float(groups[group_key]["scale_to_original"]),
            mask_score=0.0,
            mask_area=sum(box.area for box in selected_boxes),
            mask_area_ratio=0.0,
            bbox=bbox,
            crop_box=crop_box,
        )
        print(
            f"{group_key}: selected={selections[group_key]} "
            f"bbox={bbox.as_list()} crop_side={crop_box.width}"
        )
    return group_crops


def process_images(
    args: argparse.Namespace,
    dataset_root: Path,
    output_root: Path,
    crops: dict[str, ViewCrop],
    final_size: int,
) -> dict[str, int]:
    report_path = output_root / "crop_report.jsonl"
    captures_path = dataset_root / "maps" / "captures.jsonl"
    capture_rows = load_jsonl(captures_path)
    output_capture_rows: list[dict[str, Any]] = []
    counts = {
        "total_rows": len(capture_rows),
        "cropped": 0,
        "dry_run": 0,
        "missing": 0,
        "unsupported": 0,
        "unknown_view": 0,
        "errors": 0,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as report:
        for row in capture_rows:
            image_path_value = row.get("image_path", "")
            relative_path = Path(image_path_value)
            input_path = dataset_root / relative_path
            output_path = output_root / relative_path
            view_key = view_key_from_image_path(relative_path)
            group_key = group_key_from_image_path(relative_path)
            base_payload: dict[str, Any] = {
                "image_path": image_path_value,
                "input_path": str(input_path),
                "output_path": str(output_path),
                "view_key": view_key,
                "group_key": group_key,
            }

            if relative_path.suffix.lower() not in IMAGE_EXTENSIONS:
                counts["unsupported"] += 1
                write_jsonl(report, {**base_payload, "status": "unsupported_extension"})
                print_progress(sum(counts[k] for k in ("cropped", "dry_run", "missing", "unsupported", "unknown_view", "errors")), counts["total_rows"], args.progress_every)
                continue
            if group_key not in crops:
                counts["unknown_view"] += 1
                write_jsonl(report, {**base_payload, "status": "unknown_view"})
                print_progress(sum(counts[k] for k in ("cropped", "dry_run", "missing", "unsupported", "unknown_view", "errors")), counts["total_rows"], args.progress_every)
                continue
            if not input_path.exists():
                counts["missing"] += 1
                write_jsonl(report, {**base_payload, "status": "missing"})
                print_progress(sum(counts[k] for k in ("cropped", "dry_run", "missing", "unsupported", "unknown_view", "errors")), counts["total_rows"], args.progress_every)
                continue

            view_crop = crops[group_key]
            try:
                with Image.open(input_path) as raw_image:
                    image = raw_image.convert("RGB")
                    cropped = image.crop(view_crop.crop_box.as_list())
                    resized = cropped.resize((final_size, final_size), Image.Resampling.LANCZOS)

                size_bytes = int(row.get("size_bytes", 0) or 0)
                if args.dry_run:
                    counts["dry_run"] += 1
                else:
                    size_bytes = save_image(resized, output_path, args.jpeg_quality)
                    counts["cropped"] += 1

                output_row = dict(row)
                output_row["size_bytes"] = size_bytes
                output_capture_rows.append(output_row)
                write_jsonl(
                    report,
                    {
                        **base_payload,
                        "status": "dry_run" if args.dry_run else "cropped",
                        "original_size": [image.width, image.height],
                        "crop_box": view_crop.crop_box.as_list(),
                        "crop_size": [view_crop.crop_box.width, view_crop.crop_box.height],
                        "final_size": [final_size, final_size],
                    },
                )
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

            processed = sum(counts[k] for k in ("cropped", "dry_run", "missing", "unsupported", "unknown_view", "errors"))
            print_progress(processed, counts["total_rows"], args.progress_every)

    print_progress(counts["total_rows"], counts["total_rows"], args.progress_every, force=True)
    print()

    if not args.dry_run:
        output_maps_dir = output_root / "maps"
        output_maps_dir.mkdir(parents=True, exist_ok=True)
        with (output_maps_dir / "captures.jsonl").open("w", encoding="utf-8") as handle:
            for row in output_capture_rows:
                write_jsonl(handle, row)

    return counts


def write_crop_config(
    args: argparse.Namespace,
    output_root: Path,
    dataset_root: Path,
    crops: dict[str, ViewCrop],
    final_size: int,
    device: str,
    counts: dict[str, int] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "dry_run": args.dry_run,
        "device": device,
        "sam": {
            "checkpoint": str(args.sam_checkpoint.expanduser()),
            "model_type": args.sam_model_type,
            "method": args.sam_method,
            "max_side": args.sam_max_side,
            "points_per_side": args.sam_points_per_side,
            "pred_iou_thresh": args.sam_pred_iou_thresh,
            "stability_score_thresh": args.sam_stability_score_thresh,
            "crop_n_layers": args.sam_crop_n_layers,
            "min_mask_region_area": args.sam_min_mask_region_area,
        },
        "crop": {
            "target_bbox_area_ratio": args.target_bbox_area_ratio,
            "bbox_side_fraction": args.bbox_side_fraction,
            "min_crop_side": args.min_crop_side,
            "final_size": final_size,
        },
        "mask_selection": {
            "anchor_x": args.anchor_x,
            "anchor_y": args.anchor_y,
            "min_mask_area_ratio": args.min_mask_area_ratio,
            "max_mask_area_ratio": args.max_mask_area_ratio,
            "max_bbox_area_ratio": args.max_bbox_area_ratio,
            "edge_margin_ratio": args.edge_margin_ratio,
        },
        "crops": {
            view_key: view_crop_payload(view_crop)
            for view_key, view_crop in sorted(crops.items())
        },
    }
    if counts is not None:
        payload["counts"] = counts
    write_json(output_root / "crop_config.json", payload)


def process_dataset(args: argparse.Namespace) -> dict[str, int]:
    dataset_root = args.dataset_root.expanduser()
    output_root = args.output_root.expanduser()
    if args.jpeg_quality > 100:
        raise ValueError("--jpeg-quality must be <= 100")
    if not dataset_root.exists():
        raise FileNotFoundError(f"Missing dataset root: {dataset_root}")
    if not (dataset_root / "maps" / "captures.jsonl").exists():
        raise FileNotFoundError(f"Missing captures map: {dataset_root / 'maps' / 'captures.jsonl'}")
    if args.mode == "crop" and output_root.exists() and any(output_root.iterdir()) and not args.overwrite and not args.dry_run:
        raise FileExistsError(f"Output root is not empty: {output_root}. Use --overwrite to add/replace files.")
    if args.mode == "crop" and args.manual_selections is None:
        raise ValueError("--manual-selections is required when --mode crop")

    device = resolve_device(args.device)
    print(f"Using device: {device}")
    generator = load_sam_runner(args, device)
    if args.mode == "prepare-candidates":
        prepare_candidates(args, dataset_root, output_root, generator)
        return {"groups": len(iter_group_keys(dataset_root))}

    view_crops = build_group_crops_from_manual(args, dataset_root)
    final_size = min(view_crop.crop_box.width for view_crop in view_crops.values())
    if final_size < args.min_crop_side:
        raise RuntimeError(f"Computed final_size={final_size}, below min_crop_side={args.min_crop_side}")
    print(f"Final resize size: {final_size}x{final_size}")

    output_root.mkdir(parents=True, exist_ok=True)
    copy_static_maps(dataset_root / "maps", output_root / "maps", args.dry_run)
    write_crop_config(args, output_root, dataset_root, view_crops, final_size, device)

    qc_root = output_root / "crop_qc"
    for view_key, view_crop in sorted(view_crops.items()):
        draw_qc_overlay(view_crop, qc_root / f"{view_key.replace('/', '__')}.jpg", final_size)

    counts = process_images(args, dataset_root, output_root, view_crops, final_size)
    counts["crops"] = len(view_crops)
    counts["final_size"] = final_size
    write_crop_config(args, output_root, dataset_root, view_crops, final_size, device, counts)

    print(json.dumps(counts, indent=2, sort_keys=True))
    print(f"Config: {output_root / 'crop_config.json'}")
    print(f"Report: {output_root / 'crop_report.jsonl'}")
    print(f"QC overlays: {qc_root}")
    if args.dry_run:
        print("Dry run: images and rewritten captures.jsonl were not written.")
    else:
        print(f"Output dataset: {output_root}")
    return counts


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare SAM candidates or crop test_ver3 with manual candidate selections."
    )
    parser.add_argument("--mode", choices=("prepare-candidates", "crop"), default="crop")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--sam-checkpoint", type=Path, required=True)
    parser.add_argument("--sam-model-type", default=DEFAULT_SAM_MODEL_TYPE)
    parser.add_argument("--sam-method", choices=("point", "automatic", "hybrid"), default=DEFAULT_SAM_METHOD)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--dry-run", action="store_true", help="Write config/report/QC only; do not write cropped images.")
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into a non-empty output root.")
    parser.add_argument("--sam-max-side", type=positive_int, default=DEFAULT_SAM_MAX_SIDE)
    parser.add_argument("--sam-points-per-side", type=positive_int, default=16)
    parser.add_argument("--sam-pred-iou-thresh", type=ratio, default=0.88)
    parser.add_argument("--sam-stability-score-thresh", type=ratio, default=0.95)
    parser.add_argument("--sam-crop-n-layers", type=int, default=0)
    parser.add_argument("--sam-min-mask-region-area", type=positive_int, default=100)
    parser.add_argument("--candidate-source-light", default="l001")
    parser.add_argument("--candidate-masks", type=Path, default=DEFAULT_OUTPUT_ROOT / "candidate_masks.json")
    parser.add_argument("--manual-selections", type=Path)
    parser.add_argument("--max-candidates", type=positive_int, default=12)
    parser.add_argument("--candidate-thumb-size", type=positive_int, default=520)
    parser.add_argument("--candidate-columns", type=positive_int, default=3)
    parser.add_argument("--target-bbox-area-ratio", type=ratio, default=DEFAULT_TARGET_BBOX_AREA_RATIO)
    parser.add_argument("--bbox-side-fraction", type=ratio, default=DEFAULT_BBOX_SIDE_FRACTION)
    parser.add_argument("--min-crop-side", type=positive_int, default=DEFAULT_MIN_CROP_SIDE)
    parser.add_argument("--anchor-x", type=ratio, default=DEFAULT_ANCHOR_X)
    parser.add_argument("--anchor-y", type=ratio, default=DEFAULT_ANCHOR_Y)
    parser.add_argument("--min-mask-area-ratio", type=positive_float, default=0.00005)
    parser.add_argument("--max-mask-area-ratio", type=ratio, default=0.15)
    parser.add_argument("--max-bbox-area-ratio", type=ratio, default=0.35)
    parser.add_argument("--edge-margin-ratio", type=float, default=0.002)
    parser.add_argument("--jpeg-quality", type=positive_int, default=DEFAULT_JPEG_QUALITY)
    parser.add_argument("--progress-every", type=positive_int, default=DEFAULT_PROGRESS_EVERY)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    process_dataset(args)


if __name__ == "__main__":
    main()
