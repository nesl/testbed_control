"""Collect a small ImageNet-ES Diverse-style camera dataset.

This is an independent collection entry point.  It deliberately does not alter
or invoke ``control.py``'s main capture loop; it only reuses its tested Amaran
and Sony controller classes for real hardware access.

Default collection protocol (per printed sample and manual zoom position):

* two manually selected zoom positions (``z001`` and ``z002`` by default);
* one Amaran light at 1%, 20%, 50%, 70%, and 100% (API values
  10, 200, 500, 700, and 1000), fixed at 5600 K;
* three independent auto-exposure shots at each light level;
* the 27 manual settings from Table 8 of "Adaptive Camera Sensor for Vision
  Models" (ICLR 2025): aperture {f/5, f/9, f/16}, shutter
  {1/4, 1/60, 1/1000}, and ISO {250, 2000, 16000}.

Run from the repository root, normally through the Sony camera launcher::

    sonycam replicate_capture.py --plan-only
    sonycam replicate_capture.py --output-dir data_replicate/replicated_capture

After a complete light/exposure sweep at ``z001``, the script pauses so the
operator can manually change zoom before running the complete sweep at
``z002``.  The zoom count is configurable.  The script is resumable. Valid JPEGs are never overwritten, missing JSONL
records are recovered from deterministic paths, and missing/corrupt captures
are acquired again.  A capture timeout is recorded and the run continues with
the next image; other capture errors stop the run immediately.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import csv
import hashlib
import json
import os
import sys
import time
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


PAPER_TITLE = "Adaptive Camera Sensor for Vision Models"
PAPER_TABLE = "Table 8 (ImageNet-ES Diverse test set)"
SCHEMA_VERSION = 4

DEFAULT_LABELS = Path("data_replicate/manual_dataset/labels.csv")
DEFAULT_OUTPUT_DIR = Path("data_replicate/replicated_capture")
DEFAULT_CCT = 5600
DEFAULT_LIGHT_INTENSITIES = (10, 200, 500, 700, 1000)
DEFAULT_APERTURES = (5.0, 9.0, 16.0)
DEFAULT_SHUTTERS = ("1/4", "1/60", "1/1000")
DEFAULT_ISOS = (250, 2000, 16000)
DEFAULT_AE_SHOTS = 3
DEFAULT_ZOOM_COUNT = 2
DEFAULT_CAPTURE_TIMEOUT_SECONDS = 180.0
DEFAULT_PRE_CAPTURE_CLEAR_TIMEOUT_SECONDS = 5.0
DEFAULT_START_DELAY_SECONDS = 10.0
DEFAULT_LIGHT_API_SECRET_KEY = "cDdzYXNkbXM5d2V2a3EwaTJ0Z2tocHRlNjE2NWs5ODY="

# A valid 1x1 JPEG used only by --dry-run.  Keeping dry-run artifacts valid is
# important because the same resume validation is exercised in dry-run mode.
DRY_RUN_JPEG = base64.b64decode(
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////"
    "2wBDAf//////////////////////////////////////////////////////////////////////////////////////"
    "wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAf/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIQAxAAAAF//8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABBQJ//8QAFBEBAAAAAAAAAAAAAAAAAAAAAP/aAAgBAwEBPwF//8QAFBEBAAAAAAAAAAAAAAAAAAAAAP/aAAgBAgEBPwF//8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQAGPwJ//8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABPyF//9oADAMBAAIAAwAAABD/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oACAEDAQE/EB//xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oACAECAQE/EB//xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oACAEBAAE/EB//2Q=="
)

REQUIRED_LABEL_COLUMNS = {
    "sample_id",
    "original_path",
    "rendered_path",
    "pdf_page",
    "class_index",
    "wnid",
    "class_name",
    "source_relative_path",
}


def timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def session_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def parse_intensities(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("light intensities must be integers") from exc
    if not values:
        raise argparse.ArgumentTypeError("light intensity list cannot be empty")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("light intensities must be unique")
    if any(value < 0 or value > 1000 for value in values):
        raise argparse.ArgumentTypeError("light intensities must be in [0, 1000]")
    return values


def light_slug(intensity: int) -> str:
    return f"b{intensity:03d}"


def light_percent(intensity: int) -> float:
    return intensity / 10.0


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


@dataclass(frozen=True)
class Sample:
    sample_id: str
    original_path: str
    rendered_path: str
    pdf_page: int
    class_index: int
    wnid: str
    class_name: str
    source_relative_path: str


@dataclass(frozen=True)
class ManualParameter:
    parameter_number: int
    aperture: float
    shutter_speed: str
    iso: int

    @property
    def parameter_id(self) -> str:
        return f"p{self.parameter_number:03d}"


@dataclass(frozen=True)
class CaptureTask:
    plan_index: int
    sample_index: int
    sample: Sample
    zoom_index: int
    light_intensity: int
    exposure_mode: str
    ae_shot: int | None = None
    manual_parameter: ManualParameter | None = None

    @property
    def light_id(self) -> str:
        return light_slug(self.light_intensity)

    @property
    def zoom_id(self) -> str:
        return f"z{self.zoom_index:03d}"

    @property
    def capture_key(self) -> str:
        if self.exposure_mode == "auto":
            capture_id = f"ae_{self.ae_shot:02d}"
        else:
            assert self.manual_parameter is not None
            capture_id = self.manual_parameter.parameter_id
        return f"{self.sample.sample_id}|{self.zoom_id}|{self.light_id}|{capture_id}"

    def relative_path(self) -> Path:
        root = Path(self.sample.sample_id) / self.zoom_id / self.light_id
        if self.exposure_mode == "auto":
            return root / "ae" / f"ae_{self.ae_shot:02d}.jpg"
        assert self.manual_parameter is not None
        return root / "manual" / f"{self.manual_parameter.parameter_id}.jpg"


def load_samples(labels_path: Path) -> list[Sample]:
    labels_path = labels_path.expanduser().resolve()
    if not labels_path.is_file():
        raise FileNotFoundError(f"labels CSV not found: {labels_path}")

    with labels_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_LABEL_COLUMNS - columns
        if missing:
            raise ValueError(f"labels CSV is missing columns: {', '.join(sorted(missing))}")
        rows = list(reader)

    samples: list[Sample] = []
    seen_ids: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        sample_id = row["sample_id"].strip()
        if not sample_id or sample_id in seen_ids:
            raise ValueError(f"empty or duplicate sample_id at CSV row {row_number}: {sample_id!r}")
        seen_ids.add(sample_id)
        try:
            sample = Sample(
                sample_id=sample_id,
                original_path=row["original_path"].strip(),
                rendered_path=row["rendered_path"].strip(),
                pdf_page=int(row["pdf_page"]),
                class_index=int(row["class_index"]),
                wnid=row["wnid"].strip(),
                class_name=row["class_name"].strip(),
                source_relative_path=row["source_relative_path"].strip(),
            )
        except ValueError as exc:
            raise ValueError(f"invalid numeric field at CSV row {row_number}") from exc
        for label, relative in (
            ("original", sample.original_path),
            ("rendered", sample.rendered_path),
        ):
            source_path = labels_path.parent / relative
            if not source_path.is_file():
                raise FileNotFoundError(
                    f"{label} image for {sample_id} does not exist: {source_path}"
                )
        samples.append(sample)

    if not samples:
        raise ValueError(f"labels CSV contains no samples: {labels_path}")
    return samples


def build_manual_parameters() -> list[ManualParameter]:
    parameters: list[ManualParameter] = []
    number = 0
    # This ordering exactly follows Table 8: aperture blocks, shutter groups,
    # then ISO values within each shutter group.
    for aperture in DEFAULT_APERTURES:
        for shutter in DEFAULT_SHUTTERS:
            for iso in DEFAULT_ISOS:
                number += 1
                parameters.append(
                    ManualParameter(
                        parameter_number=number,
                        aperture=aperture,
                        shutter_speed=shutter,
                        iso=iso,
                    )
                )
    if number != 27:
        raise AssertionError(f"expected 27 manual parameters, generated {number}")
    return parameters


def build_capture_tasks(
    samples: Iterable[Sample], light_intensities: Iterable[int], zoom_count: int
) -> list[CaptureTask]:
    if zoom_count < 1:
        raise ValueError("zoom_count must be at least 1")
    tasks: list[CaptureTask] = []
    manual_parameters = build_manual_parameters()
    plan_index = 0
    for sample_index, sample in enumerate(samples, start=1):
        for zoom_index in range(1, zoom_count + 1):
            for intensity in light_intensities:
                for ae_shot in range(1, DEFAULT_AE_SHOTS + 1):
                    plan_index += 1
                    tasks.append(
                        CaptureTask(
                            plan_index=plan_index,
                            sample_index=sample_index,
                            sample=sample,
                            zoom_index=zoom_index,
                            light_intensity=intensity,
                            exposure_mode="auto",
                            ae_shot=ae_shot,
                        )
                    )
                for parameter in manual_parameters:
                    plan_index += 1
                    tasks.append(
                        CaptureTask(
                            plan_index=plan_index,
                            sample_index=sample_index,
                            sample=sample,
                            zoom_index=zoom_index,
                            light_intensity=intensity,
                            exposure_mode="manual",
                            manual_parameter=parameter,
                        )
                    )
    keys = [task.capture_key for task in tasks]
    paths = [str(task.relative_path()) for task in tasks]
    if len(set(keys)) != len(keys) or len(set(paths)) != len(paths):
        raise AssertionError("capture plan contains duplicate keys or output paths")
    return tasks


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_plan_manifest(
    labels_path: Path,
    samples: list[Sample],
    light_intensities: tuple[int, ...],
    cct: int,
    zoom_count: int,
) -> dict[str, Any]:
    tasks = build_capture_tasks(samples, light_intensities, zoom_count)
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": timestamp(),
        "paper": {"title": PAPER_TITLE, "parameter_source": PAPER_TABLE},
        "replication_variant": (
            "single Amaran light intensity sweep; not the paper's two-light L1-L7 geometry"
        ),
        "labels": {
            "source": str(labels_path),
            "sha256": file_sha256(labels_path),
            "samples": [asdict(sample) for sample in samples],
        },
        "capture_configuration": {
            "focus_mode": "camera/lens setting preserved; not remotely changed",
            "metering_mode": "camera setting preserved; not remotely changed",
            "exposure_mode_control": "program auto for AE shots; manual for p001-p027",
            "zoom_control": "manual between complete setting sweeps",
            "zoom_count_per_sample": zoom_count,
            "light_cct_kelvin": cct,
            "light_intensities": list(light_intensities),
            "light_percentages": [light_percent(value) for value in light_intensities],
            "auto_exposure_shots_per_light": DEFAULT_AE_SHOTS,
            "manual_parameter_order": "aperture -> shutter_speed -> iso",
            "manual_parameters": [asdict(value) for value in build_manual_parameters()],
            "trigger_policy": (
                "normal for every AE and first acquired manual shot per light; fast thereafter"
            ),
        },
        "expected_counts": {
            "samples": len(samples),
            "zooms_per_sample": zoom_count,
            "lights_per_zoom": len(light_intensities),
            "auto_per_light": DEFAULT_AE_SHOTS,
            "manual_per_light": len(build_manual_parameters()),
            "images_per_light": DEFAULT_AE_SHOTS + len(build_manual_parameters()),
            "images_per_zoom": len(light_intensities)
            * (DEFAULT_AE_SHOTS + len(build_manual_parameters())),
            "total_images": len(tasks),
        },
    }


def immutable_plan_data(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": plan.get("schema_version"),
        "paper": plan.get("paper"),
        "replication_variant": plan.get("replication_variant"),
        "labels": plan.get("labels"),
        "capture_configuration": plan.get("capture_configuration"),
        "expected_counts": plan.get("expected_counts"),
    }


def can_migrate_three_ae_plan(
    output_dir: Path,
    existing: dict[str, Any],
    expected: dict[str, Any],
    expected_tasks: Iterable[CaptureTask],
) -> bool:
    """Allow the in-progress schema-3 plan to drop unused AE shots 4 and 5."""
    if existing.get("schema_version") != 3 or expected.get("schema_version") != 4:
        return False

    existing_base = immutable_plan_data(existing)
    expected_base = immutable_plan_data(expected)
    existing_base["schema_version"] = expected_base["schema_version"]
    existing_config = dict(existing_base.get("capture_configuration") or {})
    expected_config = dict(expected_base.get("capture_configuration") or {})
    if existing_config.pop("auto_exposure_shots_per_light", None) != 5:
        return False
    if expected_config.pop("auto_exposure_shots_per_light", None) != 3:
        return False
    existing_base["capture_configuration"] = existing_config
    expected_base["capture_configuration"] = expected_config

    variable_count_fields = {
        "auto_per_light",
        "images_per_light",
        "images_per_zoom",
        "total_images",
    }
    existing_counts = dict(existing_base.get("expected_counts") or {})
    expected_counts = dict(expected_base.get("expected_counts") or {})
    for field in variable_count_fields:
        existing_counts.pop(field, None)
        expected_counts.pop(field, None)
    existing_base["expected_counts"] = existing_counts
    expected_base["expected_counts"] = expected_counts
    if existing_base != expected_base:
        return False

    task_list = list(expected_tasks)
    expected_keys = {task.capture_key for task in task_list}
    expected_paths = {str(task.relative_path()) for task in task_list}
    captures_path = output_dir / "captures.jsonl"
    records, _ = load_capture_records(captures_path)
    if any(record["capture_key"] not in expected_keys for record in records):
        return False
    existing_jpegs = {
        str(path.relative_to(output_dir)) for path in output_dir.rglob("*.jpg")
    }
    return existing_jpegs <= expected_paths


def ensure_plan_file(
    output_dir: Path,
    expected: dict[str, Any],
    expected_tasks: Iterable[CaptureTask],
) -> None:
    plan_path = output_dir / "plan.json"
    if plan_path.exists():
        try:
            existing = json.loads(plan_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"existing plan is invalid JSON: {plan_path}: {exc}") from exc
        if immutable_plan_data(existing) != immutable_plan_data(expected):
            captures_path = output_dir / "captures.jsonl"
            has_capture_records = captures_path.exists() and captures_path.stat().st_size > 0
            has_jpegs = any(output_dir.rglob("*.jpg"))
            if not has_capture_records and not has_jpegs:
                atomic_write_json(plan_path, expected)
                print(f"Updated empty capture plan to schema {SCHEMA_VERSION}: {plan_path}")
                return
            if can_migrate_three_ae_plan(output_dir, existing, expected, expected_tasks):
                atomic_write_json(plan_path, expected)
                print(
                    "Updated partial capture plan from 5 to 3 AE shots per light; "
                    "all existing captures were preserved."
                )
                return
            raise RuntimeError(
                f"capture configuration does not match existing {plan_path}. "
                "Use the original configuration or a new --output-dir."
            )
        return
    atomic_write_json(plan_path, expected)


def is_valid_jpeg(path: Path) -> bool:
    try:
        if not path.is_file() or path.stat().st_size < 128:
            return False
        with path.open("rb") as handle:
            if handle.read(2) != b"\xff\xd8":
                return False
            handle.seek(-2, os.SEEK_END)
            return handle.read(2) == b"\xff\xd9"
    except OSError:
        return False


def load_capture_records(path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    latest: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return records, latest
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict) or not isinstance(record.get("capture_key"), str):
                raise ValueError(f"invalid capture record at {path}:{line_number}")
            records.append(record)
            latest[record["capture_key"]] = record
    return records, latest


def task_record(
    task: CaptureTask,
    run_session_id: str,
    session_sequence: int,
    status: str,
    requested_trigger: str,
    size_bytes: int,
    captured_at: str,
    attempt: int,
    recovery_backup: str | None = None,
) -> dict[str, Any]:
    parameter = task.manual_parameter
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "capture_key": task.capture_key,
        "capture_status": status,
        "session_id": run_session_id,
        "session_sequence": session_sequence,
        "attempt": attempt,
        "plan_index": task.plan_index,
        "sample_index": task.sample_index,
        "sample_id": task.sample.sample_id,
        "zoom_index": task.zoom_index,
        "zoom_id": task.zoom_id,
        "zoom_control": "manual",
        "class_index": task.sample.class_index,
        "wnid": task.sample.wnid,
        "class_name": task.sample.class_name,
        "pdf_page": task.sample.pdf_page,
        "light_id": task.light_id,
        "light_intensity": task.light_intensity,
        "light_percent": light_percent(task.light_intensity),
        "cct_kelvin": None,
        "exposure_mode": task.exposure_mode,
        "ae_shot": task.ae_shot,
        "parameter_number": parameter.parameter_number if parameter else None,
        "parameter_id": parameter.parameter_id if parameter else None,
        "aperture": parameter.aperture if parameter else "auto",
        "shutter_speed": parameter.shutter_speed if parameter else "auto",
        "iso": parameter.iso if parameter else "auto",
        "requested_trigger": requested_trigger,
        "image_path": str(task.relative_path()),
        "captured_at": captured_at,
        "size_bytes": size_bytes,
    }
    if recovery_backup is not None:
        record["replaced_invalid_file"] = recovery_backup
    return record


def apply_cct(record: dict[str, Any], cct: int) -> dict[str, Any]:
    record["cct_kelvin"] = cct
    return record


def recover_unindexed_files(
    tasks: Iterable[CaptureTask],
    output_dir: Path,
    captures_path: Path,
    latest: dict[str, dict[str, Any]],
    attempt_counts: dict[str, int],
    run_session_id: str,
    cct: int,
) -> int:
    recovered = 0
    for task in tasks:
        output_path = output_dir / task.relative_path()
        if not is_valid_jpeg(output_path) or task.capture_key in latest:
            continue
        attempt = attempt_counts.get(task.capture_key, 0) + 1
        record = apply_cct(
            task_record(
                task=task,
                run_session_id=run_session_id,
                session_sequence=0,
                status="recovered_existing_file",
                requested_trigger="unknown",
                size_bytes=output_path.stat().st_size,
                captured_at=datetime.fromtimestamp(output_path.stat().st_mtime).astimezone().isoformat(),
                attempt=attempt,
            ),
            cct,
        )
        append_jsonl(captures_path, record)
        latest[task.capture_key] = record
        attempt_counts[task.capture_key] = attempt
        recovered += 1
    return recovered


def task_is_complete(task: CaptureTask, output_dir: Path, latest: dict[str, dict[str, Any]]) -> bool:
    return task.capture_key in latest and is_valid_jpeg(output_dir / task.relative_path())


def preserve_invalid_file(path: Path) -> Path | None:
    if not path.exists():
        return None
    suffix = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup = path.with_name(f"{path.name}.invalid.{suffix}")
    path.rename(backup)
    return backup


class DryLightController:
    async def __aenter__(self) -> "DryLightController":
        print("[dry-run] light connected")
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        print("[dry-run] light set to 0/1000 and disconnected")

    async def set_intensity(self, intensity: int) -> None:
        print(f"[dry-run] light intensity {intensity}/1000 ({light_percent(intensity):g}%)")


class DryCameraController(AbstractContextManager["DryCameraController"]):
    def __enter__(self) -> "DryCameraController":
        print("[dry-run] camera connected; focus and metering settings preserved")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        print("[dry-run] camera disconnected")

    def configure_for_replication(self) -> None:
        return None

    def capture_auto(self, output_path: Path, **_: Any) -> int:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(DRY_RUN_JPEG)
        return output_path.stat().st_size

    def capture(self, output_path: Path, **_: Any) -> int:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(DRY_RUN_JPEG)
        return output_path.stat().st_size


def validate_pysonycam_capabilities() -> None:
    try:
        from pysonycam.constants import F_NUMBER_TABLE, ISO_TABLE, SHUTTER_SPEED_TABLE
    except ImportError as exc:
        raise RuntimeError(
            "pysonycam is unavailable. Run this script with the same 'sonycam' launcher or "
            "control conda environment used for control.py."
        ) from exc

    aperture_values = {value.upper() for value in F_NUMBER_TABLE.values()}
    iso_values = {value.upper() for value in ISO_TABLE.values()}
    shutter_values = {value.upper() for value in SHUTTER_SPEED_TABLE.values()}
    missing: list[str] = []
    for aperture in DEFAULT_APERTURES:
        label = f"F{aperture:g}".upper()
        if label not in aperture_values:
            missing.append(label)
    for iso in DEFAULT_ISOS:
        if str(iso) not in iso_values:
            missing.append(f"ISO {iso}")
    if "AUTO" not in iso_values:
        missing.append("ISO AUTO")
    for shutter in DEFAULT_SHUTTERS:
        if shutter.upper() not in shutter_values:
            missing.append(shutter)
    if missing:
        raise RuntimeError(
            "camera parameter table cannot reproduce the paper exactly; unsupported: "
            + ", ".join(missing)
        )


def real_controllers(args: argparse.Namespace):
    from control import AmaranLightController, SonyCameraController, restore_owner

    class ReplicationSonyCameraController(SonyCameraController):
        def __enter__(self) -> "ReplicationSonyCameraController":
            from pysonycam import ExposureMode, SonyCamera
            from pysonycam.constants import (
                DeviceProperty,
                F_NUMBER_TABLE,
                ISO_TABLE,
                SHOT_OBJECT_HANDLE,
                SHUTTER_SPEED_TABLE,
                SaveMedia,
            )

            self.ExposureMode = ExposureMode
            self.DeviceProperty = DeviceProperty
            self.SHOT_OBJECT_HANDLE = SHOT_OBJECT_HANDLE
            self.SaveMedia = SaveMedia
            self.iso_table = ISO_TABLE
            self.aperture_table = F_NUMBER_TABLE
            self.shutter_table = SHUTTER_SPEED_TABLE

            self._context = SonyCamera()
            self.camera = self._context.__enter__()
            self.camera.authenticate()
            self.camera.set_mode("still")
            self._set_exposure_mode(self._auto_exposure_mode(), "auto exposure mode")
            return self

        def configure_for_replication(self) -> None:
            if self.camera is None:
                raise RuntimeError("camera is not connected")
            print(
                "Camera focus and metering modes are preserved from the camera/lens; "
                "the script controls only exposure mode and exposure parameters."
            )
            print(
                "Pre-capture stale image-state wait: "
                f"{args.pre_capture_clear_timeout:g}s; post-shutter image wait: "
                f"{args.capture_timeout:g}s."
            )

        def _capture_to_path(
            self,
            output_path: Path,
            timeout: float,
            save_media: str,
            fast_shutter: bool,
        ) -> int:
            """Capture while bounding only the stale pre-shutter state wait."""
            media = self._save_media_value(save_media)
            host_receives = media in (self.SaveMedia.HOST, self.SaveMedia.HOST_AND_CAMERA)

            self.camera.set_save_media(media)
            self._wait_for_setting(self.DeviceProperty.SAVE_MEDIA, int(media), "save media")
            self.camera._wait_for_liveview()
            self.camera._wait_for_shooting_file_info_clear(
                timeout=args.pre_capture_clear_timeout
            )
            self.camera._fire_shutter(fast=fast_shutter)

            if not host_receives:
                return 0

            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                info = self.camera.get_property(self.DeviceProperty.SHOOTING_FILE_INFO)
                shooting_file_info = (
                    info.current_value if isinstance(info.current_value, int) else 0
                )
                if shooting_file_info & 0x8000:
                    break
                time.sleep(0.2)
            else:
                raise RuntimeError("Capture timed out waiting for image")

            self.camera.get_object_info(self.SHOT_OBJECT_HANDLE)
            image_data = self.camera.get_object(self.SHOT_OBJECT_HANDLE)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(image_data)
            restore_owner(output_path)
            return len(image_data)

        # Keep this behavior local so replication does not depend on whether
        # control.py's auto-ISO hardening is present in a particular checkout.
        def capture_auto(
            self,
            output_path: Path,
            timeout: float,
            save_media: str = "host",
            fast_shutter: bool = False,
        ) -> int:
            if self.camera is None:
                raise RuntimeError("camera is not connected")
            self._set_exposure_mode(self._auto_exposure_mode(), "auto exposure mode")
            auto_iso_code = self._reverse_table(self.iso_table).get("AUTO")
            if auto_iso_code is None:
                raise RuntimeError("ISO AUTO is not supported by this camera")
            if self._last_iso != auto_iso_code:
                self.camera.set_iso(auto_iso_code)
                self._wait_for_setting(self.DeviceProperty.ISO, auto_iso_code, "ISO AUTO")
                self._last_iso = auto_iso_code
            return self._capture_to_path(output_path, timeout, save_media, fast_shutter=False)

    light = AmaranLightController(
        ws_url=args.light_ws_url,
        api_secret_key=args.light_api_secret_key,
        client_id=args.light_client_id,
        cct=args.cct,
        settle_seconds=args.settle_seconds,
    )
    return light, ReplicationSonyCameraController()


def make_controllers(args: argparse.Namespace):
    if args.dry_run:
        return DryLightController(), DryCameraController()
    return real_controllers(args)


def prompt_for_zoom(
    sample: Sample,
    sample_index: int,
    total_samples: int,
    zoom_index: int,
    zoom_count: int,
) -> bool:
    print("\n" + "=" * 72)
    print(f"Sample {sample_index}/{total_samples}: {sample.sample_id}")
    print(f"Class: {sample.class_index} / {sample.wnid} / {sample.class_name}")
    print(f"Printed source: {sample.rendered_path} (PDF page {sample.pdf_page})")
    zoom_id = f"z{zoom_index:03d}"
    if zoom_index == 1:
        instruction = f"Place and align this print; set initial zoom {zoom_id}"
    else:
        instruction = (
            f"Keep the same sample in place and manually change zoom to {zoom_id} "
            f"({zoom_index}/{zoom_count})"
        )
    while True:
        answer = input(f"{instruction}, then press Enter (q to stop): ").strip().lower()
        if not answer:
            return True
        if answer in {"q", "quit", "exit"}:
            return False
        print("Press Enter to continue, or enter q to stop.")


def selected_samples(samples: list[Sample], selected_ids: list[str] | None) -> list[Sample]:
    if not selected_ids:
        return samples
    requested = list(dict.fromkeys(selected_ids))
    by_id = {sample.sample_id: sample for sample in samples}
    unknown = [value for value in requested if value not in by_id]
    if unknown:
        raise ValueError(f"unknown --sample-id value(s): {', '.join(unknown)}")
    requested_set = set(requested)
    # Preserve labels.csv order even if CLI options are given in another order.
    return [sample for sample in samples if sample.sample_id in requested_set]


async def acquire_task(
    task: CaptureTask,
    args: argparse.Namespace,
    camera: Any,
    output_dir: Path,
    captures_path: Path,
    errors_path: Path,
    run_session_id: str,
    session_sequence: int,
    requested_trigger: str,
    attempt: int,
) -> dict[str, Any] | None:
    output_path = output_dir / task.relative_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    backup = preserve_invalid_file(output_path) if output_path.exists() else None
    partial_path = output_path.with_name(f".{output_path.name}.partial")
    partial_backup = preserve_invalid_file(partial_path) if partial_path.exists() else None

    print(
        f"[{task.plan_index}] {task.sample.sample_id} {task.zoom_id} {task.light_id} "
        f"{task.capture_key.rsplit('|', 1)[-1]} ({requested_trigger})"
    )
    try:
        if task.exposure_mode == "auto":
            camera.capture_auto(
                output_path=partial_path,
                timeout=args.capture_timeout,
                save_media=args.save_media,
                fast_shutter=False,
            )
        else:
            parameter = task.manual_parameter
            assert parameter is not None
            camera.capture(
                aperture=parameter.aperture,
                iso=parameter.iso,
                shutter=parameter.shutter_speed,
                output_path=partial_path,
                timeout=args.capture_timeout,
                save_media=args.save_media,
                fast_shutter=requested_trigger == "fast",
            )
        if not is_valid_jpeg(partial_path):
            raise RuntimeError(f"camera returned an invalid JPEG: {partial_path}")
        partial_path.replace(output_path)
    except Exception as exc:
        failed_partial = preserve_invalid_file(partial_path)
        error_record = {
            "schema_version": SCHEMA_VERSION,
            "event": "capture_failed",
            "failed_at": timestamp(),
            "session_id": run_session_id,
            "session_sequence": session_sequence,
            "capture_key": task.capture_key,
            "plan_index": task.plan_index,
            "sample_id": task.sample.sample_id,
            "zoom_index": task.zoom_index,
            "zoom_id": task.zoom_id,
            "light_id": task.light_id,
            "light_intensity": task.light_intensity,
            "light_percent": light_percent(task.light_intensity),
            "cct_kelvin": args.cct,
            "exposure_mode": task.exposure_mode,
            "ae_shot": task.ae_shot,
            "parameter_id": (
                task.manual_parameter.parameter_id if task.manual_parameter else None
            ),
            "aperture": (
                task.manual_parameter.aperture if task.manual_parameter else "auto"
            ),
            "shutter_speed": (
                task.manual_parameter.shutter_speed if task.manual_parameter else "auto"
            ),
            "iso": task.manual_parameter.iso if task.manual_parameter else "auto",
            "image_path": str(task.relative_path()),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "existing_invalid_backup": (
                str(backup.relative_to(output_dir)) if backup is not None else None
            ),
            "stale_partial_backup": (
                str(partial_backup.relative_to(output_dir)) if partial_backup is not None else None
            ),
            "failed_partial_backup": (
                str(failed_partial.relative_to(output_dir)) if failed_partial is not None else None
            ),
        }
        append_jsonl(errors_path, error_record)
        if "timed out waiting for image" in str(exc).lower():
            print(
                f"WARNING: timed out for {task.capture_key}; recorded failure and "
                "continuing with the next image."
            )
            return None
        raise RuntimeError(
            f"capture failed for {task.capture_key}; stopped for safe resume. "
            f"See {errors_path}."
        ) from exc

    record = apply_cct(
        task_record(
            task=task,
            run_session_id=run_session_id,
            session_sequence=session_sequence,
            status="captured",
            requested_trigger=requested_trigger,
            size_bytes=output_path.stat().st_size,
            captured_at=timestamp(),
            attempt=attempt,
            recovery_backup=(str(backup.relative_to(output_dir)) if backup is not None else None),
        ),
        args.cct,
    )
    append_jsonl(captures_path, record)
    return record


async def run_capture(args: argparse.Namespace) -> None:
    labels_path = args.labels.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    samples = load_samples(labels_path)
    selected = selected_samples(samples, args.sample_id)
    all_tasks = build_capture_tasks(samples, args.light_intensities, args.zoom_count)
    selected_ids = {sample.sample_id for sample in selected}
    tasks = [task for task in all_tasks if task.sample.sample_id in selected_ids]
    manifest = build_plan_manifest(
        labels_path=labels_path,
        samples=samples,
        light_intensities=args.light_intensities,
        cct=args.cct,
        zoom_count=args.zoom_count,
    )

    print(f"Paper protocol: {PAPER_TABLE}")
    print(f"Labels: {labels_path} ({len(samples)} samples)")
    print(
        "Light sweep: "
        + ", ".join(
            f"{value}/1000 ({light_percent(value):g}%)" for value in args.light_intensities
        )
        + f" at {args.cct}K"
    )
    images_per_light = DEFAULT_AE_SHOTS + len(build_manual_parameters())
    print(
        f"Per light: {DEFAULT_AE_SHOTS} auto-exposure + 27 manual = "
        f"{images_per_light} images"
    )
    print(
        f"Per sample: {args.zoom_count} manual zoom positions x "
        f"{len(args.light_intensities)} lights x {images_per_light} images"
    )
    print(f"Full plan: {len(all_tasks):,} images; selected run: {len(tasks):,} images")
    print(f"Output: {output_dir}")

    if args.plan_only:
        if args.print_plan_json:
            print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))
        return

    if not args.dry_run:
        validate_pysonycam_capabilities()

    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_plan_file(output_dir, manifest, all_tasks)
    captures_path = output_dir / "captures.jsonl"
    errors_path = output_dir / "errors.jsonl"
    records, latest = load_capture_records(captures_path)
    attempt_counts: dict[str, int] = {}
    for record in records:
        key = record["capture_key"]
        attempt_counts[key] = max(attempt_counts.get(key, 0), int(record.get("attempt", 1)))

    run_session_id = session_id()
    recovered = recover_unindexed_files(
        tasks=tasks,
        output_dir=output_dir,
        captures_path=captures_path,
        latest=latest,
        attempt_counts=attempt_counts,
        run_session_id=run_session_id,
        cct=args.cct,
    )
    if recovered:
        print(f"Recovered metadata for {recovered} valid existing JPEG(s).")

    pending = [task for task in tasks if not task_is_complete(task, output_dir, latest)]
    if not pending:
        print(f"All {len(tasks):,} selected captures are already complete and valid.")
        return

    pending_by_sample: dict[str, list[CaptureTask]] = {}
    for task in pending:
        pending_by_sample.setdefault(task.sample.sample_id, []).append(task)

    print(f"Pending: {len(pending):,}; already complete: {len(tasks) - len(pending):,}")
    light_controller, camera_controller = make_controllers(args)
    session_sequence = 0
    timed_out_count = 0

    async with light_controller as light:
        with camera_controller as camera:
            camera.configure_for_replication()
            for sample in selected:
                sample_pending = pending_by_sample.get(sample.sample_id, [])
                if not sample_pending:
                    print(f"Skipping completed sample: {sample.sample_id}")
                    continue
                print(f"Capturing {sample.sample_id}: {len(sample_pending)} pending image(s)")
                for zoom_index in range(1, args.zoom_count + 1):
                    zoom_tasks = [
                        task
                        for task in sample_pending
                        if task.zoom_index == zoom_index
                        and not task_is_complete(task, output_dir, latest)
                    ]
                    if not zoom_tasks:
                        print(f"Skipping completed zoom z{zoom_index:03d}: {sample.sample_id}")
                        continue
                    if not args.yes and not prompt_for_zoom(
                        sample=sample,
                        sample_index=samples.index(sample) + 1,
                        total_samples=len(samples),
                        zoom_index=zoom_index,
                        zoom_count=args.zoom_count,
                    ):
                        print("Stopped by user. Run the same command later to resume.")
                        return
                    if not args.yes and args.start_delay_seconds > 0:
                        print(
                            f"Starting capture in {args.start_delay_seconds:g} seconds; "
                            "keep the sample and zoom fixed..."
                        )
                        time.sleep(args.start_delay_seconds)
                    print(
                        f"Capturing zoom z{zoom_index:03d}: "
                        f"{len(zoom_tasks)} pending image(s)"
                    )
                    for intensity in args.light_intensities:
                        intensity_tasks = [
                            task
                            for task in zoom_tasks
                            if task.light_intensity == intensity
                            and not task_is_complete(task, output_dir, latest)
                        ]
                        if not intensity_tasks:
                            continue
                        await light.set_intensity(intensity)
                        manual_captures_this_light = 0
                        for task in intensity_tasks:
                            session_sequence += 1
                            requested_trigger = "normal"
                            if task.exposure_mode == "manual":
                                requested_trigger = (
                                    "normal" if manual_captures_this_light == 0 else "fast"
                                )
                            attempt = attempt_counts.get(task.capture_key, 0) + 1
                            record = await acquire_task(
                                task=task,
                                args=args,
                                camera=camera,
                                output_dir=output_dir,
                                captures_path=captures_path,
                                errors_path=errors_path,
                                run_session_id=run_session_id,
                                session_sequence=session_sequence,
                                requested_trigger=requested_trigger,
                                attempt=attempt,
                            )
                            attempt_counts[task.capture_key] = attempt
                            if record is None:
                                timed_out_count += 1
                                continue
                            latest[task.capture_key] = record
                            if task.exposure_mode == "manual":
                                manual_captures_this_light += 1
                    print(f"Completed zoom z{zoom_index:03d}: {sample.sample_id}")
                print(f"Completed sample: {sample.sample_id}")

    completed = sum(task_is_complete(task, output_dir, latest) for task in tasks)
    if completed != len(tasks):
        missing = len(tasks) - completed
        print(
            f"Capture run finished: {completed:,}/{len(tasks):,} selected images are valid; "
            f"{missing:,} missing image(s), including {timed_out_count:,} timeout(s) in "
            f"this run. Failures are recorded in {errors_path}. Run the same command "
            "again to retry only missing images."
        )
        return
    print(f"Capture run complete: {completed:,}/{len(tasks):,} selected images are valid.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect the paper's AE + 27 manual settings across a single-light sweep."
    )
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--sample-id",
        action="append",
        help="Capture only this sample ID; repeat for multiple IDs. Plan metadata still covers all labels.",
    )
    parser.add_argument("--cct", type=int, default=DEFAULT_CCT)
    parser.add_argument(
        "--zoom-count",
        type=int,
        default=DEFAULT_ZOOM_COUNT,
        help="Manual zoom positions per sample. Default: 2 (z001 and z002).",
    )
    parser.add_argument(
        "--light-intensities",
        type=parse_intensities,
        default=DEFAULT_LIGHT_INTENSITIES,
        help="Comma-separated Amaran values in [0,1000]. Default: 10,200,500,700,1000.",
    )
    parser.add_argument("--settle-seconds", type=float, default=1.0)
    parser.add_argument(
        "--capture-timeout",
        type=float,
        default=DEFAULT_CAPTURE_TIMEOUT_SECONDS,
        help="Seconds to wait for each long exposure/image transfer. Default: 180.",
    )
    parser.add_argument(
        "--pre-capture-clear-timeout",
        type=float,
        default=DEFAULT_PRE_CAPTURE_CLEAR_TIMEOUT_SECONDS,
        help=(
            "Seconds to wait for stale image-ready state before pressing the shutter. "
            "Default: 5."
        ),
    )
    parser.add_argument(
        "--start-delay-seconds",
        type=float,
        default=DEFAULT_START_DELAY_SECONDS,
        help="Wait after each manual Enter confirmation. Default: 10.",
    )
    parser.add_argument(
        "--save-media", choices=["host", "host-and-camera"], default="host"
    )
    parser.add_argument("--light-ws-url", default="ws://127.0.0.1:12345")
    parser.add_argument("--light-client-id", type=int, default=1)
    parser.add_argument(
        "--light-api-secret-key",
        default=os.environ.get("AMARAN_API_SECRET_KEY", DEFAULT_LIGHT_API_SECRET_KEY),
        help=(
            "Amaran API secret. Defaults to AMARAN_API_SECRET_KEY when available, "
            "otherwise uses the same built-in testbed key as control.py."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Use simulated camera and light.")
    parser.add_argument("--plan-only", action="store_true", help="Print plan summary without writes.")
    parser.add_argument(
        "--print-plan-json",
        action="store_true",
        help="With --plan-only, also print the complete JSON manifest.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip per-sample placement prompts (intended for automated dry-run/tests).",
    )
    args = parser.parse_args(argv)
    if args.cct <= 0:
        parser.error("--cct must be positive")
    if args.zoom_count < 1:
        parser.error("--zoom-count must be at least 1")
    if args.settle_seconds < 0:
        parser.error("--settle-seconds must be >= 0")
    if args.capture_timeout <= 0:
        parser.error("--capture-timeout must be > 0")
    if args.pre_capture_clear_timeout <= 0:
        parser.error("--pre-capture-clear-timeout must be > 0")
    if args.start_delay_seconds < 0:
        parser.error("--start-delay-seconds must be >= 0")
    if args.print_plan_json and not args.plan_only:
        parser.error("--print-plan-json requires --plan-only")
    return args


def main(argv: list[str] | None = None) -> int:
    try:
        asyncio.run(run_capture(parse_args(argv)))
    except KeyboardInterrupt:
        print("\nInterrupted. Run the same command later to resume safely.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
