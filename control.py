"""Unified capture controller for light, turntable, and Sony camera.

Run from the repository root, usually through the Sony camera launcher::

    sonycam control.py

Useful dry run for checking the capture plan and filenames::

    python control.py --dry-run

Default plan:
    lighting position: front
    lighting intensities: 0, 10, 50, 300, 500, 700, 1000
    views: 0, 90, 180, 270 degrees
    apertures: F3.2
    ISO: 800
    shutter speeds: 1/80
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import glob
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional


DEFAULT_OUTPUT_DIR = Path("dataset")
DEFAULT_TURNTABLE_PORT = "/dev/cu.usbmodem1101"
DEFAULT_CCT = 5600
DEFAULT_LIGHT_POSITION = "normal" # optional: normal, reflect, face, side
DEFAULT_LIGHT_INTENSITIES = [0, 10, 50, 300, 500, 700, 1000]
DEFAULT_APERTURES = [2.8, 4, 8, 11, 16, 22]
DEFAULT_ISOS = [100, 250, 800, 2000, 3200, 6400, 12800, 32000]
DEFAULT_SHUTTERS = ['0.5"', "1/3", "1/15", "1/60", "1/250", "1/1000"]
DEFAULT_SAVE_MEDIA = "host"
DEFAULT_START_DELAY_SECONDS = 10.0
PAD_WIDTH = 3
AUTO_PARAM_ID = 0


def parse_int_list(value: str) -> list[int]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("list cannot be empty")
    return [int(item) for item in items]


def parse_float_list(value: str) -> list[float]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("list cannot be empty")
    return [float(item) for item in items]


def parse_shutter_list(value: str) -> list[str]:
    shutters = []
    for item in [item.strip() for item in value.split(",") if item.strip()]:
        if "/" in item:
            shutters.append(item)
        else:
            shutters.append(f"1/{item}")
    if not shutters:
        raise argparse.ArgumentTypeError("list cannot be empty")
    return shutters


def light_percent(intensity: int) -> float:
    """
    This function is only for reading purpose
    Real control still use the original intensity [0, 1000]
    """
    return intensity / 10.0


def make_session_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def format_id(prefix: str, value: int) -> str:
    return f"{prefix}{value:0{PAD_WIDTH}d}"


def format_number(value: float | int | str) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def format_duration(seconds: float) -> str:
    minutes, remaining_seconds = divmod(max(0.0, seconds), 60.0)
    return f"{int(minutes)}m {remaining_seconds:.1f}s"


def restore_owner(path: Path) -> None:
    """Make files created through sudo writable by the invoking user."""
    uid_text = os.environ.get("SUDO_UID")
    gid_text = os.environ.get("SUDO_GID")
    if uid_text is None or gid_text is None:
        return
    os.chown(path, int(uid_text), int(gid_text))


def append_jsonl_record(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
    restore_owner(path)


def write_json_file(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    restore_owner(path)


def maps_dir(output_dir: Path) -> Path:
    path = output_dir / "maps"
    path.mkdir(parents=True, exist_ok=True)
    restore_owner(path)
    return path


def ensure_output_dir_for_mode(output_dir: Path, capture_mode: str) -> None:
    if capture_mode == "fresh" and output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            f"--capture-mode fresh requires an empty output directory, got {output_dir}. "
            "Use a new --output-dir or clear it manually."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    restore_owner(output_dir)


def normalize_class_id(value: str) -> int:
    class_id = value.strip()
    if not class_id.isdigit():
        raise ValueError(f"class_id must be a number, got {value!r}.")
    return int(class_id)


def prompt_class_id() -> int | None:
    while True:
        value = input("\nClass ID (number, q to quit): ").strip()
        if value.lower() in {"q", "quit", "exit"}:
            return None
        if value.isdigit():
            return int(value)
        print("Please enter a numeric class_id, or q to quit.")


def build_output_path(
    output_dir: Path,
    class_id: int,
    sample_id: int,
    light_id: int,
    view_id: int,
    param_id: int,
) -> Path:
    return (
        output_dir
        / format_id("l", light_id)
        / format_id("c", class_id)
        / format_id("s", sample_id)
        / format_id("v", view_id)
        / f"{format_id('p', param_id)}.jpg"
    )


def build_capture_record(
    *,
    session_id: str,
    sequence: int,
    class_id: int,
    sample_id: int,
    light_id: int,
    light_position: str,
    light_intensity: int,
    light_cct: int,
    view_id: int,
    angle_degrees: int,
    param_id: int,
    aperture: float | str,
    iso: int | str,
    shutter_speed: str,
    exposure_mode: str,
    output_dir: Path,
    output_path: Path,
    captured_at: str,
    size_bytes: int,
) -> dict[str, object]:
    image_path = str(output_path.relative_to(output_dir))
    return {
        "session_id": session_id,
        "sequence": sequence,
        "class_id": class_id,
        "sample_id": sample_id,
        "light_id": light_id,
        "view_id": view_id,
        "param_id": param_id,
        "position": light_position,
        "intensity": light_intensity,
        "cct": light_cct,
        "angle_degrees": angle_degrees,
        "aperture": aperture,
        "iso": iso,
        "shutter_speed": shutter_speed,
        "exposure_mode": exposure_mode,
        "image_path": image_path,
        "captured_at": captured_at,
        "size_bytes": size_bytes,
    }


def new_dataset_map() -> dict[str, object]:
    now = timestamp()
    return {
        "schema_version": 1,
        "created_at": now,
        "updated_at": now,
        "classes": [],
        "samples": [],
        "lights": [],
        "views": [],
        "params": [],
    }


def load_dataset_map(path: Path) -> dict[str, object]:
    if not path.exists() or path.stat().st_size == 0:
        return new_dataset_map()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Dataset parameter map is not valid JSON: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Dataset parameter map must be a JSON object: {path}")
    for key in ["classes", "samples", "lights", "views", "params"]:
        if not isinstance(data.get(key), list):
            data[key] = []
    data.setdefault("schema_version", 1)
    data.setdefault("created_at", timestamp())
    data["updated_at"] = timestamp()
    return data


def save_dataset_map(map_dir: Path, dataset_map: dict[str, object]) -> None:
    dataset_map["updated_at"] = timestamp()
    write_json_file(map_dir / "parameters.json", dataset_map)


def map_items(dataset_map: dict[str, object], key: str) -> list[dict[str, object]]:
    items = dataset_map.setdefault(key, [])
    if not isinstance(items, list):
        raise ValueError(f"Dataset map field {key!r} must be a list.")
    return items  # type: ignore[return-value]


def next_map_id(items: list[dict[str, object]], key: str) -> int:
    values = []
    for item in items:
        value = item.get(key)
        if isinstance(value, int):
            values.append(value)
        elif isinstance(value, str) and value.isdigit():
            values.append(int(value))
    return max(values, default=0) + 1


def ensure_class_entry(dataset_map: dict[str, object], class_id: int) -> None:
    classes = map_items(dataset_map, "classes")
    for item in classes:
        if int(item.get("class_id", -1)) == class_id:
            return
    classes.append(
        {
            "class_id": class_id,
            "class_folder": format_id("c", class_id),
            "created_at": timestamp(),
        }
    )


def sample_ids_from_dirs(output_dir: Path, light_id: int, class_id: int) -> list[int]:
    new_class_dir = output_dir / format_id("l", light_id) / format_id("c", class_id)
    sample_ids = []
    if new_class_dir.exists():
        for path in new_class_dir.iterdir():
            if path.is_dir() and path.name.startswith("s") and path.name[1:].isdigit():
                sample_ids.append(int(path.name[1:]))
    return sample_ids


def sample_light_ids(sample: dict[str, object]) -> list[int]:
    light_ids = sample.get("light_ids")
    if isinstance(light_ids, list):
        return [int(value) for value in light_ids if isinstance(value, int) or str(value).isdigit()]
    light_id = sample.get("light_id")
    if isinstance(light_id, int):
        return [light_id]
    if isinstance(light_id, str) and light_id.isdigit():
        return [int(light_id)]
    return []


def next_sample_id(
    dataset_map: dict[str, object],
    output_dir: Path,
    class_id: int,
    light_ids: list[int],
) -> int:
    light_id_set = set(light_ids)
    samples = map_items(dataset_map, "samples")
    map_sample_ids = []
    for item in samples:
        if int(item.get("class_id", -1)) != class_id:
            continue
        if not str(item.get("sample_id", "")).isdigit():
            continue
        item_light_ids = set(sample_light_ids(item))
        if item_light_ids and item_light_ids.isdisjoint(light_id_set):
            continue
        if not item_light_ids:
            continue
        map_sample_ids.append(int(item["sample_id"]))
    dir_sample_ids = []
    for light_id in light_ids:
        dir_sample_ids.extend(sample_ids_from_dirs(output_dir, light_id, class_id))
    return max([*map_sample_ids, *dir_sample_ids], default=0) + 1


def light_sample_dirs(output_dir: Path, light_ids: list[int], class_id: int, sample_id: int) -> list[Path]:
    return [
        output_dir / format_id("l", light_id) / format_id("c", class_id) / format_id("s", sample_id)
        for light_id in light_ids
    ]


def sample_folder_text(output_dir: Path, light_ids: list[int], class_id: int, sample_id: int) -> str:
    paths = light_sample_dirs(output_dir, light_ids, class_id, sample_id)
    if len(paths) == 1:
        return str(paths[0])
    return ", ".join(str(path) for path in paths)


def append_sample_entry(
    dataset_map: dict[str, object],
    class_id: int,
    sample_id: int,
    light_ids: list[int],
    session_id: str,
    total_captures: int,
    started_at: str,
) -> None:
    entry = {
        "class_id": class_id,
        "sample_id": sample_id,
        "light_ids": light_ids,
        "light_folders": [format_id("l", light_id) for light_id in light_ids],
        "class_folder": format_id("c", class_id),
        "sample_folder": format_id("s", sample_id),
        "session_id": session_id,
        "created_at": started_at,
        "started_at": started_at,
        "total_captures": total_captures,
    }
    if len(light_ids) == 1:
        entry["light_id"] = light_ids[0]
        entry["light_folder"] = format_id("l", light_ids[0])
    map_items(dataset_map, "samples").append(entry)


def validate_light_intensity(intensity: int) -> None:
    if intensity < 0 or intensity > 1000:
        raise ValueError(f"Light intensity must be in [0, 1000], got {intensity}.")

def update_sample_timing(
    dataset_map: dict[str, object],
    class_id: int,
    sample_id: int,
    session_id: str,
    capture_started_at: str,
    completed_at: str,
    object_elapsed_seconds: float,
    capture_elapsed_seconds: float,
) -> None:
    for item in map_items(dataset_map, "samples"):
        if (
            int(item.get("class_id", -1)) == class_id
            and int(item.get("sample_id", -1)) == sample_id
            and item.get("session_id") == session_id
        ):
            item.update(
                {
                    "capture_started_at": capture_started_at,
                    "completed_at": completed_at,
                    "object_elapsed_seconds": round(object_elapsed_seconds, 3),
                    "capture_elapsed_seconds": round(capture_elapsed_seconds, 3),
                }
            )
            return
    raise RuntimeError(
        f"Could not update timing for class {class_id}, sample {sample_id}, session {session_id}."
    )

def get_or_create_light_id(
    dataset_map: dict[str, object],
    position: str,
    intensity: int,
    cct: int,
) -> int:
    lights = map_items(dataset_map, "lights")
    for item in lights:
        if (
            item.get("position") == position
            and int(item.get("intensity", -1)) == intensity
            and int(item.get("cct", -1)) == cct
        ):
            return int(item["light_id"])

    light_id = next_map_id(lights, "light_id")
    lights.append(
        {
            "light_id": light_id,
            "light_folder": format_id("l", light_id),
            "position": position,
            "intensity": intensity,
            "light_percent": light_percent(intensity),
            "cct": cct,
        }
    )
    return light_id


def get_or_create_view_id(
    dataset_map: dict[str, object],
    view_index: int,
    angle_degrees: int,
) -> int:
    views = map_items(dataset_map, "views")
    for item in views:
        if (
            int(item.get("view_index", -1)) == view_index
            and int(item.get("angle_degrees", -1)) == angle_degrees
        ):
            return int(item["view_id"])

    view_id = next_map_id(views, "view_id")
    views.append(
        {
            "view_id": view_id,
            "view_folder": format_id("v", view_id),
            "view_index": view_index,
            "angle_degrees": angle_degrees,
        }
    )
    return view_id


def get_or_create_param_id(
    dataset_map: dict[str, object],
    aperture: float,
    iso: int,
    shutter: str,
) -> int:
    params = map_items(dataset_map, "params")
    aperture_value = format_number(aperture)
    for item in params:
        try:
            item_iso = int(item.get("iso", -1))
        except (TypeError, ValueError):
            continue
        if (
            str(item.get("aperture")) == aperture_value
            and item_iso == iso
            and item.get("shutter_speed") == shutter
        ):
            return int(item["param_id"])

    param_id = next_map_id(params, "param_id")
    params.append(
        {
            "param_id": param_id,
            "param_file": f"{format_id('p', param_id)}.jpg",
            "aperture": aperture_value,
            "iso": iso,
            "shutter_speed": shutter,
        }
    )
    return param_id


def ensure_auto_param_entry(dataset_map: dict[str, object]) -> None:
    params = map_items(dataset_map, "params")
    for item in params:
        if int(item.get("param_id", -1)) == AUTO_PARAM_ID:
            return
    params.append(
        {
            "param_id": AUTO_PARAM_ID,
            "param_file": f"{format_id('p', AUTO_PARAM_ID)}.jpg",
            "aperture": "auto",
            "iso": "auto",
            "shutter_speed": "auto",
            "exposure_mode": "auto",
        }
    )


def build_lighting_plan(
    dataset_map: dict[str, object],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    plan = [
        {
            "position": args.light_position,
            "intensity": intensity,
            "cct": args.cct,
        }
        for intensity in args.light_intensities
    ]

    for item in plan:
        item["light_id"] = get_or_create_light_id(
            dataset_map,
            str(item["position"]),
            int(item["intensity"]),
            int(item["cct"]),
        )
    return plan


def build_view_plan(
    dataset_map: dict[str, object],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    view_plan = []
    for view_index in range(args.views):
        angle = (view_index * args.view_step) % 360
        view_plan.append(
            {
                "view_id": get_or_create_view_id(dataset_map, view_index, angle),
                "view_index": view_index,
                "angle_degrees": angle,
            }
        )
    return view_plan


def build_param_plan(
    dataset_map: dict[str, object],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    param_plan = []
    for aperture in args.apertures:
        for iso in args.isos:
            for shutter in args.shutters:
                param_plan.append(
                    {
                        "param_id": get_or_create_param_id(
                            dataset_map,
                            aperture,
                            iso,
                            shutter,
                        ),
                        "aperture": aperture,
                        "iso": iso,
                        "shutter_speed": shutter,
                    }
                )
    return param_plan


class DryLightController:
    def __init__(self):
        self.current_cct = None

    async def __aenter__(self) -> "DryLightController":
        print("[dry-run] light connected")
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        print("[dry-run] light disconnected")

    async def set_cct(self, cct: int) -> None:
        if self.current_cct == cct:
            return
        self.current_cct = cct
        print(f"[dry-run] set light CCT {cct}K")

    async def set_intensity(self, intensity: int) -> None:
        print(f"[dry-run] set light intensity {intensity}/1000 ({light_percent(intensity):g}%)")


class AmaranLightController:
    def __init__(
        self,
        ws_url: str,
        api_secret_key: str,
        client_id: int,
        cct: int,
        settle_seconds: float,
    ):
        self.ws_url = ws_url
        self.api_secret_key = api_secret_key
        self.client_id = client_id
        self.cct = cct
        self.settle_seconds = settle_seconds
        self.ws = None
        self.node_id = None
        self.current_cct = None
        self._last_request_id = 0

    async def __aenter__(self) -> "AmaranLightController":
        """ Connect to Amaran Light """
        await self._connect()
        self._ensure_ok(await self._send("get_protocol_versions"), "get_protocol_versions")

        devices = self._extract_devices(
            self._ensure_ok(await self._send("get_fixture_list"), "get_fixture_list")
        )
        if not devices:
            devices = self._extract_devices(
                self._ensure_ok(await self._send("get_device_list"), "get_device_list")
            )
        if not devices:
            raise RuntimeError("No Amaran light found.")

        # By default, we use the first light
        light = devices[0]
        self.node_id = light["node_id"]
        print(f"Using light: {light.get('name')} ({self.node_id})")

        self._ensure_ok(
            await self._send("set_sleep", node_id=self.node_id, args={"sleep": False}),
            "set_sleep",
        )
        await asyncio.sleep(self.settle_seconds)
        await self.set_cct(self.cct)
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        if self.node_id is not None and self.ws is not None:
            try:
                self._ensure_ok(
                    await self._send(
                        "set_intensity",
                        node_id=self.node_id,
                        args={"intensity": 0},
                    ),
                    "set_intensity",
                )
                self._ensure_ok(
                    await self._send(
                        "set_sleep",
                        node_id=self.node_id,
                        args={"sleep": True},
                    ),
                    "set_sleep",
                )
                print("Light turned off and put to sleep.")
            except Exception as exc:
                print(f"Warning: failed to turn off light cleanly: {exc}")
        await self._close_ws()

    async def set_cct(self, cct: int) -> None:
        if self.node_id is None:
            raise RuntimeError("Light is not connected.")
        if self.current_cct == cct:
            return
        self._ensure_ok(
            await self._send("set_cct", node_id=self.node_id, args={"cct": cct}),
            "set_cct",
        )
        self.current_cct = cct
        print(f"Light CCT: {cct}K")
        await asyncio.sleep(self.settle_seconds)

    async def set_intensity(self, intensity: int) -> None:
        if self.node_id is None:
            raise RuntimeError("Light is not connected.")
        self._ensure_ok(
            await self._send(
                "set_intensity",
                node_id=self.node_id,
                args={"intensity": intensity},
            ),
            "set_intensity",
        )
        readback = self._ensure_ok(
            await self._send("get_intensity", node_id=self.node_id),
            "get_intensity",
        )
        print(f"Light intensity: {readback.get('data')} / 1000")
        await asyncio.sleep(self.settle_seconds)

    def _generate_token(self) -> str:
        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

        iv = os.urandom(12)
        encryptor = Cipher(
            algorithms.AES(base64.b64decode(self.api_secret_key)),
            modes.GCM(iv),
            backend=default_backend(),
        ).encryptor()
        ciphertext = encryptor.update(str(int(time.time())).encode()) + encryptor.finalize()
        return base64.b64encode(iv + encryptor.tag + ciphertext).decode()

    def _next_request_id(self) -> int:
        request_id = int(time.time() * 1000)
        if request_id <= self._last_request_id:
            request_id = self._last_request_id + 1
        self._last_request_id = request_id
        return request_id

    async def _connect(self) -> None:
        import websockets

        await self._close_ws()
        self.ws = await websockets.connect(self.ws_url)

    async def _close_ws(self) -> None:
        if self.ws is None:
            return
        try:
            await self.ws.close()
        except Exception:
            pass
        finally:
            self.ws = None

    async def _send(
        self,
        action: str,
        node_id: Optional[str] = None,
        args: Optional[dict] = None,
    ):
        for attempt in range(2):
            try:
                return await self._send_once(action, node_id=node_id, args=args)
            except Exception:
                if attempt == 1:
                    raise
                print(f"Light websocket disconnected during {action}; reconnecting and retrying...")
                await self._connect()

    async def _send_once(
        self,
        action: str,
        node_id: Optional[str] = None,
        args: Optional[dict] = None,
    ):
        if self.ws is None:
            raise RuntimeError("Light websocket is not connected.")
        request_id = self._next_request_id()
        request = {
            "version": 2,
            "type": "request",
            "client_id": self.client_id,
            "request_id": request_id,
            "action": action,
            "token": self._generate_token(),
        }
        if node_id is not None:
            request["node_id"] = node_id
        if args is not None:
            request["args"] = args

        await self.ws.send(json.dumps(request))
        while True:
            data = json.loads(await self.ws.recv())
            if data.get("type") == "event":
                continue
            if data.get("type") == "response" and data.get("request_id") == request_id:
                return data

    @staticmethod
    def _ensure_ok(resp: dict, action: str) -> dict:
        if resp.get("code") != 0:
            raise RuntimeError(f"{action} failed: {json.dumps(resp, indent=2)}")
        return resp

    @staticmethod
    def _extract_devices(resp: dict) -> list[dict]:
        data = resp.get("data", resp)
        return data if isinstance(data, list) else []


class DryTurntableController:
    def __enter__(self) -> "DryTurntableController":
        print("[dry-run] turntable connected")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        print("[dry-run] turntable disconnected")

    def home(self) -> None:
        print("[dry-run] turntable HOME")

    def set_speed(self, rpm: float) -> None:
        print(f"[dry-run] turntable SPEED {rpm}")

    def rotate(self, degrees: float) -> None:
        print(f"[dry-run] turntable ROT {degrees}")

    def goto(self, degrees: float) -> None:
        print(f"[dry-run] turntable GOTO {degrees}")


class TurntableController:
    def __init__(self, port: str, speed_rpm: float, settle_seconds: float):
        self.port = port
        self.speed_rpm = speed_rpm
        self.settle_seconds = settle_seconds
        self._turntable = None

    def __enter__(self) -> "TurntableController":
        from turntable.turntable import Turntable

        self._turntable = Turntable(self.port)
        print("Set current turntable position as HOME")
        print(self._turntable.home())
        print(f"Set turntable speed: {self.speed_rpm} rpm")
        print(self._turntable.set_speed(self.speed_rpm))
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._turntable is not None:
            self._turntable.close()

    def rotate(self, degrees: float) -> None:
        if self._turntable is None:
            raise RuntimeError("Turntable is not connected.")
        print(f"Rotate turntable by {degrees:g} degrees")
        print(self._turntable.rotate(degrees))
        time.sleep(self.settle_seconds)

    def goto(self, degrees: float) -> None:
        if self._turntable is None:
            raise RuntimeError("Turntable is not connected.")
        print(f"Move turntable to {degrees:g} degrees")
        print(self._turntable.goto(degrees))
        time.sleep(self.settle_seconds)


class DryCameraController:
    def __enter__(self) -> "DryCameraController":
        print("[dry-run] camera connected")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        print("[dry-run] camera disconnected")

    def capture(
        self,
        aperture: float,
        iso: int,
        shutter: str,
        output_path: Path,
        timeout: float,
        save_media: str = DEFAULT_SAVE_MEDIA,
        fast_shutter: bool = False,
    ) -> int:
        print(f"[dry-run] capture ISO {iso}, F{aperture:g}, {shutter} -> {output_path.name}")
        output_path.write_text("dry-run placeholder\n", encoding="utf-8")
        restore_owner(output_path)
        return output_path.stat().st_size

    def capture_auto(
        self,
        output_path: Path,
        timeout: float,
        save_media: str = DEFAULT_SAVE_MEDIA,
        fast_shutter: bool = False,
    ) -> int:
        print(f"[dry-run] capture auto exposure -> {output_path.name}")
        output_path.write_text("dry-run auto exposure placeholder\n", encoding="utf-8")
        restore_owner(output_path)
        return output_path.stat().st_size


class SonyCameraController:
    def __init__(self):
        self._context = None
        self.camera = None
        self.ExposureMode = None
        self.DeviceProperty = None
        self.SaveMedia = None
        self.iso_table = None
        self.aperture_table = None
        self.shutter_table = None
        self._last_exposure_mode = None
        self._last_iso = None
        self._last_aperture = None
        self._last_shutter = None

    def __enter__(self) -> "SonyCameraController":
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
        self._set_exposure_mode(ExposureMode.MANUAL, "manual exposure mode")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._context is not None:
            self._context.__exit__(exc_type, exc_value, traceback)

    def capture(
        self,
        aperture: float,
        iso: int,
        shutter: str,
        output_path: Path,
        timeout: float,
        save_media: str = DEFAULT_SAVE_MEDIA,
        fast_shutter: bool = False,
    ) -> int:
        if self.camera is None:
            raise RuntimeError("Camera is not connected.")

        iso_code = self._iso_code(iso)
        aperture_code = self._aperture_code(aperture)
        shutter_code = self._shutter_code(shutter)
        changed_to_manual = self._set_exposure_mode(
            self.ExposureMode.MANUAL,
            "manual exposure mode",
        )

        if self._last_iso != iso_code:
            self.camera.set_iso(iso_code)
            self._wait_for_setting(self.DeviceProperty.ISO, iso_code, "ISO")
            self._last_iso = iso_code

        if self._last_aperture != aperture_code:
            self.camera.set_aperture(aperture_code)
            self._wait_for_setting(self.DeviceProperty.F_NUMBER, aperture_code, "aperture")
            self._last_aperture = aperture_code

        if self._last_shutter != shutter_code:
            self.camera.set_shutter_speed(shutter_code)
            self._wait_for_setting(
                self.DeviceProperty.SHUTTER_SPEED,
                shutter_code,
                "shutter speed",
            )
            self._last_shutter = shutter_code
        capture_fast = fast_shutter and not changed_to_manual
        if fast_shutter and changed_to_manual:
            print("Using normal shutter for first manual capture after auto exposure; fast shutter resumes after this.")
        try:
            return self._capture_to_path(output_path, timeout, save_media, capture_fast)
        except RuntimeError as exc:
            if not changed_to_manual or "timed out waiting for image" not in str(exc):
                raise
            print("First manual capture after auto exposure timed out; retrying once with normal shutter.")
            return self._capture_to_path(output_path, timeout, save_media, fast_shutter=False)

    def capture_auto(
        self,
        output_path: Path,
        timeout: float,
        save_media: str = DEFAULT_SAVE_MEDIA,
        fast_shutter: bool = False,
    ) -> int:
        if self.camera is None:
            raise RuntimeError("Camera is not connected.")
        self._set_exposure_mode(self._auto_exposure_mode(), "auto exposure mode")
        auto_iso_code = self._auto_iso_code()
        if self._last_iso != auto_iso_code:
            self.camera.set_iso(auto_iso_code)
            self._wait_for_setting(self.DeviceProperty.ISO, auto_iso_code, "ISO AUTO")
            self._last_iso = auto_iso_code
        return self._capture_to_path(output_path, timeout, save_media, fast_shutter=False)

    def _capture_to_path(
        self,
        output_path: Path,
        timeout: float,
        save_media: str,
        fast_shutter: bool,
    ) -> int:
        media = self._save_media_value(save_media)
        host_receives = media in (self.SaveMedia.HOST, self.SaveMedia.HOST_AND_CAMERA)

        self.camera.set_save_media(media)
        self._wait_for_setting(self.DeviceProperty.SAVE_MEDIA, int(media), "save media")
        self.camera._wait_for_liveview()
        self.camera._wait_for_shooting_file_info_clear(timeout=timeout)
        self.camera._fire_shutter(fast=fast_shutter)

        if not host_receives:
            return 0

        deadline = time.monotonic() + timeout
        shooting_file_info = 0
        while time.monotonic() < deadline:
            info = self.camera.get_property(self.DeviceProperty.SHOOTING_FILE_INFO)
            shooting_file_info = info.current_value if isinstance(info.current_value, int) else 0
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

    def _set_exposure_mode(self, mode, label: str) -> bool:
        mode_value = int(mode)
        if self._last_exposure_mode == mode_value:
            return False
        attempts = 4
        last_actual = None
        for attempt in range(1, attempts + 1):
            self.camera.set_exposure_mode(mode)
            try:
                self._wait_for_setting(
                    self.DeviceProperty.EXPOSURE_MODE,
                    mode_value,
                    label,
                    timeout=5.0 + attempt * 2.0,
                )
                break
            except RuntimeError:
                last_actual = self.camera.get_property(self.DeviceProperty.EXPOSURE_MODE).current_value
                if attempt == attempts:
                    raise RuntimeError(
                        f"Camera did not apply {label} after {attempts} attempts: "
                        f"expected 0x{mode_value:X}, got {self._format_exposure_mode(last_actual)}"
                    )
                print(
                    f"Camera still in {self._format_exposure_mode(last_actual)} while setting {label}; "
                    f"retrying ({attempt}/{attempts})..."
                )
                self._settle_before_exposure_mode_retry()
        self._last_exposure_mode = mode_value
        self._last_iso = None
        self._last_aperture = None
        self._last_shutter = None
        return True

    def _settle_before_exposure_mode_retry(self) -> None:
        try:
            self.camera._wait_for_liveview()
        except Exception:
            pass
        try:
            self.camera._wait_for_shooting_file_info_clear(timeout=5.0)
        except Exception:
            pass
        try:
            self.camera.set_mode("still")
        except Exception:
            pass
        time.sleep(1.0)

    def _format_exposure_mode(self, value) -> str:
        if value is None:
            return "unknown"
        try:
            numeric_value = int(value)
        except (TypeError, ValueError):
            return repr(value)
        for name, mode in getattr(self.ExposureMode, "__members__", {}).items():
            if int(mode) == numeric_value:
                return f"{name} (0x{numeric_value:X})"
        return f"0x{numeric_value:X}"

    def _auto_exposure_mode(self):
        for name in ("PROGRAM_AUTO", "PROGRAM", "AUTO", "P", "INTELLIGENT_AUTO"):
            mode = getattr(self.ExposureMode, name, None)
            if mode is not None:
                return mode
        available = ", ".join(
            name for name in getattr(self.ExposureMode, "__members__", {}) if name != "MANUAL"
        )
        raise RuntimeError(
            "Could not find a supported auto exposure mode in pysonycam ExposureMode. "
            f"Available modes: {available or 'unknown'}."
        )

    def _reverse_table(self, table: dict[int, str]) -> dict[str, int]:
        return {label.upper(): code for code, label in table.items()}

    def _iso_code(self, value: int) -> int:
        code = self._reverse_table(self.iso_table).get(str(value))
        if code is None:
            raise ValueError(f"ISO {value} is not supported by this camera.")
        return code

    def _auto_iso_code(self) -> int:
        code = self._reverse_table(self.iso_table).get("AUTO")
        if code is None:
            raise RuntimeError("ISO AUTO is not supported by this camera.")
        return code

    def _aperture_code(self, value: float) -> int:
        label = f"F{value:g}".upper()
        code = self._reverse_table(self.aperture_table).get(label)
        if code is None:
            raise ValueError(f"Aperture F{value:g} is not supported by this lens.")
        return code

    def _shutter_code(self, value: str) -> int:
        code = self._reverse_table(self.shutter_table).get(value.strip().upper())
        if code is None:
            raise ValueError(f"Shutter speed {value!r} is not supported by this camera.")
        return code

    def _save_media_value(self, value: str):
        if value == "host":
            return self.SaveMedia.HOST
        if value == "camera":
            return self.SaveMedia.CAMERA
        if value == "host-and-camera":
            return self.SaveMedia.HOST_AND_CAMERA
        raise ValueError(f"Unsupported save media: {value!r}")

    def _wait_for_setting(
        self,
        property_code: int,
        expected: int,
        label: str,
        timeout: float = 5.0,
    ) -> None:
        deadline = time.monotonic() + timeout
        actual = None
        while time.monotonic() < deadline:
            actual = self.camera.get_property(property_code).current_value
            if actual == expected:
                return
            time.sleep(0.1)
        raise RuntimeError(
            f"Camera did not apply {label}: expected 0x{expected:X}, got {actual!r}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified dataset capture controller.")
    parser.add_argument(
        "--class-id",
        help="Numeric class id. When omitted, prompts repeatedly for continuous capture.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--light-position",
        default=DEFAULT_LIGHT_POSITION,
        help="Current physical light position label used in filenames/maps.",
    )
    parser.add_argument(
        "--light-intensities",
        type=parse_int_list,
        default=DEFAULT_LIGHT_INTENSITIES,
        help="Comma-separated light intensities for the current light position.",
    )
    parser.add_argument("--cct", type=int, default=DEFAULT_CCT)
    parser.add_argument("--apertures", type=parse_float_list, default=DEFAULT_APERTURES)
    parser.add_argument("--isos", type=parse_int_list, default=DEFAULT_ISOS)
    parser.add_argument("--shutters", type=parse_shutter_list, default=DEFAULT_SHUTTERS)
    parser.add_argument("--turntable-port", default=DEFAULT_TURNTABLE_PORT)
    parser.add_argument("--turntable-speed", type=float, default=15.0)
    parser.add_argument("--view-step", type=int, default=90)
    parser.add_argument("--views", type=int, default=4)
    parser.add_argument("--capture-timeout", type=float, default=30.0)
    parser.add_argument(
        "--start-delay-seconds",
        type=float,
        default=DEFAULT_START_DELAY_SECONDS,
        help="Seconds to wait after each class_id is entered before capture starts.",
    )
    parser.add_argument(
        "--save-media",
        choices=["host", "host-and-camera"],
        default=DEFAULT_SAVE_MEDIA,
        help="Where to save captured images. Default host means computer only.",
    )
    parser.add_argument(
        "--capture-mode",
        choices=["append", "fresh"],
        default="append",
        help="append reuses existing JSON maps; fresh requires an empty output directory.",
    )
    # settle waiting time used after changing light and view
    parser.add_argument("--settle-seconds", type=float, default=1.0)
    parser.add_argument("--light-ws-url", default="ws://127.0.0.1:12345")
    parser.add_argument(
        "--light-api-secret-key",
        default="cDdzYXNkbXM5d2V2a3EwaTJ0Z2tocHRlNjE2NWs5ODY=",
    )
    parser.add_argument("--light-client-id", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-light", action="store_true")
    parser.add_argument("--skip-turntable", action="store_true")
    parser.add_argument(
        "--skip-auto-exposure",
        action="store_true",
        help="Do not capture the p000 auto-exposure image before each manual parameter sweep.",
    )
    parser.add_argument(
        "--auto-exposure-only",
        action="store_true",
        help="Capture only p000 auto-exposure images and skip the manual parameter sweep.",
    )
    parser.add_argument(
        "--fast-shutter",
        action="store_true",
        help="Use pysonycam's fast shutter sequence to reduce S1/S2 trigger delay.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.class_id is not None:
        args.class_id = normalize_class_id(args.class_id)
    args.light_position = args.light_position.strip()
    if not args.light_position:
        raise ValueError("--light-position cannot be empty.")
    if not args.dry_run and not args.skip_turntable:
        args.turntable_port = resolve_turntable_port(args.turntable_port)
    for intensity in args.light_intensities:
        validate_light_intensity(intensity)
    if args.views < 1:
        raise ValueError("--views must be at least 1.")
    if args.start_delay_seconds < 0:
        raise ValueError("--start-delay-seconds must be >= 0.")
    if args.auto_exposure_only and args.skip_auto_exposure:
        raise ValueError("--auto-exposure-only cannot be combined with --skip-auto-exposure.")


def resolve_turntable_port(port: str) -> str:
    if Path(port).exists():
        return port

    usbmodem_ports = sorted(glob.glob("/dev/cu.usbmodem*"))
    if len(usbmodem_ports) == 1:
        detected_port = usbmodem_ports[0]
        print(f"Turntable port {port} not found; using detected port {detected_port}")
        return detected_port

    available_ports = sorted(glob.glob("/dev/cu.*"))
    available_text = "\n  ".join(available_ports) if available_ports else "(none)"
    raise RuntimeError(
        f"Turntable port {port} not found.\n"
        f"Available serial ports:\n  {available_text}\n"
        "Pass the correct one with --turntable-port /dev/cu.usbmodemXXXX, "
        "or use --skip-turntable for a camera/light-only test."
    )


def make_light_controller(args: argparse.Namespace):
    if args.dry_run or args.skip_light:
        return DryLightController()
    return AmaranLightController(
        ws_url=args.light_ws_url,
        api_secret_key=args.light_api_secret_key,
        client_id=args.light_client_id,
        cct=args.cct,
        settle_seconds=args.settle_seconds,
    )


def make_turntable_controller(args: argparse.Namespace):
    if args.dry_run or args.skip_turntable:
        return DryTurntableController()
    return TurntableController(
        port=args.turntable_port,
        speed_rpm=args.turntable_speed,
        settle_seconds=args.settle_seconds,
    )


def make_camera_controller(args: argparse.Namespace):
    if args.dry_run:
        return DryCameraController()
    return SonyCameraController()


async def capture_one_class_id(
    args: argparse.Namespace,
    class_id: int,
    output_dir: Path,
    map_dir: Path,
    dataset_map: dict[str, object],
    lighting_plan: list[dict[str, object]],
    view_plan: list[dict[str, object]],
    param_plan: list[dict[str, object]],
    light,
    turntable,
    camera,
) -> None:
    object_start_monotonic = time.monotonic()
    object_started_at = timestamp()
    session_id = make_session_id()
    auto_exposure_enabled = not args.skip_auto_exposure
    params_per_view = len(param_plan) + (1 if auto_exposure_enabled else 0)
    total_captures = len(lighting_plan) * len(view_plan) * params_per_view
    ensure_class_entry(dataset_map, class_id)
    light_ids = [int(item["light_id"]) for item in lighting_plan]
    sample_id = next_sample_id(dataset_map, output_dir, class_id, light_ids)
    for sample_dir in light_sample_dirs(output_dir, light_ids, class_id, sample_id):
        sample_dir.mkdir(parents=True, exist_ok=True)
        current_dir = sample_dir
        while current_dir != output_dir:
            restore_owner(current_dir)
            current_dir = current_dir.parent
    append_sample_entry(
        dataset_map,
        class_id,
        sample_id,
        light_ids,
        session_id,
        total_captures,
        object_started_at,
    )
    save_dataset_map(map_dir, dataset_map)

    print(f"\nClass ID: {class_id}")
    print(f"Sample: {sample_id}")
    print(f"Session: {session_id}")
    print(f"Sample folder: {sample_folder_text(output_dir, light_ids, class_id, sample_id)}")
    print(f"Capture index: {map_dir / 'captures.jsonl'}")
    print(f"Total captures: {total_captures}")
    if args.auto_exposure_only:
        plan_detail = "auto exposure only"
    else:
        plan_detail = (
            f"{len(param_plan)} manual parameter"
            f"{' + auto exposure' if auto_exposure_enabled else ''}"
        )
    print(
        "Plan: "
        f"{len(lighting_plan)} lighting x "
        f"{len(view_plan)} views x "
        f"{plan_detail}"
    )
    if args.start_delay_seconds > 0:
        print(f"Starting capture in {args.start_delay_seconds:g} seconds...")
        time.sleep(args.start_delay_seconds)

    capture_start_monotonic = time.monotonic()
    capture_started_at = timestamp()
    sequence = 0
    for light_item in lighting_plan:
        light_id = int(light_item["light_id"])
        light_position = str(light_item["position"])
        light_value = int(light_item["intensity"])
        light_cct = int(light_item["cct"])

        await light.set_cct(light_cct)
        await light.set_intensity(light_value)
        if hasattr(turntable, "goto"):
            turntable.goto(0)

        for view_item in view_plan:
            view_id = int(view_item["view_id"])
            view_index = int(view_item["view_index"])
            angle = int(view_item["angle_degrees"])
            if view_index > 0:
                turntable.rotate(args.view_step)

            if auto_exposure_enabled:
                sequence += 1
                output_path = build_output_path(
                    output_dir=output_dir,
                    class_id=class_id,
                    sample_id=sample_id,
                    light_id=light_id,
                    view_id=view_id,
                    param_id=AUTO_PARAM_ID,
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)
                restore_owner(output_path.parent)
                print(
                    f"[{sequence}/{total_captures}] "
                    f"class {class_id}, sample {sample_id}, "
                    f"light {light_id} ({light_position}, {light_value}/1000, {light_cct}K), "
                    f"view {view_id} ({angle} deg), param {AUTO_PARAM_ID} "
                    "(auto exposure)"
                )
                size = camera.capture_auto(
                    output_path=output_path,
                    timeout=args.capture_timeout,
                    save_media=args.save_media,
                    fast_shutter=args.fast_shutter,
                )
                captured_at = timestamp()
                capture_record = build_capture_record(
                    session_id=session_id,
                    sequence=sequence,
                    class_id=class_id,
                    sample_id=sample_id,
                    light_id=light_id,
                    light_position=light_position,
                    light_intensity=light_value,
                    light_cct=light_cct,
                    view_id=view_id,
                    angle_degrees=angle,
                    param_id=AUTO_PARAM_ID,
                    aperture="auto",
                    iso="auto",
                    shutter_speed="auto",
                    exposure_mode="auto",
                    output_dir=output_dir,
                    output_path=output_path,
                    captured_at=captured_at,
                    size_bytes=size,
                )
                append_jsonl_record(map_dir / "captures.jsonl", capture_record)
                print(f"Saved {output_path.relative_to(output_dir)} ({size:,} bytes)")

            for param_item in param_plan:
                param_id = int(param_item["param_id"])
                aperture = float(param_item["aperture"])
                iso = int(param_item["iso"])
                shutter = str(param_item["shutter_speed"])
                sequence += 1
                output_path = build_output_path(
                    output_dir=output_dir,
                    class_id=class_id,
                    sample_id=sample_id,
                    light_id=light_id,
                    view_id=view_id,
                    param_id=param_id,
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)
                restore_owner(output_path.parent)
                print(
                    f"[{sequence}/{total_captures}] "
                    f"class {class_id}, sample {sample_id}, "
                    f"light {light_id} ({light_position}, {light_value}/1000, {light_cct}K), "
                    f"view {view_id} ({angle} deg), param {param_id} "
                    f"(F{aperture:g}, ISO {iso}, {shutter})"
                )
                size = camera.capture(
                    aperture=aperture,
                    iso=iso,
                    shutter=shutter,
                    output_path=output_path,
                    timeout=args.capture_timeout,
                    save_media=args.save_media,
                    fast_shutter=args.fast_shutter,
                )
                captured_at = timestamp()
                capture_record = build_capture_record(
                    session_id=session_id,
                    sequence=sequence,
                    class_id=class_id,
                    sample_id=sample_id,
                    light_id=light_id,
                    light_position=light_position,
                    light_intensity=light_value,
                    light_cct=light_cct,
                    view_id=view_id,
                    angle_degrees=angle,
                    param_id=param_id,
                    aperture=aperture,
                    iso=iso,
                    shutter_speed=shutter,
                    exposure_mode="manual",
                    output_dir=output_dir,
                    output_path=output_path,
                    captured_at=captured_at,
                    size_bytes=size,
                )
                append_jsonl_record(map_dir / "captures.jsonl", capture_record)
                print(f"Saved {output_path.relative_to(output_dir)} ({size:,} bytes)")

        if args.views > 1:
            turntable.rotate(args.view_step)

    completed_at = timestamp()
    end_monotonic = time.monotonic()
    object_elapsed_seconds = end_monotonic - object_start_monotonic
    capture_elapsed_seconds = end_monotonic - capture_start_monotonic
    update_sample_timing(
        dataset_map,
        class_id,
        sample_id,
        session_id,
        capture_started_at,
        completed_at,
        object_elapsed_seconds,
        capture_elapsed_seconds,
    )
    save_dataset_map(map_dir, dataset_map)
    print(
        f"Class ID {class_id} capture complete in {format_duration(object_elapsed_seconds)} "
        f"(capture loop {format_duration(capture_elapsed_seconds)})."
    )


async def run_capture(args: argparse.Namespace) -> None:
    validate_args(args)

    output_dir = args.output_dir.expanduser().resolve()
    ensure_output_dir_for_mode(output_dir, args.capture_mode)

    map_dir = maps_dir(output_dir)
    if args.capture_mode == "fresh":
        dataset_map = new_dataset_map()
        (map_dir / "captures.jsonl").write_text("", encoding="utf-8")
        restore_owner(map_dir / "captures.jsonl")
    else:
        dataset_map = load_dataset_map(map_dir / "parameters.json")

    lighting_plan = build_lighting_plan(dataset_map, args)
    view_plan = build_view_plan(dataset_map, args)
    if not args.skip_auto_exposure:
        ensure_auto_param_entry(dataset_map)
    param_plan = [] if args.auto_exposure_only else build_param_plan(dataset_map, args)
    save_dataset_map(map_dir, dataset_map)
    light_controller = make_light_controller(args)
    turntable_controller = make_turntable_controller(args)
    camera_controller = make_camera_controller(args)

    print(f"Output folder: {output_dir}")
    print(f"Maps folder: {map_dir}")
    print("Connecting hardware once for continuous capture...")

    async with light_controller as light:
        with turntable_controller as turntable, camera_controller as camera:
            if hasattr(turntable, "home"):
                turntable.home()
            if hasattr(turntable, "set_speed"):
                turntable.set_speed(args.turntable_speed)

            while True:
                if args.class_id is not None:
                    class_id = args.class_id
                else:
                    class_id = prompt_class_id()
                    if class_id is None:
                        break

                await capture_one_class_id(
                    args=args,
                    class_id=class_id,
                    output_dir=output_dir,
                    map_dir=map_dir,
                    dataset_map=dataset_map,
                    lighting_plan=lighting_plan,
                    view_plan=view_plan,
                    param_plan=param_plan,
                    light=light,
                    turntable=turntable,
                    camera=camera,
                )

                if args.class_id is not None:
                    break

                input("\nChange object, then press Enter for the next class_id...")

    print("Capture session complete.")


def main() -> None:
    asyncio.run(run_capture(parse_args()))


if __name__ == "__main__":
    main()
