"""Unified capture controller for light, turntable, and Sony camera.

Run from the repository root, usually through the Sony camera launcher::

    sonycam control.py

Useful dry run for checking the capture plan and filenames::

    python control.py --dry-run

Default plan:
    lighting intensities: 10, 50
    views: 0, 90, 180, 270 degrees
    apertures: F3.2
    ISO: 800
    shutter speeds: 1/80
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import csv
import glob
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional


DEFAULT_OUTPUT_DIR = Path("sony_camera/captures/dataset")
DEFAULT_TURNTABLE_PORT = "/dev/cu.usbmodem1101"
DEFAULT_CCT = 5600
DEFAULT_LIGHT_INTENSITIES = [0, 10, 50, 300, 500, 700, 1000]
DEFAULT_APERTURES = [2.8, 4, 8, 13, 16, 22]
DEFAULT_ISOS = [100, 250, 800, 2000, 3200, 6400, 12800, 32000]
DEFAULT_SHUTTERS = ['0.5"', "1/3", "1/15", "1/60", "1/250", "1/1000"]
DEFAULT_SAVE_MEDIA = "host"
DEFAULT_START_DELAY_SECONDS = 10.0
PAD_WIDTH = 3

CLASS_FIELDS = [
    "class_id",
    "class_folder",
    "created_at",
]
SAMPLE_FIELDS = [
    "class_id",
    "sample_id",
    "class_folder",
    "sample_folder",
    "session_id",
    "created_at",
    "total_captures",
]
LIGHT_FIELDS = [
    "light_id",
    "light_folder",
    "intensity",
    "light_percent",
    "cct",
]
VIEW_FIELDS = [
    "view_id",
    "view_folder",
    "view_index",
    "angle_degrees",
]
PARAM_FIELDS = [
    "param_id",
    "param_file",
    "aperture",
    "iso",
    "shutter_speed",
]
IMAGE_FIELDS = [
    "session_id",
    "sequence",
    "class_id",
    "sample_id",
    "light_id",
    "view_id",
    "param_id",
    "image_path",
    "captured_at",
]


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


def restore_owner(path: Path) -> None:
    """Make files created through sudo writable by the invoking user."""
    uid_text = os.environ.get("SUDO_UID")
    gid_text = os.environ.get("SUDO_GID")
    if uid_text is None or gid_text is None:
        return
    os.chown(path, int(uid_text), int(gid_text))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def append_csv_row(path: Path, fieldnames: list[str], row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    restore_owner(path)


def next_id(rows: list[dict[str, str]], key: str) -> int:
    values = [int(row[key]) for row in rows if row.get(key, "").isdigit()]
    return max(values, default=0) + 1


def maps_dir(output_dir: Path) -> Path:
    path = output_dir / "maps"
    path.mkdir(parents=True, exist_ok=True)
    restore_owner(path)
    return path


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
        / format_id("c", class_id)
        / format_id("s", sample_id)
        / format_id("l", light_id)
        / format_id("v", view_id)
        / f"{format_id('p', param_id)}.jpg"
    )


def ensure_class_row(map_dir: Path, class_id: int) -> None:
    path = map_dir / "classes.csv"
    rows = read_csv_rows(path)
    if any(int(row["class_id"]) == class_id for row in rows if row.get("class_id", "").isdigit()):
        return
    append_csv_row(
        path,
        CLASS_FIELDS,
        {
            "class_id": class_id,
            "class_folder": format_id("c", class_id),
            "created_at": timestamp(),
        },
    )


def next_sample_id(map_dir: Path, class_id: int) -> int:
    rows = read_csv_rows(map_dir / "samples.csv")
    sample_ids = [
        int(row["sample_id"])
        for row in rows
        if row.get("class_id") == str(class_id) and row.get("sample_id", "").isdigit()
    ]
    return max(sample_ids, default=0) + 1


def append_sample_row(
    map_dir: Path,
    class_id: int,
    sample_id: int,
    session_id: str,
    total_captures: int,
) -> None:
    append_csv_row(
        map_dir / "samples.csv",
        SAMPLE_FIELDS,
        {
            "class_id": class_id,
            "sample_id": sample_id,
            "class_folder": format_id("c", class_id),
            "sample_folder": format_id("s", sample_id),
            "session_id": session_id,
            "created_at": timestamp(),
            "total_captures": total_captures,
        },
    )


def get_or_create_light_id(map_dir: Path, intensity: int, cct: int) -> int:
    path = map_dir / "lights.csv"
    rows = read_csv_rows(path)
    for row in rows:
        if row.get("intensity") == str(intensity) and row.get("cct") == str(cct):
            return int(row["light_id"])

    light_id = next_id(rows, "light_id")
    append_csv_row(
        path,
        LIGHT_FIELDS,
        {
            "light_id": light_id,
            "light_folder": format_id("l", light_id),
            "intensity": intensity,
            "light_percent": light_percent(intensity),
            "cct": cct,
        },
    )
    return light_id


def get_or_create_view_id(map_dir: Path, view_index: int, angle_degrees: int) -> int:
    path = map_dir / "views.csv"
    rows = read_csv_rows(path)
    for row in rows:
        if (
            row.get("view_index") == str(view_index)
            and row.get("angle_degrees") == str(angle_degrees)
        ):
            return int(row["view_id"])

    view_id = next_id(rows, "view_id")
    append_csv_row(
        path,
        VIEW_FIELDS,
        {
            "view_id": view_id,
            "view_folder": format_id("v", view_id),
            "view_index": view_index,
            "angle_degrees": angle_degrees,
        },
    )
    return view_id


def get_or_create_param_id(map_dir: Path, aperture: float, iso: int, shutter: str) -> int:
    path = map_dir / "params.csv"
    rows = read_csv_rows(path)
    aperture_value = format_number(aperture)
    for row in rows:
        if (
            row.get("aperture") == aperture_value
            and row.get("iso") == str(iso)
            and row.get("shutter_speed") == shutter
        ):
            return int(row["param_id"])

    param_id = next_id(rows, "param_id")
    append_csv_row(
        path,
        PARAM_FIELDS,
        {
            "param_id": param_id,
            "param_file": f"{format_id('p', param_id)}.jpg",
            "aperture": aperture_value,
            "iso": iso,
            "shutter_speed": shutter,
        },
    )
    return param_id


def build_light_plan(map_dir: Path, args: argparse.Namespace) -> list[dict[str, object]]:
    return [
        {
            "light_id": get_or_create_light_id(map_dir, intensity, args.cct),
            "intensity": intensity,
        }
        for intensity in args.light_intensities
    ]


def build_view_plan(map_dir: Path, args: argparse.Namespace) -> list[dict[str, object]]:
    view_plan = []
    for view_index in range(args.views):
        angle = (view_index * args.view_step) % 360
        view_plan.append(
            {
                "view_id": get_or_create_view_id(map_dir, view_index, angle),
                "view_index": view_index,
                "angle_degrees": angle,
            }
        )
    return view_plan


def build_param_plan(map_dir: Path, args: argparse.Namespace) -> list[dict[str, object]]:
    param_plan = []
    for aperture in args.apertures:
        for iso in args.isos:
            for shutter in args.shutters:
                param_plan.append(
                    {
                        "param_id": get_or_create_param_id(map_dir, aperture, iso, shutter),
                        "aperture": aperture,
                        "iso": iso,
                        "shutter_speed": shutter,
                    }
                )
    return param_plan


class DryLightController:
    async def __aenter__(self) -> "DryLightController":
        print("[dry-run] light connected")
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        print("[dry-run] light disconnected")

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
        self._ensure_ok(
            await self._send("set_cct", node_id=self.node_id, args={"cct": self.cct}),
            "set_cct",
        )
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
        print_timing: bool = False,
    ) -> int:
        print(f"[dry-run] capture ISO {iso}, F{aperture:g}, {shutter} -> {output_path.name}")
        start = time.monotonic()
        output_path.write_text("dry-run placeholder\n", encoding="utf-8")
        restore_owner(output_path)
        if print_timing:
            print(f"  timing camera_capture_store: {time.monotonic() - start:.4f}s")
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
        self._last_iso = None
        self._last_aperture = None
        self._last_shutter = None

    def __enter__(self) -> "SonyCameraController":
        from pysonycam import ExposureMode, SonyCamera
        from pysonycam.constants import (
            DeviceProperty,
            F_NUMBER_TABLE,
            ISO_TABLE,
            SHUTTER_SPEED_TABLE,
            SaveMedia,
        )

        self.ExposureMode = ExposureMode
        self.DeviceProperty = DeviceProperty
        self.SaveMedia = SaveMedia
        self.iso_table = ISO_TABLE
        self.aperture_table = F_NUMBER_TABLE
        self.shutter_table = SHUTTER_SPEED_TABLE

        self._context = SonyCamera()
        self.camera = self._context.__enter__()
        self.camera.authenticate()
        self.camera.set_mode("still")
        self.camera.set_exposure_mode(ExposureMode.MANUAL)
        self._wait_for_setting(
            DeviceProperty.EXPOSURE_MODE,
            int(ExposureMode.MANUAL),
            "manual exposure mode",
        )
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
        print_timing: bool = False,
    ) -> int:
        if self.camera is None:
            raise RuntimeError("Camera is not connected.")

        iso_code = self._iso_code(iso)
        aperture_code = self._aperture_code(aperture)
        shutter_code = self._shutter_code(shutter)

        if self._last_iso != iso_code:
            start = time.monotonic()
            self.camera.set_iso(iso_code)
            self._wait_for_setting(self.DeviceProperty.ISO, iso_code, "ISO")
            if print_timing:
                print(f"  timing camera_iso: {time.monotonic() - start:.4f}s")
            self._last_iso = iso_code
        elif print_timing:
            print("  timing camera_iso: skipped")

        if self._last_aperture != aperture_code:
            start = time.monotonic()
            self.camera.set_aperture(aperture_code)
            self._wait_for_setting(self.DeviceProperty.F_NUMBER, aperture_code, "aperture")
            if print_timing:
                print(f"  timing camera_aperture: {time.monotonic() - start:.4f}s")
            self._last_aperture = aperture_code
        elif print_timing:
            print("  timing camera_aperture: skipped")

        if self._last_shutter != shutter_code:
            start = time.monotonic()
            self.camera.set_shutter_speed(shutter_code)
            self._wait_for_setting(
                self.DeviceProperty.SHUTTER_SPEED,
                shutter_code,
                "shutter speed",
            )
            if print_timing:
                print(f"  timing camera_shutter: {time.monotonic() - start:.4f}s")
            self._last_shutter = shutter_code
        elif print_timing:
            print("  timing camera_shutter: skipped")

        start = time.monotonic()
        image_data = self.camera.capture(
            output_path=output_path,
            save_to_camera=self._save_media_value(save_media),
            timeout=timeout,
        )
        restore_owner(output_path)
        if print_timing:
            print(f"  timing camera_capture_store: {time.monotonic() - start:.4f}s")
        return len(image_data)

    def _reverse_table(self, table: dict[int, str]) -> dict[str, int]:
        return {label.upper(): code for code, label in table.items()}

    def _iso_code(self, value: int) -> int:
        code = self._reverse_table(self.iso_table).get(str(value))
        if code is None:
            raise ValueError(f"ISO {value} is not supported by this camera.")
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
    parser.add_argument("--light-intensities", type=parse_int_list, default=DEFAULT_LIGHT_INTENSITIES)
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
        "--print-timing",
        action="store_true",
        help="Print lightweight timing for light changes, turntable moves, camera settings, and capture/store.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.class_id is not None:
        args.class_id = normalize_class_id(args.class_id)
    if not args.dry_run and not args.skip_turntable:
        args.turntable_port = resolve_turntable_port(args.turntable_port)
    for intensity in args.light_intensities:
        if intensity < 0 or intensity > 1000:
            raise ValueError(f"Light intensity must be in [0, 1000], got {intensity}.")
    if args.views < 1:
        raise ValueError("--views must be at least 1.")
    if args.start_delay_seconds < 0:
        raise ValueError("--start-delay-seconds must be >= 0.")


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


def write_session_config(
    sample_dir: Path,
    session_id: str,
    class_id: int,
    sample_id: int,
    args: argparse.Namespace,
    total_captures: int,
) -> None:
    config_path = sample_dir / "session.json"
    config = {
        "session_id": session_id,
        "class_id": class_id,
        "sample_id": sample_id,
        "class_folder": format_id("c", class_id),
        "sample_folder": format_id("s", sample_id),
        "created_at": timestamp(),
        "sample_dir": str(sample_dir),
        "total_captures": total_captures,
        "start_delay_seconds": args.start_delay_seconds,
        "light_intensities": args.light_intensities,
        "cct": args.cct,
        "views": args.views,
        "view_step": args.view_step,
        "apertures": args.apertures,
        "isos": args.isos,
        "shutters": args.shutters,
        "save_media": args.save_media,
        "turntable_port": args.turntable_port,
        "turntable_speed": args.turntable_speed,
        "dry_run": args.dry_run,
        "skip_light": args.skip_light,
        "skip_turntable": args.skip_turntable,
        "print_timing": args.print_timing,
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    restore_owner(config_path)


async def capture_one_class_id(
    args: argparse.Namespace,
    class_id: int,
    output_dir: Path,
    map_dir: Path,
    light_plan: list[dict[str, object]],
    view_plan: list[dict[str, object]],
    param_plan: list[dict[str, object]],
    light,
    turntable,
    camera,
) -> None:
    session_id = make_session_id()
    total_captures = len(light_plan) * len(view_plan) * len(param_plan)
    ensure_class_row(map_dir, class_id)
    sample_id = next_sample_id(map_dir, class_id)
    sample_dir = output_dir / format_id("c", class_id) / format_id("s", sample_id)
    sample_dir.mkdir(parents=True, exist_ok=True)
    restore_owner(sample_dir)
    write_session_config(sample_dir, session_id, class_id, sample_id, args, total_captures)
    append_sample_row(map_dir, class_id, sample_id, session_id, total_captures)

    print(f"\nClass ID: {class_id}")
    print(f"Sample: {sample_id}")
    print(f"Session: {session_id}")
    print(f"Sample folder: {sample_dir}")
    print(f"Image index: {map_dir / 'images.csv'}")
    print(f"Total captures: {total_captures}")
    print(
        "Plan: "
        f"{len(light_plan)} lighting x "
        f"{len(view_plan)} views x "
        f"{len(param_plan)} parameter"
    )
    if args.start_delay_seconds > 0:
        print(f"Starting capture in {args.start_delay_seconds:g} seconds...")
        time.sleep(args.start_delay_seconds)

    sequence = 0
    for light_item in light_plan:
        light_id = int(light_item["light_id"])
        light_value = int(light_item["intensity"])
        start = time.monotonic()
        await light.set_intensity(light_value)
        if args.print_timing:
            print(f"  timing light_change: {time.monotonic() - start:.4f}s")
        if hasattr(turntable, "goto"):
            start = time.monotonic()
            turntable.goto(0)
            if args.print_timing:
                print(f"  timing turntable_goto_0: {time.monotonic() - start:.4f}s")

        for view_item in view_plan:
            view_id = int(view_item["view_id"])
            view_index = int(view_item["view_index"])
            angle = int(view_item["angle_degrees"])
            if view_index > 0:
                start = time.monotonic()
                turntable.rotate(args.view_step)
                if args.print_timing:
                    print(f"  timing turntable_rotate_{args.view_step:g}: {time.monotonic() - start:.4f}s")

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
                    f"light {light_id} ({light_value}/1000), "
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
                    print_timing=args.print_timing,
                )
                append_csv_row(
                    map_dir / "images.csv",
                    IMAGE_FIELDS,
                    {
                        "session_id": session_id,
                        "sequence": sequence,
                        "class_id": class_id,
                        "sample_id": sample_id,
                        "light_id": light_id,
                        "view_id": view_id,
                        "param_id": param_id,
                        "image_path": str(output_path.relative_to(output_dir)),
                        "captured_at": timestamp(),
                    },
                )
                print(f"Saved {output_path.relative_to(output_dir)} ({size:,} bytes)")

        if args.views > 1:
            start = time.monotonic()
            turntable.rotate(args.view_step)
            if args.print_timing:
                print(f"  timing turntable_rotate_return: {time.monotonic() - start:.4f}s")

    print(f"Class ID {class_id} capture complete.")


async def run_capture(args: argparse.Namespace) -> None:
    validate_args(args)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    restore_owner(output_dir)

    map_dir = maps_dir(output_dir)
    light_plan = build_light_plan(map_dir, args)
    view_plan = build_view_plan(map_dir, args)
    param_plan = build_param_plan(map_dir, args)
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
                    light_plan=light_plan,
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
