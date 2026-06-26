"""Unified capture controller for light, turntable, and Sony camera.

Run from the repository root, usually through the Sony camera launcher::

    sonycam control.py

Useful dry run for checking the capture plan and filenames::

    python control.py --dry-run --class-name test_object

Default plan:
    lighting intensities: 5
    views: 0, 90, 180, 270 degrees
    apertures: F2.8, F10
    ISO: 250, 800, 5000
    shutter speeds: 1/100, 1/1000
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import csv
import json
import os
import re
import time
from pathlib import Path
from typing import Optional


DEFAULT_OUTPUT_DIR = Path("sony_camera/captures/dataset")
DEFAULT_TURNTABLE_PORT = "/dev/cu.usbmodem1101"
DEFAULT_LIGHT_INTENSITIES = [10, 50]
DEFAULT_APERTURES = [3.2]
DEFAULT_ISOS = [800]
DEFAULT_SHUTTERS = [ "1/80"]

CSV_FIELDS = [
    "session_id",
    "sequence",
    "class_name",
    "file_name",
    "image_path",
    "light_intensity",
    "light_percent",
    "cct",
    "view_index",
    "angle_degrees",
    "aperture",
    "iso",
    "shutter_speed",
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


def slugify(value: str) -> str:
    value = re.sub(r"\s+", "_", value.strip())
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    value = value.strip("._-")
    return value or "class"


def aperture_label(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def shutter_file_label(value: str) -> str:
    return value.replace("/", "-").replace(".", "p")


def light_percent(intensity: int) -> float:
    return intensity / 10.0


def restore_owner(path: Path) -> None:
    """Make files created through sudo writable by the invoking user."""
    uid_text = os.environ.get("SUDO_UID")
    gid_text = os.environ.get("SUDO_GID")
    if uid_text is None or gid_text is None:
        return
    os.chown(path, int(uid_text), int(gid_text))


def append_manifest_row(manifest_path: Path, row: dict[str, object]) -> None:
    file_exists = manifest_path.exists()
    with manifest_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    restore_owner(manifest_path)


def build_output_path(
    output_dir: Path,
    session_id: str,
    class_slug: str,
    sequence: int,
    light: int,
    view_index: int,
    angle: int,
    aperture: float,
    iso: int,
    shutter: str,
) -> Path:
    file_name = (
        f"{session_id}__{class_slug}__{sequence:04d}"
        f"__light{light:04d}"
        f"__view{view_index}_{angle:03d}deg"
        f"__f{aperture_label(aperture)}"
        f"__iso{iso}"
        f"__sh{shutter_file_label(shutter)}.jpg"
    )
    return output_dir / file_name


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
        import websockets

        self.ws = await websockets.connect(self.ws_url)
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
        if self.ws is not None:
            await self.ws.close()

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

    async def _send(
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

    def capture(self, aperture: float, iso: int, shutter: str, output_path: Path, timeout: float) -> int:
        print(f"[dry-run] capture ISO {iso}, F{aperture:g}, {shutter} -> {output_path.name}")
        output_path.write_text("dry-run placeholder\n", encoding="utf-8")
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

    def capture(self, aperture: float, iso: int, shutter: str, output_path: Path, timeout: float) -> int:
        if self.camera is None:
            raise RuntimeError("Camera is not connected.")

        iso_code = self._iso_code(iso)
        aperture_code = self._aperture_code(aperture)
        shutter_code = self._shutter_code(shutter)

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

        image_data = self.camera.capture(
            output_path=output_path,
            save_to_camera=self.SaveMedia.HOST_AND_CAMERA,
            timeout=timeout,
        )
        restore_owner(output_path)
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
    parser.add_argument("--class-name", help="Object/class name. Prompts when omitted.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--light-intensities", type=parse_int_list, default=DEFAULT_LIGHT_INTENSITIES)
    parser.add_argument("--cct", type=int, default=3200)
    parser.add_argument("--apertures", type=parse_float_list, default=DEFAULT_APERTURES)
    parser.add_argument("--isos", type=parse_int_list, default=DEFAULT_ISOS)
    parser.add_argument("--shutters", type=parse_shutter_list, default=DEFAULT_SHUTTERS)
    parser.add_argument("--turntable-port", default=DEFAULT_TURNTABLE_PORT)
    parser.add_argument("--turntable-speed", type=float, default=15.0)
    parser.add_argument("--view-step", type=int, default=90)
    parser.add_argument("--views", type=int, default=4)
    parser.add_argument("--capture-timeout", type=float, default=30.0)
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
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for intensity in args.light_intensities:
        if intensity < 0 or intensity > 1000:
            raise ValueError(f"Light intensity must be in [0, 1000], got {intensity}.")
    if args.views < 1:
        raise ValueError("--views must be at least 1.")


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
    output_dir: Path,
    session_id: str,
    class_name: str,
    args: argparse.Namespace,
    total_captures: int,
) -> None:
    config_path = output_dir / f"{session_id}__session.json"
    config = {
        "session_id": session_id,
        "class_name": class_name,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "output_dir": str(output_dir),
        "total_captures": total_captures,
        "light_intensities": args.light_intensities,
        "cct": args.cct,
        "views": args.views,
        "view_step": args.view_step,
        "apertures": args.apertures,
        "isos": args.isos,
        "shutters": args.shutters,
        "turntable_port": args.turntable_port,
        "turntable_speed": args.turntable_speed,
        "dry_run": args.dry_run,
        "skip_light": args.skip_light,
        "skip_turntable": args.skip_turntable,
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    restore_owner(config_path)


async def run_capture(args: argparse.Namespace) -> None:
    validate_args(args)

    class_name = args.class_name or input("Class name: ").strip()
    if not class_name:
        raise ValueError("Class name cannot be empty.")
    class_slug = slugify(class_name)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    restore_owner(output_dir)

    session_id = time.strftime("%Y%m%d_%H%M%S")
    manifest_path = output_dir / "manifest.csv"
    total_captures = (
        len(args.light_intensities)
        * args.views
        * len(args.apertures)
        * len(args.isos)
        * len(args.shutters)
    )
    write_session_config(output_dir, session_id, class_name, args, total_captures)

    print(f"Class: {class_name}")
    print(f"Output folder: {output_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Total captures: {total_captures}")
    print(
        "Plan: "
        f"{len(args.light_intensities)} lighting x "
        f"{args.views} views x "
        f"{len(args.apertures)} aperture x "
        f"{len(args.isos)} ISO x "
        f"{len(args.shutters)} shutter"
    )

    sequence = 0
    light_controller = make_light_controller(args)
    turntable_controller = make_turntable_controller(args)
    camera_controller = make_camera_controller(args)

    async with light_controller as light:
        with turntable_controller as turntable, camera_controller as camera:
            if hasattr(turntable, "home"):
                turntable.home()
            if hasattr(turntable, "set_speed"):
                turntable.set_speed(args.turntable_speed)

            for light_value in args.light_intensities:
                await light.set_intensity(light_value)
                if hasattr(turntable, "goto"):
                    turntable.goto(0)

                for view_index in range(args.views):
                    if view_index > 0:
                        turntable.rotate(args.view_step)
                    angle = (view_index * args.view_step) % 360

                    for aperture in args.apertures:
                        for iso in args.isos:
                            for shutter in args.shutters:
                                sequence += 1
                                output_path = build_output_path(
                                    output_dir=output_dir,
                                    session_id=session_id,
                                    class_slug=class_slug,
                                    sequence=sequence,
                                    light=light_value,
                                    view_index=view_index,
                                    angle=angle,
                                    aperture=aperture,
                                    iso=iso,
                                    shutter=shutter,
                                )
                                print(
                                    f"[{sequence}/{total_captures}] "
                                    f"light {light_value}/1000, view {view_index} "
                                    f"({angle} deg), F{aperture:g}, ISO {iso}, {shutter}"
                                )
                                size = camera.capture(
                                    aperture=aperture,
                                    iso=iso,
                                    shutter=shutter,
                                    output_path=output_path,
                                    timeout=args.capture_timeout,
                                )
                                append_manifest_row(
                                    manifest_path,
                                    {
                                        "session_id": session_id,
                                        "sequence": sequence,
                                        "class_name": class_name,
                                        "file_name": output_path.name,
                                        "image_path": str(output_path),
                                        "light_intensity": light_value,
                                        "light_percent": light_percent(light_value),
                                        "cct": args.cct,
                                        "view_index": view_index,
                                        "angle_degrees": angle,
                                        "aperture": aperture,
                                        "iso": iso,
                                        "shutter_speed": shutter,
                                        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                    },
                                )
                                print(f"Saved {output_path.name} ({size:,} bytes)")

                if args.views > 1:
                    turntable.rotate(args.view_step)

    print("Capture session complete.")


def main() -> None:
    asyncio.run(run_capture(parse_args()))


if __name__ == "__main__":
    main()
