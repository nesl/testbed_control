"""Set manual exposure and save one photo locally and to the camera SD card.

Run from the testbed_control repository root::

    sonycam sony_camera/manual_capture_demo.py

Example with custom settings::

    sonycam sony_camera/manual_capture_demo.py \
        --iso 800 --aperture 5.6 --shutter 1/250 \
        --output sony_camera/captures/test.jpg
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

from pysonycam import ExposureMode, SonyCamera
from pysonycam.constants import (
    DeviceProperty,
    F_NUMBER_TABLE,
    ISO_TABLE,
    SHUTTER_SPEED_TABLE,
    SaveMedia,
)


DEFAULT_ISO = 400
DEFAULT_APERTURE = 4.0
DEFAULT_SHUTTER = "1/200"


def _reverse_table(table: dict[int, str]) -> dict[str, int]:
    return {label.upper(): code for code, label in table.items()}


def _iso_code(value: int) -> int:
    code = _reverse_table(ISO_TABLE).get(str(value))
    if code is None:
        raise argparse.ArgumentTypeError(f"ISO {value} is not supported")
    return code


def _aperture_code(value: float) -> int:
    label = f"F{value:g}".upper()
    code = _reverse_table(F_NUMBER_TABLE).get(label)
    if code is None:
        raise argparse.ArgumentTypeError(f"Aperture F{value:g} is not supported")
    return code


def _shutter_code(value: str) -> int:
    code = _reverse_table(SHUTTER_SPEED_TABLE).get(value.strip().upper())
    if code is None:
        raise argparse.ArgumentTypeError(f"Shutter speed {value!r} is not supported")
    return code


def _wait_for_setting(
    camera: SonyCamera,
    property_code: int,
    expected: int,
    label: str,
    timeout: float = 5.0,
) -> None:
    """Wait until the camera reports that a setting has taken effect."""
    deadline = time.monotonic() + timeout
    actual = None
    while time.monotonic() < deadline:
        actual = camera.get_property(property_code).current_value
        if actual == expected:
            return
        time.sleep(0.1)
    raise RuntimeError(
        f"Camera did not apply {label}: expected 0x{expected:X}, got {actual!r}"
    )


def _restore_output_owner(path: Path, parent_was_created: bool) -> None:
    """Make files created through sudo writable by the invoking user."""
    uid_text = os.environ.get("SUDO_UID")
    gid_text = os.environ.get("SUDO_GID")
    if uid_text is None or gid_text is None:
        return

    uid, gid = int(uid_text), int(gid_text)
    os.chown(path, uid, gid)
    if parent_was_created:
        os.chown(path.parent, uid, gid)


def parse_args() -> argparse.Namespace:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(
        description="Capture one manually exposed photo to host and SD card."
    )
    parser.add_argument("--iso", type=int, default=DEFAULT_ISO)
    parser.add_argument("--aperture", type=float, default=DEFAULT_APERTURE)
    parser.add_argument("--shutter", default=DEFAULT_SHUTTER)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("sony_camera/captures") / f"manual_{timestamp}.jpg",
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    iso_code = _iso_code(args.iso)
    aperture_code = _aperture_code(args.aperture)
    shutter_code = _shutter_code(args.shutter)
    output_path = args.output.expanduser().resolve()

    parent_was_created = not output_path.parent.exists()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Settings: ISO {args.iso}, F{args.aperture:g}, {args.shutter}\n"
        f"Local output: {output_path}\n"
        "Camera output: SD card"
    )

    with SonyCamera() as camera:
        camera.authenticate()
        camera.set_mode("still")

        camera.set_exposure_mode(ExposureMode.MANUAL)
        _wait_for_setting(
            camera,
            DeviceProperty.EXPOSURE_MODE,
            int(ExposureMode.MANUAL),
            "manual exposure mode",
        )

        camera.set_iso(iso_code)
        _wait_for_setting(camera, DeviceProperty.ISO, iso_code, "ISO")

        camera.set_aperture(aperture_code)
        _wait_for_setting(camera, DeviceProperty.F_NUMBER, aperture_code, "aperture")

        camera.set_shutter_speed(shutter_code)
        _wait_for_setting(
            camera,
            DeviceProperty.SHUTTER_SPEED,
            shutter_code,
            "shutter speed",
        )

        image_data = camera.capture(
            output_path=output_path,
            save_to_camera=SaveMedia.HOST_AND_CAMERA,
            timeout=args.timeout,
        )

    _restore_output_owner(output_path, parent_was_created)
    print(f"Capture complete: {len(image_data):,} bytes saved locally and to SD card.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
