from __future__ import annotations

import asyncio
import copy
import json
import tempfile
import unittest
from pathlib import Path

import replicate_capture as capture


REPO_ROOT = Path(__file__).resolve().parents[1]
LABELS = REPO_ROOT / "data_replicate/manual_dataset/labels.csv"
FIRST_SAMPLE_ID = "ILSVRC2012_val_00038504"


class CapturePlanTests(unittest.TestCase):
    def test_default_light_secret_matches_direct_run_configuration(self) -> None:
        args = capture.parse_args([])
        self.assertEqual(args.light_api_secret_key, capture.DEFAULT_LIGHT_API_SECRET_KEY)
        self.assertEqual(args.capture_timeout, 180.0)
        self.assertEqual(args.pre_capture_clear_timeout, 5.0)
        self.assertEqual(args.start_delay_seconds, 10.0)
        self.assertEqual(capture.DEFAULT_AE_SHOTS, 3)

    def test_paper_parameter_order_and_full_count(self) -> None:
        samples = capture.load_samples(LABELS)
        parameters = capture.build_manual_parameters()
        tasks = capture.build_capture_tasks(
            samples, capture.DEFAULT_LIGHT_INTENSITIES, capture.DEFAULT_ZOOM_COUNT
        )

        self.assertEqual(len(samples), 15)
        self.assertEqual(len(parameters), 27)
        self.assertEqual(
            parameters[0],
            capture.ManualParameter(1, aperture=5.0, shutter_speed="1/4", iso=250),
        )
        self.assertEqual(
            parameters[-1],
            capture.ManualParameter(27, aperture=16.0, shutter_speed="1/1000", iso=16000),
        )
        self.assertEqual(len(tasks), 4500)
        self.assertEqual([task.plan_index for task in tasks], list(range(1, 4501)))
        self.assertEqual(len({task.capture_key for task in tasks}), 4500)
        self.assertEqual(len({task.relative_path() for task in tasks}), 4500)
        self.assertEqual(tasks[0].zoom_id, "z001")
        self.assertEqual(tasks[150].zoom_id, "z002")
        self.assertEqual(
            tasks[0].relative_path(),
            Path("ILSVRC2012_val_00038504/z001/b010/ae/ae_01.jpg"),
        )

    def test_default_brightness_mapping(self) -> None:
        self.assertEqual(
            [capture.light_percent(value) for value in capture.DEFAULT_LIGHT_INTENSITIES],
            [1.0, 20.0, 50.0, 70.0, 100.0],
        )
        self.assertEqual(capture.light_slug(1000), "b1000")

    def test_two_zooms_have_separate_keys_and_paths(self) -> None:
        sample = capture.load_samples(LABELS)[:1]
        tasks = capture.build_capture_tasks(sample, (0,), 2)
        self.assertEqual(len(tasks), 60)
        self.assertEqual(tasks[0].zoom_id, "z001")
        self.assertEqual(tasks[30].zoom_id, "z002")
        self.assertNotEqual(tasks[0].capture_key, tasks[30].capture_key)
        self.assertNotEqual(tasks[0].relative_path(), tasks[30].relative_path())
        self.assertEqual(
            tasks[30].relative_path(),
            Path("ILSVRC2012_val_00038504/z002/b000/ae/ae_01.jpg"),
        )

    def test_plan_preserves_focus_and_metering_settings(self) -> None:
        samples = capture.load_samples(LABELS)
        plan = capture.build_plan_manifest(LABELS, samples, (0,), 5600, 1)
        configuration = plan["capture_configuration"]
        self.assertIn("preserved", configuration["focus_mode"])
        self.assertIn("preserved", configuration["metering_mode"])
        self.assertEqual(
            configuration["exposure_mode_control"],
            "program auto for AE shots; manual for p001-p027",
        )

    def test_partial_five_ae_plan_can_migrate_without_losing_ae01(self) -> None:
        samples = capture.load_samples(LABELS)
        tasks = capture.build_capture_tasks(samples, (0,), 1)
        expected = capture.build_plan_manifest(LABELS, samples, (0,), 5600, 1)
        old_plan = copy.deepcopy(expected)
        old_plan["schema_version"] = 3
        old_plan["capture_configuration"]["auto_exposure_shots_per_light"] = 5
        old_plan["expected_counts"].update(
            {
                "auto_per_light": 5,
                "images_per_light": 32,
                "images_per_zoom": 32,
                "total_images": 480,
            }
        )

        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            capture.atomic_write_json(output_dir / "plan.json", old_plan)
            first_task = tasks[0]
            image_path = output_dir / first_task.relative_path()
            image_path.parent.mkdir(parents=True)
            image_path.write_bytes(capture.DRY_RUN_JPEG)
            capture.append_jsonl(
                output_dir / "captures.jsonl",
                {"capture_key": first_task.capture_key},
            )

            capture.ensure_plan_file(output_dir, expected, tasks)

            migrated = json.loads((output_dir / "plan.json").read_text())
            self.assertEqual(migrated["schema_version"], 4)
            self.assertEqual(
                migrated["capture_configuration"]["auto_exposure_shots_per_light"], 3
            )
            self.assertTrue(image_path.exists())


class DryRunResumeTests(unittest.TestCase):
    def run_dry(self, output_dir: Path) -> None:
        args = capture.parse_args(
            [
                "--labels",
                str(LABELS),
                "--output-dir",
                str(output_dir),
                "--sample-id",
                FIRST_SAMPLE_ID,
                "--light-intensities",
                "0",
                "--zoom-count",
                "1",
                "--dry-run",
                "--yes",
            ]
        )
        asyncio.run(capture.run_capture(args))

    @staticmethod
    def records(output_dir: Path) -> list[dict[str, object]]:
        return [
            json.loads(line)
            for line in (output_dir / "captures.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        ]

    def test_dry_run_resume_and_corrupt_file_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "capture"
            self.run_dry(output_dir)
            first_records = self.records(output_dir)
            self.assertEqual(len(first_records), 30)

            # A no-op resume neither recaptures nor duplicates metadata.
            self.run_dry(output_dir)
            self.assertEqual(len(self.records(output_dir)), 30)

            # A missing JPEG is recaptured and produces a new attempt record.
            missing_path = output_dir / str(first_records[0]["image_path"])
            missing_path.unlink()
            self.run_dry(output_dir)
            self.assertTrue(capture.is_valid_jpeg(missing_path))
            self.assertEqual(len(self.records(output_dir)), 31)

            # A corrupt JPEG is retained under an .invalid name, then replaced.
            corrupt_path = output_dir / str(first_records[1]["image_path"])
            corrupt_path.write_bytes(b"not a jpeg")
            self.run_dry(output_dir)
            self.assertTrue(capture.is_valid_jpeg(corrupt_path))
            self.assertEqual(len(self.records(output_dir)), 32)
            backups = list(corrupt_path.parent.glob(f"{corrupt_path.name}.invalid.*"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_bytes(), b"not a jpeg")

    def test_valid_unindexed_file_gets_recovery_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "capture"
            samples = capture.load_samples(LABELS)
            task = capture.build_capture_tasks(samples, (0,), 1)[0]
            image_path = output_dir / task.relative_path()
            image_path.parent.mkdir(parents=True)
            image_path.write_bytes(capture.DRY_RUN_JPEG)

            self.run_dry(output_dir)
            records = self.records(output_dir)
            recovered = [item for item in records if item["capture_key"] == task.capture_key]
            self.assertEqual(len(recovered), 1)
            self.assertEqual(recovered[0]["capture_status"], "recovered_existing_file")
            self.assertEqual(len(records), 30)

    def test_capture_failure_stops_and_writes_error_log(self) -> None:
        class FailingCamera:
            def capture_auto(self, **_: object) -> int:
                raise RuntimeError("simulated camera disconnect")

        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "capture"
            task = capture.build_capture_tasks(capture.load_samples(LABELS), (0,), 1)[0]
            args = capture.parse_args(
                [
                    "--labels",
                    str(LABELS),
                    "--output-dir",
                    str(output_dir),
                    "--dry-run",
                    "--yes",
                ]
            )
            captures_path = output_dir / "captures.jsonl"
            errors_path = output_dir / "errors.jsonl"

            with self.assertRaisesRegex(RuntimeError, "stopped for safe resume"):
                asyncio.run(
                    capture.acquire_task(
                        task=task,
                        args=args,
                        camera=FailingCamera(),
                        output_dir=output_dir,
                        captures_path=captures_path,
                        errors_path=errors_path,
                        run_session_id="test-session",
                        session_sequence=1,
                        requested_trigger="normal",
                        attempt=1,
                    )
                )

            self.assertFalse(captures_path.exists())
            errors = [json.loads(line) for line in errors_path.read_text().splitlines()]
            self.assertEqual(len(errors), 1)
            self.assertEqual(errors[0]["capture_key"], task.capture_key)
            self.assertEqual(errors[0]["error"], "simulated camera disconnect")

    def test_capture_timeout_is_recorded_and_returns_none(self) -> None:
        class TimingOutCamera:
            def capture_auto(self, **_: object) -> int:
                raise RuntimeError("Capture timed out waiting for image")

        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "capture"
            task = capture.build_capture_tasks(capture.load_samples(LABELS), (0,), 1)[0]
            args = capture.parse_args(
                [
                    "--labels",
                    str(LABELS),
                    "--output-dir",
                    str(output_dir),
                    "--dry-run",
                    "--yes",
                ]
            )
            result = asyncio.run(
                capture.acquire_task(
                    task=task,
                    args=args,
                    camera=TimingOutCamera(),
                    output_dir=output_dir,
                    captures_path=output_dir / "captures.jsonl",
                    errors_path=output_dir / "errors.jsonl",
                    run_session_id="test-session",
                    session_sequence=1,
                    requested_trigger="normal",
                    attempt=1,
                )
            )

            self.assertIsNone(result)
            self.assertFalse((output_dir / "captures.jsonl").exists())
            errors = [
                json.loads(line)
                for line in (output_dir / "errors.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(errors), 1)
            self.assertEqual(errors[0]["event"], "capture_failed")
            self.assertEqual(errors[0]["error"], "Capture timed out waiting for image")
            self.assertEqual(errors[0]["sample_id"], FIRST_SAMPLE_ID)
            self.assertEqual(errors[0]["light_intensity"], 0)
            self.assertEqual(errors[0]["exposure_mode"], "auto")


if __name__ == "__main__":
    unittest.main()
