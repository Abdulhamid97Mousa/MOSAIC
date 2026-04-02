import copy
import json
import unittest

from gym_gui.config.paths import VAR_TENSORBOARD_DIR
from gym_gui.services.trainer import validate_train_run_config
from gym_gui.validations.validations_pydantic import validate_telemetry_event


def _sample_train_config() -> dict:
    return {
        "run_name": "frozenlake-v2-q-learning-20251101-151440",
        "entry_point": "python",
        "arguments": ["-m", "cleanrl_worker.cli"],
        "environment": {},
        "resources": {
            "cpus": 2,
            "memory_mb": 2048,
            "gpus": {"requested": 0, "mandatory": False},
        },
        "artifacts": {
            "output_prefix": "runs/frozenlake-v2-q-learning-20251101-151440",
            "persist_logs": True,
            "keep_checkpoints": False,
        },
        "metadata": {
            "artifacts": {
                "tensorboard": {
                    "enabled": True,
                    "relative_path": "var/trainer/runs/frozenlake-v2-q-learning-20251101-151440/tensorboard",
                }
            },
            "worker": {
                "config": {
                    "run_id": "placeholder-run-id",
                }
            },
        },
    }


class TrainerTensorboardPathTests(unittest.TestCase):
    def test_run_id_rewrites_tensorboard_paths(self) -> None:
        raw = _sample_train_config()
        validated = validate_train_run_config(raw)

        run_id = validated.metadata.run_id
        self.assertEqual(len(run_id), 26)

        metadata = validated.payload["metadata"]
        artifacts = metadata["artifacts"]
        tensorboard_meta = artifacts["tensorboard"]

        # relative_path is now just the dirname (e.g., "tensorboard")
        # The GUI resolves it relative to VAR_TRAINER_DIR/runs/<run_id>/
        expected_relative = "tensorboard"
        expected_absolute = (VAR_TENSORBOARD_DIR / run_id / "tensorboard").resolve()

        self.assertEqual(tensorboard_meta["relative_path"], expected_relative)
        self.assertEqual(tensorboard_meta["log_dir"], str(expected_absolute))

        worker_config = metadata["worker"]["config"]
        self.assertEqual(worker_config["run_id"], run_id)

    def test_tensorboard_section_created_when_missing(self) -> None:
        raw = _sample_train_config()
        raw_no_tensorboard = copy.deepcopy(raw)
        raw_no_tensorboard["metadata"].pop("artifacts")

        validated = validate_train_run_config(raw_no_tensorboard)
        run_id = validated.metadata.run_id

        metadata = validated.payload.get("metadata")
        self.assertIsInstance(metadata, dict)
        assert metadata is not None
        artifacts = metadata.get("artifacts")
        self.assertIsInstance(artifacts, dict)
        tensorboard_meta = artifacts.get("tensorboard")
        self.assertIsInstance(tensorboard_meta, dict)
        # relative_path is now just the dirname (e.g., "tensorboard")
        # The GUI resolves it relative to VAR_TRAINER_DIR/runs/<run_id>/
        expected_relative = "tensorboard"
        expected_absolute = (VAR_TENSORBOARD_DIR / run_id / "tensorboard").resolve()
        self.assertEqual(tensorboard_meta["relative_path"], expected_relative)
        self.assertEqual(tensorboard_meta["log_dir"], str(expected_absolute))

    def test_run_id_preserved_on_resubmission(self) -> None:
        raw = _sample_train_config()
        first_pass = validate_train_run_config(raw)
        payload_roundtrip = json.loads(first_pass.to_json())

        second_pass = validate_train_run_config(payload_roundtrip)

        self.assertEqual(second_pass.metadata.run_id, first_pass.metadata.run_id)


class TelemetryValidationTensorboardTests(unittest.TestCase):
    def test_tensorboard_artifact_event_allowed(self) -> None:
        event = validate_telemetry_event(
            {
                "type": "artifact",
                "run_id": "01TESTRUNID0000000000000000",
                "kind": "tensorboard",
                "path": "/tmp/tensorboard",
            }
        )

        self.assertEqual(event.kind, "tensorboard")  # type: ignore[attr-defined]


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
