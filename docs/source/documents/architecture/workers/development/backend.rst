Backend: Worker Package
=======================

This guide walks through creating the backend package for a new
MOSAIC worker under ``3rd_party/workers/``.  By the end you will have a
fully functional worker that the Daemon can spawn and that emits
standardized JSONL telemetry.

Prerequisites
-------------

- Python 3.10+
- MOSAIC installed (``pip install -e .``)
- Your RL library or agent framework installed

Overview
--------

Every worker consists of five components:

.. mermaid::

   graph LR
       A["1. Config<br/>config.py"] --> B["2. Runtime<br/>runtime.py"]
       B --> C["3. Telemetry<br/>telemetry.py"]
       B --> D["4. Analytics<br/>analytics.py"]
       E["5. Entry Point<br/>pyproject.toml"] -.->|"discovery"| A

       style A fill:#4a90d9,stroke:#2e5a87,color:#fff
       style B fill:#ff7f50,stroke:#cc5500,color:#fff
       style C fill:#50c878,stroke:#2e8b57,color:#fff
       style D fill:#50c878,stroke:#2e8b57,color:#fff
       style E fill:#9370db,stroke:#6a0dad,color:#fff

Step 1: Create the Package
--------------------------

.. code-block:: bash

   mkdir -p 3rd_party/workers/my_worker/my_worker
   touch 3rd_party/workers/my_worker/my_worker/__init__.py

Step 2: Configuration (``config.py``)
--------------------------------------

Implement the ``WorkerConfig`` protocol, A dataclass with ``run_id``,
``seed``, ``to_dict()``, and ``from_dict()``:

.. code-block:: python

   from __future__ import annotations
   from dataclasses import dataclass
   from typing import Any, Dict

   @dataclass
   class MyWorkerConfig:
       # Protocol-required fields
       run_id: str
       seed: int | None = None

       # Worker-specific fields
       env_id: str = "CartPole-v1"
       algorithm: str = "dqn"
       total_steps: int = 100_000
       learning_rate: float = 1e-4

       def __post_init__(self) -> None:
           if not self.run_id:
               raise ValueError("run_id is required")

       def to_dict(self) -> Dict[str, Any]:
           return {
               "run_id": self.run_id,
               "seed": self.seed,
               "env_id": self.env_id,
               "algorithm": self.algorithm,
               "total_steps": self.total_steps,
               "learning_rate": self.learning_rate,
           }

       @classmethod
       def from_dict(cls, data: Dict[str, Any]) -> "MyWorkerConfig":
           fields = cls.__dataclass_fields__
           return cls(**{k: v for k, v in data.items() if k in fields})

Step 3: Runtime (``runtime.py``)
---------------------------------

Manage the worker lifecycle, Emit ``run_started``, run the training
loop, and emit ``run_completed`` or ``run_failed``:

.. code-block:: python

   import json
   import sys
   import time
   from typing import Any, Dict

   class MyWorkerRuntime:
       def __init__(self, config: MyWorkerConfig):
           self.config = config

       def run(self) -> Dict[str, Any]:
           self._emit_lifecycle("run_started", {
               "env_id": self.config.env_id,
               "algorithm": self.config.algorithm,
           })

           try:
               result = self._train()
               summary = {"status": "completed", **result}
               self._emit_lifecycle("run_completed", summary)
               return summary
           except Exception as e:
               self._emit_lifecycle("run_failed", {"error": str(e)})
               raise

       def _train(self) -> Dict[str, Any]:
           """Your training logic goes here."""
           for step in range(self.config.total_steps):
               # ... train one step ...

               # Emit step telemetry every N steps
               if step % 100 == 0:
                   self._emit_step(step, reward=1.0)

               # Emit heartbeat every 60s
               if step % 10_000 == 0:
                   self._emit_lifecycle("heartbeat", {"step": step})

           return {"total_steps": self.config.total_steps}

       def _emit_step(self, step: int, reward: float):
           event = {
               "event_type": "step",
               "run_id": self.config.run_id,
               "step_index": step,
               "reward": reward,
           }
           print(json.dumps(event), file=sys.stdout, flush=True)

       def _emit_lifecycle(self, event_name: str, payload: dict):
           event = {
               "event": event_name,
               "run_id": self.config.run_id,
               "timestamp": time.time(),
               "payload": payload,
           }
           print(json.dumps(event), file=sys.stdout, flush=True)

.. important::

   Always use ``flush=True`` when printing telemetry.  The Telemetry
   Proxy reads ``stdout`` line-by-line, buffered output causes
   delayed or missing telemetry.

Step 4: Worker Metadata (``__init__.py``)
------------------------------------------

Register the worker's identity and capabilities for automatic
discovery:

.. code-block:: python

   def get_worker_metadata() -> tuple:
       from gym_gui.core.worker import WorkerMetadata, WorkerCapabilities

       metadata = WorkerMetadata(
           name="My Worker",
           version="1.0.0",
           description="My RL worker for MOSAIC",
           author="Your Name",
           homepage="https://github.com/...",
           upstream_library="mylib",
           upstream_version="1.0.0",
           license="MIT",
       )

       capabilities = WorkerCapabilities(
           worker_type="myworker",
           supported_paradigms=("sequential",),
           env_families=("gymnasium",),
           action_spaces=("discrete", "continuous"),
           observation_spaces=("vector", "image"),
           max_agents=1,
           supports_checkpointing=True,
           requires_gpu=False,
       )

       return metadata, capabilities

Step 5: Entry Point (``pyproject.toml``)
-----------------------------------------

Register the worker so MOSAIC discovers it automatically:

.. code-block:: toml

   [project]
   name = "my-worker"
   version = "1.0.0"
   requires-python = ">=3.10"

   [project.entry-points."mosaic.workers"]
   myworker = "my_worker:get_worker_metadata"

   [build-system]
   requires = ["setuptools>=64"]
   build-backend = "setuptools.backends._legacy:_Backend"

Then install in development mode:

.. code-block:: bash

   cd 3rd_party/workers/my_worker
   pip install -e .

Step 6: CLI Entry Point (``cli.py``)
--------------------------------------

Create the command-line interface that the Daemon invokes:

.. code-block:: python

   import argparse
   import json
   from my_worker.config import MyWorkerConfig
   from my_worker.runtime import MyWorkerRuntime

   def main(argv=None):
       parser = argparse.ArgumentParser()
       parser.add_argument("--config", required=True)
       parser.add_argument("--grpc", action="store_true")
       parser.add_argument("--grpc-target", default="127.0.0.1:50055")
       args = parser.parse_args(argv)

       # Load config from JSON file
       with open(args.config) as f:
           config = MyWorkerConfig.from_dict(json.load(f))

       # Run the worker
       runtime = MyWorkerRuntime(config)
       runtime.run()

   if __name__ == "__main__":
       main()

Step 7: Add Requirements
-------------------------

Create ``requirements/myworker.txt`` in the MOSAIC root:

.. code-block:: text

   # My Worker dependencies
   mylib>=1.0.0
   -e 3rd_party/workers/my_worker

And add to MOSAIC's ``pyproject.toml`` optional dependencies:

.. code-block:: toml

   [project.optional-dependencies]
   myworker = ["-r requirements/myworker.txt"]

Step 8: FastLane Integration (Optional)
---------------------------------------

FastLane is a high-performance shared-memory telemetry path for real-time
rendering. When enabled, frames bypass the gRPC/SQLite slow lane and go
directly to the GUI via ``/dev/shm`` shared memory segments.

.. important::

   **Always call ``unlink()`` after ``close()``** when closing FastLane
   writers. Failure to unlink shared memory segments will leave orphaned
   ``resource_tracker`` processes that consume CPU and memory even after
   the GUI closes.

Architecture
~~~~~~~~~~~~

.. mermaid::

   graph LR
       subgraph "Worker Process"
           ENV["Gymnasium Env"]
           W["FastLaneWrapper"]
           WR["FastLaneWriter"]
       end

       subgraph "Shared Memory (/dev/shm)"
           SHM["FastLane Buffer<br/>psm_*"]
       end

       subgraph "GUI Process"
           RD["FastLaneReader"]
           CON["FastLaneConsumer<br/>16ms poll"]
           TAB["FastLaneTab"]
       end

       ENV -->|"render()"| W
       W -->|"publish()"| WR
       WR -->|"mmap write"| SHM
       SHM -->|"mmap read"| RD
       RD -->|"QImage"| CON
       CON -->|"frame_ready"| TAB

       style SHM fill:#ffd700,stroke:#b8860b,color:#000

Implementation
~~~~~~~~~~~~~~

Create a ``fastlane.py`` module in your worker package:

.. code-block:: python

   from __future__ import annotations

   import os
   from dataclasses import dataclass
   from multiprocessing import shared_memory
   from typing import Any, Optional

   import gymnasium as gym
   import numpy as np

   try:
       from gym_gui.fastlane import FastLaneWriter, FastLaneConfig, FastLaneMetrics
       from gym_gui.fastlane.buffer import create_fastlane_name
       from gym_gui.telemetry.semconv import TelemetryEnv, VideoModes
   except ImportError:
       FastLaneWriter = None  # type: ignore
       FastLaneConfig = None  # type: ignore
       FastLaneMetrics = None  # type: ignore
       create_fastlane_name = None  # type: ignore

   @dataclass(frozen=True)
   class FastLaneConfig:
       enabled: bool
       slot: int
       run_id: str
       video_mode: str
       grid_limit: int

   def _resolve_config() -> FastLaneConfig:
       enabled = os.getenv(TelemetryEnv.FASTLANE_ONLY, "0") == "1"
       slot = int(os.getenv(TelemetryEnv.FASTLANE_SLOT, "0"))
       run_id = os.getenv("MY_WORKER_RUN_ID", "my-worker-run")
       video_mode = os.getenv(TelemetryEnv.FASTLANE_VIDEO_MODE, VideoModes.SINGLE)
       grid_limit = int(os.getenv(TelemetryEnv.FASTLANE_GRID_LIMIT, "4"))
       return FastLaneConfig(enabled, slot, run_id, video_mode, grid_limit)

   _CONFIG = _resolve_config()

   def is_fastlane_enabled() -> bool:
       return _CONFIG.enabled

   class FastLaneTelemetryWrapper(gym.Wrapper):
       """Gym wrapper that publishes frames to FastLane shared memory."""

       def __init__(self, env: gym.Env, config: FastLaneConfig) -> None:
           super().__init__(env)
           self._config = config
           self._writer: Optional[FastLaneWriter] = None

       def step(self, action: Any):
           obs, reward, terminated, truncated, info = self.env.step(action)

           if self._config.enabled:
               self._publish_frame(reward)

           return obs, reward, terminated, truncated, info

       def _publish_frame(self, reward: float) -> None:
           if FastLaneWriter is None:
               return

           frame = self.env.render()
           if frame is None:
               return

           arr = np.ascontiguousarray(frame.astype(np.uint8))
           height, width, channels = arr.shape

           writer = self._writer
           if writer is None:
               config = FastLaneConfig(
                   width=width, height=height, channels=channels,
                   pixel_format="RGB" if channels == 3 else "RGBA",
               )
               writer = self._create_writer(config)
               if writer is None:
                   return
               self._writer = writer

           metrics = FastLaneMetrics(
               last_reward=float(reward),
               rolling_return=0.0,
               step_rate_hz=0.0,
           )
           writer.publish(arr.tobytes(), metrics=metrics)

       def _create_writer(self, config: Any) -> Optional[FastLaneWriter]:
           try:
               return FastLaneWriter.create(self._config.run_id, config)
           except FileExistsError:
               try:
                   name = create_fastlane_name(self._config.run_id)
                   shm = shared_memory.SharedMemory(name=name, create=False)
                   return FastLaneWriter(shm, config)
               except Exception:
                   return None

       def close(self) -> None:
           self._close_writer()
           return self.env.close()

       def _close_writer(self) -> None:
           """Close FastLane writer and unlink shared memory."""
           if self._writer is not None:
               try:
                   self._writer.close()
                   self._writer.unlink()  # CRITICAL: Remove shared memory
               except FileNotFoundError:
                   pass  # Already unlinked
               except Exception as exc:
                   pass  # Log if needed
               finally:
                   self._writer = None

   def maybe_wrap_env(env: gym.Env) -> gym.Env:
       """Wrap environment with FastLane if enabled."""
       if not _CONFIG.enabled:
           return env
       return FastLaneTelemetryWrapper(env, _CONFIG)

Cleanup Requirements
~~~~~~~~~~~~~~~~~~~~

.. warning::

   The ``_close_writer()`` method **MUST** call both ``close()`` and
   ``unlink()``. The sequence is:

   1. ``writer.close()`` - releases the memory mapping
   2. ``writer.unlink()`` - removes the shared memory segment from ``/dev/shm``

   Without ``unlink()``, the shared memory segment persists and the
   ``multiprocessing.resource_tracker`` process keeps running to track it.
   This leads to:

   - Accumulating ``resource_tracker`` processes consuming CPU
   - Memory leaks in ``/dev/shm``
   - New worker spawns creating additional trackers

Environment Variables
~~~~~~~~~~~~~~~~~~~~~

The dispatcher sets these environment variables when FastLane is enabled:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Variable
     - Description
   * - ``GYM_GUI_FASTLANE_ONLY``
     - Set to ``"1"`` to enable FastLane mode
   * - ``GYM_GUI_FASTLANE_SLOT``
     - Vectorized env index to stream (default: ``"0"``)
   * - ``GYM_GUI_FASTLANE_VIDEO_MODE``
     - ``"single"``, ``"grid"``, or ``"off"``
   * - ``GYM_GUI_FASTLANE_GRID_LIMIT``
     - Max env slots for grid mode (default: ``"4"``)

Debugging Orphaned Processes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you see ``resource_tracker`` processes after closing the GUI:

.. code-block:: bash

   # List orphaned processes
   ps aux | grep resource_tracker

   # Kill all resource_tracker processes
   pkill -f "resource_tracker"

   # Clean up orphaned shared memory
   rm -f /dev/shm/psm_*

Testing
-------

Verify your worker passes the MOSAIC standardization tests:

.. code-block:: python

   # tests/test_my_worker_standardization.py

   def test_config_protocol():
       """Config implements WorkerConfig protocol."""
       from my_worker.config import MyWorkerConfig
       config = MyWorkerConfig(run_id="test-001", seed=42)
       assert config.run_id == "test-001"
       d = config.to_dict()
       restored = MyWorkerConfig.from_dict(d)
       assert restored.run_id == config.run_id

   def test_metadata():
       """Worker provides valid metadata."""
       from my_worker import get_worker_metadata
       metadata, capabilities = get_worker_metadata()
       assert metadata.name
       assert capabilities.worker_type

   def test_lifecycle_events(capsys):
       """Worker emits required lifecycle events."""
       from my_worker.config import MyWorkerConfig
       from my_worker.runtime import MyWorkerRuntime
       import json

       config = MyWorkerConfig(run_id="test-001", total_steps=10)
       runtime = MyWorkerRuntime(config)
       runtime.run()

       output = capsys.readouterr().out
       lines = [json.loads(l) for l in output.strip().split("\\n")]
       events = [l.get("event") for l in lines if "event" in l]
       assert "run_started" in events
       assert "run_completed" in events

