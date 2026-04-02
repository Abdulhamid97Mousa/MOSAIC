"""Integration-flavoured tests for CleanRL trainer plumbing.

These tests focus on two critical pieces of the CleanRL integration:

1. The telemetry proxy path (`trainer_telemetry_proxy.run_proxy`) which tails JSONL
   output from the worker process, performs the gRPC `RegisterWorker` handshake,
   and streams RunStep/RunEpisode payloads into the trainer service.
2. Analytics manifest ingestion via `AnalyticsTabManager`, ensuring that metadata
   emitted by the CleanRL worker yields the expected TensorBoard and WANDB tabs.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Awaitable, Callable, Iterator, List, Optional

import grpc
import pytest
from PyQt6 import QtWidgets

from gym_gui.services.trainer import trainer_telemetry_proxy
from gym_gui.services.trainer.proto import trainer_pb2, trainer_pb2_grpc
from gym_gui.ui.panels.analytics_tabs import AnalyticsTabManager


class _AsyncByteStream:
    """Minimal async iterator that yields the supplied byte lines."""

    def __init__(self, lines: List[bytes], on_complete: Optional[Callable[[], None]] = None) -> None:
        self._lines = list(lines)
        self._on_complete = on_complete
        self._iter: Optional[Iterator[bytes]] = None

    def __aiter__(self) -> "_AsyncByteStream":
        self._iter = iter(self._lines)
        return self

    async def __anext__(self) -> bytes:
        if self._iter is None:
            self._iter = iter(self._lines)
        try:
            line = next(self._iter)
        except StopIteration:
            if self._on_complete is not None:
                self._on_complete()
            raise StopAsyncIteration
        await asyncio.sleep(0)  # Let the event loop breathe
        return line


class _FakeProcess:
    """Simulated subprocess satisfying the interface used by JsonlTailer."""

    def __init__(self, stdout_lines: List[str], *, stderr_lines: Optional[List[str]] = None, returncode: int = 0) -> None:
        self._target_rc = returncode
        self._done = asyncio.Event()
        self.returncode: Optional[int] = None
        stdout_bytes = [line.encode("utf-8") if not isinstance(line, bytes) else line for line in stdout_lines]
        stderr_bytes = [line.encode("utf-8") if not isinstance(line, bytes) else line for line in (stderr_lines or [])]
        self.stdout = _AsyncByteStream(
            [line if line.endswith(b"\n") else line + b"\n" for line in stdout_bytes],  # type: ignore[arg-type, operator]
            on_complete=self._mark_done,
        )
        self.stderr = _AsyncByteStream(
            [line if line.endswith(b"\n") else line + b"\n" for line in stderr_bytes],  # type: ignore[arg-type, operator]
            on_complete=None,
        )

    def _mark_done(self) -> None:
        if self.returncode is None:
            self.returncode = self._target_rc
            self._done.set()

    async def wait(self) -> int:
        await self._done.wait()
        return int(self.returncode or 0)


class _FakeChannel:
    """Stub gRPC aio channel."""

    def __init__(self) -> None:
        self.closed = False

    async def channel_ready(self) -> None:
        return None

    async def close(self) -> None:
        self.closed = True


class _FakeTrainerStub:
    """Capture RegisterWorker + telemetry payloads for assertions."""

    def __init__(self) -> None:
        self.register_calls: List[trainer_pb2.RegisterWorkerRequest] = []
        self.steps: List[trainer_pb2.RunStep] = []
        self.episodes: List[trainer_pb2.RunEpisode] = []

    async def RegisterWorker(self, request: trainer_pb2.RegisterWorkerRequest) -> trainer_pb2.RegisterWorkerResponse:
        self.register_calls.append(request)
        return trainer_pb2.RegisterWorkerResponse(
            accepted_version="MOSAIC/1.0",
            session_token="fake-session-token",
        )

    async def PublishRunSteps(self, iterator: Awaitable[Any] | Any) -> trainer_pb2.PublishTelemetryResponse:
        async for item in iterator:  # type: ignore[union-attr]
            self.steps.append(item)
        return trainer_pb2.PublishTelemetryResponse(accepted=len(self.steps), dropped=0)

    async def PublishRunEpisodes(self, iterator: Awaitable[Any] | Any) -> trainer_pb2.PublishTelemetryResponse:
        async for item in iterator:  # type: ignore[union-attr]
            self.episodes.append(item)
        return trainer_pb2.PublishTelemetryResponse(accepted=len(self.episodes), dropped=0)


@pytest.fixture(scope="module")
def qt_app() -> QtWidgets.QApplication:
    """Provide a QApplication for widget-oriented tests."""
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app  # type: ignore[return-value]


@pytest.mark.asyncio
async def test_run_proxy_registers_worker_and_streams_events(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure telemetry proxy performs gRPC handshake and forwards step/episode telemetry."""

    run_id = "cleanrl-run-001"
    agent_id = "cleanrl-agent"
    worker_id = "cleanrl-worker"

    stdout_events = [
        json.dumps({"type": "step", "episode": 0, "step_index": 0, "reward": 1.0}),
        json.dumps({"type": "episode", "episode": 0, "steps": 1, "total_reward": 1.0}),
    ]
    fake_process = _FakeProcess(stdout_events, returncode=0)

    async def _fake_spawn(*args: Any, **kwargs: Any) -> _FakeProcess:
        return fake_process

    fake_channel = _FakeChannel()
    fake_stub = _FakeTrainerStub()

    monkeypatch.setattr(trainer_telemetry_proxy, "validated_create_subprocess_exec", _fake_spawn)
    monkeypatch.setattr(grpc.aio, "insecure_channel", lambda target, options=(): fake_channel)
    monkeypatch.setattr(trainer_pb2_grpc, "TrainerServiceStub", lambda channel: fake_stub)

    exit_code = await trainer_telemetry_proxy.run_proxy(
        target="127.0.0.1:50055",
        run_id=run_id,
        agent_id=agent_id,
        worker_id=worker_id,
        worker_argv=["python", "-m", "cleanrl_worker.cli"],
    )

    assert exit_code == 0
    assert fake_channel.closed, "Channel should be closed after proxy shutdown"

    assert len(fake_stub.register_calls) == 1
    request = fake_stub.register_calls[0]
    assert request.run_id == run_id
    assert request.worker_id == worker_id or request.worker_id == f"proxy-{worker_id}"
    assert request.worker_kind == "telemetry_proxy"
    assert request.proto_version == "MOSAIC/1.0"

    assert [step.step_index for step in fake_stub.steps] == [0]
    assert [episode.total_reward for episode in fake_stub.episodes] == [1.0]


class _RecordingRenderTabs:
    """Record dynamic tab insertions without requiring the full RenderTabs widget."""

    def __init__(self) -> None:
        self._agent_tabs: dict[str, dict[str, QtWidgets.QWidget]] = {}
        self.added: list[tuple[str, str]] = []

    def add_dynamic_tab(self, run_id: str, name: str, widget: QtWidgets.QWidget) -> None:
        self._agent_tabs.setdefault(run_id, {})[name] = widget
        self.added.append((run_id, name))


def test_analytics_tab_manager_creates_tensorboard_and_wandb_tabs(qt_app: QtWidgets.QApplication, tmp_path: Any) -> None:
    """Verify manifest-style metadata yields TensorBoard and WANDB tabs."""

    run_id = "cleanrl-run-analytics"
    agent_id = "agent-42"
    tensorboard_dir = tmp_path / "tb"
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "artifacts": {
            "tensorboard": {
                "enabled": True,
                "log_dir": str(tensorboard_dir),
            },
            "wandb": {
                "enabled": True,
                "run_path": "entity/project/runs/demo",
            },
        }
    }

    render_tabs = _RecordingRenderTabs()
    manager = AnalyticsTabManager(render_tabs, parent=QtWidgets.QWidget())

    manager.ensure_tensorboard_tab(run_id, agent_id, metadata)
    manager.ensure_wandb_tab(run_id, agent_id, metadata)

    agent_tabs = render_tabs._agent_tabs.get(run_id, {})
    expected_tb_key = f"TensorBoard-{agent_id}-{run_id[:8]}"
    assert expected_tb_key in agent_tabs
    assert "WANDB-Agent-agent-42" in agent_tabs
