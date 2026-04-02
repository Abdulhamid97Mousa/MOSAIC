"""Video frame streaming service for direct Minecraft-to-MOSAIC integration.

This service receives video frames directly from Minecraft via gRPC,
bypassing the MalmoEnv TCP layer for lower latency.

Architecture:
    Minecraft (VideoHook.java)
        ↓ gRPC StreamFrames
    VideoFrameService (Python, port 50056)
        ↓ RunEventBroadcaster
    MOSAIC Video Tab

The Java side needs modification in VideoHook.java to call this service
instead of envServer.addFrame().
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

import grpc
import numpy as np

from gym_gui.services.trainer.proto import trainer_pb2, trainer_pb2_grpc

if TYPE_CHECKING:
    from gym_gui.services.trainer.service import RunEventBroadcaster

_LOGGER = logging.getLogger("gym_gui.trainer.video_frame_service")


class VideoFrameStore:
    """Thread-safe store for the latest video frames per run."""

    def __init__(self, max_runs: int = 16):
        self._frames: dict[str, trainer_pb2.VideoFrame] = {}
        self._timestamps: dict[str, float] = {}
        self._max_runs = max_runs
        self._lock = asyncio.Lock()

    async def store(self, frame: trainer_pb2.VideoFrame) -> None:
        """Store a frame for its run_id, evicting old entries if needed."""
        async with self._lock:
            # Evict old entries if at capacity
            if len(self._frames) >= self._max_runs and frame.run_id not in self._frames:
                # Find oldest by timestamp
                oldest_run = min(self._timestamps.keys(), key=lambda k: self._timestamps[k])
                del self._frames[oldest_run]
                del self._timestamps[oldest_run]
                _LOGGER.debug("Evicted frame store for run %s", oldest_run)

            self._frames[frame.run_id] = frame
            self._timestamps[frame.run_id] = time.monotonic()

    def get_latest_frame(self) -> trainer_pb2.VideoFrame | None:
        """Get the latest frame from the most recently updated run."""
        if not self._timestamps:
            return None
        # Find run with max timestamp
        latest_run = max(self._timestamps.keys(), key=lambda k: self._timestamps[k])
        return self._frames.get(latest_run)

    def get(self, run_id: str) -> trainer_pb2.VideoFrame | None:
        """Get the latest frame for a run (non-blocking, thread-safe read)."""
        return self._frames.get(run_id)

    def get_as_numpy(self, run_id: str) -> np.ndarray | None:
        """Get the latest frame as a numpy RGB array."""
        frame = self.get(run_id)
        if frame is None or not frame.rgb_data:
            return None
        arr = np.frombuffer(frame.rgb_data, dtype=np.uint8)
        return arr.reshape((frame.height, frame.width, 3))

    def get_metadata(self, run_id: str) -> dict:
        """Get frame metadata for a run."""
        frame = self.get(run_id)
        if frame is None:
            return {}
        return {
            "frame_number": frame.frame_number,
            "timestamp_ms": frame.timestamp_ms,
            "agent_x": frame.agent_x if frame.HasField("agent_x") else None,
            "agent_y": frame.agent_y if frame.HasField("agent_y") else None,
            "agent_z": frame.agent_z if frame.HasField("agent_z") else None,
            "agent_yaw": frame.agent_yaw if frame.HasField("agent_yaw") else None,
            "agent_pitch": frame.agent_pitch if frame.HasField("agent_pitch") else None,
            "life": frame.life if frame.HasField("life") else None,
            "food": frame.food if frame.HasField("food") else None,
            "xp": frame.xp if frame.HasField("xp") else None,
            "reward": frame.reward if frame.HasField("reward") else None,
            "is_alive": frame.is_alive if frame.HasField("is_alive") else None,
        }

    def remove(self, run_id: str) -> None:
        """Remove frames for a run."""
        self._frames.pop(run_id, None)
        self._timestamps.pop(run_id, None)


class VideoFrameServicer(trainer_pb2_grpc.VideoFrameServiceServicer):
    """gRPC service receiving video frames from Minecraft."""

    def __init__(self, store: VideoFrameStore) -> None:
        self._store = store
        self._paused_runs: set[str] = set()
        self._target_fps: dict[str, int] = {}  # 0 = unlimited

    async def SendFrame(
        self,
        request: trainer_pb2.VideoFrame,
        context: grpc.aio.ServicerContext,
    ) -> trainer_pb2.VideoFrameAck:
        """Handle a single frame (unary RPC)."""
        await self._store.store(request)
        _LOGGER.debug(
            "Received frame %d for run %s (%dx%d, %d bytes)",
            request.frame_number,
            request.run_id,
            request.width,
            request.height,
            len(request.rgb_data),
        )
        return trainer_pb2.VideoFrameAck(
            received=True,
            frame_number=request.frame_number,
        )

    async def StreamFrames(
        self,
        request_iterator: AsyncIterator[trainer_pb2.VideoFrame],
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[trainer_pb2.VideoFrameControl]:
        """Handle bidirectional frame streaming.

        Receives frames from Minecraft, sends control messages back.
        """
        current_run_id: str | None = None
        frame_count = 0
        last_control_time = time.monotonic()

        try:
            async for frame in request_iterator:
                current_run_id = frame.run_id
                frame_count += 1

                # Store the frame
                await self._store.store(frame)

                # Periodically log and check for control
                now = time.monotonic()
                if now - last_control_time >= 1.0:
                    _LOGGER.debug(
                        "Streamed %d frames for run %s",
                        frame_count,
                        current_run_id,
                    )
                    last_control_time = now

                    # Send control message if needed
                    if current_run_id in self._paused_runs:
                        yield trainer_pb2.VideoFrameControl(pause=True)
                        continue

                    target_fps = self._target_fps.get(current_run_id, 0)
                    if target_fps > 0:
                        yield trainer_pb2.VideoFrameControl(target_fps=target_fps)

        except Exception as e:
            _LOGGER.error("Stream error for run %s: %s", current_run_id, e)
            raise

        finally:
            _LOGGER.info(
                "Stream closed for run %s after %d frames",
                current_run_id,
                frame_count,
            )
            if current_run_id:
                self._store.remove(current_run_id)

    def pause_run(self, run_id: str) -> None:
        """Request a run to pause frame streaming."""
        self._paused_runs.add(run_id)

    def resume_run(self, run_id: str) -> None:
        """Request a run to resume frame streaming."""
        self._paused_runs.discard(run_id)

    def set_target_fps(self, run_id: str, fps: int) -> None:
        """Set target FPS for a run (0 = unlimited)."""
        self._target_fps[run_id] = fps


# Global singleton for frame access from MOSAIC adapters
_frame_store: VideoFrameStore | None = None


def get_frame_store() -> VideoFrameStore:
    """Get the global frame store singleton."""
    global _frame_store
    if _frame_store is None:
        _frame_store = VideoFrameStore()
    return _frame_store


async def start_video_frame_server(
    listen: str = "127.0.0.1:50056",
    store: VideoFrameStore | None = None,
) -> tuple[grpc.aio.Server, VideoFrameServicer]:
    """Start the video frame gRPC server.

    Args:
        listen: Address to listen on (default: localhost:50056)
        store: Optional frame store (creates new one if None)

    Returns:
        Tuple of (server, servicer) for shutdown control
    """
    if store is None:
        store = get_frame_store()

    servicer = VideoFrameServicer(store)
    server = grpc.aio.server()
    trainer_pb2_grpc.add_VideoFrameServiceServicer_to_server(servicer, server)
    server.add_insecure_port(listen)
    await server.start()

    _LOGGER.info("Video frame server listening on %s", listen)
    return server, servicer
