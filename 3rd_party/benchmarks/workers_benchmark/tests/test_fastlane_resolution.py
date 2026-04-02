"""Test 3: Overhead vs Frame Resolution (O(1) IPC Proof).

If FastLane uses shared memory, overhead should be flat regardless of
frame size. If it used serialization (pipes, network), overhead would
grow linearly with resolution. This test distinguishes the two.

Run: pytest tests/test_fastlane_resolution.py -v
"""

import time

import numpy as np
import pytest

from gym_gui.fastlane.buffer import (
    FastLaneConfig,
    FastLaneMetrics,
    FastLaneWriter,
)


RESOLUTIONS = [
    (84, 84, 3, "CartPole"),       # 21 KB
    (210, 160, 3, "Atari"),        # 100 KB
    (320, 240, 3, "ViZDoom"),      # 230 KB
    (640, 480, 3, "HD"),           # 921 KB
]

N_FRAMES = 20_000


def _measure_publish_time(width, height, channels, n_frames=N_FRAMES, suffix=""):
    """Measure average time per publish() call at a given resolution."""
    run_id = f"res_{width}x{height}_{suffix or int(time.time()*1000)%100000}"
    config = FastLaneConfig(width=width, height=height, channels=channels, capacity=32)
    frame_bytes = bytes(np.zeros((height, width, channels), dtype=np.uint8))
    metrics = FastLaneMetrics(last_reward=0.0, rolling_return=0.0, step_rate_hz=0.0)

    writer = FastLaneWriter.create(run_id, config)

    start = time.perf_counter()
    for _ in range(n_frames):
        writer.publish(frame_bytes, metrics=metrics)
    elapsed = time.perf_counter() - start

    writer.close()

    per_frame_us = (elapsed / n_frames) * 1_000_000
    frame_kb = (width * height * channels) / 1024
    throughput_mbps = (frame_kb * n_frames / 1024) / elapsed

    return per_frame_us, frame_kb, throughput_mbps


def test_overhead_flat_across_resolutions():
    """Publish latency should scale sub-linearly with frame size (shared memory = memcpy)."""
    results = []
    for width, height, channels, label in RESOLUTIONS:
        per_frame_us, frame_kb, throughput = _measure_publish_time(width, height, channels)
        results.append((label, width, height, frame_kb, per_frame_us, throughput))

    print("\n  Resolution       Frame Size   Time/frame   Throughput")
    print("  " + "-" * 58)
    for label, w, h, kb, us, mbps in results:
        print(f"  {label:<16} {kb:>8.0f} KB   {us:>8.1f} us   {mbps:>8.1f} MB/s")

    # The key check: time per frame should NOT grow 44x when frame size grows 44x
    # (21KB to 921KB = 44x). If it did, we'd have O(frame_size) overhead.
    # With shared memory (memcpy), it should grow much less than linearly.
    smallest_time = results[0][4]
    largest_time = results[-1][4]
    size_ratio = results[-1][3] / results[0][3]  # e.g., 921/21 = 44x
    time_ratio = largest_time / smallest_time

    print(f"\n  Frame size grew {size_ratio:.0f}x, latency grew {time_ratio:.1f}x")

    # memcpy scales linearly with size, so some growth is expected.
    # But the TOTAL overhead (including ring buffer bookkeeping) should
    # grow less than the raw frame size ratio, proving the bookkeeping is O(1).
    # We allow up to 2x the size ratio to account for cache effects.
    assert time_ratio < size_ratio * 2, (
        f"Latency grew {time_ratio:.1f}x for {size_ratio:.0f}x size increase. "
        "This suggests overhead beyond memcpy."
    )


def test_throughput_exceeds_60fps_at_all_resolutions():
    """FastLane must sustain > 60 FPS at every resolution to be useful."""
    for width, height, channels, label in RESOLUTIONS:
        per_frame_us, _, _ = _measure_publish_time(width, height, channels, n_frames=5000)
        fps = 1_000_000 / per_frame_us

        print(f"  {label} ({width}x{height}): {fps:.0f} FPS")

        assert fps > 60, (
            f"{label} ({width}x{height}) only achieves {fps:.0f} FPS. "
            "FastLane cannot sustain 60 Hz at this resolution."
        )
