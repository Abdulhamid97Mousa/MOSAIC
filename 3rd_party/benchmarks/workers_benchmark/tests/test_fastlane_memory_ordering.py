"""Test 4: Cross-Core Memory Ordering Stress Test.

Pins writer and reader to separate CPU cores to maximize cache coherency
traffic. The writer alternates between all-zeros and all-255 frames.
A memory ordering bug would produce frames with mixed 0/255 values.

On x86 (TSO) this should always pass.
On ARM (relaxed ordering) it would catch missing memory barriers.

Run: pytest tests/test_fastlane_memory_ordering.py -v
"""

import multiprocessing as mp
import os
import time

import numpy as np
import pytest

from gym_gui.fastlane.buffer import (
    FastLaneConfig,
    FastLaneMetrics,
    FastLaneWriter,
    FastLaneReader,
)


WIDTH, HEIGHT, CHANNELS = 64, 64, 3
N_FRAMES = 500_000
RUN_ID = "test_memory_order"


def _try_set_affinity(core):
    """Try to pin process to a specific CPU core. Skip if not supported."""
    try:
        os.sched_setaffinity(0, {core})
        return True
    except (AttributeError, OSError):
        return False


def _stress_writer(run_id, n_frames, ready_event):
    """Alternate between all-0 and all-255 frames at maximum speed."""
    _try_set_affinity(0)

    config = FastLaneConfig(width=WIDTH, height=HEIGHT, channels=CHANNELS, capacity=64)
    writer = FastLaneWriter.create(run_id, config)
    ready_event.set()

    frame_zero = bytes(np.zeros((HEIGHT, WIDTH, CHANNELS), dtype=np.uint8))
    frame_full = bytes(np.full((HEIGHT, WIDTH, CHANNELS), 255, dtype=np.uint8))
    metrics = FastLaneMetrics(last_reward=0.0, rolling_return=0.0, step_rate_hz=0.0)

    for i in range(n_frames):
        frame = frame_zero if i % 2 == 0 else frame_full
        writer.publish(frame, metrics=metrics)

    time.sleep(0.5)
    writer.close()


def _stress_reader(run_id, n_target, ready_event, errors, frames_read):
    """Read frames and check they are internally consistent (all same value)."""
    _try_set_affinity(1)
    ready_event.wait(timeout=10)
    time.sleep(0.05)

    reader = FastLaneReader.attach(run_id)
    read_count = 0
    frame_size = WIDTH * HEIGHT * CHANNELS

    deadline = time.time() + 120
    while read_count < n_target and time.time() < deadline:
        frame = reader.latest_frame()
        if frame is None:
            continue

        # Check first 100 bytes for consistency (fast check)
        sample = frame.data[:min(100, frame_size)]
        if len(sample) > 0:
            vals = set(sample)
            # Should be either all 0 or all 255, never mixed
            if len(vals) > 1 and vals != {0} and vals != {255}:
                # Could be a legitimate 0/255 mix only if torn read
                if 0 in vals and 255 in vals:
                    errors.value += 1

        read_count += 1

    frames_read.value = read_count
    reader.close()


def test_memory_ordering_cross_core():
    """No inconsistent reads under cross-core stress."""
    errors = mp.Value("i", 0)
    frames_read = mp.Value("i", 0)
    ready = mp.Event()

    target = min(N_FRAMES // 2, 200_000)

    w = mp.Process(target=_stress_writer, args=(RUN_ID, N_FRAMES, ready))
    r = mp.Process(target=_stress_reader, args=(RUN_ID, target, ready, errors, frames_read))

    r.start()
    w.start()
    w.join(timeout=180)
    r.join(timeout=180)

    affinity_worked = True
    try:
        os.sched_setaffinity(0, {0})
        os.sched_setaffinity(0, set(range(mp.cpu_count())))
    except (AttributeError, OSError):
        affinity_worked = False

    print(f"\n  CPU affinity pinning: {'yes' if affinity_worked else 'no (skipped)'}")
    print(f"  Frames written: {N_FRAMES}")
    print(f"  Frames read:    {frames_read.value}")
    print(f"  Ordering errors: {errors.value}")

    assert errors.value == 0, (
        f"Memory ordering bug: {errors.value} inconsistent reads detected "
        f"out of {frames_read.value} frames. "
        "The sequence-number protocol may need explicit memory barriers."
    )
