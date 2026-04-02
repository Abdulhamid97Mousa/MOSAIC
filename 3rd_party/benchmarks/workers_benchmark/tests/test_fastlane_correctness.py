"""Test 1: Torn Read Detection (Seqlock Correctness Proof).

Writer continuously writes frames where every pixel = frame_number % 256.
Reader checks every received frame for internal consistency.
A torn read produces mixed pixel values -- this test catches it.

Run: pytest tests/test_fastlane_correctness.py -v
"""

import multiprocessing as mp
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
N_FRAMES = 100_000
RUN_ID = "test_torn_read"


def _writer_process(run_id, n_frames, width, height, channels, ready_event):
    """Write frames where all pixels = frame_number % 256."""
    config = FastLaneConfig(width=width, height=height, channels=channels, capacity=64)
    writer = FastLaneWriter.create(run_id, config)
    ready_event.set()

    for i in range(n_frames):
        value = i % 256
        frame = np.full((height, width, channels), value, dtype=np.uint8)
        writer.publish(
            frame.tobytes(),
            metrics=FastLaneMetrics(
                last_reward=float(value),
                rolling_return=0.0,
                step_rate_hz=0.0,
            ),
        )

    # Small delay so reader can finish
    time.sleep(0.5)
    writer.close()


def _reader_process(run_id, n_target, width, height, channels, ready_event, errors, frames_read):
    """Read frames and check all pixels are the same value."""
    ready_event.wait(timeout=10)
    time.sleep(0.1)  # Let writer get ahead

    reader = FastLaneReader.attach(run_id)
    read_count = 0
    frame_size = width * height * channels

    deadline = time.time() + 60  # 60s timeout
    while read_count < n_target and time.time() < deadline:
        frame = reader.latest_frame()
        if frame is None:
            time.sleep(0.001)
            continue

        pixels = np.frombuffer(frame.data[:frame_size], dtype=np.uint8)
        # All pixels should be the same value
        if pixels.size > 0 and not np.all(pixels == pixels[0]):
            errors.value += 1

        read_count += 1

    frames_read.value = read_count
    reader.close()


def test_no_torn_reads():
    """Zero torn reads across 100,000 frames proves sequence-number correctness."""
    errors = mp.Value("i", 0)
    frames_read = mp.Value("i", 0)
    ready = mp.Event()

    target_reads = min(N_FRAMES // 2, 50_000)

    w = mp.Process(
        target=_writer_process,
        args=(RUN_ID, N_FRAMES, WIDTH, HEIGHT, CHANNELS, ready),
    )
    r = mp.Process(
        target=_reader_process,
        args=(RUN_ID, target_reads, WIDTH, HEIGHT, CHANNELS, ready, errors, frames_read),
    )

    r.start()
    w.start()
    w.join(timeout=120)
    r.join(timeout=120)

    print(f"\n  Frames written: {N_FRAMES}")
    print(f"  Frames read:    {frames_read.value}")
    print(f"  Torn reads:     {errors.value}")

    assert errors.value == 0, (
        f"Detected {errors.value} torn reads out of {frames_read.value} frames. "
        "The sequence-number protocol is broken."
    )


def test_no_torn_reads_large_frames():
    """Same test with larger frames (320x240) to stress memcpy boundaries."""
    w, h = 320, 240
    n = 10_000
    run_id = "test_torn_large"
    errors = mp.Value("i", 0)
    frames_read = mp.Value("i", 0)
    ready = mp.Event()

    wp = mp.Process(target=_writer_process, args=(run_id, n, w, h, CHANNELS, ready))
    rp = mp.Process(target=_reader_process, args=(run_id, n // 2, w, h, CHANNELS, ready, errors, frames_read))

    rp.start()
    wp.start()
    wp.join(timeout=120)
    rp.join(timeout=120)

    print(f"\n  Large frames ({w}x{h}): {frames_read.value} read, {errors.value} torn")
    assert errors.value == 0
