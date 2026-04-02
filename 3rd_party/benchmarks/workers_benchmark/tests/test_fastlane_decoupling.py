"""Test 2: Writer-Reader Decoupling (Core FastLane Claim).

Proves that writer throughput is identical whether the reader runs at
1 Hz, 60 Hz, or does not exist. This is the mathematical definition
of zero-copy lock-free streaming.

Run: pytest tests/test_fastlane_decoupling.py -v
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


WIDTH, HEIGHT, CHANNELS = 84, 84, 3
N_FRAMES = 50_000


def _measure_writer_throughput(run_id, reader_hz, n_frames=N_FRAMES):
    """Measure how many frames/second the writer can publish.

    Args:
        run_id: unique name for the shared memory segment
        reader_hz: 0 = no reader, otherwise reader polls at this rate
        n_frames: number of frames to write

    Returns:
        frames_per_second: writer throughput
    """
    config = FastLaneConfig(width=WIDTH, height=HEIGHT, channels=CHANNELS, capacity=64)
    frame_bytes = bytes(np.zeros((HEIGHT, WIDTH, CHANNELS), dtype=np.uint8))
    metrics = FastLaneMetrics(last_reward=0.0, rolling_return=0.0, step_rate_hz=0.0)

    # Start reader in a separate process (if requested)
    reader_proc = None
    if reader_hz > 0:
        def slow_reader(rid, hz):
            try:
                reader = FastLaneReader.attach(rid)
                interval = 1.0 / hz
                deadline = time.time() + 120
                while time.time() < deadline:
                    time.sleep(interval)
                    reader.latest_frame()
            except Exception:
                pass

        reader_proc = mp.Process(target=slow_reader, args=(run_id, reader_hz), daemon=True)

    writer = FastLaneWriter.create(run_id, config)

    if reader_proc:
        reader_proc.start()
        time.sleep(0.2)  # Let reader attach

    start = time.perf_counter()
    for _ in range(n_frames):
        writer.publish(frame_bytes, metrics=metrics)
    elapsed = time.perf_counter() - start

    writer.close()

    if reader_proc and reader_proc.is_alive():
        reader_proc.terminate()
        reader_proc.join(timeout=5)

    return n_frames / elapsed


def test_writer_throughput_independent_of_reader():
    """Writer FPS should be identical with no reader, slow reader, and fast reader."""
    throughput_none = _measure_writer_throughput("decouple_none", reader_hz=0)
    throughput_1hz = _measure_writer_throughput("decouple_1hz", reader_hz=1)
    throughput_60hz = _measure_writer_throughput("decouple_60hz", reader_hz=60)

    baseline = throughput_none

    pct_1hz = abs(throughput_1hz - baseline) / baseline * 100
    pct_60hz = abs(throughput_60hz - baseline) / baseline * 100

    print(f"\n  No reader:    {throughput_none:>10.0f} fps")
    print(f"  1 Hz reader:  {throughput_1hz:>10.0f} fps  ({pct_1hz:.1f}% diff)")
    print(f"  60 Hz reader: {throughput_60hz:>10.0f} fps  ({pct_60hz:.1f}% diff)")

    # Tolerance: 10% to account for OS scheduling noise
    assert pct_1hz < 10, (
        f"Slow reader (1 Hz) caused {pct_1hz:.1f}% throughput change. "
        "Writer is NOT decoupled from reader."
    )
    assert pct_60hz < 10, (
        f"Fast reader (60 Hz) caused {pct_60hz:.1f}% throughput change. "
        "Writer is NOT decoupled from reader."
    )


def test_writer_does_not_block():
    """Each publish() call should be sub-millisecond (just a memcpy)."""
    config = FastLaneConfig(width=WIDTH, height=HEIGHT, channels=CHANNELS, capacity=64)
    frame_bytes = bytes(np.zeros((HEIGHT, WIDTH, CHANNELS), dtype=np.uint8))
    metrics = FastLaneMetrics(last_reward=0.0, rolling_return=0.0, step_rate_hz=0.0)

    writer = FastLaneWriter.create("decouple_latency", config)

    latencies = []
    for _ in range(10_000):
        t0 = time.perf_counter()
        writer.publish(frame_bytes, metrics=metrics)
        latencies.append(time.perf_counter() - t0)

    writer.close()

    latencies_us = [l * 1_000_000 for l in latencies]
    p50 = np.percentile(latencies_us, 50)
    p99 = np.percentile(latencies_us, 99)
    p999 = np.percentile(latencies_us, 99.9)

    print(f"\n  publish() latency (84x84 RGB, {len(latencies)} calls):")
    print(f"    p50:  {p50:>8.1f} us")
    print(f"    p99:  {p99:>8.1f} us")
    print(f"    p999: {p999:>8.1f} us")

    # A single memcpy of 21KB should take < 1ms even in the worst case
    assert p99 < 1000, f"p99 publish latency is {p99:.0f} us (> 1ms). Writer may be blocking."
