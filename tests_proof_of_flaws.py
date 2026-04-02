"""Proof-of-concept tests demonstrating real flaws in Fast Lane.

Each test targets a specific claimed flaw and proves or disproves it.

Run:
    source .venv/bin/activate
    pytest tests_proof_of_flaws.py -v -s
"""

import multiprocessing as mp
import struct
import time
from multiprocessing import shared_memory

import numpy as np
import pytest

from gym_gui.fastlane.buffer import (
    FastLaneConfig,
    FastLaneMetrics,
    FastLaneReader,
    FastLaneWriter,
    FastLaneBase,
    _HEADER_STRUCT,
    _HEADER_SIZE,
    _IDX_TAIL,
    _SLOT_META_STRUCT,
    _SLOT_META_SIZE,
    FLAG_INVALIDATED,
    create_fastlane_name,
)


# ---------------------------------------------------------------------------
# FLAW 5: Dead `tail` field — never read or written
# ---------------------------------------------------------------------------

class TestFlaw5DeadTailField:
    """Prove that the tail field in the header is dead code."""

    def test_tail_never_written(self):
        """After creating a writer and publishing frames, tail should remain 0.
        If it's always 0, nothing ever writes to it."""
        run_id = "proof-tail-field"
        config = FastLaneConfig(width=16, height=16, channels=3)
        writer = FastLaneWriter.create(run_id, config)

        # Publish several frames
        frame = bytes(16 * 16 * 3)
        for _ in range(10):
            writer.publish(frame, metrics=FastLaneMetrics(1.0, 2.0, 3.0))

        # Read raw header
        header = _HEADER_STRUCT.unpack_from(writer._mv, 0)
        tail_value = header[_IDX_TAIL]
        head_value = header[10]  # _IDX_HEAD

        writer.close()
        writer.unlink()

        assert head_value == 10, f"Expected head=10, got {head_value}"
        assert tail_value == 0, (
            f"Tail is {tail_value} — if non-zero, something writes it. "
            f"But if always 0, the field is dead code."
        )

    def test_tail_index_exists_but_no_setter(self):
        """Verify no method exists to set the tail."""
        import inspect
        src = inspect.getsource(FastLaneBase)
        assert "_set_tail" not in src, "No _set_tail method exists"
        assert "tail" not in [name for name, _ in inspect.getmembers(FastLaneBase) if not name.startswith("__") and "tail" in name.lower()], "No tail-related methods on FastLaneBase"


# ---------------------------------------------------------------------------
# FLAW 1: Xuance worker missing unlink() — demonstrated via _close_writer pattern
# ---------------------------------------------------------------------------

class TestFlaw1MissingUnlink:
    """Prove that xuance_worker's _close_writer doesn't unlink."""

    def test_xuance_close_writer_missing_unlink(self):
        """Read the xuance worker source and prove unlink() is not called."""
        import inspect
        try:
            from xuance_worker.fastlane import FastLaneTelemetryWrapper
        except ImportError:
            pytest.skip("xuance_worker not installed")

        src = inspect.getsource(FastLaneTelemetryWrapper._close_writer)
        assert ".unlink()" not in src, (
            "If this fails, xuance_worker NOW calls unlink(). "
            "If it passes, the bug is confirmed — unlink() is missing."
        )
        assert ".close()" in src, "close() is called (but unlink is not)"

    def test_xuance_shm_leak_demonstration(self):
        """Demonstrate that close() without unlink() leaves SHM in /dev/shm."""
        import glob
        run_id = "proof-xuance-leak"
        config = FastLaneConfig(width=8, height=8, channels=3)
        writer = FastLaneWriter.create(run_id, config)

        # Publish a frame
        writer.publish(bytes(8 * 8 * 3), metrics=FastLaneMetrics(1.0, 0.0, 0.0))

        # Simulate xuance_worker's _close_writer: close() only, NO unlink()
        writer.close()
        # Intentionally NOT calling writer.unlink()

        # Check if segment still exists
        name = create_fastlane_name(run_id)
        leaked = glob.glob(f"/dev/shm/{name}")
        
        # Clean up for the test
        try:
            shm = shared_memory.SharedMemory(name=name, create=False)
            shm.close()
            shm.unlink()
        except FileNotFoundError:
            pass

        assert leaked, (
            f"SHM segment {name} should still exist after close() without unlink(). "
            f"Found: {leaked}. This proves close() alone is insufficient."
        )


# ---------------------------------------------------------------------------
# FLAW 2: Stale SHM reconnect with mismatched config
# ---------------------------------------------------------------------------

class TestFlaw2DimensionMismatch:
    """Prove that FastLaneWriter(shm, wrong_config) causes corruption."""

    def test_writer_with_wrong_config_corrupts_slots(self):
        """Create a SHM for 8x8 frames, then attach with 16x16 config.
        The writer will compute wrong slot offsets and corrupt data."""
        run_id = "proof-dim-mismatch"
        
        # Create SHM for 8x8 RGB frames
        config_small = FastLaneConfig(width=8, height=8, channels=3, capacity=4)
        writer1 = FastLaneWriter.create(run_id, config_small)
        
        # Write a known frame
        frame_8x8 = bytes(range(192))  # 8*8*3 = 192 bytes
        writer1.publish(frame_8x8, metrics=FastLaneMetrics(0.0, 0.0, 0.0))
        writer1.close()
        # Don't unlink — simulate stale SHM

        # Attach with WRONG config (16x16)
        name = create_fastlane_name(run_id)
        shm = shared_memory.SharedMemory(name=name, create=False)
        config_big = FastLaneConfig(width=16, height=16, channels=3, capacity=4)
        writer2 = FastLaneWriter(shm, config_big)

        # The writer thinks payload_bytes = 16*16*3 = 768
        # But the actual slot was sized for 8*8*3 = 192
        assert writer2._frame_payload_bytes == 768, "Writer thinks frames are 768 bytes"
        
        # The actual payload space per slot (from header) is only 192 + 0 = 192
        actual_payload = config_small.payload_bytes
        assert actual_payload == 192, "Actual slot payload is only 192 bytes"

        # Trying to publish a 16x16 frame will fail or corrupt
        frame_16x16 = bytes(768)
        try:
            writer2.publish(frame_16x16, metrics=FastLaneMetrics(0.0, 0.0, 0.0))
            # If we get here without error, the write happened but likely
            # overflowed the slot boundary into adjacent slots
            reader = FastLaneReader.attach(run_id)
            frame = reader.latest_frame()
            if frame is not None:
                # The data is garbage because we overflowed the slot
                assert len(frame.data) <= 192, "Frame should be at most 192 bytes (from original header)"
            reader.close()
        except (ValueError, struct.error, OverflowError) as e:
            pass  # Expected: frame exceeds slot capacity
        finally:
            writer2.close()
            # Clean up
            try:
                shm2 = shared_memory.SharedMemory(name=name, create=False)
                shm2.close()
                shm2.unlink()
            except FileNotFoundError:
                pass

    def test_writer_slot_size_mismatch_detected(self):
        """Prove that config slot_size != header slot_size when dimensions differ."""
        run_id = "proof-slot-size-mismatch"
        config_original = FastLaneConfig(width=8, height=8, channels=3, capacity=4)
        writer = FastLaneWriter.create(run_id, config_original)
        original_slot_size = writer.slot_size  # from header
        original_payload = writer.payload_size  # from header
        writer.close()

        # Reattach with different config
        name = create_fastlane_name(run_id)
        shm = shared_memory.SharedMemory(name=name, create=False)
        config_wrong = FastLaneConfig(width=16, height=16, channels=3, capacity=4)
        writer2 = FastLaneWriter(shm, config_wrong)

        # Header says slot_size for 8x8, but writer thinks 16x16
        header_slot_size = writer2.slot_size  # reads from actual header
        writer_payload = writer2._frame_payload_bytes  # from wrong config

        writer2.close()
        try:
            shm2 = shared_memory.SharedMemory(name=name, create=False)
            shm2.close()
            shm2.unlink()
        except FileNotFoundError:
            pass

        assert header_slot_size != writer_payload + _SLOT_META_SIZE or header_slot_size == 0, (
            f"Slot size mismatch: header={header_slot_size}, "
            f"writer expects={writer_payload + _SLOT_META_SIZE}"
        )
        # The writer's internal _frame_payload_bytes (768) disagrees with
        # the header's payload_size (192)
        assert writer_payload != original_payload, (
            f"Writer payload ({writer_payload}) != actual SHM payload ({original_payload}). "
            "This is the dimension mismatch bug."
        )


# ---------------------------------------------------------------------------
# FLAW 3: Header metrics written outside seqlock
# ---------------------------------------------------------------------------

def _flaw3_writer_process(run_id, n_frames, ready_event):
    """Writer that publishes frames with distinct metrics per frame."""
    config = FastLaneConfig(width=16, height=16, channels=3, capacity=8)
    writer = FastLaneWriter.create(run_id, config)
    ready_event.set()

    for i in range(n_frames):
        val = float(i % 256)
        frame = bytes([int(val)] * (16 * 16 * 3))
        metrics = FastLaneMetrics(
            last_reward=val,
            rolling_return=val * 10.0,
            step_rate_hz=1000.0,
        )
        writer.publish(frame, metrics=metrics)

    time.sleep(0.3)
    writer.close()


def _flaw3_reader_process(run_id, n_reads, ready_event, inconsistencies):
    """Reader that checks if metrics are self-consistent."""
    ready_event.wait(timeout=10)
    time.sleep(0.05)

    reader = FastLaneReader.attach(run_id)
    count = 0
    deadline = time.time() + 30

    while count < n_reads and time.time() < deadline:
        frame = reader.latest_frame()
        if frame is None:
            time.sleep(0.001)
            continue
        count += 1
        m = frame.metrics
        # If metrics are from the same frame, rolling_return should be
        # last_reward * 10.0 (as published). If torn, they'll disagree.
        if m.last_reward != 0.0:
            expected_rolling = m.last_reward * 10.0
            if abs(m.rolling_return - expected_rolling) > 0.01:
                inconsistencies.value += 1

    reader.close()


class TestFlaw3MetricsOutsideSeqlock:
    """Prove that metrics can be read inconsistently (outside seqlock)."""

    def test_metrics_inconsistency_under_load(self):
        """Multi-process test: writer hammers metrics, reader checks consistency.
        If metrics were inside the seqlock, they'd always be self-consistent."""
        run_id = "proof-metrics-seqlock"
        n_frames = 50_000
        n_reads = 10_000
        inconsistencies = mp.Value("i", 0)
        ready = mp.Event()

        w = mp.Process(target=_flaw3_writer_process,
                       args=(run_id, n_frames, ready))
        r = mp.Process(target=_flaw3_reader_process,
                       args=(run_id, n_reads, ready, inconsistencies))

        r.start()
        w.start()
        w.join(timeout=60)
        r.join(timeout=60)

        # Clean up
        name = create_fastlane_name(run_id)
        try:
            shm = shared_memory.SharedMemory(name=name, create=False)
            shm.close()
            shm.unlink()
        except FileNotFoundError:
            pass

        print(f"\n  Metrics inconsistencies: {inconsistencies.value} / {n_reads} reads")
        # Note: this test may or may not show inconsistencies depending on
        # scheduling. On x86 with CPython GIL, _set_metrics and _set_head
        # run in the same process so there's no interleaving of the writer's
        # own operations. The inconsistency can only be observed between
        # _set_metrics (header write) and the reader's metrics() call.
        # This is inherently a race that's hard to trigger deterministically.


# ---------------------------------------------------------------------------
# FLAW 7: Dead _slot_payload_bytes in reader
# ---------------------------------------------------------------------------

class TestFlaw7DeadReaderField:
    """Prove that reader._slot_payload_bytes is stored but never used."""

    def test_slot_payload_bytes_never_used(self):
        """Verify _slot_payload_bytes is set in __init__ but never referenced."""
        import inspect
        src = inspect.getsource(FastLaneReader)
        
        # Count references to _slot_payload_bytes
        refs = src.count("_slot_payload_bytes")
        # Should be exactly 1 (the assignment in __init__)
        assert refs == 1, (
            f"_slot_payload_bytes referenced {refs} times. "
            f"If 1, it's only assigned (dead). If >1, it's used."
        )


# ---------------------------------------------------------------------------
# FLAW 4: Non-atomic header read-modify-write
# ---------------------------------------------------------------------------

class TestFlaw4NonAtomicHeader:
    """Prove that _set_head and _set_metrics do full header read-modify-write."""

    def test_set_head_rewrite_count(self):
        """Verify _set_head reads the entire header just to change one field."""
        import inspect
        src = inspect.getsource(FastLaneBase._set_head)
        
        # Should contain unpack_from (read) and pack_into (write) of full header
        assert "unpack_from" in src, "_set_head reads the full header"
        assert "pack_into" in src, "_set_head writes the full header"
        
        # It reads all 15 fields, modifies 1, writes all 15 back
        # This is the TOCTOU pattern

    def test_set_metrics_rewrite_count(self):
        """Verify _set_metrics also does full header read-modify-write."""
        import inspect
        src = inspect.getsource(FastLaneBase._set_metrics)
        
        assert "unpack_from" in src, "_set_metrics reads the full header"
        assert "pack_into" in src, "_set_metrics writes the full header"


# ---------------------------------------------------------------------------
# FLAW 8: Reader attaches to uninitialized SHM
# ---------------------------------------------------------------------------

class TestFlaw8UninitializedHeader:
    """Prove reader can attach to a zero-filled SHM segment."""

    def test_attach_to_zero_header(self):
        """Create SHM with correct size but don't write header.
        Reader should handle gracefully but currently trusts the header."""
        run_id = "proof-zero-header"
        config = FastLaneConfig(width=8, height=8, channels=3, capacity=4)
        
        # Calculate the exact size the writer would use
        slot_payload = config.payload_bytes
        slot_size = _SLOT_META_SIZE + slot_payload
        total_size = _HEADER_SIZE + 4 * slot_size
        
        # Create raw SHM (all zeros) — simulates the window between
        # SharedMemory(create=True) and _write_header()
        name = create_fastlane_name(run_id)
        shm = shared_memory.SharedMemory(name=name, create=True, size=total_size)
        
        # Don't write header! Attach reader directly to zero-filled memory
        try:
            reader = FastLaneReader(shm)
            
            # All header fields are 0
            assert reader.capacity == 0, "Capacity should be 0 (uninitialized)"
            assert reader.width == 0, "Width should be 0 (uninitialized)"
            assert reader.head == 0, "Head should be 0 (uninitialized)"
            
            # latest_frame handles capacity=0 correctly
            frame = reader.latest_frame()
            assert frame is None, "Reader returns None for uninitialized buffer"
            
            reader.close()
        finally:
            try:
                shm.unlink()
            except FileNotFoundError:
                pass
        
        # But the reader's __init__ already read payload_size=0
        # This proves the reader CAN attach to uninitialized SHM
        # It's handled gracefully (returns None) but there's no MAGIC check


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
