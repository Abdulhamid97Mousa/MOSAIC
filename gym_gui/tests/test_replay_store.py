"""Unit tests for HDF5 replay storage.

Tests the ReplayWriter and ReplayReader classes for storing and retrieving
RL experience data (frames, observations, actions, rewards).
"""

import numpy as np
import pytest

from gym_gui.replay import (
    FrameRef,
    FrameResolver,
    ReplayReader,
    ReplayWriter,
    make_frame_ref,
)


class TestFrameRef:
    """Tests for FrameRef parsing and serialization."""

    def test_parse_valid_ref(self):
        """Parse a valid frame reference URI."""
        ref = FrameRef.parse("h5://run_abc123/frames/1523")
        assert ref is not None
        assert ref.run_id == "run_abc123"
        assert ref.dataset == "frames"
        assert ref.index == 1523

    def test_parse_observations_ref(self):
        """Parse a reference to observations dataset."""
        ref = FrameRef.parse("h5://run_xyz/observations/42")
        assert ref is not None
        assert ref.run_id == "run_xyz"
        assert ref.dataset == "observations"
        assert ref.index == 42

    def test_parse_invalid_ref_returns_none(self):
        """Invalid reference formats return None."""
        assert FrameRef.parse("") is None
        assert FrameRef.parse("invalid") is None
        assert FrameRef.parse("h5://") is None
        assert FrameRef.parse("h5://run/dataset") is None
        assert FrameRef.parse("http://run/dataset/0") is None

    def test_to_uri_roundtrip(self):
        """FrameRef can be converted back to URI."""
        ref = FrameRef(run_id="test_run", dataset="frames", index=100)
        uri = ref.to_uri()
        assert uri == "h5://test_run/frames/100"

        # Roundtrip
        parsed = FrameRef.parse(uri)
        assert parsed == ref

    def test_make_frame_ref_helper(self):
        """make_frame_ref creates valid URI strings."""
        uri = make_frame_ref("my_run", "frames", 999)
        assert uri == "h5://my_run/frames/999"


class TestReplayWriter:
    """Tests for ReplayWriter."""

    def test_writer_creates_hdf5_file(self, tmp_path):
        """Writer creates HDF5 file on start."""
        run_id = "test_run_001"

        with ReplayWriter(run_id, tmp_path) as writer:
            assert writer.path.exists()
            assert writer.run_id == run_id

    def test_writer_records_steps(self, tmp_path):
        """Writer records steps and returns frame refs."""
        run_id = "test_run_002"
        frame_shape = (32, 32, 3)
        obs_shape = (84, 84, 4)

        with ReplayWriter(run_id, tmp_path) as writer:
            for i in range(10):
                frame = np.random.randint(0, 255, frame_shape, dtype=np.uint8)
                obs = np.random.randint(0, 255, obs_shape, dtype=np.uint8)
                ref = writer.record_step(
                    frame=frame,
                    observation=obs,
                    action=i % 4,
                    reward=float(i) * 0.1,
                    done=(i == 9),
                )
                assert ref == f"h5://{run_id}/frames/{i}"

            writer.flush()
            assert writer.step_count == 10

    def test_writer_handles_none_frame(self, tmp_path):
        """Writer handles steps with no frame data."""
        run_id = "test_run_003"

        with ReplayWriter(run_id, tmp_path) as writer:
            for i in range(5):
                ref = writer.record_step(
                    frame=None,
                    observation=None,
                    action=i,
                    reward=1.0,
                    done=False,
                )
                assert "frames" in ref

            writer.flush()
            assert writer.step_count == 5

    def test_writer_episode_boundaries(self, tmp_path):
        """Writer tracks episode boundaries."""
        run_id = "test_run_004"
        frame_shape = (16, 16, 3)

        with ReplayWriter(run_id, tmp_path) as writer:
            # Episode 0: steps 0-4
            for i in range(5):
                frame = np.zeros(frame_shape, dtype=np.uint8)
                writer.record_step(frame, None, i, 0.0, False)
            writer.mark_episode_end()

            # Episode 1: steps 5-9
            for i in range(5):
                frame = np.zeros(frame_shape, dtype=np.uint8)
                writer.record_step(frame, None, i, 0.0, i == 4)
            writer.mark_episode_end()

            writer.flush()

        # Verify with reader
        with ReplayReader(tmp_path / f"{run_id}.h5") as reader:
            assert reader.num_episodes == 2
            assert reader.num_steps == 10

    def test_writer_not_started_raises(self, tmp_path):
        """Recording without start() raises RuntimeError."""
        writer = ReplayWriter("test", tmp_path)
        with pytest.raises(RuntimeError, match="not started"):
            writer.record_step(None, None, 0, 0.0, False)


class TestReplayReader:
    """Tests for ReplayReader."""

    @pytest.fixture
    def sample_hdf5(self, tmp_path):
        """Create a sample HDF5 file for testing."""
        run_id = "reader_test"
        frame_shape = (32, 32, 3)
        obs_shape = (84, 84, 4)

        with ReplayWriter(run_id, tmp_path) as writer:
            # Episode 0: 10 steps
            for i in range(10):
                frame = np.full(frame_shape, i * 10, dtype=np.uint8)
                obs = np.full(obs_shape, i * 5, dtype=np.uint8)
                writer.record_step(frame, obs, i % 4, float(i), i == 9)
            writer.mark_episode_end()

            # Episode 1: 5 steps
            for i in range(5):
                frame = np.full(frame_shape, 100 + i * 10, dtype=np.uint8)
                obs = np.full(obs_shape, 50 + i * 5, dtype=np.uint8)
                writer.record_step(frame, obs, i % 4, float(i) * 2, i == 4)
            writer.mark_episode_end()

            writer.flush()

        return tmp_path / f"{run_id}.h5"

    def test_reader_metadata(self, sample_hdf5):
        """Reader exposes file metadata."""
        with ReplayReader(sample_hdf5) as reader:
            assert reader.num_steps == 15
            assert reader.num_episodes == 2
            assert reader.has_frames
            assert reader.has_observations

            meta = reader.metadata
            assert meta["run_id"] == "reader_test"
            assert meta["total_steps"] == 15
            assert "created_at" in meta

    def test_reader_get_step(self, sample_hdf5):
        """Reader retrieves individual steps."""
        with ReplayReader(sample_hdf5) as reader:
            step = reader.get_step(0)
            assert step["action"] == 0
            assert step["reward"] == 0.0
            assert step["done"] is False
            assert "frame" in step
            assert "observation" in step

            # Check frame values
            assert step["frame"][0, 0, 0] == 0  # First frame filled with 0*10

            step5 = reader.get_step(5)
            assert step5["frame"][0, 0, 0] == 50  # 5th frame filled with 5*10

    def test_reader_get_episode(self, sample_hdf5):
        """Reader retrieves entire episodes."""
        with ReplayReader(sample_hdf5) as reader:
            ep0 = reader.get_episode(0)
            assert ep0["length"] == 10
            assert ep0["start_step"] == 0
            assert len(ep0["actions"]) == 10
            assert len(ep0["rewards"]) == 10
            assert "frames" in ep0

            ep1 = reader.get_episode(1)
            assert ep1["length"] == 5
            assert ep1["start_step"] == 10

    def test_reader_get_episode_invalid_index(self, sample_hdf5):
        """Getting invalid episode raises IndexError."""
        with ReplayReader(sample_hdf5) as reader:
            with pytest.raises(IndexError):
                reader.get_episode(999)

    def test_reader_get_frame(self, sample_hdf5):
        """Reader retrieves individual frames."""
        with ReplayReader(sample_hdf5) as reader:
            frame = reader.get_frame(0)
            assert frame is not None
            assert frame.shape == (32, 32, 3)

            # Out of bounds returns None
            assert reader.get_frame(9999) is None

    def test_reader_get_observation(self, sample_hdf5):
        """Reader retrieves individual observations."""
        with ReplayReader(sample_hdf5) as reader:
            obs = reader.get_observation(0)
            assert obs is not None
            assert obs.shape == (84, 84, 4)

    def test_reader_iter_batches(self, sample_hdf5):
        """Reader iterates in batches."""
        with ReplayReader(sample_hdf5) as reader:
            batches = list(reader.iter_batches(batch_size=4))

            # 15 steps / 4 batch_size = 4 batches (3 full + 1 partial)
            assert len(batches) == 4

            # First batch should have 4 items
            assert len(batches[0]["actions"]) == 4
            assert len(batches[0]["rewards"]) == 4

            # Last batch has remainder
            assert len(batches[-1]["actions"]) == 3

    def test_reader_iter_batches_with_shuffle(self, sample_hdf5):
        """Reader can shuffle batches."""
        with ReplayReader(sample_hdf5) as reader:
            batches1 = list(reader.iter_batches(batch_size=4, shuffle=True))
            batches2 = list(reader.iter_batches(batch_size=4, shuffle=True))

            # With shuffle, batches should differ (probabilistically)
            # We just verify it doesn't crash
            assert len(batches1) == len(batches2) == 4

    def test_reader_not_open_raises(self, sample_hdf5):
        """Operations on unopened reader raise RuntimeError."""
        reader = ReplayReader(sample_hdf5)
        with pytest.raises(RuntimeError, match="not open"):
            reader.get_step(0)


class TestFrameResolver:
    """Tests for FrameResolver."""

    @pytest.fixture
    def sample_runs(self, tmp_path):
        """Create multiple run files for testing."""
        runs = {}
        frame_shape = (16, 16, 3)

        for run_idx, run_id in enumerate(["run_a", "run_b"]):
            with ReplayWriter(run_id, tmp_path) as writer:
                for i in range(5):
                    frame = np.full(frame_shape, run_idx * 100 + i, dtype=np.uint8)
                    writer.record_step(frame, None, i, float(i), False)
                writer.flush()
            runs[run_id] = tmp_path / f"{run_id}.h5"

        return tmp_path, runs

    def test_resolver_resolve_single(self, sample_runs):
        """Resolver resolves single frame reference."""
        replay_dir, runs = sample_runs

        with FrameResolver(replay_dir) as resolver:
            frame = resolver.resolve("h5://run_a/frames/0")
            assert frame is not None
            assert frame.shape == (16, 16, 3)
            assert frame[0, 0, 0] == 0  # run_a, frame 0: 0*100+0

            frame = resolver.resolve("h5://run_a/frames/3")
            assert frame is not None
            assert frame[0, 0, 0] == 3  # run_a, frame 3: 0*100+3

            frame = resolver.resolve("h5://run_b/frames/0")
            assert frame is not None
            assert frame[0, 0, 0] == 100  # run_b, frame 0: 1*100+0

    def test_resolver_resolve_invalid(self, sample_runs):
        """Resolver returns None for invalid references."""
        replay_dir, _ = sample_runs

        with FrameResolver(replay_dir) as resolver:
            assert resolver.resolve("invalid") is None
            assert resolver.resolve("h5://nonexistent/frames/0") is None
            assert resolver.resolve("h5://run_a/frames/9999") is None
            assert resolver.resolve("h5://run_a/invalid_dataset/0") is None

    def test_resolver_resolve_batch(self, sample_runs):
        """Resolver resolves multiple references efficiently."""
        replay_dir, _ = sample_runs

        refs = [
            "h5://run_a/frames/0",
            "h5://run_a/frames/2",
            "h5://run_b/frames/1",
            "invalid_ref",
            "h5://run_a/frames/4",
        ]

        with FrameResolver(replay_dir) as resolver:
            results = resolver.resolve_batch(refs)

            assert len(results) == 5
            assert results[0] is not None
            assert results[0][0, 0, 0] == 0  # run_a, frame 0
            assert results[1] is not None
            assert results[1][0, 0, 0] == 2  # run_a, frame 2
            assert results[2] is not None
            assert results[2][0, 0, 0] == 101  # run_b, frame 1
            assert results[3] is None  # invalid
            assert results[4] is not None
            assert results[4][0, 0, 0] == 4  # run_a, frame 4

    def test_resolver_has_run(self, sample_runs):
        """Resolver checks if run exists."""
        replay_dir, _ = sample_runs

        with FrameResolver(replay_dir) as resolver:
            assert resolver.has_run("run_a")
            assert resolver.has_run("run_b")
            assert not resolver.has_run("nonexistent")

    def test_resolver_list_runs(self, sample_runs):
        """Resolver lists available runs."""
        replay_dir, _ = sample_runs

        with FrameResolver(replay_dir) as resolver:
            runs = resolver.list_runs()
            assert "run_a" in runs
            assert "run_b" in runs
            assert len(runs) == 2

    def test_resolver_get_run_info(self, sample_runs):
        """Resolver retrieves run metadata."""
        replay_dir, _ = sample_runs

        with FrameResolver(replay_dir) as resolver:
            info = resolver.get_run_info("run_a")
            assert info is not None
            assert info["run_id"] == "run_a"
            assert info["total_steps"] == 5
            assert info["has_frames"] is True


class TestIntegration:
    """Integration tests for the complete replay pipeline."""

    def test_write_read_roundtrip(self, tmp_path):
        """Complete write/read roundtrip preserves data."""
        run_id = "roundtrip_test"
        frame_shape = (64, 64, 3)
        num_steps = 50

        # Write
        original_frames = []
        original_actions = []
        original_rewards = []

        with ReplayWriter(run_id, tmp_path) as writer:
            for i in range(num_steps):
                frame = np.random.randint(0, 255, frame_shape, dtype=np.uint8)
                original_frames.append(frame.copy())
                original_actions.append(i % 8)
                original_rewards.append(float(i) * 0.5)

                writer.record_step(
                    frame=frame,
                    observation=None,
                    action=original_actions[-1],
                    reward=original_rewards[-1],
                    done=(i == num_steps - 1),
                )

            writer.flush()

        # Read back
        with ReplayReader(tmp_path / f"{run_id}.h5") as reader:
            assert reader.num_steps == num_steps

            for i in range(num_steps):
                step = reader.get_step(i)
                np.testing.assert_array_equal(step["frame"], original_frames[i])
                assert step["action"] == original_actions[i]
                assert step["reward"] == pytest.approx(original_rewards[i])

    def test_resolver_with_multiple_runs(self, tmp_path):
        """Resolver handles multiple concurrent runs."""
        num_runs = 3
        steps_per_run = 10

        # Create multiple runs
        for r in range(num_runs):
            run_id = f"multi_run_{r}"
            with ReplayWriter(run_id, tmp_path) as writer:
                for i in range(steps_per_run):
                    frame = np.full((8, 8, 3), r * 100 + i, dtype=np.uint8)
                    writer.record_step(frame, None, i, 0.0, False)
                writer.flush()

        # Resolve refs across all runs
        with FrameResolver(tmp_path) as resolver:
            for r in range(num_runs):
                run_id = f"multi_run_{r}"
                for i in range(steps_per_run):
                    ref = f"h5://{run_id}/frames/{i}"
                    frame = resolver.resolve(ref)
                    assert frame is not None
                    expected_value = r * 100 + i
                    assert frame[0, 0, 0] == expected_value

    def test_large_batch_write_performance(self, tmp_path):
        """Large batch writes complete efficiently."""
        run_id = "perf_test"
        frame_shape = (84, 84, 4)
        num_steps = 1000

        with ReplayWriter(
            run_id,
            tmp_path,
            batch_size=100,
            chunk_size=100,
        ) as writer:
            for i in range(num_steps):
                frame = np.random.randint(0, 255, frame_shape, dtype=np.uint8)
                writer.record_step(frame, None, i % 4, 1.0, False)

            writer.flush()
            assert writer.step_count == num_steps

        # Verify file is readable
        with ReplayReader(tmp_path / f"{run_id}.h5") as reader:
            assert reader.num_steps == num_steps
