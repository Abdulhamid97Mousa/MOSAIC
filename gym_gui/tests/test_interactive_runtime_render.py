"""Tests for InteractiveRuntime render_payload functionality.

This test verifies that the InteractiveRuntime correctly emits render_payload
with the structure expected by the GUI (mode, rgb, width, height).
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

# Policy and environment configuration
PROJECT_ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = (
    PROJECT_ROOT / "var/trainer/runs/"
    "01KE2NMTG89BRFQBSVH7MNZY5H/runs/"
    "MiniGrid-Empty-8x8-v0__ppo_with_save__1__1767468860/"
    "ppo_with_save.cleanrl_model"
)
ENV_ID = "MiniGrid-Empty-8x8-v0"
ALGORITHM = "ppo"


@pytest.fixture
def interactive_process():
    """Launch an interactive cleanrl_worker process."""
    cmd = [
        sys.executable,
        "-m", "cleanrl_worker.cli",
        "--interactive",
        "--run-id", "test_render_payload",
        "--algo", ALGORITHM,
        "--env-id", ENV_ID,
        "--policy-path", str(POLICY_PATH),
    ]

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    yield proc

    # Cleanup
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def read_json_response(proc, timeout=10.0):
    """Read a single JSON line from the process stdout."""
    import select

    start = time.time()
    while time.time() - start < timeout:
        readable, _, _ = select.select([proc.stdout], [], [], 0.1)
        if readable:
            line = proc.stdout.readline()
            if line:
                line = line.strip()
                if line:
                    return json.loads(line)
    return None


def read_all_responses(proc, timeout=2.0):
    """Read all available JSON lines from process stdout."""
    responses = []
    start = time.time()
    import select

    while time.time() - start < timeout:
        readable, _, _ = select.select([proc.stdout], [], [], 0.1)
        if readable:
            line = proc.stdout.readline()
            if line:
                line = line.strip()
                if line:
                    try:
                        responses.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        else:
            if responses:
                break
    return responses


class TestInteractiveRuntimeRenderPayload:
    """Test suite for InteractiveRuntime render_payload."""

    @pytest.mark.skipif(not POLICY_PATH.exists(), reason="Policy not found")
    def test_policy_exists(self):
        """Verify the policy file exists."""
        assert POLICY_PATH.exists(), f"Policy not found at {POLICY_PATH}"

    @pytest.mark.skipif(not POLICY_PATH.exists(), reason="Policy not found")
    def test_init_message_emitted(self, interactive_process):
        """Test that init message is emitted on startup."""
        proc = interactive_process

        # Read init message
        response = read_json_response(proc, timeout=10)

        assert response is not None, "No init response received"
        assert response.get("type") == "init", f"Expected 'init' type, got: {response}"
        assert response.get("env_id") == ENV_ID
        assert response.get("algo") == ALGORITHM

    @pytest.mark.skipif(not POLICY_PATH.exists(), reason="Policy not found")
    def test_reset_emits_ready(self, interactive_process):
        """Test that reset command emits ready response with stats and render."""
        proc = interactive_process

        # Wait for init
        init = read_json_response(proc, timeout=10)
        assert init is not None
        assert init.get("type") == "init"

        # Send reset command
        reset_cmd = json.dumps({"cmd": "reset", "seed": 42}) + "\n"
        proc.stdin.write(reset_cmd)
        proc.stdin.flush()

        # Read ready response
        response = read_json_response(proc, timeout=10)

        assert response is not None, "No ready response received"
        assert response.get("type") == "ready", f"Expected 'ready' type, got: {response}"
        assert response.get("seed") == 42

        # Verify stats are included for GUI reset
        assert "step_index" in response, "Missing step_index in ready response"
        assert response.get("step_index") == 0, f"Expected step_index=0, got {response.get('step_index')}"
        assert "episode_index" in response, "Missing episode_index in ready response"
        assert "episode_reward" in response, "Missing episode_reward in ready response"
        assert response.get("episode_reward") == 0.0, f"Expected episode_reward=0.0, got {response.get('episode_reward')}"

        # Verify render_payload is included
        render_payload = response.get("render_payload")
        assert render_payload is not None, "Missing render_payload in ready response"
        assert render_payload.get("mode") == "rgb_array"
        assert "rgb" in render_payload
        assert "width" in render_payload
        assert "height" in render_payload
        print(f"Ready response includes: step_index={response.get('step_index')}, "
              f"episode_index={response.get('episode_index')}, "
              f"render_payload width={render_payload.get('width')}")

    @pytest.mark.skipif(not POLICY_PATH.exists(), reason="Policy not found")
    def test_step_emits_render_payload(self, interactive_process):
        """Test that step command emits render_payload with correct structure."""
        proc = interactive_process

        # Wait for init
        init = read_json_response(proc, timeout=10)
        assert init is not None and init.get("type") == "init"

        # Send reset
        proc.stdin.write(json.dumps({"cmd": "reset", "seed": 42}) + "\n")
        proc.stdin.flush()

        ready = read_json_response(proc, timeout=10)
        assert ready is not None and ready.get("type") == "ready"

        # Send step command
        proc.stdin.write(json.dumps({"cmd": "step"}) + "\n")
        proc.stdin.flush()

        # Read step response
        response = read_json_response(proc, timeout=10)

        assert response is not None, "No step response received"
        assert response.get("type") == "step", f"Expected 'step' type, got: {response}"

        # Check render_payload exists and has correct structure
        render_payload = response.get("render_payload")
        print("\n=== STEP RESPONSE ===")
        print(f"Keys: {list(response.keys())}")
        print(f"render_payload: {render_payload is not None}")

        if render_payload is None:
            print(f"Full response: {json.dumps(response, indent=2, default=str)[:1000]}")
            pytest.fail("render_payload is None - rendering not working!")

        assert "mode" in render_payload, f"Missing 'mode' in render_payload: {render_payload.keys()}"
        assert render_payload["mode"] == "rgb_array", f"Expected mode='rgb_array', got: {render_payload['mode']}"
        assert "rgb" in render_payload, f"Missing 'rgb' in render_payload: {render_payload.keys()}"
        assert "width" in render_payload, "Missing 'width' in render_payload"
        assert "height" in render_payload, "Missing 'height' in render_payload"

        # Verify dimensions
        rgb = render_payload["rgb"]
        assert isinstance(rgb, list), f"Expected rgb to be list, got: {type(rgb)}"
        assert len(rgb) > 0, "RGB array is empty"

        height = render_payload["height"]
        width = render_payload["width"]

        print(f"RGB dimensions: height={height}, width={width}")
        print(f"RGB array length: {len(rgb)}")

        # For MiniGrid, typical render size is around 160x160 or similar
        assert height > 0, f"Invalid height: {height}"
        assert width > 0, f"Invalid width: {width}"

        print("=== render_payload structure is CORRECT ===")

    @pytest.mark.skipif(not POLICY_PATH.exists(), reason="Policy not found")
    def test_multiple_steps_with_render(self, interactive_process):
        """Test multiple steps all emit render_payload."""
        proc = interactive_process

        # Init and reset
        init = read_json_response(proc, timeout=10)
        assert init is not None

        proc.stdin.write(json.dumps({"cmd": "reset", "seed": 42}) + "\n")
        proc.stdin.flush()
        ready = read_json_response(proc, timeout=10)
        assert ready is not None

        # Run 5 steps
        for i in range(5):
            proc.stdin.write(json.dumps({"cmd": "step"}) + "\n")
            proc.stdin.flush()

            response = read_json_response(proc, timeout=10)
            assert response is not None, f"No response for step {i}"

            if response.get("type") == "step":
                render_payload = response.get("render_payload")
                print(f"Step {i}: render_payload={'present' if render_payload else 'MISSING'}")

                if render_payload is None:
                    pytest.fail(f"Step {i}: render_payload is None!")

                assert "mode" in render_payload
                assert "rgb" in render_payload
            elif response.get("type") == "episode_done":
                print(f"Episode done at step {i}")
                # After episode_done, we may need to reset
                break

    @pytest.mark.skipif(not POLICY_PATH.exists(), reason="Policy not found")
    def test_episode_reset_counters(self, interactive_process):
        """Test that step and episode counters reset properly after episode ends."""
        proc = interactive_process

        # Init and reset
        init = read_json_response(proc, timeout=10)
        assert init is not None

        proc.stdin.write(json.dumps({"cmd": "reset", "seed": 42}) + "\n")
        proc.stdin.flush()
        ready = read_json_response(proc, timeout=10)
        assert ready is not None

        # Run until episode ends (max 200 steps to be safe)
        episode_done = False
        last_step_index = 0
        last_episode_index = 0

        for i in range(200):
            proc.stdin.write(json.dumps({"cmd": "step"}) + "\n")
            proc.stdin.flush()

            response = read_json_response(proc, timeout=10)
            assert response is not None, f"No response for step {i}"

            if response.get("type") == "step":
                last_step_index = response.get("step_index", 0)
                last_episode_index = response.get("episode_index", 0)
                print(f"Step {i}: step_index={last_step_index}, episode_index={last_episode_index}")

                # Check for episode_done that might come right after
                # (SyncVectorEnv might emit it separately)
            elif response.get("type") == "episode_done":
                episode_done = True
                print(f"Episode done! episode_number={response.get('episode_number')}, "
                      f"episode_length={response.get('episode_length')}")
                break

        assert episode_done, "Episode did not end within 200 steps"

        # Now step again and verify counters reset
        proc.stdin.write(json.dumps({"cmd": "step"}) + "\n")
        proc.stdin.flush()

        response = read_json_response(proc, timeout=10)
        assert response is not None

        if response.get("type") == "step":
            new_step_index = response.get("step_index", -1)
            new_episode_index = response.get("episode_index", -1)
            print(f"After reset: step_index={new_step_index}, episode_index={new_episode_index}")

            # Step index should be 1 (first step of new episode)
            assert new_step_index == 1, f"Expected step_index=1 after episode reset, got {new_step_index}"
            # Episode index should have incremented
            assert new_episode_index == last_episode_index + 1, \
                f"Expected episode_index={last_episode_index + 1}, got {new_episode_index}"

            # Check render_payload is still valid after episode reset
            render_payload = response.get("render_payload")
            if render_payload:
                print(f"After reset render_payload: mode={render_payload.get('mode')}, "
                      f"width={render_payload.get('width')}, height={render_payload.get('height')}")
                rgb = render_payload.get("rgb")
                if rgb:
                    # Check for valid RGB data (not all zeros)
                    import numpy as np
                    arr = np.array(rgb, dtype=np.uint8)
                    non_zero = np.count_nonzero(arr)
                    print(f"RGB array: shape={arr.shape}, non_zero_pixels={non_zero}, "
                          f"min={arr.min()}, max={arr.max()}")
                    assert non_zero > 0, "RGB frame is all zeros after episode reset!"
                    assert arr.max() > 0, "RGB frame has no brightness after episode reset!"
            else:
                pytest.fail("render_payload is None after episode reset!")

            print("=== Episode reset counters are CORRECT ===")


class TestInteractiveRuntimeDirect:
    """Direct tests of InteractiveRuntime internals."""

    def test_import_interactive_runtime(self):
        """Test that InteractiveRuntime can be imported."""
        from cleanrl_worker.runtime import InteractiveRuntime
        assert InteractiveRuntime is not None

    @pytest.mark.skipif(not POLICY_PATH.exists(), reason="Policy not found")
    def test_minigrid_env_renders(self):
        """Test that MiniGrid environment can render."""
        import gymnasium as gym
        import minigrid  # noqa: F401 - needed for registration

        env = gym.make(ENV_ID, render_mode="rgb_array")
        env.reset(seed=42)

        frame = env.render()

        assert frame is not None, "MiniGrid render returned None"
        print(f"Frame shape: {frame.shape}")  # type: ignore[union-attr]
        print(f"Frame dtype: {frame.dtype}")  # type: ignore[union-attr]

        assert len(frame.shape) == 3, f"Expected 3D array, got shape: {frame.shape}"  # type: ignore[union-attr]
        assert frame.shape[2] == 3, f"Expected RGB (3 channels), got: {frame.shape[2]}"  # type: ignore[union-attr]

        env.close()

    @pytest.mark.skipif(not POLICY_PATH.exists(), reason="Policy not found")
    def test_sync_vector_env_renders(self):
        """Test that SyncVectorEnv can render (used by InteractiveRuntime)."""
        import gymnasium as gym
        import minigrid  # noqa: F401
        from minigrid.wrappers import ImgObsWrapper

        def make_env():
            env = gym.make(ENV_ID, render_mode="rgb_array")
            env = ImgObsWrapper(env)
            env = gym.wrappers.FlattenObservation(env)
            return env

        envs = gym.vector.SyncVectorEnv([make_env])
        envs.reset(seed=42)

        # This is how InteractiveRuntime calls render
        frames = envs.call("render")

        print(f"envs.call('render') returned: {type(frames)}")
        print(f"frames length: {len(frames) if frames else 'None'}")

        if frames is not None and len(frames) > 0:
            frame = frames[0]
            print(f"Frame type: {type(frame)}")
            if frame is not None:
                print(f"Frame shape: {frame.shape}")
                assert len(frame.shape) == 3
        else:
            pytest.fail("SyncVectorEnv.call('render') returned None or empty!")

        envs.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
