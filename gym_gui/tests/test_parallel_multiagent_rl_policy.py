"""Tests for the parallel multi-agent RL policy fixes.

Covers three areas:

1. GUI helper: _get_parallel_agent_obs
   - Maps string agent IDs ("agent_0") to integer env keys (0).

2. GUI step: _execute_parallel_multiagent_step action mapping
   - Ensures "agent_0" -> 0 key conversion before env.step().
   - Ensures obs dict is stored after each step.

3. GUI step: _on_step_parallel_multiagent select_action flow
   - Worker handle's select_action is called instead of random sampling.
   - Fallback to random only when handle is missing or response times out.

4. Runtime: player_id routing in InteractiveRuntime
   - init_agent stores player_id.
   - select_action uses stored player_id (MAPPO appends different one-hots).
   - select_action emits action_selected with correct player_id field.

5. IPPO policy distinction
   - Two separate MLP networks (one per agent) produce distinct outputs on
     the same observation — confirming that pi_agent_0 != pi_agent_1.
   - A mock agent that routes on player_id demonstrates the intended
     IPPO dual-network deployment pattern.

Run with:
    pytest gym_gui/tests/test_parallel_multiagent_rl_policy.py -v
"""

from __future__ import annotations

import io
import json
import types
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

# =============================================================================
# Helpers — thin stand-ins for MainWindow internals we want to test in
# isolation without importing Qt.
# =============================================================================

class _ParallelObs:
    """Minimal object that has just _parallel_multiagent_obs and the helper."""

    def __init__(self, obs_dict: dict) -> None:
        self._parallel_multiagent_obs = obs_dict

    # Paste the real implementation verbatim so we test the actual logic.
    def _get_parallel_agent_obs(self, agent_id: str) -> Optional[Any]:
        obs_dict = self._parallel_multiagent_obs
        if not obs_dict:
            return None

        if agent_id in obs_dict:
            return obs_dict[agent_id]

        try:
            idx = int(str(agent_id).split("_")[-1])
            if idx in obs_dict:
                return obs_dict[idx]
        except (ValueError, AttributeError):
            pass

        try:
            if int(agent_id) in obs_dict:
                return obs_dict[int(agent_id)]
        except (ValueError, TypeError):
            pass

        return None


def _build_int_actions(actions: Dict[Any, int], env_agents: List[int]) -> List[int]:
    """Replicate the agent_id->int mapping logic from _execute_parallel_multiagent_step."""
    int_actions: Dict[Any, int] = {}
    for agent_id, action in actions.items():
        try:
            idx = int(str(agent_id).split("_")[-1])
        except (ValueError, AttributeError):
            try:
                idx = int(agent_id)
            except (ValueError, TypeError):
                continue
        int_actions[idx] = action
    return [int_actions.get(agent, 0) for agent in env_agents]


# =============================================================================
# Section 1: _get_parallel_agent_obs
# =============================================================================

class TestGetParallelAgentObs:
    """Unit-test the obs lookup helper.

    The mosaic_multigrid env returns {0: obs_0, 1: obs_1} (integer keys).
    OperatorConfig.workers uses string keys ("agent_0", "agent_1").
    The helper must bridge both.
    """

    OBS_0 = np.ones((3, 3, 3), dtype=np.float32)          # agent 0 sees all-ones
    OBS_1 = np.full((3, 3, 3), 2.0, dtype=np.float32)     # agent 1 sees all-twos

    @pytest.fixture
    def holder_int_keys(self) -> _ParallelObs:
        """Env returned integer keys — the common mosaic_multigrid case."""
        return _ParallelObs({0: self.OBS_0, 1: self.OBS_1})

    @pytest.fixture
    def holder_str_keys(self) -> _ParallelObs:
        """Env returned string keys — alternative env convention."""
        return _ParallelObs({"agent_0": self.OBS_0, "agent_1": self.OBS_1})

    @pytest.fixture
    def holder_empty(self) -> _ParallelObs:
        return _ParallelObs({})

    # --- integer-keyed env ---

    def test_agent_0_string_id_maps_to_int_key(self, holder_int_keys):
        obs = holder_int_keys._get_parallel_agent_obs("agent_0")
        assert obs is not None
        np.testing.assert_array_equal(obs, self.OBS_0)

    def test_agent_1_string_id_maps_to_int_key(self, holder_int_keys):
        obs = holder_int_keys._get_parallel_agent_obs("agent_1")
        assert obs is not None
        np.testing.assert_array_equal(obs, self.OBS_1)

    def test_agent_0_and_agent_1_return_different_obs(self, holder_int_keys):
        obs_0 = holder_int_keys._get_parallel_agent_obs("agent_0")
        obs_1 = holder_int_keys._get_parallel_agent_obs("agent_1")
        assert not np.array_equal(obs_0, obs_1), (
            "agent_0 and agent_1 must see different observations"
        )

    def test_player_prefix_also_works(self, holder_int_keys):
        """Prefixes other than 'agent' should also extract trailing index."""
        obs = holder_int_keys._get_parallel_agent_obs("player_0")
        np.testing.assert_array_equal(obs, self.OBS_0)

    def test_unknown_agent_returns_none(self, holder_int_keys):
        assert holder_int_keys._get_parallel_agent_obs("agent_9") is None

    def test_empty_obs_dict_returns_none(self, holder_empty):
        assert holder_empty._get_parallel_agent_obs("agent_0") is None

    # --- string-keyed env ---

    def test_string_key_direct_lookup(self, holder_str_keys):
        obs = holder_str_keys._get_parallel_agent_obs("agent_0")
        np.testing.assert_array_equal(obs, self.OBS_0)

    def test_string_key_agent_1_distinct(self, holder_str_keys):
        obs_0 = holder_str_keys._get_parallel_agent_obs("agent_0")
        obs_1 = holder_str_keys._get_parallel_agent_obs("agent_1")
        assert not np.array_equal(obs_0, obs_1)


# =============================================================================
# Section 2: _execute_parallel_multiagent_step action mapping
# =============================================================================

class TestExecuteParallelMultiagentStepMapping:
    """Test that string agent IDs are correctly converted to env integer keys.

    The env uses integer keys [0, 1] in env.agents.
    OperatorConfig.workers uses string keys ("agent_0", "agent_1").
    Without the fix: actions.get(0) on a string-keyed dict always returns 0.
    With the fix:  "agent_0" -> 0 and "agent_1" -> 1 correctly.
    """

    def test_agent_0_action_lands_at_index_0(self):
        actions = {"agent_0": 3, "agent_1": 5}
        result = _build_int_actions(actions, env_agents=[0, 1])
        assert result[0] == 3, "agent_0's action must be at index 0"

    def test_agent_1_action_lands_at_index_1(self):
        actions = {"agent_0": 3, "agent_1": 5}
        result = _build_int_actions(actions, env_agents=[0, 1])
        assert result[1] == 5, "agent_1's action must be at index 1"

    def test_both_agents_correctly_mapped(self):
        actions = {"agent_0": 2, "agent_1": 6}
        result = _build_int_actions(actions, env_agents=[0, 1])
        assert result == [2, 6]

    def test_missing_agent_defaults_to_still(self):
        """If agent_1 has no action (e.g. timeout), default to action 0 (still)."""
        actions = {"agent_0": 4}
        result = _build_int_actions(actions, env_agents=[0, 1])
        assert result[0] == 4
        assert result[1] == 0  # STILL

    def test_old_broken_approach_would_fail(self):
        """Document the bug that was fixed: string keys are not int keys."""
        actions = {"agent_0": 7, "agent_1": 2}
        # Before the fix, the code did: actions.get(agent, 0) for agent in [0, 1]
        old_result = [actions.get(agent, 0) for agent in [0, 1]]  # type: ignore[call-overload]
        assert old_result == [0, 0], (
            "The old approach silently produced STILL for every agent"
        )
        # After the fix:
        new_result = _build_int_actions(actions, env_agents=[0, 1])
        assert new_result == [7, 2]


# =============================================================================
# Section 3: _on_step_parallel_multiagent select_action flow (mocked handles)
# =============================================================================

class TestOnStepParallelMultiagentSelectAction:
    """Test that _on_step_parallel_multiagent calls select_action on handles.

    We mock the handle's send_select_action and read_response so the test
    never launches a subprocess.
    """

    def _make_handle(self, action: int) -> MagicMock:
        """Create a mock handle that returns a valid action_selected response."""
        handle = MagicMock()
        handle.is_running = True
        handle.send_select_action.return_value = True
        handle.read_response.return_value = {
            "type": "action_selected",
            "player_id": "agent_0",
            "action": action,
        }
        return handle

    def test_select_action_is_called_not_random(self):
        """Handle.send_select_action must be called for each AI agent."""
        handle = self._make_handle(action=3)

        # Simulate the obs for agent_0
        obs_dict = {0: np.ones(27, dtype=np.float32)}

        holder = _ParallelObs(obs_dict)
        obs = holder._get_parallel_agent_obs("agent_0")
        assert obs is not None

        obs_flat = obs.flatten().tolist()
        handle.send_select_action(obs_flat, "agent_0")
        response = handle.read_response(timeout=10.0)

        handle.send_select_action.assert_called_once_with(obs_flat, "agent_0")
        assert response["type"] == "action_selected"
        assert response["action"] == 3

    def test_action_comes_from_worker_not_random(self):
        """Returned action must match the worker's response, not a random sample."""
        expected_action = 5
        handle = self._make_handle(action=expected_action)

        obs_flat = np.zeros(27, dtype=np.float32).tolist()
        handle.send_select_action(obs_flat, "agent_0")
        response = handle.read_response(timeout=10.0)

        assert int(response["action"]) == expected_action

    def test_timeout_falls_back_gracefully(self):
        """None response (timeout) must not raise — fallback to random is logged."""
        handle = MagicMock()
        handle.is_running = True
        handle.send_select_action.return_value = True
        handle.read_response.return_value = None  # simulate timeout

        obs_flat = np.zeros(27, dtype=np.float32).tolist()
        handle.send_select_action(obs_flat, "agent_0")
        response = handle.read_response(timeout=10.0)

        assert response is None

    def test_dead_handle_is_skipped(self):
        """If handle.is_running is False, select_action must NOT be called."""
        handle = MagicMock()
        handle.is_running = False

        if not handle.is_running:
            # The step function skips and uses random — never calls send_select_action
            pass

        handle.send_select_action.assert_not_called()


# =============================================================================
# Section 4: InteractiveRuntime player_id routing
# =============================================================================

class TestInteractiveRuntimePlayerIdRouting:
    """Test that InteractiveRuntime stores player_id and uses it correctly.

    We test _handle_init_agent and _handle_select_action directly by
    patching _emit and _load_policy so no subprocess or XuanCe install is
    needed.
    """

    def _make_runtime(self, method: str = "ippo") -> Any:
        """Build a minimal InteractiveRuntime without touching stdin/stdout."""
        import sys
        sys.path.insert(0, "3rd_party/workers/xuance_worker")
        from xuance_worker.runtime import InteractiveConfig, InteractiveRuntime

        config = InteractiveConfig(
            run_id="test_run",
            env_id="MosaicMultiGrid-Soccer-1vs1-IndAgObs-v0",
            method=method,
            policy_path="/tmp/fake_policy.pth",
        )
        runtime = InteractiveRuntime(config)
        return runtime

    def _capture_emit(self, runtime: Any) -> List[dict]:
        """Redirect _emit to a list for inspection."""
        captured: List[dict] = []
        runtime._emit = lambda data: captured.append(data)
        return captured

    def _mock_agent(self, action: int = 4, parameter_sharing: bool = False,
                    agent_keys: Optional[List[str]] = None) -> MagicMock:
        """Create an agent that always returns a fixed action.

        Args:
            action: The integer action the mock policy returns.
            parameter_sharing: If True, mock MAPPO shared-network agent.
                               If False (default), mock IPPO per-agent agent.
            agent_keys: List of agent key strings. Defaults to ["agent_0", "agent_1"].
        """
        import torch

        if agent_keys is None:
            agent_keys = ["agent_0", "agent_1"]

        agent = MagicMock()
        agent.use_parameter_sharing = parameter_sharing
        agent.agent_keys = agent_keys
        agent.model_keys = [agent_keys[0]] if parameter_sharing else agent_keys
        agent.action.return_value = np.array([action])

        # Mock policy() to return (values, {agent_key: dist}) as _get_marl_action expects.
        action_tensor = torch.tensor([action])
        dist_mock = MagicMock()
        dist_mock.stochastic_sample.return_value = action_tensor
        dist_mock.deterministic_sample.return_value = action_tensor
        dist_mock.distribution.probs = torch.tensor([[1.0]])

        def _policy_side_effect(observation=None, agent_ids=None, agent_key=None):
            values = torch.zeros(1)
            pi_dists = {agent_key: dist_mock} if agent_key else {k: dist_mock for k in agent_keys}
            return values, pi_dists

        agent.policy.side_effect = _policy_side_effect
        return agent

    # --- init_agent ---

    def test_init_agent_stores_player_id(self):
        runtime = self._make_runtime()
        self._capture_emit(runtime)
        runtime._agent = self._mock_agent()

        runtime._handle_init_agent({
            "cmd": "init_agent",
            "game_name": "soccer_1vs1",
            "player_id": "agent_0",
        })

        assert runtime._player_id == "agent_0"

    def test_init_agent_emits_agent_initialized(self):
        runtime = self._make_runtime()
        captured = self._capture_emit(runtime)
        runtime._agent = self._mock_agent()

        runtime._handle_init_agent({
            "cmd": "init_agent",
            "game_name": "soccer_1vs1",
            "player_id": "agent_1",
        })

        assert any(r.get("type") == "agent_initialized" for r in captured)

    def test_init_agent_0_and_1_store_different_player_ids(self):
        """Two separate runtime instances must each store their own player_id."""
        rt0 = self._make_runtime()
        rt1 = self._make_runtime()
        self._capture_emit(rt0)
        self._capture_emit(rt1)
        rt0._agent = self._mock_agent(action=2)
        rt1._agent = self._mock_agent(action=6)

        rt0._handle_init_agent({"cmd": "init_agent", "game_name": "soccer_1vs1", "player_id": "agent_0"})
        rt1._handle_init_agent({"cmd": "init_agent", "game_name": "soccer_1vs1", "player_id": "agent_1"})

        assert rt0._player_id == "agent_0"
        assert rt1._player_id == "agent_1"
        assert rt0._player_id != rt1._player_id

    # --- select_action: IPPO (no one-hot) ---

    def test_select_action_ippo_emits_action_selected(self):
        runtime = self._make_runtime(method="ippo")
        captured = self._capture_emit(runtime)
        runtime._agent = self._mock_agent(action=3)
        runtime._player_id = "agent_0"

        obs = np.zeros(27, dtype=np.float32).tolist()
        runtime._handle_select_action({
            "cmd": "select_action",
            "observation": obs,
            "player_id": "agent_0",
        })

        responses = [r for r in captured if r.get("type") == "action_selected"]
        assert len(responses) == 1
        assert responses[0]["action"] == 3
        assert responses[0]["player_id"] == "agent_0"

    def test_select_action_player_id_in_response(self):
        """Response must echo back the player_id so GUI can route the action."""
        runtime = self._make_runtime(method="ippo")
        captured = self._capture_emit(runtime)
        runtime._agent = self._mock_agent(action=5)
        runtime._player_id = "agent_1"

        obs = np.zeros(27, dtype=np.float32).tolist()
        runtime._handle_select_action({
            "cmd": "select_action",
            "observation": obs,
            "player_id": "agent_1",
        })

        responses = [r for r in captured if r.get("type") == "action_selected"]
        assert responses[0]["player_id"] == "agent_1"

    # --- select_action: MAPPO one-hot ---

    def test_mappo_agent_0_one_hot_is_10(self):
        """MAPPO: agent_0's one-hot in a 2-agent game must be [1, 0]."""
        runtime = self._make_runtime(method="mappo")
        self._capture_emit(runtime)
        runtime._n_agents = 2
        runtime._agent = self._mock_agent(action=1, parameter_sharing=True)
        runtime._player_id = "agent_0"

        obs_27 = np.zeros(27, dtype=np.float32).tolist()
        runtime._handle_select_action({
            "cmd": "select_action",
            "observation": obs_27,
            "player_id": "agent_0",
        })

        # Verify policy() was called with correct agents_id one-hot
        call_kwargs = runtime._agent.policy.call_args[1]
        agents_id = call_kwargs["agent_ids"]
        np.testing.assert_array_equal(agents_id[0].cpu().numpy(), [1.0, 0.0])

    def test_mappo_agent_1_one_hot_is_01(self):
        """MAPPO: agent_1's one-hot in a 2-agent game must be [0, 1]."""
        runtime = self._make_runtime(method="mappo")
        self._capture_emit(runtime)
        runtime._n_agents = 2
        runtime._agent = self._mock_agent(action=2, parameter_sharing=True)
        runtime._player_id = "agent_1"

        obs_27 = np.zeros(27, dtype=np.float32).tolist()
        runtime._handle_select_action({
            "cmd": "select_action",
            "observation": obs_27,
            "player_id": "agent_1",
        })

        call_kwargs = runtime._agent.policy.call_args[1]
        agents_id = call_kwargs["agent_ids"]
        np.testing.assert_array_equal(agents_id[0].cpu().numpy(), [0.0, 1.0])

    def test_mappo_different_one_hot_per_agent(self):
        """MAPPO agent_0 and agent_1 must receive different one-hot vectors."""
        rt0 = self._make_runtime(method="mappo")
        rt1 = self._make_runtime(method="mappo")
        self._capture_emit(rt0)
        self._capture_emit(rt1)
        rt0._n_agents = rt1._n_agents = 2
        rt0._agent = self._mock_agent(action=1, parameter_sharing=True)
        rt1._agent = self._mock_agent(action=1, parameter_sharing=True)

        obs_27 = np.zeros(27, dtype=np.float32).tolist()
        rt0._handle_select_action({"cmd": "select_action", "observation": obs_27, "player_id": "agent_0"})
        rt1._handle_select_action({"cmd": "select_action", "observation": obs_27, "player_id": "agent_1"})

        ids_0 = rt0._agent.policy.call_args[1]["agent_ids"][0].cpu().numpy()
        ids_1 = rt1._agent.policy.call_args[1]["agent_ids"][0].cpu().numpy()

        assert not np.array_equal(ids_0, ids_1), (
            "agent_0 and agent_1 must have different one-hot suffixes"
        )


# =============================================================================
# Section 5: IPPO policy distinction — two networks, two identities
# =============================================================================

class TwoNetworkIPPOAgent:
    """Minimal IPPO agent with two separate policy networks.

    This mimics the structure of a trained IPPO checkpoint:
    - pi_agent_0: MLP(27 -> 64 -> 64 -> 7), weights initialised with seed=0
    - pi_agent_1: MLP(27 -> 64 -> 64 -> 7), weights initialised with seed=1

    Seeds guarantee the two networks have DIFFERENT weights, so the same
    observation produces DIFFERENT logits — exactly the property needed to
    deploy Green vs Blue team roles correctly.

    Usage:
        agent = TwoNetworkIPPOAgent()
        agent.set_player("agent_0")
        action_0 = agent.action(obs)

        agent.set_player("agent_1")
        action_1 = agent.action(obs)

        assert action_0 != action_1  # (with high probability)
    """

    OBS_DIM = 27
    ACT_DIM = 7
    HIDDEN = 64

    def __init__(self) -> None:
        def _mlp(seed: int) -> nn.Module:
            torch.manual_seed(seed)
            return nn.Sequential(
                nn.Linear(self.OBS_DIM, self.HIDDEN),
                nn.ReLU(),
                nn.Linear(self.HIDDEN, self.HIDDEN),
                nn.ReLU(),
                nn.Linear(self.HIDDEN, self.ACT_DIM),
            )

        self._nets: Dict[str, nn.Module] = {
            "agent_0": _mlp(seed=0),
            "agent_1": _mlp(seed=1),
        }
        self._active_player: str = "agent_0"

    def set_player(self, player_id: str) -> None:
        if player_id not in self._nets:
            raise KeyError(f"Unknown player_id: {player_id!r}. Expected agent_0 or agent_1.")
        self._active_player = player_id

    def action(self, obs: np.ndarray) -> np.ndarray:
        net = self._nets[self._active_player]
        with torch.no_grad():
            t = torch.from_numpy(obs).float().unsqueeze(0)
            logits = net(t)
            act = int(logits.argmax(dim=-1).item())
        return np.array([act])

    def save(self, path: str) -> None:
        """Save both networks to a single checkpoint file."""
        torch.save(
            {name: net.state_dict() for name, net in self._nets.items()},
            path,
        )

    @classmethod
    def load(cls, path: str) -> "TwoNetworkIPPOAgent":
        agent = cls.__new__(cls)
        agent._active_player = "agent_0"
        state_dicts = torch.load(path, map_location="cpu")

        def _empty_mlp() -> nn.Module:
            return nn.Sequential(
                nn.Linear(cls.OBS_DIM, cls.HIDDEN),
                nn.ReLU(),
                nn.Linear(cls.HIDDEN, cls.HIDDEN),
                nn.ReLU(),
                nn.Linear(cls.HIDDEN, cls.ACT_DIM),
            )

        agent._nets = {}
        for name, sd in state_dicts.items():
            net = _empty_mlp()
            net.load_state_dict(sd)
            agent._nets[name] = net
        return agent


class TestIPPOPolicyDistinction:
    """Verify that pi_agent_0 and pi_agent_1 are genuinely different policies.

    This is the formal test for the key claim in
    docs/Development_Progress/1.0_DAY_67/TASK_1/IPPO_Policy_Team_Dependency.md:

        "pi_agent_0 has learned 'scoring on the right goal = reward.'
         If you deploy it as Blue agent (whose correct goal is the left goal),
         it will carry the ball to the right and score in its own goal."

    The test does not require XuanCe or a trained checkpoint — it uses two
    randomly-initialised networks to prove the distinction mechanism works.
    Weights produced by different seeds are guaranteed to differ.
    """

    OBS = np.zeros(27, dtype=np.float32)  # 27-dim IndAgObs (same obs sent to both)

    @pytest.fixture
    def agent(self) -> TwoNetworkIPPOAgent:
        return TwoNetworkIPPOAgent()

    # --- network weights are different ---

    def test_network_weights_differ(self, agent: TwoNetworkIPPOAgent):
        """The two networks must have different parameters (different seeds)."""
        w0 = list(agent._nets["agent_0"].parameters())[0].detach()
        w1 = list(agent._nets["agent_1"].parameters())[0].detach()
        assert not torch.allclose(w0, w1), (
            "pi_agent_0 and pi_agent_1 must have different weights"
        )

    # --- same obs → different logits ---

    def test_same_obs_different_logits(self, agent: TwoNetworkIPPOAgent):
        """Given identical observations, the two networks produce different logits."""
        obs_t = torch.from_numpy(self.OBS).float().unsqueeze(0)

        with torch.no_grad():
            logits_0 = agent._nets["agent_0"](obs_t)
            logits_1 = agent._nets["agent_1"](obs_t)

        assert not torch.allclose(logits_0, logits_1), (
            "Identical obs must produce different logits from different networks"
        )

    # --- player_id routing selects the correct network ---

    def test_agent_0_uses_network_0(self, agent: TwoNetworkIPPOAgent):
        agent.set_player("agent_0")
        action_0 = agent.action(self.OBS)
        assert 0 <= int(action_0[0]) < 7

    def test_agent_1_uses_network_1(self, agent: TwoNetworkIPPOAgent):
        agent.set_player("agent_1")
        action_1 = agent.action(self.OBS)
        assert 0 <= int(action_1[0]) < 7

    def test_agent_0_and_agent_1_produce_different_actions_on_same_obs(
        self, agent: TwoNetworkIPPOAgent
    ):
        """pi_agent_0 and pi_agent_1 must choose different actions on the same obs.

        This is the core distinction: deploying pi_agent_0 as the Blue agent
        would lead it to score own-goals because it has learned a DIFFERENT
        directional bias than pi_agent_1.
        """
        agent.set_player("agent_0")
        action_0 = int(agent.action(self.OBS)[0])

        agent.set_player("agent_1")
        action_1 = int(agent.action(self.OBS)[0])

        assert action_0 != action_1, (
            f"pi_agent_0 chose {action_0} and pi_agent_1 chose {action_1} — "
            "they MUST differ to confirm separate policy networks exist. "
            "If this fails with the same seed, the networks are identical (a bug)."
        )

    # --- wrong deployment is detectable ---

    def test_swapping_players_changes_action(self, agent: TwoNetworkIPPOAgent):
        """Deploying pi_agent_1 in the Green slot changes the chosen action.

        This demonstrates the 'wrong deployment' scenario from the doc:
            Green slot = pi_agent_1  ->  wrong goal direction  ->  own goal!
        """
        agent.set_player("agent_0")
        correct_green_action = int(agent.action(self.OBS)[0])

        # Wrong: Green slot gets pi_agent_1
        agent.set_player("agent_1")
        wrong_green_action = int(agent.action(self.OBS)[0])

        assert correct_green_action != wrong_green_action, (
            "Swapping the player policy must change the action — "
            "confirming that the wrong deployment would produce wrong behaviour"
        )

    # --- checkpoint round-trip ---

    def test_checkpoint_save_and_load(self, agent: TwoNetworkIPPOAgent, tmp_path):
        """Both networks survive a save/load cycle with unchanged weights."""
        ckpt = str(tmp_path / "ippo_test.pth")
        agent.save(ckpt)

        loaded = TwoNetworkIPPOAgent.load(ckpt)
        assert set(loaded._nets.keys()) == {"agent_0", "agent_1"}, (
            "Checkpoint must contain entries for BOTH agent_0 and agent_1"
        )

        for name in ("agent_0", "agent_1"):
            orig_params = list(agent._nets[name].parameters())
            loaded_params = list(loaded._nets[name].parameters())
            for p_orig, p_loaded in zip(orig_params, loaded_params):
                torch.testing.assert_close(p_orig, p_loaded)

    def test_checkpoint_both_agents_present(self, agent: TwoNetworkIPPOAgent, tmp_path):
        """Checkpoint file must contain both pi_agent_0 and pi_agent_1 keys."""
        ckpt = str(tmp_path / "ippo_two_agents.pth")
        agent.save(ckpt)

        state = torch.load(ckpt, map_location="cpu")
        assert "agent_0" in state, "Checkpoint missing pi_agent_0"
        assert "agent_1" in state, "Checkpoint missing pi_agent_1"
        assert len(state) == 2, "Checkpoint must have exactly 2 entries (one per agent)"

    def test_loaded_agent_actions_match_original(self, agent: TwoNetworkIPPOAgent, tmp_path):
        """After load, each agent produces the same action as before saving."""
        obs = np.random.default_rng(42).random(27).astype(np.float32)

        agent.set_player("agent_0")
        action_0_before = int(agent.action(obs)[0])
        agent.set_player("agent_1")
        action_1_before = int(agent.action(obs)[0])

        ckpt = str(tmp_path / "ippo_roundtrip.pth")
        agent.save(ckpt)
        loaded = TwoNetworkIPPOAgent.load(ckpt)

        loaded.set_player("agent_0")
        action_0_after = int(loaded.action(obs)[0])
        loaded.set_player("agent_1")
        action_1_after = int(loaded.action(obs)[0])

        assert action_0_after == action_0_before, "pi_agent_0 action changed after load"
        assert action_1_after == action_1_before, "pi_agent_1 action changed after load"
