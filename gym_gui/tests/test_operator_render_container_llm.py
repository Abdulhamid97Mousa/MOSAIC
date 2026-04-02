"""Tests for LLM conversation tracking in OperatorRenderContainer.

Tests cover:
- LLM conversation history tracking
- System prompt extraction from payloads
- Dialog creation and display
- Conversation reset on new episode
- Button visibility based on operator type
"""

import os

# Ensure Qt renders offscreen in CI environments
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from unittest.mock import patch

import pytest
from qtpy import QtWidgets

from gym_gui.services.operator import OperatorConfig
from gym_gui.ui.widgets.operator_render_container import OperatorRenderContainer


# Ensure QApplication exists for all widget tests
@pytest.fixture(scope="module")
def qapp():
    """Ensure QApplication exists for widget tests."""
    app = QtWidgets.QApplication.instance()
    if app is None:
        # Use offscreen platform for headless CI environments
        app = QtWidgets.QApplication(["-platform", "offscreen"])
    return app


@pytest.fixture
def llm_operator_config():
    """Create an LLM operator config for testing."""
    return OperatorConfig.single_agent(
        operator_id="op_llm_test",
        display_name="Test LLM Operator",
        worker_id="balrog_worker",
        worker_type="llm",
        env_name="minigrid",
        task="MiniGrid-Empty-8x8-v0",
        settings={
            "client_name": "vllm",
            "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        },
    )


@pytest.fixture
def rl_operator_config():
    """Create an RL operator config for testing."""
    return OperatorConfig.single_agent(
        operator_id="op_rl_test",
        display_name="Test RL Operator",
        worker_id="cleanrl_worker",
        worker_type="rl",
        env_name="CartPole-v1",
        task="CartPole-v1",
    )


@pytest.fixture
def human_operator_config():
    """Create a Human operator config for testing."""
    return OperatorConfig.single_agent(
        operator_id="op_human_test",
        display_name="Test Human Operator",
        worker_id="human_worker",
        worker_type="human",
        env_name="FrozenLake-v1",
        task="FrozenLake-v1",
    )


@pytest.fixture
def llm_render_container(qapp, llm_operator_config):
    """Create an LLM OperatorRenderContainer for testing."""
    container = OperatorRenderContainer(llm_operator_config)
    yield container
    container.cleanup()
    container.deleteLater()


@pytest.fixture
def rl_render_container(qapp, rl_operator_config):
    """Create an RL OperatorRenderContainer for testing."""
    container = OperatorRenderContainer(rl_operator_config)
    yield container
    container.cleanup()
    container.deleteLater()


class TestLLMConversationTracking:
    """Test LLM conversation history tracking."""

    def test_initial_conversation_empty(self, llm_render_container):
        """Conversation history should be empty initially."""
        assert llm_render_container._conversation_history == []
        assert llm_render_container._system_prompt == ""

    def test_update_llm_data_adds_observation(self, llm_render_container):
        """update_llm_data should add observation to conversation history."""
        llm_render_container.update_llm_data(observation="You see a red ball.")

        assert len(llm_render_container._conversation_history) == 1
        assert llm_render_container._conversation_history[0]["role"] == "user"
        assert llm_render_container._conversation_history[0]["content"] == "You see a red ball."

    def test_update_llm_data_adds_action(self, llm_render_container):
        """update_llm_data should add action to conversation history."""
        llm_render_container.update_llm_data(action="go forward")

        assert len(llm_render_container._conversation_history) == 1
        assert llm_render_container._conversation_history[0]["role"] == "assistant"
        assert llm_render_container._conversation_history[0]["content"] == "go forward"

    def test_update_llm_data_full_exchange(self, llm_render_container):
        """update_llm_data should track a full observation-action exchange."""
        llm_render_container.update_llm_data(
            observation="Current Observation: nothing special.",
            action="turn left"
        )

        assert len(llm_render_container._conversation_history) == 2
        assert llm_render_container._conversation_history[0]["role"] == "user"
        assert llm_render_container._conversation_history[1]["role"] == "assistant"

    def test_system_prompt_extraction(self, llm_render_container):
        """System prompt should be extracted from first observation."""
        system_prompt = "You are an agent playing a navigation game. PLAY!"
        llm_render_container.update_llm_data(system_prompt=system_prompt)

        assert llm_render_container._system_prompt == system_prompt

    def test_system_prompt_set_once(self, llm_render_container):
        """System prompt should only be set once (first time)."""
        llm_render_container.update_llm_data(system_prompt="First prompt")
        llm_render_container.update_llm_data(system_prompt="Second prompt")

        assert llm_render_container._system_prompt == "First prompt"

    def test_multi_step_conversation(self, llm_render_container):
        """Conversation should accumulate across multiple steps."""
        # Step 1
        llm_render_container.update_llm_data(
            observation="Observation: nothing special.",
            action="turn left"
        )
        # Step 2
        llm_render_container.update_llm_data(
            observation="Observation: you see a wall.",
            action="turn right"
        )
        # Step 3
        llm_render_container.update_llm_data(
            observation="Observation: green goal ahead.",
            action="go forward"
        )

        assert len(llm_render_container._conversation_history) == 6
        # Check alternating pattern
        roles = [msg["role"] for msg in llm_render_container._conversation_history]
        assert roles == ["user", "assistant", "user", "assistant", "user", "assistant"]


class TestConversationFromPayload:
    """Test conversation extraction from telemetry payloads."""

    def test_payload_extracts_observation_and_action(self, llm_render_container):
        """Payload with observation and action should update conversation."""
        payload = {
            "episode_index": 0,
            "step_index": 0,
            "observation": "You see a red door.",
            "action": "toggle",
            "reward": 0.0,
        }
        llm_render_container._update_stats_from_payload(payload)

        assert len(llm_render_container._conversation_history) == 2

    def test_payload_extracts_llm_response(self, llm_render_container):
        """Payload with llm_response field should be captured."""
        payload = {
            "episode_index": 0,
            "step_index": 0,
            "observation": "Maze entrance.",
            "llm_response": "go forward",
            "reward": 0.0,
        }
        llm_render_container._update_stats_from_payload(payload)

        # Check action was captured (llm_response fallback)
        actions = [m for m in llm_render_container._conversation_history if m["role"] == "assistant"]
        assert len(actions) == 1
        assert actions[0]["content"] == "go forward"

    def test_system_prompt_detected_from_observation(self, llm_render_container):
        """System prompt markers in observation should be detected."""
        payload = {
            "episode_index": 0,
            "step_index": 0,
            "observation": "You are an agent playing a game. Your goal is to reach the goal. PLAY!",
            "action": "turn left",
            "reward": 0.0,
        }
        llm_render_container._update_stats_from_payload(payload)

        # Should extract system prompt from observation containing markers
        assert "You are an agent" in llm_render_container._system_prompt

    def test_conversation_clears_on_new_episode(self, llm_render_container):
        """Conversation should clear when episode changes."""
        # Episode 0, Step 0
        payload1 = {
            "episode_index": 0,
            "step_index": 0,
            "observation": "Start of episode 0",
            "action": "turn left",
        }
        llm_render_container._update_stats_from_payload(payload1)
        assert len(llm_render_container._conversation_history) == 2

        # Episode 1, Step 0 (new episode)
        payload2 = {
            "episode_index": 1,
            "step_index": 0,
            "observation": "Start of episode 1",
            "action": "go forward",
        }
        llm_render_container._update_stats_from_payload(payload2)

        # Should have cleared and added new conversation
        assert len(llm_render_container._conversation_history) == 2
        assert "episode 1" in llm_render_container._conversation_history[0]["content"]


class TestButtonVisibility:
    """Test LLM button visibility based on operator type.

    Note: We use isHidden() instead of isVisible() because isVisible() returns
    False for widgets whose parent hasn't been shown yet. isHidden() checks
    the explicit visibility flag set by setVisible().
    """

    def test_llm_operator_shows_buttons(self, llm_render_container):
        """LLM operators should show Prompt and Chat buttons."""
        # isHidden() returns False if setVisible(True) was called
        assert not llm_render_container._prompt_btn.isHidden()
        assert not llm_render_container._chat_btn.isHidden()

    def test_rl_operator_hides_buttons(self, rl_render_container):
        """RL operators should hide Prompt and Chat buttons."""
        # isHidden() returns True if setVisible(False) was called
        assert rl_render_container._prompt_btn.isHidden()
        assert rl_render_container._chat_btn.isHidden()

    def test_button_visibility_updates_on_config_change(self, qapp, rl_render_container, llm_operator_config):
        """Button visibility should update when config changes to LLM type."""
        # Start as RL (buttons hidden)
        assert rl_render_container._prompt_btn.isHidden()

        # Change to LLM config
        rl_render_container.set_config(llm_operator_config)

        # Buttons should now be visible (not hidden)
        assert not rl_render_container._prompt_btn.isHidden()
        assert not rl_render_container._chat_btn.isHidden()


class TestDialogCreation:
    """Test dialog creation for Prompt and Chat buttons."""

    def test_prompt_dialog_shows_system_prompt(self, llm_render_container):
        """Prompt dialog should display the system prompt."""
        llm_render_container._system_prompt = "You are a navigation agent."

        # Mock QDialog.exec to avoid blocking
        with patch.object(QtWidgets.QDialog, 'exec', return_value=None):
            llm_render_container._show_prompt_dialog()

    def test_chat_dialog_shows_conversation(self, llm_render_container):
        """Chat dialog should display conversation history."""
        llm_render_container.update_llm_data(
            observation="Test observation",
            action="test action"
        )

        # Mock QDialog.exec to avoid blocking
        with patch.object(QtWidgets.QDialog, 'exec', return_value=None):
            llm_render_container._show_chat_dialog()

    def test_prompt_dialog_empty_state(self, llm_render_container):
        """Prompt dialog should show placeholder when no prompt captured."""
        # No system prompt set
        assert llm_render_container._system_prompt == ""

        # Mock QDialog.exec to avoid blocking
        with patch.object(QtWidgets.QDialog, 'exec', return_value=None):
            llm_render_container._show_prompt_dialog()


class TestResetBehavior:
    """Test reset and cleanup behavior."""

    def test_reset_stats_clears_conversation(self, llm_render_container):
        """reset_stats should clear conversation history."""
        llm_render_container.update_llm_data(
            observation="Test",
            action="test"
        )
        assert len(llm_render_container._conversation_history) > 0

        llm_render_container.reset_stats()

        assert len(llm_render_container._conversation_history) == 0

    def test_reset_preserves_system_prompt(self, llm_render_container):
        """reset_stats should preserve system prompt (it's episode-independent)."""
        llm_render_container._system_prompt = "Game instructions"
        llm_render_container.update_llm_data(observation="Step data", action="action")

        llm_render_container.reset_stats()

        # System prompt should remain
        assert llm_render_container._system_prompt == "Game instructions"
        # Conversation should be cleared
        assert len(llm_render_container._conversation_history) == 0


class TestOperatorIdentification:
    """Test operator identification for correct container targeting."""

    def test_operator_id_accessible(self, llm_render_container):
        """operator_id should be accessible from container."""
        assert llm_render_container.operator_id == "op_llm_test"
        assert llm_render_container.config.operator_id == "op_llm_test"

    def test_config_property_returns_current_config(self, llm_render_container, llm_operator_config):
        """config property should return the current OperatorConfig."""
        assert llm_render_container.config == llm_operator_config

    def test_different_operators_have_different_ids(self, qapp, llm_operator_config, rl_operator_config):
        """Different operators should have distinct IDs."""
        llm_container = OperatorRenderContainer(llm_operator_config)
        rl_container = OperatorRenderContainer(rl_operator_config)

        try:
            assert llm_container.operator_id != rl_container.operator_id
            assert llm_container.operator_id == "op_llm_test"
            assert rl_container.operator_id == "op_rl_test"
        finally:
            llm_container.cleanup()
            rl_container.cleanup()
            llm_container.deleteLater()
            rl_container.deleteLater()


class TestVLMOperator:
    """Test VLM operator type (should behave like LLM)."""

    @pytest.fixture
    def vlm_operator_config(self):
        """Create a VLM operator config for testing."""
        return OperatorConfig.single_agent(
            operator_id="op_vlm_test",
            display_name="Test VLM Operator",
            worker_id="balrog_worker",
            worker_type="vlm",
            env_name="minigrid",
            task="MiniGrid-Empty-8x8-v0",
            settings={
                "client_name": "openrouter",
                "model_id": "openai/gpt-4o-mini",
                "max_image_history": 3,
            },
        )

    def test_vlm_shows_buttons(self, qapp, vlm_operator_config):
        """VLM operators should also show Prompt and Chat buttons."""
        container = OperatorRenderContainer(vlm_operator_config)

        try:
            # Use isHidden() since container isn't shown on screen
            assert not container._prompt_btn.isHidden()
            assert not container._chat_btn.isHidden()
        finally:
            container.cleanup()
            container.deleteLater()

    def test_vlm_conversation_tracking(self, qapp, vlm_operator_config):
        """VLM should track conversation like LLM."""
        container = OperatorRenderContainer(vlm_operator_config)

        try:
            container.update_llm_data(
                observation="Image observation",
                action="describe image"
            )

            assert len(container._conversation_history) == 2
        finally:
            container.cleanup()
            container.deleteLater()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
