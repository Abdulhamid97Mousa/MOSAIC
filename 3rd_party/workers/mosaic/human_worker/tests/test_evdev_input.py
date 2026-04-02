"""Tests for pure-Python evdev input and keycode-to-action resolution.

Tests cover:
1. Linux keycode constants and directional sets
2. All keycode resolvers (MultiGrid, INI MultiGrid, MiniGrid, MeltingPot)
3. Resolver factory (create_resolver)
4. EvdevDeviceReader (mocked os.read with struct-packed events)
5. HumanKeyboardRuntime stdin/stdout protocol
6. Keyboard discovery (mocked /dev/input)
"""

from __future__ import annotations

import json
import os
import struct
from io import StringIO
from typing import Set
from unittest.mock import MagicMock, mock_open, patch

import pytest

from human_worker.evdev_input import (
    EV_KEY,
    EV_VALUE_PRESS,
    EV_VALUE_RELEASE,
    KEY_A,
    KEY_D,
    KEY_DOWN,
    KEY_E,
    KEY_F,
    KEY_G,
    KEY_H,
    KEY_LEFT,
    KEY_Q,
    KEY_RETURN,
    KEY_RIGHT,
    KEY_S,
    KEY_SPACE,
    KEY_UP,
    KEY_W,
    KEY_1,
    KEY_2,
    KEY_3,
    KEYS_DOWN,
    KEYS_LEFT,
    KEYS_RIGHT,
    KEYS_UP,
    EvdevDeviceReader,
    INIMultiGridKeycodeResolver,
    KeyboardDeviceInfo,
    LinuxKeycodeActionResolver,
    MalmoKeycodeResolver,
    MeltingPotKeycodeResolver,
    MiniGridKeycodeResolver,
    MultiGridKeycodeResolver,
    create_resolver,
    discover_keyboards,
)


# ============================================================================
# 1. Keycode Constants
# ============================================================================

class TestKeycodeConstants:
    def test_wasd_keycodes_match_kernel(self):
        assert KEY_W == 17
        assert KEY_A == 30
        assert KEY_S == 31
        assert KEY_D == 32

    def test_arrow_keycodes_match_kernel(self):
        assert KEY_UP == 103
        assert KEY_DOWN == 108
        assert KEY_LEFT == 105
        assert KEY_RIGHT == 106

    def test_directional_sets(self):
        assert KEYS_UP == {KEY_UP, KEY_W}
        assert KEYS_DOWN == {KEY_DOWN, KEY_S}
        assert KEYS_LEFT == {KEY_LEFT, KEY_A}
        assert KEYS_RIGHT == {KEY_RIGHT, KEY_D}


# ============================================================================
# 2. MultiGrid Resolver
# ============================================================================

class TestMultiGridKeycodeResolver:
    @pytest.fixture
    def resolver(self):
        return MultiGridKeycodeResolver()

    def test_empty_returns_none(self, resolver):
        assert resolver.resolve(set()) is None

    def test_w_forward(self, resolver):
        assert resolver.resolve({KEY_W}) == 3

    def test_up_forward(self, resolver):
        assert resolver.resolve({KEY_UP}) == 3

    def test_a_left(self, resolver):
        assert resolver.resolve({KEY_A}) == 1

    def test_d_right(self, resolver):
        assert resolver.resolve({KEY_D}) == 2

    def test_s_still(self, resolver):
        assert resolver.resolve({KEY_S}) == 0

    def test_space_pickup(self, resolver):
        assert resolver.resolve({KEY_SPACE}) == 4

    def test_g_pickup(self, resolver):
        assert resolver.resolve({KEY_G}) == 4

    def test_h_drop(self, resolver):
        assert resolver.resolve({KEY_H}) == 5

    def test_e_toggle(self, resolver):
        assert resolver.resolve({KEY_E}) == 6

    def test_return_toggle(self, resolver):
        assert resolver.resolve({KEY_RETURN}) == 6

    def test_q_done(self, resolver):
        assert resolver.resolve({KEY_Q}) == 7

    def test_action_priority_over_movement(self, resolver):
        # Space (PICKUP) should win over W (FORWARD)
        assert resolver.resolve({KEY_SPACE, KEY_W}) == 4


# ============================================================================
# 3. INI MultiGrid Resolver
# ============================================================================

class TestINIMultiGridKeycodeResolver:
    @pytest.fixture
    def resolver(self):
        return INIMultiGridKeycodeResolver()

    def test_empty_returns_none(self, resolver):
        assert resolver.resolve(set()) is None

    def test_w_forward(self, resolver):
        assert resolver.resolve({KEY_W}) == 2

    def test_a_left(self, resolver):
        assert resolver.resolve({KEY_A}) == 0

    def test_d_right(self, resolver):
        assert resolver.resolve({KEY_D}) == 1

    def test_s_returns_none(self, resolver):
        """INI MultiGrid has no backward action."""
        assert resolver.resolve({KEY_S}) is None

    def test_space_pickup(self, resolver):
        assert resolver.resolve({KEY_SPACE}) == 3

    def test_q_done(self, resolver):
        assert resolver.resolve({KEY_Q}) == 6


# ============================================================================
# 4. MiniGrid Resolver
# ============================================================================

class TestMiniGridKeycodeResolver:
    @pytest.fixture
    def resolver(self):
        return MiniGridKeycodeResolver()

    def test_w_forward(self, resolver):
        assert resolver.resolve({KEY_W}) == 2

    def test_a_left(self, resolver):
        assert resolver.resolve({KEY_A}) == 0

    def test_d_right(self, resolver):
        assert resolver.resolve({KEY_D}) == 1


# ============================================================================
# 5. MeltingPot Resolver
# ============================================================================

class TestMeltingPotKeycodeResolver:
    @pytest.fixture
    def resolver(self):
        return MeltingPotKeycodeResolver()

    def test_empty_returns_none(self, resolver):
        assert resolver.resolve(set()) is None

    def test_w_forward(self, resolver):
        assert resolver.resolve({KEY_W}) == 1

    def test_s_backward(self, resolver):
        assert resolver.resolve({KEY_S}) == 2

    def test_a_turn_left(self, resolver):
        assert resolver.resolve({KEY_A}) == 5

    def test_d_turn_right(self, resolver):
        assert resolver.resolve({KEY_D}) == 6

    def test_space_interact(self, resolver):
        assert resolver.resolve({KEY_SPACE}) == 7

    def test_1_fire1(self, resolver):
        assert resolver.resolve({KEY_1}) == 8

    def test_2_fire2(self, resolver):
        assert resolver.resolve({KEY_2}) == 9

    def test_3_fire3(self, resolver):
        assert resolver.resolve({KEY_3}) == 10

    def test_q_strafe_left(self, resolver):
        assert resolver.resolve({KEY_Q}) == 3

    def test_f_strafe_right(self, resolver):
        assert resolver.resolve({KEY_F}) == 4


# ============================================================================
# 5b. Malmo Resolver
# ============================================================================

class TestMalmoKeycodeResolver:
    """Test Malmo resolver with a typical TreasureHunt action space."""

    TREASURE_HUNT_ACTIONS = [
        "move 1", "move -1", "turn 1", "turn -1",
        "jump 1", "attack 1", "use 1", "noop 1",
    ]

    @pytest.fixture
    def resolver(self):
        return MalmoKeycodeResolver(self.TREASURE_HUNT_ACTIONS)

    def test_empty_returns_none(self, resolver):
        assert resolver.resolve(set()) is None

    def test_w_move_forward(self, resolver):
        assert resolver.resolve({KEY_W}) == 0  # "move 1" is index 0

    def test_up_move_forward(self, resolver):
        assert resolver.resolve({KEY_UP}) == 0

    def test_s_move_backward(self, resolver):
        assert resolver.resolve({KEY_S}) == 1  # "move -1" is index 1

    def test_d_turn_right(self, resolver):
        assert resolver.resolve({KEY_D}) == 2  # "turn 1" is index 2

    def test_a_turn_left(self, resolver):
        assert resolver.resolve({KEY_A}) == 3  # "turn -1" is index 3

    def test_space_jump(self, resolver):
        assert resolver.resolve({KEY_SPACE}) == 4  # "jump 1" is index 4

    def test_return_attack(self, resolver):
        assert resolver.resolve({KEY_RETURN}) == 5  # "attack 1" is index 5

    def test_e_use(self, resolver):
        assert resolver.resolve({KEY_E}) == 6  # "use 1" is index 6

    def test_f_use(self, resolver):
        assert resolver.resolve({KEY_F}) == 6  # "use 1" is index 6

    def test_attack_priority_over_movement(self, resolver):
        """Attack should take priority over movement when both pressed."""
        assert resolver.resolve({KEY_RETURN, KEY_W}) == 5  # attack

    def test_jump_priority_over_movement(self, resolver):
        assert resolver.resolve({KEY_SPACE, KEY_W}) == 4  # jump

    def test_missing_action_returns_none(self):
        """If the mission doesn't have 'jump 1', pressing Space returns None."""
        resolver = MalmoKeycodeResolver(["move 1", "move -1", "turn 1", "turn -1"])
        assert resolver.resolve({KEY_SPACE}) is None


# ============================================================================
# 6. Resolver Factory
# ============================================================================

class TestResolverFactory:
    def test_multigrid(self):
        assert isinstance(create_resolver("multigrid"), MultiGridKeycodeResolver)

    def test_mosaic_multigrid(self):
        assert isinstance(create_resolver("mosaic_multigrid"), MultiGridKeycodeResolver)

    def test_ini_multigrid(self):
        assert isinstance(create_resolver("ini_multigrid"), INIMultiGridKeycodeResolver)

    def test_minigrid(self):
        assert isinstance(create_resolver("minigrid"), MiniGridKeycodeResolver)

    def test_babyai(self):
        assert isinstance(create_resolver("babyai"), MiniGridKeycodeResolver)

    def test_meltingpot(self):
        assert isinstance(create_resolver("meltingpot"), MeltingPotKeycodeResolver)

    def test_malmo(self):
        actions = ["move 1", "move -1", "turn 1", "turn -1"]
        resolver = create_resolver("malmo", action_list=actions)
        assert isinstance(resolver, MalmoKeycodeResolver)

    def test_mosaic_malmo(self):
        actions = ["move 1", "move -1", "turn 1", "turn -1"]
        resolver = create_resolver("mosaic_malmo", action_list=actions)
        assert isinstance(resolver, MalmoKeycodeResolver)

    def test_malmo_without_action_list_raises(self):
        with pytest.raises(ValueError, match="requires action_list"):
            create_resolver("malmo")

    def test_unknown_defaults_to_multigrid(self):
        resolver = create_resolver("unknown_env")
        assert isinstance(resolver, MultiGridKeycodeResolver)


# ============================================================================
# 7. EvdevDeviceReader (mocked I/O)
# ============================================================================

def _pack_event(ev_type: int, ev_code: int, ev_value: int) -> bytes:
    """Pack a struct input_event (24 bytes on x86_64)."""
    return struct.pack("llHHi", 0, 0, ev_type, ev_code, ev_value)


class TestEvdevDeviceReader:
    def test_open_close(self):
        reader = EvdevDeviceReader("/dev/input/event99")
        assert not reader.is_open

        with patch("os.open", return_value=42):
            reader.open()
            assert reader.is_open

        with patch("os.close"):
            reader.close()
            assert not reader.is_open

    def test_wait_for_key_press_returns_keycode(self):
        reader = EvdevDeviceReader("/dev/input/event99")
        reader._fd = 42

        event_data = _pack_event(EV_KEY, KEY_W, EV_VALUE_PRESS)

        with patch("select.select", return_value=([42], [], [])):
            with patch("os.read", return_value=event_data):
                keycode = reader.wait_for_key_press(timeout=0.1)

        assert keycode == KEY_W

    def test_wait_for_key_press_timeout(self):
        reader = EvdevDeviceReader("/dev/input/event99")
        reader._fd = 42

        with patch("select.select", return_value=([], [], [])):
            keycode = reader.wait_for_key_press(timeout=0.01)

        assert keycode is None

    def test_wait_for_key_press_no_fd(self):
        reader = EvdevDeviceReader("/dev/input/event99")
        assert reader.wait_for_key_press() is None

    def test_read_all_pressed(self):
        reader = EvdevDeviceReader("/dev/input/event99")
        reader._fd = 42

        # Two key presses, one key release
        events = (
            _pack_event(EV_KEY, KEY_W, EV_VALUE_PRESS)
            + _pack_event(EV_KEY, KEY_D, EV_VALUE_PRESS)
            + _pack_event(EV_KEY, KEY_W, EV_VALUE_RELEASE)
        )

        call_count = 0

        def mock_read(fd, size):
            nonlocal call_count
            offset = call_count * 24
            call_count += 1
            if offset + 24 <= len(events):
                return events[offset : offset + 24]
            raise BlockingIOError

        with patch("select.select", return_value=([42], [], [])):
            with patch("os.read", side_effect=mock_read):
                pressed = reader.read_all_pressed(timeout=0.0)

        # W was pressed then released, D is still pressed
        assert pressed == {KEY_D}

    def test_read_ignores_non_key_events(self):
        reader = EvdevDeviceReader("/dev/input/event99")
        reader._fd = 42

        # EV_SYN event (type 0) followed by a key press
        events = (
            _pack_event(0, 0, 0)  # EV_SYN
            + _pack_event(EV_KEY, KEY_A, EV_VALUE_PRESS)
        )

        call_count = 0

        def mock_read(fd, size):
            nonlocal call_count
            offset = call_count * 24
            call_count += 1
            if offset + 24 <= len(events):
                return events[offset : offset + 24]
            raise BlockingIOError

        with patch("select.select", return_value=([42], [], [])):
            with patch("os.read", side_effect=mock_read):
                pressed = reader.read_all_pressed(timeout=0.0)

        assert pressed == {KEY_A}


# ============================================================================
# 8. Keyboard Discovery (mocked filesystem)
# ============================================================================

class TestDiscoverKeyboards:
    @patch("human_worker.evdev_input.glob.glob")
    @patch("os.path.realpath")
    @patch(
        "builtins.open",
        mock_open(
            read_data=(
                'I: Bus=0003 Vendor=046d Product=c31c Version=0110\n'
                'N: Name="Logitech USB Keyboard"\n'
                'H: Handlers=sysrq kbd event16\n'
                '\n'
            )
        ),
    )
    def test_discovers_keyboard(self, mock_realpath, mock_glob):
        mock_glob.side_effect = [
            ["/dev/input/by-path/pci-0000:00:14.0-usb-0:5.1:1.0-event-kbd"],
            [],  # by-id returns nothing
        ]
        mock_realpath.return_value = "/dev/input/event16"

        keyboards = discover_keyboards()

        assert len(keyboards) == 1
        kbd = keyboards[0]
        assert kbd.device_path == "/dev/input/by-path/pci-0000:00:14.0-usb-0:5.1:1.0-event-kbd"
        assert kbd.event_path == "/dev/input/event16"
        assert kbd.name == "Logitech USB Keyboard"

    @patch("human_worker.evdev_input.glob.glob")
    @patch("os.path.realpath")
    def test_deduplicates_by_real_path(self, mock_realpath, mock_glob):
        mock_glob.side_effect = [
            ["/dev/input/by-path/kbd1"],
            ["/dev/input/by-id/kbd1"],  # same device via different symlink
        ]
        mock_realpath.return_value = "/dev/input/event16"

        with patch("builtins.open", mock_open(read_data="N: Name=\"Test\"\nH: Handlers=event16\n\n")):
            keyboards = discover_keyboards()

        assert len(keyboards) == 1


# ============================================================================
# 9. HumanKeyboardRuntime Protocol (mocked)
# ============================================================================

class TestHumanKeyboardRuntimeProtocol:
    """Test the stdin/stdout JSON protocol of HumanKeyboardRuntime."""

    def test_init_emits_message(self):
        from human_worker.config import HumanWorkerConfig
        from human_worker.runtime import HumanKeyboardRuntime

        config = HumanWorkerConfig(
            run_id="test_run",
            player_name="Player 1",
            env_name="multigrid",
        )
        runtime = HumanKeyboardRuntime(config, device_path="/dev/input/event99")

        # Capture stdout
        emitted = []
        runtime._emit = lambda data: emitted.append(data)

        # Simulate init_agent command
        with patch("human_worker.evdev_input.EvdevDeviceReader") as MockReader:
            mock_reader_instance = MockReader.return_value
            runtime._handle_init_agent({
                "cmd": "init_agent",
                "game_name": "MultiGrid-Soccer-v0",
                "player_id": "agent_0",
            })

        assert len(emitted) == 1
        assert emitted[0]["type"] == "agent_initialized"
        assert emitted[0]["player_id"] == "agent_0"
        assert emitted[0]["device_path"] == "/dev/input/event99"

    def test_select_action_returns_resolved_action(self):
        from human_worker.config import HumanWorkerConfig
        from human_worker.runtime import HumanKeyboardRuntime

        config = HumanWorkerConfig(
            run_id="test_run",
            player_name="Player 1",
            env_name="multigrid",
        )
        runtime = HumanKeyboardRuntime(config, device_path="/dev/input/event99")

        # Setup mocks
        emitted = []
        runtime._emit = lambda data: emitted.append(data)

        mock_reader = MagicMock()
        mock_reader._fd = 42  # Fake fd for select()
        # drain_key_events returns list of (keycode, pressed) tuples
        mock_reader.drain_key_events.return_value = [(KEY_W, True)]
        runtime._reader = mock_reader

        mock_resolver = MagicMock()
        mock_resolver.resolve.return_value = 3  # FORWARD
        runtime._resolver = mock_resolver

        # select.select is called twice per loop iteration:
        # 1. stdin check: return nothing readable
        # 2. keyboard+mouse fd check: return keyboard fd as readable
        call_count = [0]
        def fake_select(rlist, wlist, xlist, timeout=None):
            call_count[0] += 1
            if call_count[0] % 2 == 1:
                # stdin check
                return ([], [], [])
            else:
                # keyboard fd check
                return ([42], [], [])

        with patch("select.select", side_effect=fake_select):
            runtime._handle_select_action({
                "cmd": "select_action",
                "player_id": "agent_0",
            })

        assert len(emitted) == 1
        assert emitted[0]["type"] == "action_selected"
        assert emitted[0]["action"] == 3
        assert emitted[0]["source"] == "keyboard"
        # Verify raw_keys are included in the response
        assert "raw_keys" in emitted[0]
        assert len(emitted[0]["raw_keys"]) == 1
        assert emitted[0]["raw_keys"][0]["keycode"] == KEY_W
        assert emitted[0]["raw_keys"][0]["pressed"] is True

    def test_select_action_without_init_returns_error(self):
        from human_worker.config import HumanWorkerConfig
        from human_worker.runtime import HumanKeyboardRuntime

        config = HumanWorkerConfig(run_id="test", env_name="multigrid")
        runtime = HumanKeyboardRuntime(config, device_path="/dev/input/event99")

        emitted = []
        runtime._emit = lambda data: emitted.append(data)

        runtime._handle_select_action({"cmd": "select_action", "player_id": "agent_0"})

        assert len(emitted) == 1
        assert emitted[0]["type"] == "error"


# ============================================================================
# 10. Integration: Resolver + Reader pipeline
# ============================================================================

class TestResolverReaderIntegration:
    """End-to-end test: evdev event bytes to game action index."""

    def test_w_key_event_to_forward_action(self):
        """Simulate a W key press raw event and resolve to FORWARD."""
        resolver = MultiGridKeycodeResolver()

        # Pack a W key press event
        event_data = _pack_event(EV_KEY, KEY_W, EV_VALUE_PRESS)
        _tv_sec, _tv_usec, ev_type, ev_code, ev_value = struct.unpack(
            "llHHi", event_data
        )

        assert ev_type == EV_KEY
        assert ev_code == KEY_W
        assert ev_value == EV_VALUE_PRESS

        action = resolver.resolve({ev_code})
        assert action == 3  # FORWARD

    def test_simultaneous_d_from_two_devices(self):
        """Two keyboards pressing D: both resolve to RIGHT independently."""
        resolver = MultiGridKeycodeResolver()

        action_0 = resolver.resolve({KEY_D})
        action_1 = resolver.resolve({KEY_D})

        assert action_0 == 2  # RIGHT
        assert action_1 == 2  # RIGHT
        assert action_0 == action_1


# ============================================================================
# 11. Mouse device discovery
# ============================================================================

class TestDiscoverMice:
    """Tests for discover_mice() function."""

    def test_discovers_mouse_by_path(self):
        from human_worker.evdev_input import MouseDeviceInfo, discover_mice

        with patch("human_worker.evdev_input.glob.glob") as mock_glob, \
             patch("human_worker.evdev_input.os.path.realpath") as mock_real, \
             patch("human_worker.evdev_input._get_device_name", return_value="Test Mouse"), \
             patch("human_worker.evdev_input._extract_usb_port", return_value="2"), \
             patch("human_worker.evdev_input._get_device_ids", return_value=("1234", "5678")):

            mock_glob.side_effect = [
                ["/dev/input/by-path/usb-0:2:1.0-event-mouse"],  # by-path
                [],  # by-id
            ]
            mock_real.return_value = "/dev/input/event20"

            mice = discover_mice()

        assert len(mice) == 1
        assert mice[0].name == "Test Mouse"
        assert mice[0].event_path == "/dev/input/event20"
        assert mice[0].usb_port == "2"

    def test_deduplicates_by_real_path(self):
        from human_worker.evdev_input import discover_mice

        with patch("human_worker.evdev_input.glob.glob") as mock_glob, \
             patch("human_worker.evdev_input.os.path.realpath") as mock_real, \
             patch("human_worker.evdev_input._get_device_name", return_value="Mouse"), \
             patch("human_worker.evdev_input._extract_usb_port", return_value=None), \
             patch("human_worker.evdev_input._get_device_ids", return_value=(None, None)):

            # Same real path from two different symlinks
            mock_glob.side_effect = [
                ["/dev/input/by-path/usb-0:2:1.0-event-mouse"],
                ["/dev/input/by-id/usb-TestMouse-event-mouse"],
            ]
            mock_real.return_value = "/dev/input/event20"

            mice = discover_mice()

        assert len(mice) == 1

    def test_empty_when_no_mice(self):
        from human_worker.evdev_input import discover_mice

        with patch("human_worker.evdev_input.glob.glob", return_value=[]):
            mice = discover_mice()

        assert mice == []


# ============================================================================
# 12. Mouse device reader
# ============================================================================

class TestMouseDeviceReader:
    """Tests for MouseDeviceReader class."""

    def test_open_and_close(self):
        from human_worker.evdev_input import MouseDeviceReader

        reader = MouseDeviceReader("/dev/input/event99")
        assert reader.is_open is False
        assert reader.fd is None

        with patch("human_worker.evdev_input.os.open", return_value=42):
            reader.open()
        assert reader.is_open is True
        assert reader.fd == 42

        with patch("human_worker.evdev_input.os.close"):
            reader.close()
        assert reader.is_open is False
        assert reader.fd is None

    def test_drain_deltas_accumulates_xy(self):
        from human_worker.evdev_input import (
            EV_REL, REL_X, REL_Y, MouseDeviceReader,
        )

        reader = MouseDeviceReader("/dev/input/event99")
        reader._fd = 42  # Bypass open()

        # Pack 3 events: REL_X +10, REL_Y -5, REL_X +3
        events = b""
        events += struct.pack("llHHi", 0, 0, EV_REL, REL_X, 10)
        events += struct.pack("llHHi", 0, 0, EV_REL, REL_Y, -5)
        events += struct.pack("llHHi", 0, 0, EV_REL, REL_X, 3)

        call_count = [0]
        def fake_read(fd, size):
            nonlocal events
            event_size = 24
            if call_count[0] < 3:
                start = call_count[0] * event_size
                call_count[0] += 1
                return events[start:start + event_size]
            raise BlockingIOError()

        with patch("human_worker.evdev_input.os.read", side_effect=fake_read):
            dx, dy = reader.drain_deltas()

        assert dx == 13   # 10 + 3
        assert dy == -5

    def test_drain_deltas_ignores_non_rel_events(self):
        from human_worker.evdev_input import (
            EV_KEY, EV_REL, EV_VALUE_PRESS, REL_X, MouseDeviceReader,
        )

        reader = MouseDeviceReader("/dev/input/event99")
        reader._fd = 42

        # One EV_KEY event (should be ignored) followed by EV_REL
        events = b""
        events += struct.pack("llHHi", 0, 0, EV_KEY, 30, EV_VALUE_PRESS)  # KEY_A
        events += struct.pack("llHHi", 0, 0, EV_REL, REL_X, 7)

        call_count = [0]
        def fake_read(fd, size):
            event_size = 24
            if call_count[0] < 2:
                start = call_count[0] * event_size
                call_count[0] += 1
                return events[start:start + event_size]
            raise BlockingIOError()

        with patch("human_worker.evdev_input.os.read", side_effect=fake_read):
            dx, dy = reader.drain_deltas()

        assert dx == 7
        assert dy == 0

    def test_drain_deltas_returns_zero_when_closed(self):
        from human_worker.evdev_input import MouseDeviceReader

        reader = MouseDeviceReader("/dev/input/event99")
        # fd is None (not opened)
        dx, dy = reader.drain_deltas()
        assert dx == 0
        assert dy == 0

    def test_drain_deltas_handles_blocking_io(self):
        from human_worker.evdev_input import MouseDeviceReader

        reader = MouseDeviceReader("/dev/input/event99")
        reader._fd = 42

        with patch("human_worker.evdev_input.os.read", side_effect=BlockingIOError()):
            dx, dy = reader.drain_deltas()

        assert dx == 0
        assert dy == 0


# ============================================================================
# 13. Runtime with mouse (keyboard + mouse combined)
# ============================================================================

class TestRuntimeWithMouse:
    """Tests for HumanKeyboardRuntime with mouse_path."""

    def test_init_with_mouse_path(self):
        from human_worker.config import HumanWorkerConfig
        from human_worker.runtime import HumanKeyboardRuntime

        config = HumanWorkerConfig(run_id="test", env_name="multigrid")
        runtime = HumanKeyboardRuntime(
            config,
            device_path="/dev/input/event5",
            mouse_path="/dev/input/event3",
        )
        assert runtime._mouse_path == "/dev/input/event3"
        assert runtime._mouse_reader is None  # Not opened until init_agent

    def test_init_without_mouse_path(self):
        from human_worker.config import HumanWorkerConfig
        from human_worker.runtime import HumanKeyboardRuntime

        config = HumanWorkerConfig(run_id="test", env_name="multigrid")
        runtime = HumanKeyboardRuntime(config, device_path="/dev/input/event5")
        assert runtime._mouse_path is None
        assert runtime._mouse_reader is None

    def test_select_action_includes_mouse_delta(self):
        from human_worker.config import HumanWorkerConfig
        from human_worker.runtime import HumanKeyboardRuntime

        config = HumanWorkerConfig(run_id="test", env_name="multigrid")
        runtime = HumanKeyboardRuntime(
            config,
            device_path="/dev/input/event5",
            mouse_path="/dev/input/event3",
        )

        emitted = []
        runtime._emit = lambda data: emitted.append(data)

        # Mock keyboard reader
        mock_reader = MagicMock()
        mock_reader._fd = 42
        mock_reader.drain_key_events.return_value = [(KEY_W, True)]
        runtime._reader = mock_reader

        # Mock mouse reader
        mock_mouse = MagicMock()
        mock_mouse.is_open = True
        mock_mouse.fd = 43
        mock_mouse.drain_deltas.return_value = (15, -8)
        runtime._mouse_reader = mock_mouse

        # Mock resolver
        mock_resolver = MagicMock()
        mock_resolver.resolve.return_value = 3  # FORWARD
        runtime._resolver = mock_resolver

        # select: stdin empty, then both kbd+mouse readable
        call_count = [0]
        def fake_select(rlist, wlist, xlist, timeout=None):
            call_count[0] += 1
            if call_count[0] % 2 == 1:
                return ([], [], [])  # stdin check
            else:
                return ([42, 43], [], [])  # kbd + mouse both readable

        with patch("select.select", side_effect=fake_select):
            runtime._handle_select_action({
                "cmd": "select_action",
                "player_id": "agent_0",
            })

        assert len(emitted) == 1
        assert emitted[0]["type"] == "action_selected"
        assert emitted[0]["action"] == 3
        assert emitted[0]["mouse_dx"] == 15
        assert emitted[0]["mouse_dy"] == -8

    def test_select_action_no_mouse_delta_when_zero(self):
        from human_worker.config import HumanWorkerConfig
        from human_worker.runtime import HumanKeyboardRuntime

        config = HumanWorkerConfig(run_id="test", env_name="multigrid")
        runtime = HumanKeyboardRuntime(config, device_path="/dev/input/event5")

        emitted = []
        runtime._emit = lambda data: emitted.append(data)

        mock_reader = MagicMock()
        mock_reader._fd = 42
        mock_reader.drain_key_events.return_value = [(KEY_D, True)]
        runtime._reader = mock_reader

        mock_resolver = MagicMock()
        mock_resolver.resolve.return_value = 2  # RIGHT
        runtime._resolver = mock_resolver

        call_count = [0]
        def fake_select(rlist, wlist, xlist, timeout=None):
            call_count[0] += 1
            if call_count[0] % 2 == 1:
                return ([], [], [])
            else:
                return ([42], [], [])

        with patch("select.select", side_effect=fake_select):
            runtime._handle_select_action({
                "cmd": "select_action",
                "player_id": "agent_0",
            })

        assert len(emitted) == 1
        assert emitted[0]["type"] == "action_selected"
        assert emitted[0]["action"] == 2
        # No mouse delta keys when movement is zero
        assert "mouse_dx" not in emitted[0]
        assert "mouse_dy" not in emitted[0]
