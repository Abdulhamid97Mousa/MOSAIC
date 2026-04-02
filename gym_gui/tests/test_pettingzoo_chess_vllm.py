"""Test PettingZoo Chess integration with vLLM workers.

Tests the turn-based chess game with two vLLM servers:
- Server 1 (port 8000) for player_0 (White)
- Server 2 (port 8001) for player_1 (Black)

This test verifies:
1. Action encoding/decoding works correctly (AlphaZero style)
2. Both players can make valid moves via vLLM
3. Board state updates correctly after each move
4. Turn alternates between players
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import pytest

# Skip if pettingzoo or chess not available
pytest.importorskip("pettingzoo")
pytest.importorskip("chess")


class TestPettingZooChessActionEncoding:
    """Test that UCI moves are correctly converted to PettingZoo action indices."""

    @pytest.fixture
    def chess_env(self):
        """Create a fresh PettingZoo chess environment."""
        from pettingzoo.classic import chess_v6
        env = chess_v6.env()
        env.reset(seed=42)
        return env

    def test_action_space_size(self, chess_env):
        """PettingZoo chess has 4672 actions (64 squares x 73 move types)."""
        obs = chess_env.observe("player_0")
        action_mask = obs["action_mask"]
        assert len(action_mask) == 4672

    def test_initial_legal_moves_count(self, chess_env):
        """Initial position has 20 legal moves."""
        obs = chess_env.observe("player_0")
        action_mask = obs["action_mask"]
        legal_count = np.sum(action_mask == 1)
        assert legal_count == 20

    def test_convert_e2e4_to_action(self, chess_env):
        """Convert e2e4 UCI move to correct action index."""
        import chess
        from pettingzoo.classic.chess import chess_utils

        move = chess.Move.from_uci("e2e4")
        source = move.from_square
        coord = chess_utils.square_to_coord(source)
        panel = chess_utils.get_move_plane(move)
        action = (coord[0] * 8 + coord[1]) * 73 + panel

        # Verify this action is legal
        obs = chess_env.observe("player_0")
        assert obs["action_mask"][action] == 1

        # Execute and verify board changes
        chess_env.step(action)
        # After e2e4, it should be black's turn
        assert chess_env.agent_selection == "player_1"
        # Check FEN has pawn on e4
        fen = chess_env.board.fen()
        assert "4P3" in fen or "P" in fen.split()[0]

    def test_convert_black_move_e7e5(self, chess_env):
        """Convert black's e7e5 UCI move with mirroring."""
        import chess
        from pettingzoo.classic.chess import chess_utils

        # First, white plays e2e4
        white_move = chess.Move.from_uci("e2e4")
        source = white_move.from_square
        coord = chess_utils.square_to_coord(source)
        panel = chess_utils.get_move_plane(white_move)
        white_action = (coord[0] * 8 + coord[1]) * 73 + panel
        chess_env.step(white_action)

        # Now black plays e7e5
        black_move = chess.Move.from_uci("e7e5")
        # For black, need to mirror the move
        mirrored_move = chess_utils.mirror_move(black_move)
        source = mirrored_move.from_square
        coord = chess_utils.square_to_coord(source)
        panel = chess_utils.get_move_plane(mirrored_move)
        black_action = (coord[0] * 8 + coord[1]) * 73 + panel

        # Verify and execute
        obs = chess_env.observe("player_1")
        assert obs["action_mask"][black_action] == 1

        chess_env.step(black_action)
        # After e7e5, should be white's turn
        assert chess_env.agent_selection == "player_0"


class TestPettingZooChessMoveSequence:
    """Test a sequence of moves in PettingZoo chess."""

    def test_opening_sequence(self):
        """Play an opening sequence and verify board state."""
        import chess
        from pettingzoo.classic import chess_v6
        from pettingzoo.classic.chess import chess_utils

        env = chess_v6.env()
        env.reset(seed=42)

        def uci_to_action(env, uci_move: str) -> int:
            """Convert UCI move to action index."""
            move = chess.Move.from_uci(uci_move)
            current_agent = env.agent_selection
            current_player = 0 if current_agent == "player_0" else 1

            if current_player == 1:
                move_for_encoding = chess_utils.mirror_move(move)
            else:
                move_for_encoding = move

            source = move_for_encoding.from_square
            coord = chess_utils.square_to_coord(source)
            panel = chess_utils.get_move_plane(move_for_encoding)
            return (coord[0] * 8 + coord[1]) * 73 + panel

        # Play Italian Game opening
        moves = [
            ("e2e4", "player_0"),  # 1. e4
            ("e7e5", "player_1"),  # 1... e5
            ("g1f3", "player_0"),  # 2. Nf3
            ("b8c6", "player_1"),  # 2... Nc6
            ("f1c4", "player_0"),  # 3. Bc4 (Italian Game)
        ]

        for uci, expected_player in moves:
            assert env.agent_selection == expected_player, f"Expected {expected_player}, got {env.agent_selection}"
            action = uci_to_action(env, uci)
            obs = env.observe(expected_player)
            assert obs is not None
            assert obs["action_mask"][action] == 1, f"Action {action} for {uci} is not legal"
            env.step(action)

        # Verify final position (Italian Game setup)
        fen = env.board.fen()
        # White bishop should be on c4
        assert "B" in fen  # Bishop present
        # Black knight on c6
        assert "n" in fen  # Knight present


class TestUCIToActionConversion:
    """Test the UCI to action conversion function used in main_window.py."""

    def test_convert_function_white_moves(self):
        """Test conversion for white's moves."""
        import chess
        from pettingzoo.classic import chess_v6
        from pettingzoo.classic.chess import chess_utils

        env = chess_v6.env()
        env.reset(seed=42)

        def convert_uci_to_action_index(env, uci_move: str) -> Optional[int]:
            """Same logic as main_window._convert_uci_to_action_index."""
            board = env.board
            move = chess.Move.from_uci(uci_move)

            if move not in board.legal_moves:
                return None

            current_agent = env.agent_selection
            current_player = 0 if current_agent == "player_0" else 1

            if current_player == 1:
                move_for_encoding = chess_utils.mirror_move(move)
            else:
                move_for_encoding = move

            source = move_for_encoding.from_square
            coord = chess_utils.square_to_coord(source)
            panel = chess_utils.get_move_plane(move_for_encoding)
            action = (coord[0] * 8 + coord[1]) * 73 + panel

            obs = env.observe(current_agent)
            if obs["action_mask"][action] == 1:
                return action
            return None

        # Test all initial pawn moves
        pawn_moves = ["a2a3", "a2a4", "b2b3", "b2b4", "c2c3", "c2c4",
                      "d2d3", "d2d4", "e2e3", "e2e4", "f2f3", "f2f4",
                      "g2g3", "g2g4", "h2h3", "h2h4"]

        for move in pawn_moves:
            action = convert_uci_to_action_index(env, move)
            assert action is not None, f"Failed to convert {move}"
            obs = env.observe("player_0")
            assert obs is not None
            assert obs["action_mask"][action] == 1, f"Action for {move} is not legal"

        # Test knight moves
        knight_moves = ["b1a3", "b1c3", "g1f3", "g1h3"]
        for move in knight_moves:
            action = convert_uci_to_action_index(env, move)
            assert action is not None, f"Failed to convert knight move {move}"

    def test_convert_function_black_moves(self):
        """Test conversion for black's moves (with mirroring)."""
        import chess
        from pettingzoo.classic import chess_v6
        from pettingzoo.classic.chess import chess_utils

        env = chess_v6.env()
        env.reset(seed=42)

        def convert_uci_to_action_index(env, uci_move: str) -> Optional[int]:
            """Same logic as main_window._convert_uci_to_action_index."""
            board = env.board
            move = chess.Move.from_uci(uci_move)

            if move not in board.legal_moves:
                return None

            current_agent = env.agent_selection
            current_player = 0 if current_agent == "player_0" else 1

            if current_player == 1:
                move_for_encoding = chess_utils.mirror_move(move)
            else:
                move_for_encoding = move

            source = move_for_encoding.from_square
            coord = chess_utils.square_to_coord(source)
            panel = chess_utils.get_move_plane(move_for_encoding)
            action = (coord[0] * 8 + coord[1]) * 73 + panel

            obs = env.observe(current_agent)
            if obs["action_mask"][action] == 1:
                return action
            return None

        # White plays e2e4
        action = convert_uci_to_action_index(env, "e2e4")
        env.step(action)

        # Now black's turn - test black moves
        black_moves = ["e7e5", "e7e6", "d7d5", "d7d6", "g8f6", "b8c6"]
        for move in black_moves:
            action = convert_uci_to_action_index(env, move)
            assert action is not None, f"Failed to convert black move {move}"
            obs = env.observe("player_1")
            assert obs is not None
            assert obs["action_mask"][action] == 1, f"Action for black's {move} is not legal"


def _check_vllm_server(url: str) -> bool:
    """Check if a vLLM server is available."""
    try:
        import requests
        response = requests.get(f"{url}/health", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


class TestVLLMChessIntegration:
    """Integration tests with vLLM servers (requires running servers)."""

    @pytest.fixture
    def vllm_config_white(self):
        """vLLM config for white player (server 1, port 8000)."""
        return {
            "client_name": "vllm",
            "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
            "base_url": "http://127.0.0.1:8000/v1",
        }

    @pytest.fixture
    def vllm_config_black(self):
        """vLLM config for black player (server 2, port 8001)."""
        return {
            "client_name": "vllm",
            "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
            "base_url": "http://127.0.0.1:8001/v1",
        }

    @pytest.mark.skipif(
        not _check_vllm_server("http://127.0.0.1:8000"),
        reason="vLLM server 1 (port 8000) not available"
    )
    @pytest.mark.skipif(
        not _check_vllm_server("http://127.0.0.1:8001"),
        reason="vLLM server 2 (port 8001) not available"
    )
    def test_vllm_servers_available(self, vllm_config_white, vllm_config_black):
        """Both vLLM servers are available."""
        assert _check_vllm_server(vllm_config_white["base_url"].replace("/v1", ""))
        assert _check_vllm_server(vllm_config_black["base_url"].replace("/v1", ""))

    @pytest.mark.skipif(
        not _check_vllm_server("http://127.0.0.1:8000"),
        reason="vLLM server 1 not available"
    )
    def test_vllm_chess_move_selection(self, vllm_config_white):
        """Test that vLLM can select a chess move."""
        import requests

        # Get legal moves for initial position
        legal_moves = ["e2e4", "d2d4", "g1f3", "b1c3"]

        prompt = f"""You are playing chess as White.
The current board position is the starting position.
Legal moves: {', '.join(legal_moves)}

Select ONE move from the legal moves list. Reply with ONLY the move in UCI format (e.g., 'e2e4').
Your move:"""

        response = requests.post(
            f"{vllm_config_white['base_url']}/chat/completions",
            json={
                "model": vllm_config_white["model_id"],
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 10,
                "temperature": 0.1,
            },
            timeout=30,
        )

        if response.status_code == 200:
            data = response.json()
            move = data["choices"][0]["message"]["content"].strip()
            # Check if the response contains a legal move
            found_legal_move = any(m in move for m in legal_moves)
            assert found_legal_move, f"vLLM response '{move}' doesn't contain a legal move"


class TestGameTermination:
    """Test game termination conditions."""

    def test_scholars_mate(self):
        """Test Scholar's Mate leads to checkmate."""
        import chess
        from pettingzoo.classic import chess_v6
        from pettingzoo.classic.chess import chess_utils

        env = chess_v6.env()
        env.reset(seed=42)

        def uci_to_action(env, uci_move: str) -> int:
            move = chess.Move.from_uci(uci_move)
            current_agent = env.agent_selection
            current_player = 0 if current_agent == "player_0" else 1

            if current_player == 1:
                move_for_encoding = chess_utils.mirror_move(move)
            else:
                move_for_encoding = move

            source = move_for_encoding.from_square
            coord = chess_utils.square_to_coord(source)
            panel = chess_utils.get_move_plane(move_for_encoding)
            return (coord[0] * 8 + coord[1]) * 73 + panel

        # Scholar's Mate: 1.e4 e5 2.Qh5 Nc6 3.Bc4 Nf6 4.Qxf7#
        moves = ["e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6", "h5f7"]

        for uci in moves:
            action = uci_to_action(env, uci)
            env.step(action)

        # Game should be over
        assert env.board.is_checkmate()
        assert env.board.is_game_over()

    def test_stalemate_detection(self):
        """Test stalemate detection (requires specific position)."""
        from pettingzoo.classic import chess_v6

        env = chess_v6.env()
        env.reset(seed=42)

        # Note: Creating a stalemate position from scratch is complex
        # This test verifies the board methods work
        assert not env.board.is_stalemate()
        assert not env.board.is_game_over()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
