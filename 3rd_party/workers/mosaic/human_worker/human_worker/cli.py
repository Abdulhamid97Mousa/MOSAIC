"""Human Worker CLI - Entry point for the human worker."""

import argparse
import sys

from .config import HumanWorkerConfig
from .runtime import HumanInteractiveRuntime, HumanKeyboardRuntime, HumanWorkerRuntime


def main() -> int:
    """Main entry point for human_worker CLI."""
    parser = argparse.ArgumentParser(
        description="Human Worker - Human-in-the-loop action selection via GUI"
    )

    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        choices=["interactive", "board-game", "keyboard"],
        default="interactive",
        help="Worker mode: 'interactive' owns the environment (MiniGrid, etc.), "
             "'board-game' for games where GUI owns the environment (chess, etc.), "
             "'keyboard' reads a physical USB keyboard via evdev (Linux only).",
    )

    # Run identification
    parser.add_argument("--run-id", type=str, default="", help="Unique run identifier")

    # Player settings
    parser.add_argument(
        "--player-name",
        type=str,
        default="Human",
        help="Human player's display name",
    )

    # Environment settings (for interactive mode)
    parser.add_argument(
        "--env-name",
        type=str,
        default="",
        help="Environment family (minigrid, babyai, etc.)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="",
        help="Specific environment task (MiniGrid-Empty-8x8-v0, etc.)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for environment",
    )
    parser.add_argument(
        "--render-mode",
        type=str,
        default="rgb_array",
        help="Render mode for environment",
    )
    parser.add_argument(
        "--game-resolution",
        type=str,
        default="512x512",
        help="Game render resolution (e.g., '64x64', '512x512'). Only affects Crafter.",
    )

    # Timeout settings
    parser.add_argument(
        "--timeout",
        type=float,
        default=0.0,
        help="Timeout for human input in seconds (0 = no timeout)",
    )

    # UI hints
    parser.add_argument(
        "--show-legal-moves",
        action="store_true",
        default=True,
        help="Highlight legal moves in UI",
    )
    parser.add_argument(
        "--no-show-legal-moves",
        action="store_false",
        dest="show_legal_moves",
        help="Don't highlight legal moves in UI",
    )
    parser.add_argument(
        "--confirm-moves",
        action="store_true",
        default=False,
        help="Require move confirmation before submitting",
    )

    # Keyboard mode settings
    parser.add_argument(
        "--device-path",
        type=str,
        default="",
        help="Path to evdev keyboard device (e.g., /dev/input/by-path/...-event-kbd). "
             "Required for --mode keyboard.",
    )
    parser.add_argument(
        "--mouse-path",
        type=str,
        default="",
        help="Path to evdev mouse device (e.g., /dev/input/by-path/...-event-mouse). "
             "Optional. Enables mouse delta for Malmo/ViZDoom turn/look.",
    )

    # Telemetry
    parser.add_argument(
        "--telemetry-dir",
        type=str,
        default="var/telemetry",
        help="Directory for telemetry output",
    )

    args = parser.parse_args()

    # Parse game resolution string (e.g., "512x512") to tuple
    game_resolution = (512, 512)  # Default
    if args.game_resolution:
        try:
            parts = args.game_resolution.lower().split("x")
            if len(parts) == 2:
                game_resolution = (int(parts[0]), int(parts[1]))
        except ValueError:
            pass  # Use default if parsing fails

    # Create config from args
    config = HumanWorkerConfig(
        run_id=args.run_id,
        player_name=args.player_name,
        env_name=args.env_name,
        task=args.task,
        seed=args.seed,
        render_mode=args.render_mode,
        game_resolution=game_resolution,
        timeout_seconds=args.timeout,
        show_legal_moves=args.show_legal_moves,
        confirm_moves=args.confirm_moves,
        telemetry_dir=args.telemetry_dir,
    )

    # Create and run the appropriate runtime
    if args.mode == "keyboard":
        # Keyboard mode - reads evdev device, returns actions via pipe
        if not args.device_path:
            parser.error("--device-path is required for --mode keyboard")
        runtime = HumanKeyboardRuntime(
            config,
            device_path=args.device_path,
            mouse_path=args.mouse_path or None,
        )
        runtime.run()
    elif args.mode == "interactive":
        # New interactive mode - worker owns the environment
        runtime = HumanInteractiveRuntime(config)
        runtime.run()
    else:
        # Legacy board-game mode - GUI owns the environment
        runtime = HumanWorkerRuntime(config)
        runtime.run_interactive()

    return 0


if __name__ == "__main__":
    sys.exit(main())
