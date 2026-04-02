Keyboard Input System
=====================

The keyboard input system translates physical key presses into
environment actions.  It is implemented primarily in
``gym_gui.controllers.human_input`` and covers two input modes,
per-family key combination resolvers, shortcut mappings, and
multi-keyboard support for multi-agent play.

Two Input Modes
---------------

MOSAIC offers two fundamentally different ways to process keyboard input.
The mode is selectable in the Game Configuration panel and is stored as
an ``InputMode`` enum value.

.. list-table::
   :widths: 20 40 40
   :header-rows: 1

   * - Mode
     - Mechanism
     - Best For
   * - ``InputMode.SHORTCUT_BASED``
     - Qt ``QShortcut`` objects.  Each key press immediately fires a single
       action.  No key-state tracking, pressing two keys at once fires
       whichever arrives first.
     - Turn-based games: FrozenLake, MiniGrid, Chess, MiniHack, Jumanji
       puzzles.
   * - ``InputMode.STATE_BASED``
     - ``eventFilter`` on ``keyPressEvent`` / ``keyReleaseEvent``.  All
       currently pressed keys are tracked in a set.  A
       ``KeyCombinationResolver`` examines the set on each tick and
       produces a combined action (e.g., Up+Right = diagonal).
     - Real-time arcade games: Procgen, Atari, ViZDoom, Box2D, MeltingPot.

.. mermaid::

   sequenceDiagram
       participant User
       participant Widget as Qt Widget
       participant QS as QShortcut
       participant HIC as HumanInputController
       participant Res as KeyCombinationResolver
       participant SC as SessionController
       participant Adapt as Adapter

       alt Shortcut-Based
           User->>Widget: keyPressEvent
           Widget->>QS: triggered signal
           QS->>SC: perform_human_action(action)
       else State-Based
           User->>Widget: keyPressEvent
           Widget->>HIC: eventFilter (KeyPress)
           HIC->>Res: resolve(pressed_keys)
           Res-->>HIC: action index
           HIC->>SC: perform_human_action(action)
       end
       SC->>Adapt: _apply_action(action)

Key Combination Resolvers
-------------------------

In state-based mode, a ``KeyCombinationResolver`` subclass inspects the
set of currently pressed keys and returns the appropriate action index.
Each resolver targets a specific environment family.

.. list-table::
   :widths: 28 30 42
   :header-rows: 1

   * - Resolver
     - Environments
     - Actions
   * - ``MiniGridKeyCombinationResolver``
     - MiniGrid (7 actions)
     - Left / Right / Forward / Pickup / Drop / Toggle / Done
   * - ``MultiGridKeyCombinationResolver``
     - mosaic_multigrid (8 actions)
     - Noop / Left / Right / Forward / Pickup / Drop / Toggle / Done
   * - ``INIMultiGridKeyCombinationResolver``
     - INI multigrid (7 actions)
     - Left / Right / Forward / Pickup / Drop / Toggle / Done
   * - ``RWAREKeyCombinationResolver``
     - RWARE warehouse (5 actions)
     - Noop / Forward / Left / Right / Toggle
   * - ``MeltingPotKeyCombinationResolver``
     - MeltingPot (8 to 11 actions)
     - Noop / Forward / Backward / Strafe / Turn / Interact / Fire
   * - ``ProcgenKeyCombinationResolver``
     - Procgen (15 actions)
     - 8 directions + 6 action buttons
   * - ``AleKeyCombinationResolver``
     - Atari / ALE (18 actions)
     - 4 dirs + fire + all direction-fire combos
   * - ``LunarLanderKeyCombinationResolver``
     - LunarLander (4 actions)
     - Idle / Left engine / Main engine / Right engine
   * - ``CarRacingKeyCombinationResolver``
     - CarRacing (5 actions)
     - Coast / Right / Left / Accel / Brake
   * - ``BipedalWalkerKeyCombinationResolver``
     - BipedalWalker (5 actions)
     - Neutral / Forward / Back / Crouch / Extend
   * - ``ViZDoomKeyCombinationResolver``
     - ViZDoom scenarios
     - Scenario-specific button sets

The factory function ``get_key_combination_resolver(game_id, action_space)``
selects the correct resolver based on the ``GameId`` enum and the
environment's action space.

Shortcut Mappings
-----------------

In shortcut-based mode, MOSAIC registers ``QShortcut`` objects using
the ``ShortcutMapping`` dataclass, which pairs a ``QKeySequence`` with
an action index and a human-readable label.  The helper function
``_mapping(key_str, action, label)`` constructs these entries.

Each environment family defines its own mapping dictionary:

- ``_TOY_TEXT_MAPPINGS`` : FrozenLake, Taxi, CliffWalking, Blackjack
- ``_MINIG_GRID_MAPPINGS`` : MiniGrid family
- ``_MULTIGRID_MAPPINGS`` : MOSAIC MultiGrid and INI MultiGrid
- ``_BOX_2D_MAPPINGS`` : LunarLander, CarRacing, BipedalWalker
- ``_VIZDOOM_MAPPINGS`` : ViZDoom scenarios
- ``_MINIHACK_MAPPINGS`` : MiniHack dungeon crawling
- ``_NETHACK_MAPPINGS`` : NetHack challenge
- ``_CRAFTER_MAPPINGS`` : Crafter survival
- ``_BABAISAI_MAPPINGS`` : BabaIsAI puzzles
- ``_PROCGEN_MAPPINGS`` : Procgen games
- ``_ALE_MAPPINGS`` : Atari / ALE
- ``_JUMANJI_MAPPINGS`` : Jumanji logic games

Common Key Reference
^^^^^^^^^^^^^^^^^^^^

The table below summarises keys shared across most environment families.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Key
     - Common Action
   * - Arrow Up / W
     - Move forward / Up
   * - Arrow Down / S
     - Move backward / Down
   * - Arrow Left / A
     - Turn or move left
   * - Arrow Right / D
     - Turn or move right
   * - Space
     - Fire / Interact / Pickup
   * - E / Enter
     - Toggle / Use / Interact
   * - G
     - Pickup (grid worlds)
   * - H
     - Drop (grid worlds)
   * - Q
     - Done / Noop

Multi-Keyboard Support
----------------------

For multi-agent environments where multiple humans each control a
separate agent, MOSAIC supports routing physical USB keyboards to
different agents via Linux evdev.  A USB hub (4+ ports) with one
keyboard per agent lets each player press the same keys (WASD, Space,
etc.) on their own keyboard while only their agent responds.

On Linux, X11 merges all keyboards into a single virtual device, so
Qt's ``QInputDevice.systemId()`` cannot distinguish them.  MOSAIC
bypasses X11 entirely by reading raw ``/dev/input/eventX`` file
descriptors through ``EvdevKeyboardMonitor``, a background ``QThread``
that emits per-device ``key_pressed`` / ``key_released`` signals.

**Worker-based keyboard mode (subprocess):** For environments where
``adapter.step()`` blocks the GUI, each keyboard can be assigned to a
``human_worker`` subprocess (``--mode keyboard``).  The subprocess reads
evdev events in its own process using pure Python (no Qt dependency),
resolves keys to actions, and returns them via JSON pipes.  The GUI
stays responsive because it never blocks on keyboard I/O or env.step().

For the full architecture, data flow, hardware requirements, setup
instructions, and troubleshooting guide, see the dedicated page:
:doc:`multi_keyboard_evdev`.

.. tip::

   For multi-agent environments, MOSAIC automatically forces
   **state-based** input mode because shortcut-based mode conflicts with
   per-device keyboard routing.
