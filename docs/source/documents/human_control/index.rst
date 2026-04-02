Overview
========

.. raw:: html

   <video style="width:100%; max-width:100%; height:auto; border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,0.15);" controls autoplay muted loop playsinline>
     <source src="../../_static/videos/human_vs_human.mp4" type="video/mp4">
     Your browser does not support the video tag.
   </video>
   <br/><br/>

Human Control is the interactive play layer of MOSAIC's keyboard input,
mouse interaction, and the render view that displays environment state in
real time.  Together these subsystems let a human player step into any
supported environment and interact with it through the GUI.

Signal Flow
-----------

.. mermaid::

   graph TB
       KB[Keyboard] --> HIC[HumanInputController]
       HIC --> SC[SessionController]
       SC --> ADAPT[Adapter]
       ADAPT --> ENV[Environment]
       ENV --> |obs| ADAPT
       ADAPT --> |payload| REG[RendererRegistry]
       REG --> STRAT[RendererStrategy]
       STRAT --> RV[Render View]

       style KB fill:#4a90d9,stroke:#2e5a87,color:#fff
       style RV fill:#4a90d9,stroke:#2e5a87,color:#fff
       style HIC fill:#50c878,stroke:#2e8b57,color:#fff
       style SC fill:#50c878,stroke:#2e8b57,color:#fff
       style REG fill:#ff7f50,stroke:#cc5500,color:#fff
       style STRAT fill:#ff7f50,stroke:#cc5500,color:#fff
       style ADAPT fill:#9370db,stroke:#6a0dad,color:#fff
       style ENV fill:#9370db,stroke:#6a0dad,color:#fff

| Blue = GUI widgets | Green = Controllers | Orange = Rendering | Purple = Environment / Adapter |

Input Modes
-----------

The **Input Mode** selector in the Game Configuration panel determines how
MOSAIC translates physical key presses into environment actions.  Two modes
are available, each targeting a different class of environment.

.. list-table::
   :widths: 25 40 35
   :header-rows: 1

   * - Mode
     - Behaviour
     - Recommended For
   * - | **Shortcut-Based** 
       |  **(Immediate)**

     - Each key press triggers a single, immediate action via Qt
       ``QShortcut`` objects.  No key-state tracking is performed; pressing
       two keys simultaneously dispatches whichever event Qt delivers first.
     - Turn-based and step-by-step environments: FrozenLake, MiniGrid,
       Chess, MiniHack, Jumanji puzzles, BabaIsAI.
   * - **State-Based (Real-time)**
     - A ``keyPressEvent`` / ``keyReleaseEvent`` event filter tracks all
       currently held keys in a set.  On each tick a
       ``KeyCombinationResolver`` inspects the set and produces a combined
       action (e.g., Up + Right = diagonal movement).  Supports WASD and
       arrow keys interchangeably.
     - Real-time and arcade-style environments: Procgen, Atari/ALE,
       ViZDoom, Box2D (LunarLander, CarRacing), MeltingPot.

.. mermaid::

   %%{init: {"flowchart": {"curve": "linear", "padding": 14, "subGraphTitleMargin": {"top": 9, "bottom": 5}}} }%%
   graph TD
       subgraph SB["&ensp; Shortcut-Based (Immediate) <br/>&nbsp;"]
           direction TB
           SB1["Key press"] --> SB2["Qt QShortcut triggered"]
           SB2 --> SB3["SessionController.perform_human_action()"]
       end

       subgraph STB["&ensp; State-Based (Real-time) &ensp;"]
           direction TB

           STB1["Key press / release"] --> STB2["eventFilter updates\npressed-keys set"]
           STB2 --> STB3["KeyCombinationResolver.resolve()"]
           STB3 --> STB4["SessionController.perform_human_action()"]
       end

       style SB fill:#e3f2fd,stroke:#1565c0,color:#333
       style STB fill:#e8f5e9,stroke:#2e8b57,color:#333

.. note::

   Multi-agent environments with per-keyboard routing (via Linux ``evdev``)
   **require** state-based mode.  Shortcut-based mode uses Qt's global
   shortcut system, which cannot distinguish between physical keyboards.
   MOSAIC automatically enforces this constraint when a multi-agent
   control mode is selected.

See :doc:`keyboard_input` for the full technical reference, including
per-family key combination resolvers and shortcut mapping tables.

Subsystems
----------

**Keyboard Input:** Per-environment key mappings covering all 26 environment
families, and multi-keyboard support for multi-agent play via Linux ``evdev``.
For multi-agent environments with blocking ``adapter.step()`` calls, each
keyboard can be assigned to a ``human_worker`` subprocess
(``--mode keyboard``) that reads evdev events in its own process,
keeping the GUI responsive.
See :doc:`keyboard_input` for the full reference.

**Render View:** A strategy-pattern rendering pipeline that converts environment
observations into visual output.  Three built-in strategies cover grid
tile maps, RGB pixel arrays, and interactive board games.  See
:doc:`render_view` for the full reference.

Control Modes
-------------

The ``ControlMode`` enum determines who provides actions for each agent.
Human input is active in every mode except ``AGENT_ONLY``.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Control Mode
     - Description
   * - ``HUMAN_ONLY``
     - All actions come from the human keyboard.  No AI agent is involved.
   * - ``AGENT_ONLY``
     - All actions come from an AI agent.  The keyboard is disabled.
   * - ``MULTI_AGENT_COOP``
     - Multiple human players cooperate, each with a dedicated USB keyboard.
   * - ``MULTI_AGENT_COMPETITIVE``
     - Multiple human players compete against each other, each with a
       dedicated USB keyboard.

.. toctree::
   :maxdepth: 1

   keyboard_input
   render_view
   multi_keyboard_evdev
   single_agent/index
