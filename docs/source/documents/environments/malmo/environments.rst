Environments Reference
======================

All 13 MalmoEnv missions available in MOSAIC are listed here. Each mission is defined by
an XML file bundled in ``3rd_party/environments/malmo/MalmoEnv/missions/``.

Action Space
------------

MalmoEnv action spaces are **mission-specific** — each mission XML defines which
commands are allowed.  The default action filter is ``{move, turn, use, attack}``
but individual missions may deny specific commands (e.g. ``attack``).

Three action-space profiles exist across the 13 missions:

**Discrete(4)** — move + turn only (FindTheGoal, TreasureHunt agent 0):

.. list-table::
   :header-rows: 1
   :widths: 10 25 65

   * - Index
     - Command
     - Description
   * - 0
     - ``move 1``
     - Move forward
   * - 1
     - ``move -1``
     - Move backward
   * - 2
     - ``turn 1``
     - Turn right
   * - 3
     - ``turn -1``
     - Turn left

**Discrete(6)** — move + turn + use (Attic, MobChase, Vertical, CliffWalking,
CatchTheMob, Eating, Obstacles, TrickyArena):

.. list-table::
   :header-rows: 1
   :widths: 10 25 65

   * - Index
     - Command
     - Description
   * - 0
     - ``move 1``
     - Move forward
   * - 1
     - ``move -1``
     - Move backward
   * - 2
     - ``turn 1``
     - Turn right
   * - 3
     - ``turn -1``
     - Turn left
   * - 4
     - ``use 1``
     - Use / interact
   * - 5
     - ``use 0``
     - Stop use

**Discrete(8)** — move + turn + attack + use (MazeRunner, DefaultFlatWorld,
DefaultWorld):

.. list-table::
   :header-rows: 1
   :widths: 10 25 65

   * - Index
     - Command
     - Description
   * - 0
     - ``move 1``
     - Move forward
   * - 1
     - ``move -1``
     - Move backward
   * - 2
     - ``turn 1``
     - Turn right
   * - 3
     - ``turn -1``
     - Turn left
   * - 4
     - ``attack 1``
     - Attack / break block
   * - 5
     - ``attack 0``
     - Stop attack
   * - 6
     - ``use 1``
     - Use / place
   * - 7
     - ``use 0``
     - Stop use

Observation Space
-----------------

All missions return an **RGB image** ``(H, W, 3)`` as the observation.  The default
resolution is ``84 × 84`` pixels.  The frame is rendered by Minecraft and streamed over TCP
to the Python agent.

.. note::

   The frame dimensions are controlled by the ``<VideoProducer>`` element in the mission XML.
   You can edit the XML to change the resolution; just remember to re-init the env.

Movement Types
--------------

Malmo missions use one of two movement command handlers, defined in the mission XML.
This fundamentally affects how actions behave:

**DiscreteMovementCommands** — each action is a one-shot, block-based movement:

- ``move 1`` moves the agent exactly **one block** forward, then stops.
- ``turn 1`` rotates the agent exactly **90 degrees**, then stops.
- No persistence — the agent is stationary between commands.
- Best for: grid-world style reasoning, turn-based evaluation.

**ContinuousMovementCommands** — each action sets a persistent velocity:

- ``move 1`` sets forward velocity to 1.0 — the agent **keeps moving** until ``move 0`` is sent.
- ``turn 1`` sets rotation velocity — the agent **keeps turning** until ``turn 0`` is sent.
- Values between -1 and 1 control speed (e.g. ``move 0.5`` = half speed).
- The agent has inertia (velocity is interpolated over ~6 ticks).
- Best for: smooth navigation, FPS-style control, realistic physics.

.. note::

   For **RL training**, the agent must learn to manage velocity in continuous missions
   (e.g. send ``move 0`` to stop). For **human play**, MOSAIC routes keyboard input
   through a native side-channel (TCP port 9001) that handles press/release naturally.

Missions
--------

MalmoEnv-MobChase-v0
~~~~~~~~~~~~~~~~~~~~~

:XML: ``mobchase_single_agent.xml``
:Movement: **Discrete:** one block per ``move``, 90° per ``turn``
:Actions: Discrete(6): move, turn, use
:Objective: Chase and reach a mob (pig/cow) in an open flat arena.
:Reward: Positive reward when the agent reaches within a threshold distance of the mob.
:Termination: Fixed time limit (in Malmo ticks).

**Use case:** Testing pursuit / chasing behaviours.  The mob moves randomly, providing a
moving target.


MalmoEnv-MazeRunner-v0
~~~~~~~~~~~~~~~~~~~~~~~~

:XML: ``mazerunner.xml``
:Movement: **Continuous:** persistent velocity
:Actions: Discrete(8): move, turn, attack, use
:Objective: Navigate from the start position to the goal block at the exit of a maze.
:Reward: Positive reward on reaching the goal; small negative step penalty.
:Termination: Time limit or goal reached.

**Use case:** Pathfinding and navigation benchmarks in a structured environment.


MalmoEnv-Vertical-v0
~~~~~~~~~~~~~~~~~~~~~~

:XML: ``vertical.xml``
:Movement: **Continuous:** persistent velocity
:Actions: Discrete(6): move, turn, use
:Objective: Climb a vertical tower of blocks placed on a platform over a void.
:Reward: Reward proportional to height gained.
:Termination: Agent falls off or time limit expires.

**Use case:** Training agents to climb and jump in 3-D environments.


MalmoEnv-CliffWalking-v0
~~~~~~~~~~~~~~~~~~~~~~~~~

:XML: ``cliffwalking.xml``
:Movement: **Discrete:** one block per ``move``, 90° per ``turn``
:Actions: Discrete(6): move, turn, use
:Objective: Walk along the top of a cliff from start to goal without falling.
:Reward: +1 for each step toward the goal; large negative reward for falling.
:Termination: Agent falls or reaches the goal.

**Use case:** Safety-aware navigation; dense reward signal for curriculum learning.


MalmoEnv-CatchTheMob-v0
~~~~~~~~~~~~~~~~~~~~~~~~

:XML: ``catchthemob.xml``
:Movement: **Discrete:** one block per ``move``, 90° per ``turn``
:Actions: Discrete(6): move, turn, use
:Objective: Catch a mob that is enclosed in a small arena.
:Reward: Reward on contact with the mob.
:Termination: Time limit.

**Use case:** Simpler mob-chasing variant with a confined space; easier exploration problem.


MalmoEnv-FindTheGoal-v0
~~~~~~~~~~~~~~~~~~~~~~~~

:XML: ``findthegoal.xml``
:Movement: **Continuous:** persistent velocity
:Actions: Discrete(4): move, turn only
:Objective: Locate and stand on a gold block hidden somewhere in a large flat world.
:Reward: Large positive reward on reaching the goal block.
:Termination: Time limit.

**Use case:** Sparse-reward exploration.  The agent must search a wide area with no
intermediate guidance.


MalmoEnv-Attic-v0
~~~~~~~~~~~~~~~~~~

:XML: ``attic.xml``
:Movement: **Continuous:** persistent velocity
:Actions: Discrete(6): move, turn, use
:Objective: Navigate an indoor "attic" layout (corridors, rooms, furniture).
:Reward: Positive reward on reaching the designated exit.
:Termination: Time limit.

**Use case:** Indoor navigation with obstacles; closer to real-world room layout.


MalmoEnv-DefaultFlatWorld-v0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:XML: ``defaultflatworld.xml``
:Movement: **Continuous:** persistent velocity
:Actions: Discrete(8): move, turn, attack, use
:Objective: Open-ended flat creative world — no specific mission goal.
:Reward: No default reward (extendable via XML).
:Termination: Time limit.

**Use case:** Free exploration, block placement experiments, custom reward shaping.


MalmoEnv-DefaultWorld-v0
~~~~~~~~~~~~~~~~~~~~~~~~~

:XML: ``defaultworld.xml``
:Movement: **Continuous:** persistent velocity
:Actions: Discrete(8): move, turn, attack, use
:Objective: Open-ended default Minecraft world generation (survival-style terrain).
:Reward: No default reward.
:Termination: Time limit.

**Use case:** Exploration in a rich procedurally generated landscape; closest to
vanilla Minecraft.


MalmoEnv-Eating-v0
~~~~~~~~~~~~~~~~~~~

:XML: ``eating.xml``
:Movement: **Continuous:** persistent velocity
:Actions: Discrete(6): move, turn, use
:Objective: Collect food items (bread, carrots, etc.) placed in the world.
:Reward: Positive reward for each food item collected.
:Termination: All items collected or time limit.

**Use case:** Reward-dense item collection for initial policy bootstrapping.


MalmoEnv-Obstacles-v0
~~~~~~~~~~~~~~~~~~~~~~

:XML: ``obstacles.xml``
:Movement: **Continuous:** persistent velocity
:Actions: Discrete(6): move, turn, use
:Objective: Navigate from start to goal while bypassing a series of obstacles
            (walls, pits, lava).
:Reward: Positive reward on reaching the goal.
:Termination: Agent dies or time limit.

**Use case:** Multi-hazard navigation; tests the agent's ability to plan around traps.


MalmoEnv-TrickyArena-v0
~~~~~~~~~~~~~~~~~~~~~~~~

:XML: ``trickyarena.xml``
:Movement: **Continuous:** persistent velocity
:Actions: Discrete(6): move, turn, use
:Objective: Survive in an arena with pits, moving platforms, and hazards.
:Reward: Reward for time survived; penalty for falling into pits.
:Termination: Agent dies or time limit.

**Use case:** Robustness testing — the agent must be reactive to environmental hazards.


MalmoEnv-TreasureHunt-v0
~~~~~~~~~~~~~~~~~~~~~~~~~

:XML: ``treasurehunt.xml``
:Movement: **Discrete (Turn-Based):** one block per ``move``, 90° per ``turn``
:Actions: Discrete(4): move, turn only (Agent 0)
:Agents: **2 agents required:** Agent 0 (deny attack) + Agent 1 (deny use). Single-agent mode not supported.
:Objective: Find and collect treasure chests scattered across the world.
:Reward: Positive reward for each chest opened.
:Termination: All chests collected or time limit.

.. warning::

   This is a **2-agent mission**. The Malmo server waits for both agents to connect
   before starting the episode. Running in single-agent mode causes a timeout
   (``ERROR_TIMED_OUT_WAITING_FOR_EPISODE_START``).

**Use case:** Multi-target collection with sparse rewards and exploration requirement.


MarLo Mission History
----------------------

These missions were originally part of the **MarLo** benchmark (the MOSAIC predecessor
used the Go-based ``mosaic_malmo`` package).  They are now served identically through
MalmoEnv by sending the same mission XML files to Minecraft.  The MOSAIC game IDs have
been renamed from ``MosaicMarLo-*-v0`` to ``MalmoEnv-*-v0`` to reflect the new backend.

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Old ID (Go backend, removed)
     - New ID (MalmoEnv backend)
   * - ``MosaicMalmo-Navigate-v0``
     - ``MalmoEnv-DefaultWorld-v0``
   * - ``MosaicMarLo-Vertical-v0``
     - ``MalmoEnv-Vertical-v0``
   * - ``MosaicMarLo-MazeRunner-v0``
     - ``MalmoEnv-MazeRunner-v0``
   * - ``MosaicMarLo-CliffWalking-v0``
     - ``MalmoEnv-CliffWalking-v0``
   * - ``MosaicMarLo-CatchTheMob-v0``
     - ``MalmoEnv-CatchTheMob-v0``
   * - ``MosaicMarLo-FindTheGoal-v0``
     - ``MalmoEnv-FindTheGoal-v0``
   * - ``MosaicMarLo-Attic-v0``
     - ``MalmoEnv-Attic-v0``
   * - ``MosaicMarLo-DefaultFlatWorld-v0``
     - ``MalmoEnv-DefaultFlatWorld-v0``
   * - ``MosaicMarLo-DefaultWorld-v0``
     - ``MalmoEnv-DefaultWorld-v0``
   * - ``MosaicMarLo-Eating-v0``
     - ``MalmoEnv-Eating-v0``
   * - ``MosaicMarLo-Obstacles-v0``
     - ``MalmoEnv-Obstacles-v0``
   * - ``MosaicMarLo-TrickyArena-v0``
     - ``MalmoEnv-TrickyArena-v0``
