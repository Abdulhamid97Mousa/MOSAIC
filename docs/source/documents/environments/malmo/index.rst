Project Malmo (Microsoft Research)
===================================

.. image:: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/06/malmo_human_ai_interaction-web.png
   :width: 400px
   :align: center

**STATUS: Experimental - Under Active Development**

Project Malmo is a sophisticated AI experimentation platform built on top of
Minecraft, designed to support fundamental research in artificial intelligence.
It consists of a Forge mod for Minecraft Java Edition (1.11.2) and a Python
environment interface (MalmoEnv) that exposes a Gymnasium-compatible API.

MOSAIC integrates Malmo via **MalmoEnv**, the TCP-based Python wrapper that
communicates with the Minecraft Java client. The 13 mission environments
available in MOSAIC originate from the `MarLo <https://github.com/crowdAI/marlo>`_
benchmark (2018 MarLo Challenge) but are served directly through MalmoEnv.
The MarLo Python package itself is not required.

Quick Start
-----------

.. code-block:: bash

   ./setup_malmo.sh        # One-time setup (Java, assets, build)
   ./run_malmo.sh           # Terminal 1: Start Minecraft headless
   ./run.sh                 # Terminal 2: Start MOSAIC GUI

Two Control Modes
-----------------

**Human Play** (``Human Only``): Keyboard and mouse go directly to Minecraft
via a native TCP side-channel (port 9001). Press W to walk, release to stop.
Feels like playing Minecraft natively. The Java mod auto-detects the human
connection and switches to ``InputType.HUMAN``.

**RL Training** (``Agent Only``): An RL worker sends discrete actions
(``move 1``, ``turn 1``) through the MalmoEnv step loop (port 9000).
The Java mod stays in ``InputType.AI`` mode where movement commands set
persistent velocities. The agent must learn to send ``move 0`` to stop.

Both modes use port 9000 for observations. Only human play uses port 9001.

Architecture
------------

.. mermaid::

   graph TB
       subgraph Minecraft["Minecraft Java Client (1.11.2)"]
           Forge["Forge 13.20.0.2228"]
           MalmoMod["Malmo Mod 0.37.0"]
           MalmoEnvServer["MalmoEnv Server<br/>Port 9000"]
           NativeInput["NativeInputHandler<br/>Port 9001"]
       end

       subgraph MOSAIC["MOSAIC GUI (PyQt6)"]
           Adapter["MalmoEnvAdapter"]
           Interaction["MalmoInteractionController"]
       end

       subgraph MarLo["MarLo Missions"]
           XMLs["Mission XML files<br/>(13 environments)"]
       end

       Adapter -- "observations, rewards, actions<br/>TCP :9000" --> MalmoEnvServer
       Interaction -- "keyboard, mouse<br/>TCP :9001" --> NativeInput
       XMLs -- "loaded at reset()" --> Adapter

Installation
------------

.. code-block:: bash

   pip install -e ".[malmo]"

See :doc:`installation` for the full setup guide (Java 8, Gradle build, headless
launch).

Movement Types
--------------

Missions use either **Discrete** or **Continuous** movement commands.
See :doc:`environments` for details on each mission's movement type.

- **Discrete**: One-shot block-based movement (``move 1`` = one block forward).
- **Continuous**: Persistent velocity (``move 1`` = keep moving until ``move 0``).

Available Environments (13)
---------------------------

**Discrete Movement** (one-shot, block-based):

.. list-table::
   :header-rows: 0
   :widths: 2 2 2
   :align: center

   * - ``MalmoEnv-MobChase-v0``
         .. image:: https://media.giphy.com/media/9A1gHZrWcaS4AYzcIU/giphy.gif
           :align: center
           :width: 200

     - ``MalmoEnv-CatchTheMob-v0``
         .. image:: https://media.giphy.com/media/9A1gHZrWcaS4AYzcIU/giphy.gif
           :align: center
           :width: 200

     - ``MalmoEnv-CliffWalking-v0``
         .. image:: https://media.giphy.com/media/ef4lPGNqaLlKr45rWB/giphy.gif
           :align: center
           :width: 200

**Continuous Movement** (persistent velocity):

.. list-table::
   :header-rows: 0
   :widths: 2 2 2
   :align: center

   * - ``MalmoEnv-MazeRunner-v0``
         .. image:: https://media.giphy.com/media/u45fNQxG59wfnRpzwJ/giphy.gif
           :align: center
           :width: 200

     - ``MalmoEnv-FindTheGoal-v0``
         .. image:: https://media.giphy.com/media/1gWkQbDsHOfo4kZXZv/giphy.gif
           :align: center
           :width: 200

     - ``MalmoEnv-Attic-v0``
         .. image:: https://media.giphy.com/media/47C7AYB3FA6kgrMiQ3/giphy.gif
           :align: center
           :width: 200

   * - ``MalmoEnv-Vertical-v0``
         .. image:: https://media.giphy.com/media/ZcaMeSnzLrMY1NWM7f/giphy.gif
           :align: center
           :width: 200

     - ``MalmoEnv-DefaultFlatWorld-v0``
         .. image:: https://media.giphy.com/media/L0s9QXuR6vIJh6A0dq/giphy.gif
           :align: center
           :width: 200

     - ``MalmoEnv-DefaultWorld-v0``
         .. image:: https://media.giphy.com/media/4Nx7gYiM9NDrMrMao7/giphy.gif
           :align: center
           :width: 200

   * - ``MalmoEnv-Eating-v0``
         .. image:: https://media.giphy.com/media/pObNMjjfcGI5tVhmX6/giphy.gif
           :align: center
           :width: 200

     - ``MalmoEnv-Obstacles-v0``
         .. image:: https://media.giphy.com/media/5sYmFFkq7aEMKTbKP4/giphy.gif
           :align: center
           :width: 200

     - ``MalmoEnv-TrickyArena-v0``
         .. image:: https://media.giphy.com/media/1g1bxw2nD3G9fz2WVV/giphy.gif
           :align: center
           :width: 200

**Multi-Agent** (turn-based, discrete):

.. list-table::
   :header-rows: 0
   :widths: 2 2
   :align: center

   * - ``MalmoEnv-TreasureHunt-v0``

     - *Requires 2 agents. Single-agent mode not supported.*

.. warning::

   **TreasureHunt** requires 2 agents. The Malmo server waits for both agents
   to connect before starting. Running in single-agent mode causes a timeout.

References
----------

- `Microsoft Research - Project Malmo <https://www.microsoft.com/en-us/research/project/project-malmo/>`_
- `Project Malmo GitHub <https://github.com/Microsoft/malmo>`_
- `Malmo Documentation <https://microsoft.github.io/malmo/>`_
- `MarLo Challenge (2018) <https://www.crowdai.org/challenges/marlo-2018>`_

.. toctree::
   :maxdepth: 1
   :hidden:

   environments
   installation
