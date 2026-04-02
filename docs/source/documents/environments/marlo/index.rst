MarLo (Multi-Agent Reinforcement Learning in Malmo)
====================================================

.. image:: https://raw.githubusercontent.com/crowdAI/crowdai/master/app/assets/images/misc/crowdai-logo-smile.svg?sanitize=true
   :width: 150px

**MarLO** (short for *Multi-Agent Reinforcement Learning in Malmo*) is a
high-level API built on top of Project Malmo for RL research in Minecraft.
It was used in the **2018 MarLo Challenge** and provides pre-built mission
environments with Gym-compatible interfaces.

Relationship to MOSAIC
----------------------

MOSAIC uses MarLo's **mission XML files** directly through the MalmoEnv backend.
The MarLo Python package is not imported at runtime, but the MarLo repository
(``3rd_party/environments/marLo``) is kept as an upstream reference for mission
definitions, launcher utilities, and the MarLo Challenge documentation.

.. mermaid::

   graph TB
       subgraph MarLo["MarLo (upstream reference)"]
           MarLoPkg["marlo Python package"]
           Missions["Mission XML files<br/>(Attic, MazeRunner, etc.)"]
           Launchers["Launcher utilities"]
       end

       subgraph Malmo["Project Malmo (runtime)"]
           JavaMod["Java Minecraft Mod"]
           MalmoEnv["MalmoEnv TCP Protocol"]
       end

       subgraph MOSAIC["MOSAIC"]
           Adapter["MalmoEnvAdapter"]
           GUI["PyQt6 GUI"]
       end

       MarLoPkg -. "wraps" .-> MalmoEnv
       Missions -- "loaded by" --> Adapter
       Adapter -- "TCP :9000" --> MalmoEnv
       MalmoEnv --> JavaMod

       style MarLoPkg stroke-dasharray: 5 5
       style MarLoPkg fill:#f5f5f5

ID Migration
~~~~~~~~~~~~

The original MarLo environment IDs have been renamed in MOSAIC:

.. list-table::
   :header-rows: 1
   :widths: 40 40

   * - MarLo ID (legacy)
     - MalmoEnv ID (current)
   * - ``MarLo-MazeRunner-v0``
     - ``MalmoEnv-MazeRunner-v0``
   * - ``MarLo-CliffWalking-v0``
     - ``MalmoEnv-CliffWalking-v0``
   * - ``MarLo-CatchTheMob-v0``
     - ``MalmoEnv-CatchTheMob-v0``
   * - ``MarLo-FindTheGoal-v0``
     - ``MalmoEnv-FindTheGoal-v0``
   * - ``MarLo-Attic-v0``
     - ``MalmoEnv-Attic-v0``
   * - ``MarLo-DefaultFlatWorld-v0``
     - ``MalmoEnv-DefaultFlatWorld-v0``
   * - ``MarLo-DefaultWorld-v0``
     - ``MalmoEnv-DefaultWorld-v0``
   * - ``MarLo-Eating-v0``
     - ``MalmoEnv-Eating-v0``
   * - ``MarLo-Obstacles-v0``
     - ``MalmoEnv-Obstacles-v0``
   * - ``MarLo-TrickyArena-v0``
     - ``MalmoEnv-TrickyArena-v0``
   * - ``MarLo-Vertical-v0``
     - ``MalmoEnv-Vertical-v0``

For full environment details (action spaces, movement types, objectives), see
:doc:`../malmo/environments`.

Mission Previews
----------------

.. list-table::
  :header-rows: 0
  :widths: 2 2 2
  :align: center

  * - ``MazeRunner``
        .. image:: https://media.giphy.com/media/u45fNQxG59wfnRpzwJ/giphy.gif
          :align: center
          :width: 200

    - ``CliffWalking``
        .. image:: https://media.giphy.com/media/ef4lPGNqaLlKr45rWB/giphy.gif
          :align: center
          :width: 200

    - ``CatchTheMob``
        .. image:: https://media.giphy.com/media/9A1gHZrWcaS4AYzcIU/giphy.gif
          :align: center
          :width: 200

  * - ``FindTheGoal``
        .. image:: https://media.giphy.com/media/1gWkQbDsHOfo4kZXZv/giphy.gif
          :align: center
          :width: 200

    - ``Attic``
        .. image:: https://media.giphy.com/media/47C7AYB3FA6kgrMiQ3/giphy.gif
          :align: center
          :width: 200

    - ``DefaultFlatWorld``
        .. image:: https://media.giphy.com/media/L0s9QXuR6vIJh6A0dq/giphy.gif
          :align: center
          :width: 200

  * - ``DefaultWorld``
        .. image:: https://media.giphy.com/media/4Nx7gYiM9NDrMrMao7/giphy.gif
          :align: center
          :width: 200

    - ``Eating``
        .. image:: https://media.giphy.com/media/pObNMjjfcGI5tVhmX6/giphy.gif
          :align: center
          :width: 200

    - ``Obstacles``
        .. image:: https://media.giphy.com/media/5sYmFFkq7aEMKTbKP4/giphy.gif
          :align: center
          :width: 200

  * - ``TrickyArena``
        .. image:: https://media.giphy.com/media/1g1bxw2nD3G9fz2WVV/giphy.gif
          :align: center
          :width: 200

    - ``Vertical``
        .. image:: https://media.giphy.com/media/ZcaMeSnzLrMY1NWM7f/giphy.gif
          :align: center
          :width: 200

    -

References
----------

- `MarLo Documentation <https://marlo.readthedocs.io/>`_
- `MarLo GitHub <https://github.com/crowdAI/marlo>`_
- `2018 MarLo Challenge <https://www.crowdai.org/challenges/marlo-2018>`_

Citation
--------

If you use these missions in your research, please cite:

.. code-block:: bibtex

   @misc{perez2019marlo,
     title={The Multi-Agent Reinforcement Learning in Malm{\"o} (MARL{\"O}) Competition},
     author={Perez-Liebana, Diego and Hofmann, Katja and Mohanty, Sharada Prasanna and Kuno, Noburu and Kramer, Andre and Devlin, Sam and Gaina, Raluca D.},
     journal={arXiv preprint arXiv:1901.08129},
     year={2019}
   }
