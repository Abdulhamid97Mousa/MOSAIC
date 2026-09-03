HeMAC
=====

Overview
--------

**HeMAC (Heterogeneous Multi-Agent Challenge)** is a standardized, PettingZoo-based
benchmark environment for Heterogeneous Multi-Agent Reinforcement Learning (HeMARL).
It proposes multiple scenarios where agents with diverse sensors, resources, and
capabilities must cooperate to solve complex tasks under partial observability.

.. list-table::
   :widths: 20 80
   :header-rows: 0

   * - **Paper**
     - `Dansereau et al. (2025) <https://arxiv.org/abs/2509.19512>`_
   * - **Repo**
     - https://github.com/ThalesGroup/hemac
   * - **Location**
     - ``3rd_party/environments/hemac/``

Key Features
------------

- **Rich Heterogeneity**: Multiple distinct agent types (Quadcopters, Observers, Provisioners) with unique observation and action spaces
- **Multi-Stage Benchmarking**: Three challenges with increasing difficulty and heterogeneity
- **Partial Observability**: Agents perceive the world through unique, limited sensors
- **Flexible Spaces**: Both discrete and continuous action spaces supported
- **Extensibility**: Easily add new agent types, capabilities, and scenarios

Why HeMAC?
----------

Traditional MARL benchmarks focus on homogeneous teams, falling short when representing
real-world heterogeneous agent systems. HeMAC provides:

- A controlled environment where agents must specialize and cooperate based on unique abilities
- Standardized tasks to facilitate reproducible, comparable HeMARL research
- Rich partial observability and coordination challenges

Research shows that while state-of-the-art methods (like MAPPO) excel at simpler tasks,
their performance degrades with increased heterogeneity, with simpler algorithms (like IPPO)
sometimes outperforming them under these conditions.

Environment Overview
--------------------

In HeMAC, a team of autonomous agents works together to find and reach moving targets
in a randomly generated map featuring obstacles and special structures.

Agent Types
~~~~~~~~~~~

**Quadcopter**
   Low-altitude, agile agents that can reach targets but have limited energy and capacity.

**Observer**
   High-altitude, fast agents with broad forward-facing views; guide Quadcopters but
   cannot directly reach targets.

**Provisioner**
   Ground vehicles navigating a road network to recharge/support aerial agents and
   assist with target retrieval.

Challenges and Scenarios
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 30 35 15

   * - Challenge
     - Agents
     - Description
     - Scenarios
   * - **Simple Fleet**
     - Quadcopters, Observers
     - Reach as many moving targets as possible. Observers must guide Quadcopters.
     - 1q1o, 3q1o, 5q2o
   * - **Fleet**
     - Quadcopters, Observers
     - Multi-target, energy constraints, obstacles, limited communication range.
     - 3q1o, 10q3o, 20q5o
   * - **Complex Fleet**
     - Quadcopters, Observers, Provisioners
     - High heterogeneity: energy/capacity limits, provisioners restricted to roads.
     - 3q1o1p, 5q2o1p

Agents receive different local observations according to their sensors and roles.

Installation
------------

HeMAC is included as a git submodule in MOSAIC. To install:

.. code-block:: bash

   # Initialize the submodule
   git submodule update --init 3rd_party/environments/hemac

   # Install HeMAC
   pip install -e 3rd_party/environments/hemac

Usage Example
-------------

HeMAC uses PettingZoo's AEC API:

.. code-block:: python

   from hemac import HeMAC_v0

   env = HeMAC_v0.env(render_mode="human")
   env.reset(seed=0)

   for agent in env.agent_iter():
       observation, reward, termination, truncation, info = env.last()
       if termination or truncation:
           action = None
       else:
           # Insert your policy here
           action = env.action_space(agent).sample()
       env.step(action)
   env.close()

MOSAIC Integration
------------------

HeMAC environments can be used with MOSAIC's multi-agent operators:

.. code-block:: python

   from gym_gui.config.operator_config import OperatorConfig, WorkerAssignment

   config = OperatorConfig.multi_agent(
       operator_id="hemac_simple_fleet",
       display_name="HeMAC Simple Fleet 3q1o",
       env_name="hemac",
       task="simple_fleet_3q1o",
       player_workers={
           # 3 Quadcopters
           "quadcopter_0": WorkerAssignment(
               worker_id="xuance_worker",
               worker_type="rl",
               settings={"algorithm": "mappo"},
           ),
           "quadcopter_1": WorkerAssignment(
               worker_id="xuance_worker",
               worker_type="rl",
               settings={"algorithm": "mappo"},
           ),
           "quadcopter_2": WorkerAssignment(
               worker_id="xuance_worker",
               worker_type="rl",
               settings={"algorithm": "mappo"},
           ),
           # 1 Observer
           "observer_0": WorkerAssignment(
               worker_id="xuance_worker",
               worker_type="rl",
               settings={"algorithm": "mappo"},
           ),
       },
   )

Research Applications
---------------------

HeMAC enables research in:

**Heterogeneous MARL**
   Study how agents with different capabilities learn to cooperate

**Specialization and Division of Labor**
   Investigate how agents develop specialized roles based on their unique abilities

**Partial Observability**
   Research coordination under limited and asymmetric information

**Transfer Learning**
   Test how policies trained on simpler scenarios transfer to more complex ones

**Algorithm Comparison**
   Benchmark MARL algorithms (MAPPO, IPPO, QMIX) on standardized heterogeneous tasks

Citation
--------

If you use HeMAC in your research, please cite:

.. code-block:: bibtex

   @inproceedings{dansereau2025hemac,
     title     = {The Heterogeneous Multi-Agent Challenge},
     author    = {Dansereau, Charles and Lopez Yepez, Junior Samuel and
                  Soma, Karthik and Fagette, Antoine},
     booktitle = {Proceedings of the 27th European Conference on
                  Artificial Intelligence (ECAI 2025)},
     series    = {Frontiers in Artificial Intelligence and Applications},
     volume    = {413},
     pages     = {3290--3296},
     year      = {2025},
     publisher = {IOS Press},
     doi       = {10.3233/FAIA251197}
   }

See Also
--------

- :doc:`../smac/index` - StarCraft Multi-Agent Challenge (homogeneous teams)
- :doc:`../smacv2/index` - SMAC v2 with improved scenarios
- :doc:`../mosaic_multigrid/index` - Multi-agent grid worlds
- :doc:`../melting_pot/index` - Multi-agent social dilemmas
