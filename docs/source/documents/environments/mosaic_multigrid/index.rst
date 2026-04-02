MOSAIC MultiGrid
================

Competitive team-based multi-agent grid-world games.  Developed as part of
MOSAIC with ``view_size=3`` (agent-centric partial observability).

**New in v6.3.0:** American Football environments with brown field rendering,
end zone scoring mechanics, ball stealing, and touchdown detection. Agents cannot
score in their own end zones.

:Install: ``pip install -e ".[mosaic_multigrid]"``
:Paradigm: Multi-agent (simultaneous)
:Stepping: ``SIMULTANEOUS``
:PyPI: `mosaic-multigrid v6.3.0 <https://pypi.org/project/mosaic-multigrid/>`_

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Environment
     - Description
   * - Soccer-2vs2-IndAgObs
     - 2v2 soccer on a 16×11 FIFA grid, ball respawn, first-to-2-goals
   * - Soccer-1vs1-IndAgObs
     - 1v1 soccer variant
   * - Collect-IndAgObs
     - 3 agents, individual ball collection competition
   * - Collect-2vs2-IndAgObs
     - 2v2 teams, 7 balls (no draws)
   * - Collect-1vs1-IndAgObs
     - 1v1 variant, 3 balls
   * - Basketball-3vs3-IndAgObs
     - 6 agents, 3v3 basketball with court rendering
   * - **AmericanFootball-1v1-IndAgObs** (v6.3.0)
     - 1v1 American Football on 16×11 brown field, touchdown scoring, ball stealing
   * - **AmericanFootball-2v2-IndAgObs** (v6.3.0)
     - 2v2 American Football variant
   * - **AmericanFootball-3v3-IndAgObs** (v6.3.0)
     - 3v3 American Football variant
   * - **AmericanFootball-Solo-Green** (v6.3.0)
     - Solo training environment (no opponent) for curriculum pre-training
   * - **AmericanFootball-Solo-Blue** (v6.3.0)
     - Solo training environment (no opponent) for curriculum pre-training
   * - *TeamObs variants*
     - Soccer-2vs2, Collect-2vs2, Basketball-3vs3, AmericanFootball-2v2, AmericanFootball-3v3 with SMAC-style teammate awareness


Environment IDs
---------------

American Football (v6.3.0)
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import gymnasium as gym
   import mosaic_multigrid.envs

   # Solo training (curriculum pre-training)
   env = gym.make('MosaicMultiGrid-AmericanFootball-Solo-Green-v0')
   env = gym.make('MosaicMultiGrid-AmericanFootball-Solo-Blue-v0')

   # 1v1 competitive
   env = gym.make('MosaicMultiGrid-AmericanFootball-1v1-v0')

   # 2v2 competitive
   env = gym.make('MosaicMultiGrid-AmericanFootball-2v2-v0')
   env = gym.make('MosaicMultiGrid-AmericanFootball-2v2-TeamObs-v0')  # With teammate awareness

   # 3v3 competitive
   env = gym.make('MosaicMultiGrid-AmericanFootball-3v3-v0')
   env = gym.make('MosaicMultiGrid-AmericanFootball-3v3-TeamObs-v0')  # With teammate awareness

**Features:**
- 16×11 brown field with alternating stripes
- White boundary lines and yard lines
- Colored end zones (green and blue)
- Touchdown scoring: walk into opponent's end zone while carrying ball
- Ball stealing: use pickup action on opponent carrying ball
- Agents cannot score in their own end zones (verified by tests)
- Custom renderer with optional HUD (agent labels, FOV highlights)

Soccer, Basketball, Collect
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Soccer
   env = gym.make('MosaicMultiGrid-Soccer-2vs2-IndAgObs-v0')
   env = gym.make('MosaicMultiGrid-Soccer-1vs1-IndAgObs-v0')
   env = gym.make('MosaicMultiGrid-Soccer-2vs2-TeamObs-v0')

   # Basketball
   env = gym.make('MosaicMultiGrid-Basketball-3vs3-IndAgObs-v0')
   env = gym.make('MosaicMultiGrid-Basketball-3vs3-TeamObs-v0')

   # Collect
   env = gym.make('MosaicMultiGrid-Collect-IndAgObs-v0')
   env = gym.make('MosaicMultiGrid-Collect-2vs2-IndAgObs-v0')
   env = gym.make('MosaicMultiGrid-Collect-1vs1-IndAgObs-v0')
   env = gym.make('MosaicMultiGrid-Collect-2vs2-TeamObs-v0')


Action Space
------------

All environments use the same action space (Discrete(8)):

.. list-table::
   :header-rows: 1
   :widths: 10 20 70

   * - ID
     - Action
     - Description
   * - 0
     - NOOP
     - No operation (AEC compatibility)
   * - 1
     - LEFT
     - Rotate 90° counter-clockwise
   * - 2
     - RIGHT
     - Rotate 90° clockwise
   * - 3
     - FORWARD
     - Move one cell in facing direction
   * - 4
     - PICKUP
     - Pick up ball / steal from opponent
   * - 5
     - DROP
     - Drop ball / pass to teammate / shoot
   * - 6
     - TOGGLE
     - Toggle/activate object
   * - 7
     - DONE
     - Signal task completion


Gameplay Rules
--------------

American Football (v6.3.0)
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Objective:** Score touchdowns by carrying the ball into the opponent's end zone.

**Field Layout:**
- 16×11 brown field with white yard lines
- Green end zone (column 1): Defended by Team 0, scored on by Team 1
- Blue end zone (column 14): Defended by Team 1, scored on by Team 0

**How to Play:**
1. **Pick up the ball:** Use PICKUP (action 4) when facing the ball
2. **Carry the ball:** Move with FORWARD (action 3) while carrying
3. **Score touchdown:** Walk into the opponent's end zone while carrying the ball
4. **Ball stealing:** Use PICKUP (action 4) when facing an opponent carrying the ball
5. **Pass to teammate:** Use DROP (action 5) when facing a teammate (ball teleports to them)
6. **Important:** You CANNOT score in your own end zone

**Scoring:**
- Touchdown = +1 point for scoring team
- Zero-sum: Opponent receives -1 point
- Episode terminates after touchdown
- Ball respawns in midfield after touchdown

Soccer
~~~~~~

**Objective:** Score goals by shooting the ball into the opponent's goal.

**Field Layout:**
- 16×11 FIFA-style green field
- Goals at left and right ends of the field
- First team to 2 goals wins

**How to Play:**
1. **Pick up the ball:** Use PICKUP (action 4) when facing the ball
2. **Dribble:** Move with FORWARD (action 3) while carrying
3. **Shoot:** Use DROP (action 5) when facing the opponent's goal
4. **Pass:** Use DROP (action 5) when facing a teammate
5. **Steal:** Use PICKUP (action 4) when facing an opponent with the ball

**Scoring:**
- Goal = +1 point for scoring team
- Ball respawns at center after each goal
- Episode continues until one team reaches 2 goals

Basketball
~~~~~~~~~~

**Objective:** Score baskets by shooting the ball into the opponent's hoop.

**Field Layout:**
- 19×11 court with basketball markings
- Hoops at left and right ends of the court

**How to Play:**
1. **Pick up the ball:** Use PICKUP (action 4) when facing the ball
2. **Dribble:** Move with FORWARD (action 3) while carrying
3. **Shoot:** Use DROP (action 5) when facing the opponent's hoop
4. **Pass:** Use DROP (action 5) when facing a teammate
5. **Steal:** Use PICKUP (action 4) when facing an opponent with the ball

**Scoring:**
- Basket = +1 point for scoring team
- Ball respawns at center after each basket

Collect
~~~~~~~

**Objective:** Collect more balls than opponents before time runs out.

**Field Layout:**
- Multiple colored balls scattered on the field
- Each ball has a designated collection zone

**How to Play:**
1. **Pick up a ball:** Use PICKUP (action 4) when facing a ball
2. **Carry to zone:** Move with FORWARD (action 3) to the ball's collection zone
3. **Score:** Use DROP (action 5) in the correct collection zone
4. **Steal:** Use PICKUP (action 4) when facing an opponent with a ball

**Scoring:**
- Each ball collected = +1 point
- Episode ends when all balls are collected or time limit reached
- Agent/team with most points wins


Citation
--------


.. code-block:: bibtex 
  
    @article{mousa2026mosaic,
      title = {MOSAIC MultiGrid: Research-Grade Multi-Agent Gridworld Environments},
      author = {Mousa, Abdulhamid},
      journal = {GitHub repository},
      year = {2026},
      url = {https://github.com/Abdulhamid97Mousa/mosaic_multigrid},
    }

    @misc{gym_multigrid,
      author = {Fickinger, Arnaud},
      title = {Multi-Agent Gridworld Environment for OpenAI Gym},
      year = {2020},
      publisher = {GitHub},
      journal = {GitHub repository},
      howpublished = {\url{https://github.com/ArnaudFickinger/gym-multigrid}},
    }

    @article{oguntola2023theory,
      title = {Theory of Mind as Intrinsic Motivation for Multi-Agent Reinforcement Learning},
      author = {Oguntola, Ini and Campbell, Joseph and Stepputtis, Simon and Sycara, Katia},
      journal = {arXiv preprint arXiv:2307.01158},
      year = {2023},
      url = {https://github.com/ini/multigrid},
    }

    @misc{mosaic_multigrid,
      author = {Mousa, Abdulhamid},
      title = {mosaic\_multigrid: Research-Grade Multi-Agent Gridworld Environments},
      year = {2026},
      publisher = {GitHub},
      journal = {GitHub repository},
      howpublished = {\url{https://github.com/Abdulhamid97Mousa/mosaic_multigrid}},
    }

    @article{terry2021pettingzoo,
      title = {PettingZoo: Gym for Multi-Agent Reinforcement Learning},
      author = {Terry, J. K and Black, Benjamin and Grammel, Nathaniel and Jayakumar, Mario
                and Hari, Ananth and Sullivan, Ryan and Santos, Luis S and Dieffendahl, Clemens
                and Horsch, Caroline and Perez-Vicente, Rodrigo and Williams, Niall L
                and Lokesh, Yashas and Ravi, Praveen},
      journal = {Advances in Neural Information Processing Systems},
      volume = {34},
      pages = {2242--2254},
      year = {2021},
      url = {https://pettingzoo.farama.org/},
    }