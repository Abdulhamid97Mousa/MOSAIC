Integrated Workers
==================

MOSAIC ships with twelve production-ready workers that wrap major RL
frameworks, LLM evaluation suites, VLM multimodal agents, multi-agent LLM
coordination, LLM chess play, human-in-the-loop control, and baseline agents.  Each worker follows the
:doc:`shim pattern <../concept>`: upstream libraries are **never
modified**; a thin integration layer translates between MOSAIC and
the library.

.. list-table::
   :header-rows: 1
   :widths: 18 18 22 22 20

   * - Worker
     - Paradigm
     - Algorithms / Models
     - Environments
     - Execution Model
   * - :doc:`MOSAIC LLM <MOSAIC_LLM_Worker/index>`
     - Multi-Agent LLM
     - OpenRouter, GPT-4o, Claude 3, Gemini, vLLM
     - MultiGrid Soccer/Collect, Melting Pot, Google Research Football, Minecraft
     - Subprocess
   * - :doc:`MOSAIC VLM <MOSAIC_VLM_Worker/index>`
     - Multi-Agent VLM
     - OpenRouter, GPT-4o, Claude 3, Gemini, vLLM (multimodal)
     - MultiGrid, Melting Pot, Google Research Football, Minecraft
     - Subprocess 
   * - :doc:`MOSAIC Human <MOSAIC_Human_Worker/index>`
     - Human-in-the-Loop
     - Human action selection via GUI
     - MiniGrid, Crafter, PettingZoo, Classic Control
     - Subprocess
   * - :doc:`MOSAIC Random <MOSAIC_Random_Worker/index>`
     - Baseline Agent
     - random (uniform sampling, no training)
     - All Gymnasium-compatible environments
     - Subprocess
   * - :doc:`MOSAIC Passive <MOSAIC_Passive_Worker/index>`
     - Passive Baseline
     - noop / still (env-aware, no training)
     - All Gymnasium-compatible environments
     - Subprocess
   * - :doc:`CleanRL <CleanRL_Worker/index>`
     - Single-Agent
     - PPO, DQN, SAC, TD3, DDPG, C51
     - Gymnasium, Atari, MiniGrid, BabyAI, Procgen
     - Subprocess
   * - :doc:`XuanCe <XuanCe_Worker/index>`
     - Multi-Agent
     - MAPPO, QMIX, MADDPG, VDN, COMA + 40 more
     - PettingZoo, SMAC, MultiGrid, MPE, Google Research Football
     - Subprocess
   * - :doc:`Ray RLlib <RLlib_Worker/index>`
     - Both
     - PPO, IMPALA, APPO, DQN, A2C
     - PettingZoo (SISL, Classic, Butterfly, MPE)
     - Subprocess
   * - :doc:`BALROG <BALROG_Worker/index>`
     - Single-Agent, LLM/VLM  
     - GPT-4o, Claude 3, Gemini, vLLM (local)
     - NetHack, MiniHack, BabyAI, Crafter, TextWorld
     - Subprocess 
   * - :doc:`Chess LLM <Chess_LLM_Worker/index>`
     - LLM Chess
     - GPT-4o, Claude 3, Gemini, vLLM (local)
     - PettingZoo Chess (chess_v6)
     - Subprocess 
   * - :doc:`Tianshou <Tianshou_Worker/index>`
     - Sing-Agent, Multi-Agent, MARL, Model-based RL
     - DQN, C51, Rainbow, IQN, PG, A2C, TRPO, PPO, DDPG, TD3, SAC, REDQ, BCQ, CQL, GAIL + more
     - Gymnasium, Atari, MuJoCo, Classic Control, Box2D
     - Subprocess
   * - :doc:`Jumanji <Jumanji_Worker/index>`
     - A suite of scalable reinforcement learning environments written in JAX
     - A2C, PPO (hardware-accelerated via JAX)
     - BinPack, TSP, CVRP, Knapsack, Game2048, Routing, Cleaner
     - Subprocess

Each worker provides:

- **CLI entry point** for subprocess launching by the Trainer Daemon
- **Configuration dataclass** implementing the ``WorkerConfig`` protocol
- **Runtime orchestrator** managing the training lifecycle
- **FastLane telemetry** for real-time frame streaming to the GUI
- **GUI form widgets** for visual experiment configuration
- **Automatic discovery** via Python entry points

.. mermaid::

   graph TB
       subgraph "MOSAIC GUI"
           FORM["Training Form<br/>(per-worker UI)"]
           DAEMON["Trainer Daemon"]
       end

       subgraph "Worker Subprocess"
           CLI["cli.py"]
           CFG["config.py"]
           RT["runtime.py"]
           FL["fastlane.py"]
           SITE["sitecustomize.py"]
       end

       subgraph "Upstream Library"
           LIB["CleanRL / XuanCe / RLlib<br/>(unmodified)"]
       end

       FORM -->|"config JSON"| DAEMON
       DAEMON -->|"spawn"| CLI
       CLI --> CFG --> RT
       RT --> FL
       RT --> LIB
       SITE -.->|"import-time patches"| LIB

       style FORM fill:#4a90d9,stroke:#2e5a87,color:#fff
       style DAEMON fill:#50c878,stroke:#2e8b57,color:#fff
       style CLI fill:#ff7f50,stroke:#cc5500,color:#fff
       style CFG fill:#ff7f50,stroke:#cc5500,color:#fff
       style RT fill:#ff7f50,stroke:#cc5500,color:#fff
       style FL fill:#ff7f50,stroke:#cc5500,color:#fff
       style SITE fill:#ff7f50,stroke:#cc5500,color:#fff
       style LIB fill:#e8e8e8,stroke:#999

GUI Integration
---------------

Each worker has dedicated GUI form widgets for experiment configuration:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Worker
     - Form Widgets
     - Purpose
   * - **CleanRL**
     - ``cleanrl_train_form.py``
       ``cleanrl_script_form.py``
       ``cleanrl_resume_form.py``
       ``cleanrl_policy_form.py``
     - Standard training, custom scripts,
       checkpoint resume, policy evaluation
   * - **XuanCe**
     - ``xuance_train_form.py``
       ``xuance_script_form.py``
     - Standard training (with backend selection),
       custom scripts
   * - **Tianshou**
     - ``tianshou_train_form.py``
       ``tianshou_script_form.py``
       ``tianshou_resume_form.py``
       ``tianshou_policy_form.py``
     - Standard training, custom scripts,
       checkpoint resume, policy evaluation
   * - **Ray RLlib**
     - (Configured via Advanced Config)
     - Distributed training setup

.. toctree::
   :maxdepth: 1

   MOSAIC_LLM_Worker/index
   MOSAIC_VLM_Worker/index
   Chess_LLM_Worker/index
   MOSAIC_Human_Worker/index
   MOSAIC_Random_Worker/index
   MOSAIC_Passive_Worker/index
   CleanRL_Worker/index
   XuanCe_Worker/index
   Tianshou_Worker/index
   Jumanji_Worker/index
   RLlib_Worker/index
   BALROG_Worker/index
