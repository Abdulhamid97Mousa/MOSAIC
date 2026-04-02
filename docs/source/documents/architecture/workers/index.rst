Workers
=======

A **Worker** is a process-isolated wrapper around a reinforcement
learning library or framework.  Workers are the execution layer of
the platform. These workers are the main decision makers, they run the actual training, evaluation, custom scripts and so on. Whereas MOSAIC handles orchestration, telemetry, and visualization.

Some workers are designed to facilitate training reinforcement learning algorithms. These workers often bundle training scripts, evaluation scripts, and benchmark files. While other Workers are designed for evaluation only, such as LLM related benchmarks. The common thread is that they all implement the same simple interface, which allows the GUI to interact with them in a consistent way. 

.. mermaid::

   graph TB
       subgraph "MOSAIC Core"
           GUI["Qt6 GUI<br/>(Main Process)"]
           Daemon["Trainer Daemon<br/>(AsyncIO)"]
       end

       subgraph "Worker Sub-Processes"
           W1["CleanRL Worker<br/>PPO · DQN · SAC"]
           W2["XuanCe Worker<br/>MAPPO · QMIX"]
           W3["RLlib Worker<br/>PPO · IMPALA"]
           W4["Tianshou Worker<br/>PPO · DQN"]
           W5["Jumanji Worker<br/>JAX Combinatorial"]
           W6["BALROG Worker<br/>Single-Agent LLM"]
           W7["MOSAIC LLM Worker<br/>Multi-Agent LLM"]
       end

       GUI -- "gRPC" --> Daemon
       Daemon -- "spawn + JSONL" --> W1
       Daemon -- "spawn + JSONL" --> W2
       Daemon -- "spawn + JSONL" --> W3
       Daemon -- "spawn + JSONL" --> W4
       Daemon -- "spawn + JSONL" --> W5
       Daemon -- "spawn + JSONL" --> W6
       Daemon -- "spawn + JSONL" --> W7

       style GUI fill:#4a90d9,stroke:#2e5a87,color:#fff
       style Daemon fill:#50c878,stroke:#2e8b57,color:#fff
       style W1 fill:#ff7f50,stroke:#cc5500,color:#fff
       style W2 fill:#ff7f50,stroke:#cc5500,color:#fff
       style W3 fill:#ff7f50,stroke:#cc5500,color:#fff
       style W4 fill:#ff7f50,stroke:#cc5500,color:#fff
       style W5 fill:#ff7f50,stroke:#cc5500,color:#fff
       style W6 fill:#ff7f50,stroke:#cc5500,color:#fff
       style W7 fill:#ff7f50,stroke:#cc5500,color:#fff

Key Principles
--------------

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - **Process Isolation**
     - Each worker runs as a separate OS process.  A worker crash never
       takes down the GUI or other workers.
   * - **Zero Modification**
     - Upstream libraries (CleanRL, Ray, XuanCe) are **never modified**.
       A thin "shim" layer translates between MOSAIC and the library.
   * - **JSONL Telemetry**
     - Workers emit structured JSON lines to ``stdout``. This is the simplest
       possible output mechanism.  No gRPC client code required inside
       the worker itself.
   * - **Automatic Discovery**
     - Workers register via Python entry points
       (``[project.entry-points."mosaic.workers"]``).  The GUI discovers
       them at startup.
   * - **Protocol-Based**
     - Workers implement Python ``Protocol`` classes
       instead of inheriting from base classes.

Available Workers
-----------------

.. list-table::
   :header-rows: 1
   :widths: 18 15 32 35

   * - Worker
     - Paradigm
     - Algorithms
     - Use Case
   * - :doc:`CleanRL <integrated_workers/CleanRL_Worker/index>`
     - Single-Agent
     - PPO, DQN, SAC, TD3, DDPG, C51
     - Simple single-file RL training
   * - :doc:`XuanCe <integrated_workers/XuanCe_Worker/index>`
     - Multi-Agent
     - MAPPO, QMIX, MADDPG, VDN, COMA
     - Multi-agent RL research
   * - :doc:`Tianshou <integrated_workers/Tianshou_Worker/index>`
     - Single-Agent
     - DQN, C51, Rainbow, IQN, PPO, A2C, TRPO, DDPG, TD3, SAC, REDQ, BCQ, CQL, GAIL + more
     - Modular PyTorch RL (type-safe, dual API)
   * - :doc:`Jumanji <integrated_workers/Jumanji_Worker/index>`
     - JAX-based
     - A2C, PPO (hardware-accelerated via JAX)
     - Combinatorial & logistics environments (BinPack, TSP, Routing)
   * - :doc:`RLlib <integrated_workers/RLlib_Worker/index>`
     - Multi-Agent
     - PPO, IMPALA, APPO, SAC, DQN
     - Distributed training at scale
   * - :doc:`BALROG <integrated_workers/BALROG_Worker/index>`
     - Evaluation
     - GPT-4, Claude, Llama (single-agent)
     - Single-agent LLM benchmarking (MiniGrid, BabyAI)
   * - :doc:`MOSAIC LLM <integrated_workers/MOSAIC_LLM_Worker/index>`
     - Evaluation
     - GPT-4, Claude, Llama (multi-agent)
     - Multi-agent LLM with coordination strategies and Theory of Mind

.. tip::

   Install a specific worker with ``pip install -e ".[<worker>]"``.
   For example: ``pip install -e ".[cleanrl]"``,
   ``pip install -e ".[xuance]"``, or
   ``pip install -e ".[tianshou]"``.


.. toctree::
   :maxdepth: 3

   concept
   architecture
   lifecycle
   integrated_workers/index
   development/index
   examples
