Operators
=========

An **Operator** is the agent-level interface of MOSAIC, the unified
abstraction that lets the GUI assign a worker to each individual agent
or a group of agents.  While :doc:`Workers <../workers/index>` handle
process-level concerns (training, telemetry, GPU isolation), Operators
are strictly for **evaluation and interactive play**. Then, the worker inside
an Operator loads a trained policy (or calls an LLM API, or reads
keyboard input) and computes actions step-by-step.  The Operator wraps
this and answers the question *"given this observation, what action
should I take?"*

.. mermaid::

   %%{init: {"flowchart": {"curve": "linear"}} }%%
   graph TB
       GUI["Qt6 GUI<br/>(Main Process)"]
       LAUNCHER["OperatorLauncher<br/>(Subprocess Manager)"]

       GUI --> LAUNCHER

       LAUNCHER -- "stdin/stdout JSON" --> H_OP
       LAUNCHER -- "stdin/stdout JSON" --> L_OP
       LAUNCHER -- "stdin/stdout JSON" --> V_OP
       LAUNCHER -- "stdin/stdout JSON" --> R_OP
       LAUNCHER -- "stdin/stdout JSON" --> RND_OP
       LAUNCHER -- "stdin/stdout JSON" --> P_OP

       subgraph H_OP["Human Operator"]
           HW["human_worker<br/>Keyboard Input"]
       end

       subgraph L_OP["LLM Operator"]
           LW1["balrog_worker<br/>Single-Agent"]
           LW2["llm_worker<br/>MOSAIC Native"]
           LW3["chess_worker<br/>Two-Player"]
       end

       subgraph V_OP["VLM Operator"]
           VW["vlm_worker<br/>Vision-Language"]
       end

       subgraph R_OP["RL Operator"]
           RW1["cleanrl_worker<br/>PPO / DQN"]
           RW2["xuance_worker<br/>MAPPO / QMIX"]
           RW3["ray_worker<br/>PPO / IMPALA"]
       end

       subgraph RND_OP["Random Operator"]
           RNDW["random_worker<br/>Uniform Random"]
       end

       subgraph P_OP["Passive Operator"]
           PW["passive_worker<br/>NOOP / STILL"]
       end

       style GUI fill:#4a90d9,stroke:#2e5a87,color:#fff
       style LAUNCHER fill:#50c878,stroke:#2e8b57,color:#fff
       style H_OP fill:#9370db,stroke:#6a0dad,color:#fff
       style L_OP fill:#9370db,stroke:#6a0dad,color:#fff
       style V_OP fill:#9370db,stroke:#6a0dad,color:#fff
       style R_OP fill:#9370db,stroke:#6a0dad,color:#fff
       style RND_OP fill:#9370db,stroke:#6a0dad,color:#fff
       style P_OP fill:#9370db,stroke:#6a0dad,color:#fff
       style HW fill:#ff7f50,stroke:#cc5500,color:#fff
       style LW1 fill:#ff7f50,stroke:#cc5500,color:#fff
       style LW2 fill:#ff7f50,stroke:#cc5500,color:#fff
       style LW3 fill:#ff7f50,stroke:#cc5500,color:#fff
       style VW fill:#ff7f50,stroke:#cc5500,color:#fff
       style RW1 fill:#ff7f50,stroke:#cc5500,color:#fff
       style RW2 fill:#ff7f50,stroke:#cc5500,color:#fff
       style RW3 fill:#ff7f50,stroke:#cc5500,color:#fff
       style RNDW fill:#ff7f50,stroke:#cc5500,color:#fff
       style PW fill:#ff7f50,stroke:#cc5500,color:#fff

Key Principles
--------------

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - **Protocol-Based**
     - Operators implement Python ``Protocol`` classes -- no base class
       inheritance required.  Any object with ``select_action(obs)`` is
       a valid operator.
   * - **Category System**
     - Every operator belongs to a category: ``human``, ``llm``, ``vlm``,
       ``rl``, ``random``, or ``passive``.  The GUI adapts its
       configuration UI based on category.
   * - **Interactive Mode**
     - Operators run as subprocesses with ``--interactive`` flag, enabling
       step-by-step JSON commands over stdin/stdout.  This keeps the GUI
       responsive while operators compute.
   * - **Multi-Operator Comparison**
     - Multiple operators can run side-by-side on the same environment
       with shared seeds for scientific comparison (e.g., LLM vs RL on
       the same MiniGrid layout).
   * - **Decoupled Execution**
     - Manual mode (click-to-step) and Script mode (automated experiments)
       are fully independent code paths with separate state machines.

Available Operators
-------------------

.. list-table::
   :header-rows: 1
   :widths: 18 15 32 35

   * - Operator
     - Category
     - Backend
     - Use Case
   * - **Human**
     - human
     - Keyboard input via GUI
     - Manual play and debugging
   * - **BALROG LLM**
     - llm
     - balrog_worker (vLLM, OpenRouter)
     - Single-agent LLM benchmarking on MiniGrid/BabyAI
   * - **MOSAIC LLM**
     - llm
     - mosaic_llm_worker (vLLM, OpenRouter, OpenAI, Anthropic)
     - Multi-agent LLM with coordination and Theory of Mind
   * - **Chess LLM**
     - llm
     - chess_worker (llm_chess prompting)
     - LLM chess play with multi-turn dialog
   * - **CleanRL**
     - rl
     - cleanrl_worker (PPO, DQN)
     - Trained single-agent RL policy evaluation
   * - **XuanCe**
     - rl
     - xuance_worker (MAPPO, QMIX)
     - Trained multi-agent RL policy evaluation
   * - **Ray RLlib**
     - rl
     - ray_worker (PPO, IMPALA)
     - Distributed RL policy evaluation
   * - **MOSAIC Random Worker**
     - random
     - random_worker (random action)
     - Random action selection for experiments
   * - **MOSAIC Passive Worker**
     - passive
     - passive_worker (NOOP/STILL)
     - Do-nothing agent for experiments

.. tip::

   An Operator *wraps* one or more Workers.  The Operator is the
   agent-level interface (``select_action(obs) -> action``) that the
   GUI interacts with.  The Worker is the process-level engine that
   runs inside the Operator.  This separation is what enables heterogeneous
   teams -- e.g., an RL-trained policy and an LLM playing side-by-side
   in the same multi-agent environment.  See :doc:`concept` for the
   full motivation and diagrams.

.. note::

   **Policy Mappings for Multi-Agent RL:** When deploying RL policies
   in multi-agent scenarios, MOSAIC supports flexible policy-to-agent
   mappings through link groups. This enables **one-to-one** mappings
   (each agent has its own policy) and **one-to-many** mappings
   (multiple agents share a single policy checkpoint). Link groups are
   essential for MAPPO/IPPO evaluation because these algorithms store
   all agents' policies in a single checkpoint file. See
   :doc:`../policy_mapping` for complete documentation.


.. toctree::
   :maxdepth: 2

   concept
   homogenous_decision_makers/index
   heterogeneous_decision_maker/index
   policy_mappings
   architecture
   lifecycle
   development
   examples
