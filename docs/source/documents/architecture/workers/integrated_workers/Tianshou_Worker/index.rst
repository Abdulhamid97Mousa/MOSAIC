Tianshou Worker
===============

.. image:: /images/workers/tianshou_logo.png
   :alt: Tianshou Logo
   :align: center
   :width: 60%

.. raw:: html

   <br>

The Tianshou worker is MOSAIC's integration of the
`Tianshou <https://github.com/thu-ml/tianshou>`_ deep reinforcement learning
platform.  Tianshou (v2.0) provides a modular, type-safe PyTorch framework
with clear separation between ``Algorithm`` and ``Policy`` abstractions,
supporting online (on- and off-policy), offline, and imitation learning
behind the standard :doc:`shim pattern <../../concept>`.

.. list-table::
   :widths: 25 75

   * - **Paradigm**
     - Single-agent (sequential)
   * - **Algorithms**
     - PPO, DQN (integrated); 30+ available upstream (SAC, TD3, DDPG,
       A2C, TRPO, C51, Rainbow, IQN, FQF, BCQ, CQL, GAIL, ICM, and more)
   * - **Environments**
     - `Gymnasium <https://gymnasium.farama.org/>`_, Atari, MuJoCo,
       Classic Control, Box2D, MiniGrid, Toy Text
   * - **Execution**
     - Subprocess (one OS process per training run)
   * - **GPU required**
     - No (optional CUDA acceleration)
   * - **Upstream version**
     - 2.0.0 (integrated as git submodule)
   * - **Source**
     - ``3rd_party/workers/tianshou_worker/tianshou_worker/``

.. note::

   **Early integration.**  The Tianshou worker currently has **PPO and DQN**
   wired end-to-end.  The remaining algorithms from Tianshou's catalog are
   available in the submodule but have not yet been connected to the MOSAIC
   launcher and GUI forms.  See `Current Limitations`_ for details.

About Tianshou
--------------

Tianshou (meaning "divinely ordained" in Chinese) is developed by
Tsinghua University and the appliedAI Institute.  Version 2.0 is a
complete overhaul that introduces:

- Clear separation between ``Algorithm`` (learning logic) and ``Policy``
  (action selection), replacing the monolithic ``BasePolicy`` of v1.
- Renamed, more intuitive parameters (e.g. ``n_step_return_horizon``
  instead of ``n_step``).
- Type-level separation between on-policy, off-policy, and offline
  algorithms in the class hierarchy.
- High-level ``ExperimentBuilder`` API for declarative experiment setup
  alongside the low-level procedural API for maximum control.

Tianshou's full algorithm catalog includes:

.. list-table::
   :header-rows: 1
   :widths: 22 50 28

   * - Family
     - Algorithms
     - Notes
   * - **Q-Learning**
     - DQN, Double DQN, Dueling DQN, Branching DQN, C51, Rainbow,
       QRDQN, IQN, FQF
     - Discrete action spaces
   * - **Policy Gradient**
     - PG (REINFORCE), NPG, A2C, TRPO, PPO
     - On-policy, discrete and continuous
   * - **Continuous Control**
     - DDPG, TD3, SAC, REDQ, Discrete SAC
     - Off-policy actor-critic
   * - **Offline RL**
     - BCQ, CQL, TD3+BC, CRR, Discrete BCQ/CQL/CRR
     - Learning from static datasets
   * - **Imitation Learning**
     - IL (vanilla), GAIL
     - Learning from demonstrations
   * - **Exploration**
     - ICM, PER, HER, PSRL
     - Curiosity, prioritized replay, hindsight

Architecture
------------

.. mermaid::

   graph TB
       subgraph "MOSAIC GUI"
           FORM["Training Form<br/>(Tianshou widgets)"]
           DAEMON["Trainer Daemon"]
       end

       subgraph "Worker Subprocess"
           CLI["cli.py<br/>entry point"]
           CFG["config.py<br/>TianshouWorkerConfig"]
           RT["runtime.py<br/>TianshouWorkerRuntime"]
           LAUNCH["launcher.py<br/>algorithm dispatch"]
       end

       subgraph "Upstream Tianshou (v2.0)"
           ALGO["PPO / DQN / ...<br/>Algorithm + Policy + Collector + Trainer"]
       end

       FORM -->|"config JSON"| DAEMON
       DAEMON -->|"spawn"| CLI
       CLI --> CFG --> RT
       RT -->|"subprocess"| LAUNCH
       LAUNCH --> ALGO

       style FORM fill:#4a90d9,stroke:#2e5a87,color:#fff
       style DAEMON fill:#50c878,stroke:#2e8b57,color:#fff
       style CLI fill:#ff7f50,stroke:#cc5500,color:#fff
       style CFG fill:#ff7f50,stroke:#cc5500,color:#fff
       style RT fill:#ff7f50,stroke:#cc5500,color:#fff
       style LAUNCH fill:#ff7f50,stroke:#cc5500,color:#fff
       style ALGO fill:#e8e8e8,stroke:#999

**Lifecycle of a training run:**

1. The GUI form (``TianshouTrainForm``) builds a ``TianshouWorkerConfig``
   and hands it to the Trainer Daemon as JSON.
2. The daemon spawns ``python -m tianshou_worker.launcher --config-file <path>``.
3. ``launcher.py`` loads the config, looks up the algorithm in ``ALGO_MAP``,
   and calls the corresponding runner function (``run_ppo`` or ``run_dqn``).
4. The runner creates ``SubprocVectorEnv``, builds the Tianshou 2.0
   component stack (``Net`` -> ``Actor``/``Critic`` -> ``Policy`` ->
   ``Algorithm``), sets up ``Collector`` + ``VectorReplayBuffer``, and
   launches training via ``algorithm.run_training()``.
5. TensorBoard metrics are written to ``var/trainer/runs/{run_id}/``.
6. FastLane environment variables are configured by ``runtime.py``
   before spawning the subprocess.

Tianshou 2.0 Component Stack
-----------------------------

Tianshou 2.0 introduces a clean separation of concerns.  The MOSAIC
launcher constructs the following component stack for each algorithm:

.. code-block:: text

   Environment (gymnasium.Env)
     -> SubprocVectorEnv (parallelized)
       -> Collector (data collection)
         -> VectorReplayBuffer (storage)
           -> Algorithm (learning logic)
             -> Policy (action selection)
               -> Network (neural network)
                 -> Net / Actor / Critic

**PPO stack (on-policy):**

.. code-block:: python

   # Network
   net = Net(state_shape, hidden_sizes=[64, 64])
   actor = Actor(net, action_shape)
   critic = Critic(net)

   # Policy
   policy = ProbabilisticActorPolicy(actor, dist_fn, action_space)

   # Algorithm
   algorithm = PPO(policy, critic, optim, eps_clip=0.2, ...)

   # Training
   algorithm.run_training(OnPolicyTrainerParams(...))

**DQN stack (off-policy):**

.. code-block:: python

   # Network
   net = Net(state_shape, action_shape, hidden_sizes=[128, 128])

   # Policy
   policy = DiscreteQLearningPolicy(net, action_space, eps_training=0.1)

   # Algorithm
   algorithm = DQN(policy, optim, gamma=0.99, target_update_freq=320)

   # Training
   algorithm.run_training(OffPolicyTrainerParams(...))

Configuration
-------------

The ``TianshouWorkerConfig`` dataclass (``config.py``) is a frozen
dataclass implementing the MOSAIC ``WorkerConfig`` protocol:

.. code-block:: python

   @dataclass(frozen=True)
   class TianshouWorkerConfig:
       run_id: str                     # ULID-format unique run identifier
       algo: str                       # Algorithm name ("ppo", "dqn")
       env_id: str                     # Gymnasium environment ID
       total_timesteps: int            # Training budget
       seed: Optional[int] = None      # Random seed
       extras: dict[str, Any] = ...    # Algorithm-specific hyperparameters
       worker_id: Optional[str] = None
       raw: dict[str, Any] = ...       # Full raw payload

Key ``extras`` fields:

- ``lr``: learning rate
- ``hidden_sizes``: network architecture (e.g. ``[64, 64]``)
- ``batch_size``: optimization batch size
- ``epoch``: number of training epochs
- ``buffer_size``: replay buffer capacity (off-policy)
- ``num_envs``: number of parallel environments
- ``step_per_collect``: steps per collection phase (on-policy)
- ``eps_train`` / ``eps_test``: exploration rates (DQN)
- ``fastlane_enabled``: enable real-time frame streaming
- ``video_mode``: FastLane video mode (``"single"``)
- ``eval_only``: run evaluation instead of training
- ``eval_episodes``: number of evaluation episodes
- ``policy_path``: path to trained policy checkpoint
- ``resume_from``: path to checkpoint for resume

The config supports:

- ``to_dict()`` / ``from_dict()``: JSON serialization
- ``with_overrides()``: create a new config with selective field updates
- Nested format loading: extracts config from ``metadata.worker.config``

FastLane Telemetry
------------------

FastLane environment variables are set by ``runtime.py`` via
``apply_fastlane_environment()`` before spawning the subprocess:

- ``GYM_GUI_FASTLANE_ONLY``: ``1`` to stream, ``0`` to disable
- ``GYM_GUI_FASTLANE_SLOT``: which parallel env to probe
- ``GYM_GUI_FASTLANE_VIDEO_MODE``: ``"single"`` (default)
- ``GYM_GUI_FASTLANE_GRID_LIMIT``: max envs to tile

GUI Integration
---------------

The Tianshou worker provides four dedicated form widgets in
``gym_gui/ui/widgets/`` and a presenter in ``gym_gui/ui/presenters/workers/``:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Form
     - Purpose
   * - ``tianshou_train_form.py``
     - Primary training dialog.  Algorithm selection (PPO, DQN),
       environment family and ID selection, hyperparameter tuning
       (dynamically generated from ``_ALGO_PARAM_SPECS``), seed,
       timesteps, FastLane toggle.
   * - ``tianshou_script_form.py``
     - Custom Python script launcher.  Discovers ``.py`` scripts from
       ``TIANSHOU_SCRIPTS_DIR`` and allows selection/execution.
   * - ``tianshou_resume_form.py``
     - Resume training from a checkpoint.  Browses for ``.pth``/``.pt``
       files and auto-populates algorithm/environment by reading
       ``config-*.json`` from the checkpoint directory.
   * - ``tianshou_policy_form.py``
     - Policy evaluation dialog.  Loads a trained checkpoint, configures
       evaluation episodes, and optionally enables FastLane rendering.
   * - ``tianshou_worker_presenter.py``
     - Creates ``FastLaneTab`` for live video streaming when
       ``fastlane_enabled`` is set in the run config.

All four forms self-register with the ``WorkerFormFactory`` at import
time via the factory pattern at the bottom of each module.

Worker Discovery
----------------

The worker registers itself via the ``mosaic.workers`` entry point
in ``pyproject.toml``:

.. code-block:: toml

   [project.entry-points."mosaic.workers"]
   tianshou = "tianshou_worker:get_worker_metadata"

``get_worker_metadata()`` returns:

.. code-block:: python

   WorkerCapabilities(
       worker_type="tianshou",
       supported_paradigms=("sequential",),
       env_families=("gymnasium", "atari", "mujoco", "pettingzoo"),
       action_spaces=("discrete", "continuous"),
       observation_spaces=("vector", "image"),
       max_agents=1,
       supports_checkpointing=True,
       supports_pause_resume=False,
       requires_gpu=False,
       estimated_memory_mb=512,
   )

Current Limitations
-------------------

The Tianshou worker is an early integration with known gaps compared to
the CleanRL and XuanCe workers:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Gap
     - Description
   * - **Algorithm coverage**
     - Only PPO and DQN are wired; Tianshou provides 30+ algorithms
       upstream.  The launcher uses a static ``ALGO_MAP`` dict instead of
       dynamic registry-based lookup.
   * - **No sitecustomize.py**
     - Missing import-time patches for ``gym.make()`` wrapping, TensorBoard
       redirect, ``torch.save()`` auto-mkdir, and checkpoint resume hooks.
   * - **No dedicated fastlane.py**
     - No ``FastLaneTelemetryWrapper``, no frame throttling, no grid mode.
       Only basic environment variable setup via ``apply_fastlane_environment``.
   * - **No analytics manifest**
     - No ``analytics.json`` written on run completion (TensorBoard paths,
       checkpoint locations, etc.).
   * - **No dry-run validation**
     - CLI does not support ``--dry-run``.  No
       ``validation_tianshou_worker_form.py`` for pre-flight config checks.
   * - **No interactive runtime**
     - No step-by-step JSON IPC protocol for GUI-driven policy evaluation.
   * - **No curriculum training**
     - No environment switching with weight preservation across phases.
   * - **No WANDB integration**
     - No Weights & Biases logging support.
   * - **No telemetry emitter**
     - No ``run_started`` / ``heartbeat`` / ``run_completed`` lifecycle events.
   * - **Limited test coverage**
     - 9 test cases (vs 1000+ in XuanCe worker).

See the development progress report at
``docs/Development_Progress/1.0_DAY_70/TASK_2/TIANSHOU_WORKER_TECHNICAL_REPORT.md``
for the full gap analysis and implementation roadmap.

Metadata & Schemas
------------------

Algorithm hyperparameter schemas are defined in
``metadata/tianshou/2.0.0/schemas.json``:

.. code-block:: json

   {
     "algorithms": {
       "ppo": {
         "fields": [
           {"name": "lr", "type": "float", "default": 3e-4, "help": "Learning rate"},
           {"name": "hidden_sizes", "type": "list[int]", "default": [64, 64]},
           {"name": "eps_clip", "type": "float", "default": 0.2},
           {"name": "gae_lambda", "type": "float", "default": 0.95},
           {"name": "batch_size", "type": "int", "default": 64}
         ]
       },
       "dqn": {
         "fields": [
           {"name": "lr", "type": "float", "default": 1e-3},
           {"name": "hidden_sizes", "type": "list[int]", "default": [128, 128]},
           {"name": "gamma", "type": "float", "default": 0.99},
           {"name": "target_update_freq", "type": "int", "default": 320},
           {"name": "is_double", "type": "bool", "default": true}
         ]
       }
     }
   }

The ``TianshouTrainForm`` uses ``_ALGO_PARAM_SPECS`` (hardcoded in the
form widget) to dynamically generate hyperparameter input fields when the
algorithm selection changes.  Future work will drive this from
``schemas.json`` instead.

Dependencies
------------

The worker depends on:

- **Tianshou** v2.0.0 (git submodule at
  ``3rd_party/workers/tianshou_worker/tianshou``)
- **PyTorch** (deep learning backend)
- **Gymnasium** (environment API)
- **NumPy** (numerical operations)
- **ULID** (time-sortable unique run identifiers)

Install with:

.. code-block:: bash

   pip install -e ".[tianshou]"
   pip install -e 3rd_party/workers/tianshou_worker/tianshou


.. toctree::
   :maxdepth: 1

   installation
   common_errors
