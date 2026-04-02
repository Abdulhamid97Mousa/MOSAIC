Overview
========

MOSAIC is built on a layered architecture that separates concerns
and enables extensibility.

.. figure:: /_static/figures/A_Full_Architecture.png
   :alt: MOSAIC Full Architecture
   :align: center
   :width: 100%

   Full architecture: Evaluation Phase (left), Training Phase (right),
   Daemon Process (gRPC Server, RunRegistry, Dispatcher, Broadcasters),
   and Worker Processes (CleanRL, XuanCe, Ray RLlib, BALROG, MOSAIC LLM).

System Layers
-------------

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Layer
     - Key Components
   * - **Visual Layer**
     - MainWindow, ControlPanel, RenderTabs, AdvancedConfigTab
   * - **Service Layer**
     - PolicyMappingService, ActorService, TelemetryService, OperatorService, SessionSeedManager, StorageRecorderService
   * - **Controller Layer**
     - SessionController, HumanInputController, InteractionController, LiveTelemetryController
   * - **Adapter Layer**
     - EnvironmentAdapter (base), PettingZooAdapter, ALEAdapter, MiniGridAdapter, ViZDoomAdapter, SMACAdapter, and 50+ environment-specific adapters
   * - **Worker Layer (gRPC/IPC)**
     - CleanRL, XuanCe, RLlib, BALROG, MOSAIC LLM, Chess LLM
   * - **Fast Lane (Shared Memory)**
     - FastLaneWriter, FastLaneReader, FastLaneConsumer, SPSC ring buffer for real-time frame delivery

Visual Layer
------------

The visual layer provides the user interface built with PyQt6.  The screenshot
below shows the four main regions of the MOSAIC Qt Shell, each highlighted with
a distinct colour:

.. figure:: /images/architecture/visual_layer.png
   :alt: MOSAIC Visual Layer, annotated screenshot
   :align: center
   :width: 100%

   Annotated screenshot of the MOSAIC Qt Shell showing all four visual regions.

.. list-table::
   :widths: 8 20 72
   :header-rows: 1

   * - Colour
     - Region
     - Description
   * - .. raw:: html

          <span style="color:#c0392b; font-weight:bold;">&#9679;</span>
     - **Main Window** 
     - The top-level application shell (``MainWindow``).  Hosts the menu bar
       (*Settings, Control Panel, Render View, Game Info, Runtime Log, Chat*),
       the three content panes below, and the global Dark Mode toggle.
   * - .. raw:: html

          <span style="color:#e67e22; font-weight:bold;">&#9679;</span>
     - **Control Panel** 
     - Left sidebar (``ControlPanel``).  Contains environment selection
       (*Family, Environment, Seed*), game configuration (*Input Mode, Display
       Resolution, Control Mode*), the game-flow buttons (*Start / Pause /
       Continue / Terminate / Agent Step / Reset*), and keyboard assignment
       for multi-human play via ``evdev``.
   * - .. raw:: html

          <span style="color:#2980b9; font-weight:bold;">&#9679;</span>
     - **Render Tabs** 
     - Centre area (``RenderTabs``, a ``QTabWidget``).  Displays the live
       environment frame through switchable tabs: *Grid, Raw, Video,
       Human Replay, Multi-Operator, Management, Tensorboard*.  Dynamic
       per-run tabs (e.g. ``FastLaneTab``, ``LiveTelemetryTab``) are added
       automatically when training starts.
       See :doc:`/documents/rendering_tabs/index` for the full rendering
       architecture.
   * - .. raw:: html

          <span style="color:#27ae60; font-weight:bold;">&#9679;</span>
     - **Runtime Logs** 
     - Right panel (``RuntimeLogPanel``).  Streams structured log messages
       with *Component* and *Severity* filters.  Every log line carries a
       ``LOG###`` code (see :doc:`/documents/runtime_logging/log_constants`)
       for fast searching.

**Component summary:**

- **MainWindow**: Application shell with tab management and menu bar
- **ControlPanel**: Environment selection and actor configuration
- **RenderTabs**: Display environment renders (RGB, ASCII, etc.)
- **RuntimeLogPanel**: Filterable structured log viewer
- **AdvancedConfigTab**: Fine-grained experiment configuration (accessible via the *Settings* menu)

Service Layer
-------------

Services provide business logic independent of the UI:

- **PolicyMappingService**: Per-agent policy binding with paradigm awareness and link groups for one-to-one and one-to-many policy mappings in multi-agent RL
- **ActorService**: Actor registration and action selection — see :doc:`/documents/architecture/actors/index`
- **TelemetryService**: Aggregates telemetry events and forwards to storage backends
- **OperatorService**: Multi-agent environment orchestration during evaluation
- **SessionSeedManager**: Deterministic seeding across Python, NumPy, and Qt for reproducibility
- **StorageRecorderService**: HDF5-based session recording and replay
- **ServiceLocator**: Central registry for service discovery

Controller Layer
----------------

Controllers coordinate between services and the UI via Qt signals:

- **SessionController**: Manages the adapter lifecycle and evaluation loop
- **HumanInputController**: Captures keyboard and mouse input for human agents
- **InteractionController**: Abstract base with environment-specific subclasses (Box2D, TurnBased, ALE, ViZDoom, SMAC, Procgen, Jumanji)
- **LiveTelemetryController**: Real-time telemetry display and updates

Adapter Layer
-------------

Adapters provide a unified ``EnvironmentAdapter`` interface to different environment types.
MOSAIC uses an adapter factory pattern to instantiate the correct adapter at runtime:

- **EnvironmentAdapter**: Abstract base class defining the step/reset/render contract
- **PettingZooAdapter**: PettingZoo multi-agent environments (AEC and Parallel)
- **ALEAdapter**: Atari 2600 games via the Arcade Learning Environment
- **MiniGridAdapter**: Procedural grid-world navigation (25+ variants)
- **BabyAIAdapter**: Language-grounded instruction following (35+ variants)
- **ViZDoomAdapter**: Doom-based first-person visual RL
- **SMACAdapter / SMACv2Adapter**: StarCraft Multi-Agent Challenge
- **JumanjiAdapter**: JAX-accelerated environments (20+ variants)
- **And 50+ more** covering Gymnasium, Box2D, MuJoCo, Crafter, MiniHack, NetHack, TextWorld, Procgen, Overcooked, RWARE, Melting Pot, PyBullet Drones, and others

Worker Layer
------------

External training and inference backends communicate via gRPC/IPC:

- **CleanRL**: Single-agent RL (PPO, DQN, SAC, TD3, DDPG, C51, Rainbow)
- **XuanCe**: Multi-agent RL (MAPPO, QMIX, MADDPG, VDN, COMA)
- **RLlib**: Distributed RL with Ray (PPO, IMPALA, APPO)
- **BALROG**: Single-agent LLM benchmarking (MiniGrid, BabyAI, MiniHack, Crafter)
- **MOSAIC LLM**: Multi-agent LLM with coordination strategies and Theory of Mind
- **Chess LLM**: LLM chess play with multi-turn dialog

Fast Lane
---------

The :doc:`/documents/rendering_tabs/fastlane` provides real-time frame delivery from workers to the GUI
via a shared-memory SPSC ring buffer, bypassing the gRPC/SQLite slow lane
for rendering. Key components:

- **FastLaneWriter / FastLaneReader**: Shared-memory ring buffer with seqlock semantics
- **FastLaneConsumer**: Qt-side poller that converts shared-memory frames to ``QImage``
- **tile_frames()**: Composites vectorized environment frames into a single image
- **worker_helpers**: Injects fast-lane environment variables into worker subprocess launch

See the :doc:`/documents/rendering_tabs/index` section for the full rendering architecture.
