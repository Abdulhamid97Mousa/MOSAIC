Overview
========

MOSAIC's rendering subsystem delivers environment frames to the screen through
two independent data paths: a :doc:`fastlane` for real-time display and a
:doc:`slow_lane` for durable telemetry, plus a pluggable
:doc:`strategies` layer that adapts pixel, grid, and board-game payloads to
Qt widgets.  All rendering surfaces converge inside :doc:`render_tabs` (the
central ``QTabWidget``) through individual :doc:`views` (``FastLaneTab``,
``LiveTelemetryTab``, ``OperatorRenderContainer``,
``MultiOperatorRenderView``).

.. mermaid::

   %%{init: {"flowchart": {"curve": "linear"}} }%%
   graph TB
       subgraph Sources["Frame Sources"]
           W1["CleanRL Worker"]
           W2["XuanCe Worker"]
           W3["Ray RLlib Worker"]
           W4["BALROG Worker"]
           W5["Session Controller<br/>(human / eval)"]
       end

       subgraph FastLane["Fast Lane · shared memory"]
           FLW["FastLaneWriter<br/>SPSC Ring Buffer"]
           FLC["FastLaneConsumer<br/>~60 Hz poll · 16 ms timer"]
           FLT["FastLaneTab<br/>Qt Quick / QML"]
       end

       subgraph SlowLane["Slow Lane · gRPC → SQLite"]
           PROXY["TrainerTelemetryProxy<br/>JSONL → gRPC"]
           GRPC["TrainerService (gRPC)"]
           BUS["RunBus<br/>pub-sub · 2 048 queue"]
           LTC["LiveTelemetryController<br/>background thread"]
           CM["CreditManager<br/>200 credits/stream"]
           RSR["RenderingSpeedRegulator<br/>100 ms · ~10 FPS"]
           LTT["LiveTelemetryTab"]
       end

       subgraph Strategies["Rendering Strategies"]
           REG["RendererRegistry"]
           RGB["RgbRendererStrategy<br/>mouse capture · scroll"]
           GRID["GridRendererStrategy<br/>FrozenLake · Taxi · Cliff"]
           BG["BoardGameRendererStrategy<br/>Chess · Go · Connect4 · …"]
       end

       subgraph Views["Render Views"]
           RT["RenderTabs<br/>(QTabWidget)"]
           ORC["OperatorRenderContainer"]
           MORV["MultiOperatorRenderView<br/>dynamic grid layout"]
       end

       W1 & W2 & W3 --> FLW --> FLC --> FLT
       W1 & W2 & W3 & W4 --> PROXY --> GRPC --> BUS
       BUS --> LTC
       LTC --> CM --> RSR --> LTT
       W5 --> LTT

       LTT --> REG
       REG --> RGB & GRID & BG

       FLT --> RT
       LTT --> RT
       ORC --> MORV --> RT

       style FastLane fill:#e8f5e9,stroke:#2e8b57,color:#333
       style SlowLane fill:#e3f2fd,stroke:#1565c0,color:#333
       style Strategies fill:#fff3e0,stroke:#e65100,color:#333
       style Views fill:#f3e5f5,stroke:#7b1fa2,color:#333

Why Two Lanes?
--------------

.. list-table::
   :widths: 12 44 44
   :header-rows: 1

   * - Lane
     - Mechanism
     - Use Case
   * - :doc:`Fast <fastlane>`
     - SPSC shared-memory ring buffer (magic ``FLAN``, sequence-number
       consistency).  ``FastLaneConsumer`` polls every **16 ms** (~60 Hz).
       The first shared-memory live visualization path in RL.
     - Live training frames: zero serialisation, lossy by design.
       CleanRL, XuanCe, Ray RLlib, SB3, SBX, Tianshou, and TorchRL
       workers produce fast-lane frames.
   * - :doc:`Slow <slow_lane>`
     - gRPC → ``RunBus`` (queue 2 048) → ``LiveTelemetryController`` →
       ``CreditManager`` (200 credits/stream) →
       ``RenderingSpeedRegulator`` (100 ms drain).
     - Durable step/episode telemetry, replay, W&B / TensorBoard integration.
       All worker types, including :doc:`BALROG </documents/architecture/workers/integrated_workers/BALROG_Worker/index>`, publish here.

Both lanes converge in :doc:`render_tabs`: ``FastLaneTab`` for the fast lane,
``LiveTelemetryTab`` for the slow lane, and ``OperatorRenderContainer`` for
multi-operator evaluation.  The :doc:`strategies` layer (``RendererRegistry``)
decides *how* each frame is painted: pixel, grid, or board game.

Relationship to Other Sections
------------------------------

- :doc:`/documents/architecture/overview`: system-level layer diagram that
  positions rendering within the broader MOSAIC stack.
- :doc:`/documents/runtime_logging/index`: the
  :doc:`/documents/runtime_logging/log_constants` file defines ``LOG_*``
  codes emitted by rendering components (``LOG1001``–``LOG1015`` for operator UI,
  ``LOG700+`` for workers).
- :doc:`/documents/runtime_logging/constants`: ``RenderDefaults``
  (``min_delay_ms=10``, ``default_delay_ms=100``, ``queue_size=32``),
  ``BufferDefaults``, and telemetry queue sizes all live in
  ``gym_gui/constants/constants_ui.py`` and ``constants_telemetry.py``.

.. toctree::
   :maxdepth: 2

   fastlane
   slow_lane
   strategies
   render_tabs
   views
