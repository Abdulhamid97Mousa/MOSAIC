Developing a Worker
===================

This guide walks through creating a new MOSAIC worker from scratch.
A complete worker has two sides:

.. mermaid::

   graph LR
       subgraph "Backend (3rd_party/workers/)"
           CFG["Config<br/>config.py"]
           RT["Runtime<br/>runtime.py"]
           CLI["CLI<br/>cli.py"]
           EP["Entry Point<br/>pyproject.toml"]
       end

       subgraph "Frontend (gym_gui/ui/)"
           FORM["Train Form<br/>widgets/"]
           FAC["Form Factory<br/>forms/"]
           CAT["Worker Catalog<br/>worker_catalog/"]
           PRES["Presenter<br/>presenters/"]
       end

       subgraph "Bridge"
           HANDLER["TrainingFormHandler"]
           CLIENT["TrainerClient (gRPC)"]
       end

       FORM -->|"get_config()"| HANDLER
       HANDLER -->|"submit_run()"| CLIENT
       CLIENT -->|"gRPC"| RT
       EP -.->|"discovery"| CFG
       FAC -.->|"creates"| FORM
       CAT -.->|"metadata"| FAC

       style CFG fill:#ff7f50,stroke:#cc5500,color:#fff
       style RT fill:#ff7f50,stroke:#cc5500,color:#fff
       style CLI fill:#ff7f50,stroke:#cc5500,color:#fff
       style EP fill:#ff7f50,stroke:#cc5500,color:#fff
       style FORM fill:#4a90d9,stroke:#2e5a87,color:#fff
       style FAC fill:#4a90d9,stroke:#2e5a87,color:#fff
       style CAT fill:#4a90d9,stroke:#2e5a87,color:#fff
       style PRES fill:#4a90d9,stroke:#2e5a87,color:#fff
       style HANDLER fill:#50c878,stroke:#2e8b57,color:#fff
       style CLIENT fill:#50c878,stroke:#2e8b57,color:#fff

- **Backend:** the worker package under ``3rd_party/workers/``. Handles training
  logic, telemetry, and the CLI entry point that the Daemon spawns.
- **Frontend:** the Qt6 UI integration under ``gym_gui/ui/``. Handles
  the training form dialog, worker catalog entry, presenter, and how
  user configuration reaches the backend via gRPC.

Telemetry Paths
---------------

MOSAIC workers support two telemetry paths for different use cases:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Path
     - Use Case
     - Data Flow
   * - **Slow Lane**
     - Persistent storage, replays, analytics
     - Worker → gRPC → SQLite → GUI
   * - **FastLane**
     - Real-time rendering, low latency
     - Worker → Shared Memory → GUI (~60 Hz)

.. note::

   When ``fastlane_only=True``, the worker skips the slow lane entirely
   and only streams frames via shared memory. This is ideal for training
   runs where you want real-time visualization without disk I/O overhead.

   See :doc:`backend` for FastLane implementation details and the critical
   cleanup requirements for shared memory segments.

.. toctree::
   :maxdepth: 2

   backend
   frontend
