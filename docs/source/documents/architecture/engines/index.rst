Engines
=======

Engines are standalone game-playing programs that MOSAIC integrates as
built-in opponents. Unlike :doc:`Workers </documents/architecture/workers/index>`,
engines do not train policies. They provide a fixed-strength adversary
so humans or RL agents can play against a competent opponent without
launching a full training run.

.. mermaid::

   graph LR
       subgraph "MOSAIC GUI"
           HC["Human Control"]
           RL["RL Agent"]
       end

       subgraph "Engines"
           SF["Stockfish<br/>Chess"]
           KG["KataGo<br/>Go"]
           GG["GNU Go<br/>Go"]
       end

       HC -- "play vs" --> SF
       HC -- "play vs" --> KG
       HC -- "play vs" --> GG
       RL -- "train vs" --> SF
       RL -- "train vs" --> KG

       style HC fill:#4a90d9,stroke:#2e5a87,color:#fff
       style RL fill:#50c878,stroke:#2e8b57,color:#fff
       style SF fill:#ff7f50,stroke:#cc5500,color:#fff
       style KG fill:#ff7f50,stroke:#cc5500,color:#fff
       style GG fill:#ff7f50,stroke:#cc5500,color:#fff

Integrated Engines
------------------

.. list-table::
   :widths: 20 15 30 35
   :header-rows: 1

   * - Engine
     - Game
     - Service Module
     - Notes
   * - **Stockfish**
     - Chess
     - ``gym_gui.services.chess_ai.stockfish_service``
     - UCI protocol. Requires ``sudo apt install stockfish`` or ``brew install stockfish``.
       Used as opponent in PettingZoo Chess (``chess_v6``).
   * - **KataGo**
     - Go
     - ``gym_gui.services.go_ai.katago_service``
     - GTP protocol. Neural-network Go engine. Model weights stored in
       ``var/models/katago/``.
   * - **GNU Go**
     - Go
     - ``gym_gui.services.go_ai.gnugo_service``
     - GTP protocol. Classical Go engine (no neural network). Lighter alternative
       to KataGo.

Engine vs Worker
----------------

.. list-table::
   :widths: 25 37 38
   :header-rows: 1

   * -
     - Engine
     - Worker
   * - **Purpose**
     - Fixed-strength opponent
     - Trains RL policies
   * - **Learns?**
     - No
     - Yes
   * - **Protocol**
     - UCI / GTP
     - gRPC + JSONL (Fast Lane)
   * - **Example**
     - Stockfish, KataGo
     - CleanRL, XuanCe, Tianshou

.. toctree::
   :maxdepth: 2

   stockfish
   katago
   gnugo
