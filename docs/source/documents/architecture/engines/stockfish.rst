Stockfish
=========

`Stockfish <https://stockfishchess.org/>`_ is an open-source chess engine and
one of the strongest chess programs in the world. MOSAIC uses it as a built-in
opponent for the PettingZoo Chess environment (``chess_v6``).

Stockfish is **not** part of PettingZoo. PettingZoo's Chess environment uses
``python-chess`` for move validation and rendering. Stockfish is an optional
external engine that provides a strong opponent.

Installation
------------

.. code-block:: bash

   # Ubuntu / Debian / WSL
   sudo apt-get install -y stockfish

   # macOS
   brew install stockfish

   # Python wrapper (included in MOSAIC dependencies)
   pip install stockfish

Service Module
--------------

The Stockfish integration lives in:

- ``gym_gui/services/chess_ai/stockfish_service.py``

This service wraps the Stockfish binary via the UCI (Universal Chess Interface)
protocol and exposes it to MOSAIC's human-control and agent pipelines.

Usage in MOSAIC
---------------

- **Human vs Stockfish**: Play chess against the engine using the GUI.
  Select the Chess environment, choose Human-Control mode, and Stockfish
  acts as your opponent.
- **Agent vs Stockfish**: Train an RL agent or LLM agent against Stockfish
  as a fixed-strength sparring partner.
