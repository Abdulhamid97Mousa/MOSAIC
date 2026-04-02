KataGo
======

`KataGo <https://github.com/lightvector/KataGo>`_ is a neural-network-based Go
engine. It is one of the strongest open-source Go programs, using deep learning
and Monte Carlo Tree Search (MCTS).

MOSAIC integrates KataGo as a built-in opponent for Go environments
(OpenSpiel Go, PettingZoo Go).

Installation
------------

KataGo requires a binary and model weights:

.. code-block:: bash

   # Ubuntu / Debian
   sudo apt-get install -y katago

   # Or download from GitHub releases
   # https://github.com/lightvector/KataGo/releases

Model weights are stored in ``var/models/katago/``.

Service Module
--------------

The KataGo integration lives in:

- ``gym_gui/services/go_ai/katago_service.py``
- ``gym_gui/services/go_ai/gtp_engine.py`` (shared GTP base class)

KataGo communicates via the GTP (Go Text Protocol).

Usage in MOSAIC
---------------

- **Human vs KataGo**: Play Go against the engine using the GUI.
- **Agent vs KataGo**: Train or evaluate RL/LLM agents against KataGo.
