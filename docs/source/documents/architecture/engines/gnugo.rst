GNU Go
======

`GNU Go <https://www.gnu.org/software/gnugo/>`_ is a classical Go engine that
does not use neural networks. It is a lighter alternative to KataGo, useful
when GPU resources are limited or when a weaker opponent is desired for
curriculum training.

Installation
------------

.. code-block:: bash

   # Ubuntu / Debian / WSL
   sudo apt-get install -y gnugo

   # macOS
   brew install gnugo

Service Module
--------------

The GNU Go integration lives in:

- ``gym_gui/services/go_ai/gnugo_service.py``
- ``gym_gui/services/go_ai/gtp_engine.py`` (shared GTP base class)

GNU Go communicates via the GTP (Go Text Protocol), sharing the same
base class as KataGo.

Usage in MOSAIC
---------------

- **Human vs GNU Go**: Play Go against a classical engine using the GUI.
- **Curriculum training**: Use GNU Go as an easier opponent before
  graduating agents to play against KataGo.
