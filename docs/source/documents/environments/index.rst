Environment Families
====================

MOSAIC organises environments into **families** — logical groupings that share
an underlying engine, API, or research benchmark.  Each family is an optional
dependency; install only the ones you need.

.. tip::

   **Base module** (required): ``pip install -e .``
   installs the MOSAIC GUI, gRPC, and core utilities.

   **Environment families** (optional) add environment support:
   ``pip install -e ".[smac]"`` or ``pip install -e ".[minigrid,crafter]"``

   **Workers** (optional) add training backends:
   ``pip install -e ".[cleanrl]"`` or ``pip install -e ".[xuance]"``

   Environment families and workers are independent -- install any
   combination you need.  See the
   :doc:`Installation Guide </documents/tutorials/installation/index>` for details.

.. toctree::
   :maxdepth: 2
   :caption: Gymnasium

   gymnasium/index
   atari_ale/index

.. toctree::
   :maxdepth: 1
   :caption: Grid Worlds

   minigrid/index
   babyai/index
   mosaic_multigrid/index
   ini_multigrid/index
   griddly/index

.. toctree::
   :maxdepth: 1
   :caption: First-Person & Roguelike

   vizdoom/index
   malmo/index
   marlo/index
   minihack/index
   nethack/index
   crafter/index

.. toctree::
   :maxdepth: 1
   :caption: Procedural & Text

   procgen/index
   textworld/index
   babaisai/index

.. toctree::
   :maxdepth: 1
   :caption: JAX & Physics

   jumanji/index
   pybullet_drones/index

.. toctree::
   :maxdepth: 2
   :caption: Board Games

   pettingzoo/index
   openspiel/index

.. toctree::
   :maxdepth: 1
   :caption: Multi-Agent Benchmarks

   melting_pot/index
   overcooked/index
   smac/index
   smacv2/index
   rware/index
   hemac/index
