Installation
============

Prerequisites
-------------

- Python 3.10+
- MOSAIC installed (``pip install -e .``)
- PyTorch (CPU or CUDA)

Install from Repository
-----------------------

The Tianshou worker is included in the MOSAIC repository as a git
submodule.  Install it with the optional dependency group:

.. code-block:: bash

   # Install MOSAIC with Tianshou dependencies
   pip install -e ".[tianshou]"

   # Install the Tianshou library from the submodule
   pip install -e 3rd_party/workers/tianshou_worker/tianshou

   # Install the worker harness itself
   pip install -e 3rd_party/workers/tianshou_worker

Or use the requirements file:

.. code-block:: bash

   pip install -r requirements/tianshou_worker.txt

Verify Installation
-------------------

.. code-block:: python

   # Verify Tianshou library
   import tianshou
   print(tianshou.__version__)  # Should print 2.0.0 or similar

   # Verify worker harness
   from tianshou_worker import get_worker_metadata
   metadata, capabilities = get_worker_metadata()
   print(f"{metadata.name} v{metadata.version}")
   print(f"Algorithms: {capabilities.action_spaces}")

   # Verify MOSAIC integration
   from tianshou_worker.config import TianshouWorkerConfig
   config = TianshouWorkerConfig(
       run_id="test",
       algo="ppo",
       env_id="CartPole-v1",
       total_timesteps=1000,
   )
   print(config.to_dict())

Git Submodule Setup
-------------------

If the Tianshou submodule is not initialized (e.g. after a fresh clone):

.. code-block:: bash

   git submodule update --init 3rd_party/workers/tianshou_worker/tianshou

The submodule tracks the ``master`` branch of
`thu-ml/tianshou <https://github.com/thu-ml/tianshou>`_.

GPU Support
-----------

Tianshou uses PyTorch for GPU acceleration.  To enable CUDA:

1. Install PyTorch with CUDA support
   (see `pytorch.org <https://pytorch.org/get-started/locally/>`_).
2. Set ``device: "cuda:0"`` in the training form or pass
   ``"device": "cuda:0"`` in the config extras.

.. note::

   GPU is optional.  All algorithms work on CPU.  For small environments
   like CartPole or MiniGrid, CPU training is often faster due to reduced
   data transfer overhead.
