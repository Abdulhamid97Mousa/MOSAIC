Jumanji Worker
==============

The **Jumanji Worker** wraps `Jumanji <https://github.com/instadeepai/jumanji>`_,
a suite of JAX-based combinatorial and logistics environments developed by InstaDeep.

Jumanji provides hardware-accelerated environments for problems such as
bin-packing, routing (TSP, CVRP), game 2048, and more. All environments
are fully compatible with JAX's ``jit``, ``vmap``, and ``pmap`` for
massively parallel rollouts on GPU/TPU.

Key Features
------------

- **Hardware-accelerated**: Environments run natively on GPU/TPU via JAX.
- **Combinatorial focus**: BinPack, TSP, CVRP, Knapsack, JobShop, FlatPack, etc.
- **Gymnasium-compatible**: Standard ``reset`` / ``step`` API.
- **Vectorised rollouts**: Use ``jax.vmap`` for thousands of parallel environments.

Supported Algorithms
--------------------

Any JAX-compatible RL algorithm can be used. Typical choices include:

- A2C
- PPO (with JAX backend)
