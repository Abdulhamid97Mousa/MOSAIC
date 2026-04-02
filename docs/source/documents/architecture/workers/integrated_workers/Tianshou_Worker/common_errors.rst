Common Errors
=============

ImportError: No module named 'tianshou'
---------------------------------------

**Cause:** The Tianshou submodule is not installed.

**Fix:**

.. code-block:: bash

   git submodule update --init 3rd_party/workers/tianshou_worker/tianshou
   pip install -e 3rd_party/workers/tianshou_worker/tianshou

Algorithm "xxx" not supported by Tianshou worker yet
-----------------------------------------------------

**Cause:** The launcher's ``ALGO_MAP`` only contains ``ppo`` and ``dqn``.
Tianshou supports 30+ algorithms upstream, but they have not been wired
into the MOSAIC launcher yet.

**Workaround:** Use the Script form (``TianshouScriptForm``) to run a
custom Python script that uses the Tianshou API directly.

**Long-term fix:** Expand ``ALGO_MAP`` in ``launcher.py`` and add runner
functions for additional algorithms (SAC, TD3, A2C, etc.).

TianshouWorkerConfig validation error: missing 'run_id'
--------------------------------------------------------

**Cause:** The config JSON does not contain a ``run_id`` field, or
it is empty.

**Fix:** Ensure the config JSON includes all required fields:

.. code-block:: json

   {
     "run_id": "tianshou-ppo-01ARZ...",
     "algo": "ppo",
     "env_id": "CartPole-v1",
     "total_timesteps": 100000
   }

When using the GUI forms, the run ID is auto-generated using ULID.

Subprocess hangs or deadlocks
-----------------------------

**Cause:** The runtime reads stdout line-by-line in a loop while stderr
is captured separately.  If the subprocess writes a large amount to
stderr before stdout is drained, the pipe buffer fills and the process
blocks.

**Workaround:** This is a known limitation.  If the worker appears to
hang, check if the subprocess is producing excessive stderr output
(e.g. deprecation warnings from PyTorch or Gymnasium).

**Long-term fix:** Switch to file-based stdout/stderr capture (as used
by the CleanRL worker) or use threading to read both streams concurrently.

FastLane not showing video
--------------------------

**Cause:** The Tianshou worker does not yet have a dedicated
``fastlane.py`` module with ``FastLaneTelemetryWrapper``.  FastLane
environment variables are set, but the training subprocess does not
automatically wrap environments with frame capture.

**Workaround:** FastLane integration is partial.  The environment
variables are configured correctly, but without ``sitecustomize.py``
patching ``gym.make()``, environments are not automatically wrapped.

DQN with continuous action space
---------------------------------

**Cause:** DQN only supports discrete action spaces.  The launcher
validates this and raises:

.. code-block:: text

   ValueError: DQN only supports discrete action spaces, but {env_id} has Box(...)

**Fix:** Use PPO for continuous action spaces (e.g. MuJoCo environments),
or wait for SAC/TD3 support to be added.

Gym vs Gymnasium API Compatibility
------------------------------------

.. important::

   This is a critical compatibility issue that affects all MOSAIC workers
   and must be understood when integrating RL libraries.

**Background:**  The Python RL ecosystem underwent a major API transition
from the legacy ``gym`` package (maintained by OpenAI, now archived) to
the ``gymnasium`` package (maintained by the Farama Foundation).  The key
differences include:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Aspect
     - ``gym`` (legacy, pre-0.26)
     - ``gymnasium`` (modern, 0.26+)
   * - **Package import**
     - ``import gym``
     - ``import gymnasium as gym``
   * - **step() return**
     - ``obs, reward, done, info``
       (4 values)
     - ``obs, reward, terminated, truncated, info``
       (5 values)
   * - **reset() return**
     - ``obs``
       (1 value)
     - ``obs, info``
       (2 values)
   * - **Render mode**
     - ``env.render(mode="rgb_array")``
       (per-call argument)
     - ``gym.make(env_id, render_mode="rgb_array")``
       (set at construction time)
   * - **Truncation**
     - Encoded in ``done`` flag; distinguished
       via ``info["TimeLimit.truncated"]``
     - Explicit ``truncated`` boolean return value
   * - **Spaces**
     - ``gym.spaces.Discrete``, etc.
     - ``gymnasium.spaces.Discrete`` (same API,
       different package)
   * - **Wrappers**
     - ``gym.wrappers.RecordVideo``
     - ``gymnasium.wrappers.RecordVideo``
       (different constructor signature)
   * - **Maintenance**
     - Archived (no updates)
     - Actively maintained by Farama Foundation

**How MOSAIC handles this:**

MOSAIC standardises on the **gymnasium** API (5-value ``step()``,
2-value ``reset()``).  All MOSAIC core code, GUI components, and
worker runtime infrastructure use ``gymnasium``.

However, upstream RL libraries have mixed adoption:

- **Tianshou v2.0** uses ``gymnasium`` natively.  All Tianshou 2.0
  components (``Collector``, ``VectorEnv``, ``Algorithm``) expect the
  gymnasium 5-tuple ``step()`` return format.  This is a clean match
  with MOSAIC.

- **CleanRL** has a mixed codebase.  Some algorithm scripts (the
  newer ones like ``ppo.py``, ``dqn.py``, ``sac_continuous_action.py``)
  use ``import gymnasium as gym`` directly.  However, several older
  scripts (``ppo_procgen.py``, ``ppg_procgen.py``, envpool-based variants
  like ``ppo_atari_envpool.py``) still use the legacy ``import gym``
  because they depend on libraries (Procgen, EnvPool) that have not
  migrated to gymnasium.  MOSAIC's ``sitecustomize.py`` patches help
  bridge some of these differences at import time.

- **XuanCe** uses ``gymnasium`` internally but wraps environments
  through its own ``VecEnv`` abstraction, which normalises the API.

**Common symptoms of API mismatch:**

1. ``ValueError: not enough values to unpack (expected 5, got 4)`` —
   a gymnasium-style caller receives old gym 4-tuple from ``step()``.

2. ``ValueError: too many values to unpack (expected 4, got 5)`` —
   a legacy gym caller receives gymnasium 5-tuple from ``step()``.

3. ``TypeError: render() got an unexpected keyword argument 'mode'`` —
   using the old ``render(mode=...)`` call on a gymnasium env that
   expects ``render_mode`` at construction time.

4. ``AttributeError: 'tuple' object has no attribute 'shape'`` —
   ``reset()`` now returns ``(obs, info)`` instead of just ``obs``.

**Impact on Tianshou worker:**

Since both Tianshou v2.0 and MOSAIC use gymnasium natively, there are
no fundamental API mismatches for the Tianshou worker.  However, be
aware of these edge cases:

- **Environment wrappers:** If you apply custom wrappers, ensure they
  use the gymnasium wrapper base classes (``gymnasium.Wrapper``), not
  the legacy ``gym.Wrapper``.  Mixing wrapper hierarchies causes
  attribute errors.

- **Third-party environments:** Some environment packages (e.g. older
  versions of MiniGrid, Procgen, EnvPool) may still register with the
  legacy ``gym`` registry.  Use ``gymnasium.register()`` or the
  ``shimmy`` compatibility package to bridge these:

  .. code-block:: bash

     pip install shimmy[gym-v21]  # Bridge old gym envs to gymnasium

- **Checkpoint portability:** Policies trained with Tianshou v1.x
  (which used the old gym API) cannot be directly loaded in Tianshou
  v2.0 without migration.  The policy/algorithm class hierarchy changed
  completely in v2.0.

- **FastLane wrapping:** The ``sitecustomize.py`` module (when
  implemented) patches ``gymnasium.make()`` to inject
  ``render_mode="rgb_array"`` for FastLane frame capture.  This only
  works with the gymnasium API; legacy ``gym.make()`` calls are not
  patched.

Cannot find policy file (.pth / .pt)
-------------------------------------

**Cause:** The policy evaluation form expects a PyTorch checkpoint file
(``.pth`` or ``.pt``).  Tianshou saves policies via
``torch.save(policy.state_dict(), path)``.

**Fix:** Ensure the checkpoint was saved using ``torch.save()`` and has
the correct extension.  The default search directory is
``var/trainer/`` under the MOSAIC root.
