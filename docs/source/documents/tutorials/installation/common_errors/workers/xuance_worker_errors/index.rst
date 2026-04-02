XuanCe Worker Errors
====================

Errors specific to the **XuanCe worker** (``mosaic-xuance-worker``).


Installation Errors
-------------------

``ModuleNotFoundError: No module named 'xuance'``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Error (at runtime when training or evaluating):**

.. code-block:: text

   [worker stderr] Could not register MOSAIC environments with XuanCe: No module named 'xuance'
   ...
   File ".../xuance_worker/multi_agent_curriculum_training.py", line 146, in _create_runner
       from xuance import get_runner
   ModuleNotFoundError: No module named 'xuance'

**Cause:** The XuanCe worker has a **two-part installation**:

1. **The MOSAIC harness** (``mosaic-xuance-worker``) -- installed via
   ``pip install -e 3rd_party/workers/xuance_worker``.  This provides the
   ``xuance_worker`` package that bridges MOSAIC's trainer daemon to XuanCe.

2. **The XuanCe library itself** (``xuance``) -- lives in a git submodule
   at ``3rd_party/workers/xuance_worker/xuance/`` and must be installed separately.

Unlike the CleanRL worker (which maps the upstream library into its own
``pyproject.toml``), the XuanCe worker does **not** bundle the upstream
``xuance`` package.  Installing only the harness gives you ``xuance_worker``
but not ``xuance``, so the ``from xuance import get_runner`` call fails.

**Fix -- initialise the submodule and install both parts:**

.. code-block:: bash

   cd /path/to/mosaic
   source .venv/bin/activate

   # 1. Initialise the XuanCe submodule
   git submodule update --init 3rd_party/workers/xuance_worker/xuance

   # If the above fails with "did not match any file(s) known to git",
   # clone the submodule manually:
   git clone --depth 1 https://github.com/agi-brain/xuance.git \
       3rd_party/workers/xuance_worker/xuance

   # 2. Install the XuanCe library (with PyTorch backend)
   pip install -e "3rd_party/workers/xuance_worker/xuance[torch]"

   # 3. Install the MOSAIC harness
   pip install -e 3rd_party/workers/xuance_worker

   # 4. Install MOSAIC's XuanCe Python dependencies
   pip install -e ".[xuance]"

   # 5. Verify
   python -c "import xuance; print('xuance OK:', xuance.__version__)"
   python -c "import xuance_worker; print('xuance_worker OK')"

.. important::

   The main project's ``pyproject.toml`` extra ``pip install -e ".[xuance]"``
   installs the **Python dependencies** (torch, scipy, mpi4py, etc.) but does
   **not** install the ``xuance`` library itself.  You must install the
   submodule separately as shown above.

   **Installation order matters:** install ``xuance`` (the library) before
   ``mosaic-xuance-worker`` (the harness) so that the harness can detect and
   register MOSAIC environments with XuanCe at import time.

``Cannot compile MPI programs`` (mpi4py build failure)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Error:**

.. code-block:: text

   error: Cannot compile MPI programs. Check your configuration!

Or:

.. code-block:: text

   mpicc: No such file or directory

**Cause:** The XuanCe extras include ``mpi4py``, which compiles against MPI
libraries.  The MPI compiler wrapper ``mpicc`` is not installed by default.

**Fix:**

.. code-block:: bash

   # Ubuntu / Debian / WSL
   sudo apt-get install -y libopenmpi-dev

   # macOS
   brew install open-mpi

   # Then retry
   pip install mpi4py

.. tip::

   MOSAIC sets ``MPI4PY_RC_INITIALIZE=0`` in the ``.env`` file to prevent
   ``MPI_Init()`` from blocking when XuanCe is imported outside of ``mpirun``.
   Make sure your ``.env`` file includes this setting.

-----

Version Compatibility
---------------------

.. important::

   **MOSAIC requires XuanCe v1.4.0 or later.**

   XuanCe v1.4.0 introduced several breaking API changes compared to v1.3.x.
   The MOSAIC harness (``mosaic-xuance-worker``) has been updated to use the
   v1.4.0 API.  If you install an older version of XuanCe, you will encounter
   runtime errors.

   Verify your version:

   .. code-block:: bash

      python -c "import xuance; print(xuance.__version__)"

   If the version is below 1.4.0, update by re-installing from the submodule:

   .. code-block:: bash

      cd /path/to/mosaic
      source .venv/bin/activate
      cd 3rd_party/workers/xuance_worker/xuance
      git pull origin main
      pip install -e ".[torch]"


``TypeError: get_runner() got an unexpected keyword argument 'method'``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Error:**

.. code-block:: text

   TypeError: get_runner() got an unexpected keyword argument 'method'

**Cause:** XuanCe v1.4.0 renamed the ``method`` parameter to ``algo`` in
``get_runner()``.  If your MOSAIC harness code still uses ``method=``, it is
out of date.

**Fix:** Ensure you are on the latest version of the MOSAIC harness.  The
corrected calls use ``algo=`` instead of ``method=``:

.. code-block:: python

   # Old (v1.3.x) -- no longer works
   runner = get_runner(method=..., env=..., env_id=..., is_test=False, ...)

   # New (v1.4.0+) -- current MOSAIC code
   runner = get_runner(algo=..., env=..., env_id=..., ...)

Note that the ``is_test`` parameter was also removed in v1.4.0.


``'RunnerMARL' object has no attribute 'agents'``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Error:**

.. code-block:: text

   AttributeError: 'RunnerMARL' object has no attribute 'agents'

**Cause:** XuanCe v1.4.0 changed ``runner.agents`` (plural) to
``runner.agent`` (singular).  This affects all runner types (single-agent
and multi-agent).

**Fix:** Ensure you are on the latest version of the MOSAIC harness.  All
references have been updated:

.. code-block:: python

   # Old (v1.3.x)
   runner.agents.train(n_steps)
   runner.agents.save_model(path)
   runner.agents.load_model(path)
   runner.agents.finish()

   # New (v1.4.0+)
   runner.agent.train(n_steps)
   runner.agent.save_model(path)
   runner.agent.load_model(path)
   runner.agent.finish()


``'MAPPO_Learner' object has no attribute 'use_cnn'``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Error:**

.. code-block:: text

   File ".../xuance/torch/learners/multi_agent_rl/iac_learner.py", line 88, in build_training_data
       if self.use_cnn and len(obs_tensor.shape) > 3:
          ^^^^^^^^^^^^
   AttributeError: 'MAPPO_Learner' object has no attribute 'use_cnn'

**Cause:** This is a bug in XuanCe v1.4.0 itself.  The ``use_cnn`` attribute
is set in the **agent** base class (``MARLAgents``) but is never initialised
in the **learner** base class (``LearnerMAS``).  When the learner's
``build_training_data()`` or ``update()`` method runs, it accesses
``self.use_cnn`` which does not exist.

This error typically appears during multi-agent curriculum training with
MAPPO on multigrid environments (e.g. ``collect_1vs1``, ``soccer_1vs1``).

**Fix:** The MOSAIC harness applies a monkey-patch at the shim layer
(``_patches.py``) that adds the missing attribute to
``LearnerMAS.__init__``.  Both ``runtime.py`` (standard training) and
``multi_agent_curriculum_training.py`` (curriculum training) call this
patch automatically before creating any runner.  Ensure you are on the
latest version of the harness.

If you still encounter this error, verify the patches are being applied
before runner creation:

.. code-block:: python

   from xuance_worker._patches import apply_xuance_patches
   apply_xuance_patches()  # Must be called before get_runner()


Curriculum environment swap does not switch the environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Symptom:** After Phase 1 completes, the logs say the environment was
swapped to the new ``env_id`` (e.g. ``soccer_1vs1``), but the FastLane
visualization and training behaviour remain on the Phase 1 environment
(e.g. ``collect_1vs1``).  Rewards and episode lengths do not change.
The process eventually crashes because the agent is stepping a closed
environment handle.

**Root cause:** XuanCe's on-policy training loop (``on_policy_marl.py``)
reads exclusively from ``self.envs``:

.. code-block:: python

   # XuanCe on_policy_marl.py -- the attribute that actually matters
   obs_dict = self.envs.buf_obs
   next_obs_dict, ... = self.envs.step(actions_dict)

The MOSAIC harness was setting ``runner.agents.train_envs = new_envs``.
The attribute ``train_envs`` does **not exist** on XuanCe's
``MARLAgents``.  Python silently creates a new attribute on the object,
the training loop never sees it, and the agent continues stepping the
already-closed Phase 1 environment handles until the process crashes.

**Fix:** The swap in ``multi_agent_curriculum_training.py`` now sets
``runner.agents.envs``, the attribute XuanCe actually reads, with a
runtime assertion to catch any future regression:

.. code-block:: python

   # Before (bug): train_envs does not exist in XuanCe
   runner.agents.train_envs = new_envs   # silently ignored

   # After (fix): envs is what on_policy_marl.py reads
   runner.agents.envs = new_envs
   assert runner.agents.envs is new_envs

A regression test in ``tests/test_env_swap.py`` confirms that the swap
code targets ``runner.agents.envs`` and that the attribute is identical
to the newly created environment after the swap.

-----

Runtime Errors
--------------

``buffer_size must be divisible by the number of envs (parallels)``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Error:**

.. code-block:: text

   AssertionError: buffer_size must be divisible by the number of envs (parallels)

**Cause:** XuanCe's on-policy replay buffer (``MARL_OnPolicyBuffer``) requires
that ``buffer_size % n_envs == 0``.  The number of parallel environments
(``n_envs``) is set by the **Parallel Envs** field in the training form or by
the ``XUANCE_PARALLELS`` / ``XUANCE_NUM_ENVS`` environment variables in custom
scripts.

This error typically appears when the **Grid Limit** in FastLane is set to a
value that is not a divisor of the buffer size.  For example, the default
buffer size is **256**.  Setting Grid Limit (and therefore parallel envs) to
**6** causes ``256 % 6 = 4 ≠ 0``, which triggers the assertion.

.. important::

   **Use power-of-2 values for Parallel Envs and Grid Limit.**

   The following values are safe with the default buffer size of 256:

   - **1:** single environment (slowest, useful for debugging)
   - **2:** two parallel environments
   - **4:** four parallel environments (default)
   - **8:** eight parallel environments (recommended for grid view)

   Values like **3**, **5**, **6**, **7** will crash because 256 is not
   divisible by them.

**Fix — choose a compatible value:**

If you are using the **training form**, set **Parallel Envs** to 1, 2, 4, or 8.

If you are using a **custom script**, set the environment variable to a
power-of-2 value:

.. code-block:: bash

   # In your custom script
   NUM_ENVS="${XUANCE_NUM_ENVS:-4}"   # safe default
   export XUANCE_PARALLELS="$NUM_ENVS"

If you need a specific number of environments that does not divide the buffer
size, you can adjust the buffer size in your XuanCe configuration to a
multiple of ``n_envs``.  For example, for 6 environments use
``buffer_size = 252`` (252 / 6 = 42) or ``buffer_size = 256`` with 8 envs.

.. tip::

   The **Grid Limit** field in FastLane controls how many environments are
   displayed in the grid view.  It does **not** override the number of
   parallel environments used for training.  If you want to see 8
   environments in the grid, set **both** Parallel Envs and Grid Limit to 8.

   The grid display is clamped: ``grid_limit = min(grid_limit, parallels)``.
   Setting Grid Limit higher than Parallel Envs will show at most
   ``parallels`` environments.


``Unknown command: init_agent`` when using CleanRL worker for multi-agent environments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Error (when launching multi-agent parallel environments like soccer_2vs2):**

.. code-block:: text

   2026-02-27 10:58:16,768 | WARNING | gym_gui.operators.main_window | Unexpected init_agent response for agent_0: {'type': 'error', 'message': 'Unknown command: init_agent'}
   2026-02-27 10:58:16,769 | WARNING | gym_gui.operators.main_window | Unexpected init_agent response for agent_1: {'type': 'error', 'message': 'Unknown command: init_agent'}
   ...
   2026-02-27 10:59:04,769 | ERROR   | gym_gui.operators.main_window | Failed to get action from AI agent agent_0 after 5 attempts. Last response: None

**Cause:** The **CleanRL worker** (``cleanrl_worker``) only supports single-agent
environments where it owns the environment. It has only two command handlers:

- ``_handle_reset`` - Initialize environment with seed
- ``_handle_step`` - Execute action and return next observation

For **parallel multi-agent environments** (MosaicMultiGrid soccer, MeltingPot, Overcooked),
the GUI owns the shared environment and sends observations to multiple worker processes
for action selection. This requires the **action-selector protocol**:

- ``init_agent`` - Initialize worker in action-selector mode (no env ownership)
- ``select_action`` - Given observation, return action for specific agent

The CleanRL worker does **not** implement these commands, so it responds with
``Unknown command: init_agent`` and all agents timeout waiting for actions.

**Fix -- use XuanCe worker for multi-agent environments:**

Multi-agent parallel environments require the **XuanCe worker** (``mosaic-xuance-worker``),
which implements the full action-selector protocol including ``init_agent`` and
``select_action`` commands.

When configuring operators for multi-agent environments:

.. code-block:: python

   # ❌ WRONG - CleanRL worker doesn't support multi-agent parallel mode
   operators = [
       {
           "id": "soccer_team",
           "name": "Soccer 2v2",
           "env_name": "mosaic_multigrid",
           "task": "MosaicMultiGrid-Soccer-2vs2-IndAgObs-v0",
           "workers": {
               "agent_0": {"type": "rl", ...},  # cleanrl_worker
               "agent_1": {"type": "rl", ...},  # cleanrl_worker
               "agent_2": {"type": "rl", ...},  # cleanrl_worker
               "agent_3": {"type": "rl", ...},  # cleanrl_worker
           }
       }
   ]

   # ✅ CORRECT - XuanCe worker supports multi-agent parallel mode
   operators = [
       {
           "id": "soccer_team",
           "name": "Soccer 2v2",
           "env_name": "mosaic_multigrid",
           "task": "MosaicMultiGrid-Soccer-2vs2-IndAgObs-v0",
           "workers": {
               "agent_0": {"type": "xuance", "method": "ippo", "policy_path": "..."},
               "agent_1": {"type": "xuance", "method": "ippo", "policy_path": "..."},
               "agent_2": {"type": "xuance", "method": "ippo", "policy_path": "..."},
               "agent_3": {"type": "xuance", "method": "ippo", "policy_path": "..."},
           }
       }
   ]

.. important::

   **Worker compatibility by environment type:**

   - **Single-agent environments** (MiniGrid, Atari, Gymnasium):
     Use either CleanRL worker or XuanCe worker

   - **Multi-agent parallel environments** (MosaicMultiGrid, MeltingPot, Overcooked):
     Must use XuanCe worker (CleanRL worker will fail with "Unknown command: init_agent")

   The system detects parallel multi-agent mode when:

   1. Environment name is in: ``mosaic_multigrid``, ``ini_multigrid``, ``meltingpot``, ``overcooked``
   2. Operator has multiple workers (``len(config.workers) > 1``)

   See ``main_window.py:1069`` (``_is_parallel_multiagent()``) for detection logic.

.. tip::

   If you see "Unknown command: init_agent" errors, check your operator configuration
   and verify you're using ``"type": "xuance"`` for all agents in multi-agent environments.

``Could not register MOSAIC environments with XuanCe`` (non-fatal warning)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Warning (at worker startup):**

.. code-block:: text

   [worker stderr] Could not register MOSAIC environments with XuanCe: No module named 'xuance'

**Cause:** This is the same missing ``xuance`` library issue described above.
The harness tries to register custom MOSAIC environment adapters with XuanCe
at import time.  If the library is missing, it logs a warning and continues,
but the actual training call will fail shortly after.

**Fix:** Install the ``xuance`` submodule as described in the installation
section above.

``optional_deps`` shows ``"xuance_worker": false``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Symptom:** The MOSAIC settings log at startup shows:

.. code-block:: json

   "optional_deps": {
     "xuance_worker": false
   }

**Cause:** MOSAIC uses ``importlib.util.find_spec("xuance_worker")`` at
startup.  If the harness package is not installed, the worker appears as
unavailable.  The GUI will still launch, but XuanCe training forms will not
appear.

**Fix:** Install both the library and the harness as described in the
installation section above.  After restarting MOSAIC, the setting should show
``"xuance_worker": true``.
