Common Errors
=============

This page lists frequently encountered errors when working with the
CleanRL worker, along with their causes and fixes.


ModuleNotFoundError: No module named 'tyro'
--------------------------------------------

.. code-block:: text

   ModuleNotFoundError: No module named 'tyro'

**Cause:** The ``cleanrl`` optional dependencies were not installed.
``tyro`` is the CLI argument parser used by upstream CleanRL scripts and
is included in the ``cleanrl`` extras group.

**Fix:**

.. code-block:: bash

   pip install -e ".[cleanrl]"

This installs ``tyro``, ``torch``, ``tensorboard``, ``wandb``,
``tenacity``, and ``moviepy`` in one step.

ModuleNotFoundError: No module named 'cleanrl'
-----------------------------------------------

.. code-block:: text

   ModuleNotFoundError: No module named 'cleanrl'

**Cause:** The upstream CleanRL package is not installed.  MOSAIC's
``cleanrl`` extras group installs the *worker shim* dependencies but
not the CleanRL library itself (it is expected to be available in the
environment).

**Fix:** Install CleanRL:

.. code-block:: bash

   pip install cleanrl

If you are using algorithms that require CleanRL's Atari or EnvPool
extras, install those as well:

.. code-block:: bash

   pip install "cleanrl[atari]"
   pip install "cleanrl[envpool]"

TensorBoard: "No module named 'pkg_resources'"
--------------------------------------------------

.. code-block:: text

   ModuleNotFoundError: No module named 'pkg_resources'

**Cause:** ``setuptools`` version 78+ removed the bundled ``pkg_resources``
package. TensorBoard imports ``pkg_resources`` at startup, so it fails
when ``setuptools>=78`` is installed.

**Fix:**

.. code-block:: bash

   pip install "setuptools<78"

This constraint is included in ``requirements/base.txt`` and
``requirements/cleanrl_worker.txt``.

CUDA / GPU Errors
-----------------

**"CUDA out of memory"**

.. code-block:: text

   torch.cuda.OutOfMemoryError: CUDA out of memory.

**Cause:** The training run requires more GPU memory than is available.
This commonly happens with large batch sizes, many parallel environments,
or Atari/image-based observations.

**Fixes:**

- Reduce ``num_envs`` in the algorithm parameters.
- Reduce ``num_steps`` (rollout buffer length).
- Disable CUDA and train on CPU (uncheck the GPU toggle in the GUI,
  or set ``"cuda": false`` in the config extras).
- Close other GPU-consuming processes.

**"CUDA not available"**

.. code-block:: text

   RuntimeError: Attempting to use CUDA, but torch.cuda.is_available() is False

**Cause:** PyTorch was installed without CUDA support, or the CUDA
toolkit / GPU drivers are missing.

**Fixes:**

- Install the CUDA-enabled PyTorch build:
  ``pip install torch --index-url https://download.pytorch.org/whl/cu121``
- Verify with: ``python -c "import torch; print(torch.cuda.is_available())"``
- Alternatively, disable CUDA in the run config and train on CPU.

FastLane Telemetry Issues
-------------------------

**No frames appearing in the GUI**

**Possible causes:**

1. ``fastlane_video_mode`` is set to ``"off"``.  Change it to
   ``"single"`` or ``"grid"`` in the training form.
2. The environment does not support ``render(mode="rgb_array")``.
   FastLane calls ``env.render()`` on every step to capture frames.
3. ``GYM_GUI_FASTLANE_ONLY`` is not set.  The ``sitecustomize.py``
   patch checks this environment variable before wrapping environments.
4. The shared-memory segment is not accessible.  Ensure the GUI and the
   worker subprocess are running under the same user.

**Frames are too slow / choppy**

- Increase ``CLEANRL_FASTLANE_INTERVAL_MS`` to reduce frame rate and
  lower overhead (e.g. set to ``100`` for ~10 FPS).
- Set ``CLEANRL_FASTLANE_MAX_DIM`` to downscale large frames before
  publishing (e.g. ``128``).
- Switch from ``grid`` to ``single`` video mode to reduce the number
  of environments rendering simultaneously.

**Leaked shared-memory segments (``/dev/shm/psm_*``)**

.. code-block:: text

   resource_tracker: There appear to be N leaked shared_memory objects
   to clean up at shutdown

**Cause:** When a training subprocess is killed (SIGKILL, OOM, or IDE
crash), Python's ``resource_tracker`` cannot clean up the POSIX
shared-memory semaphore files (``/dev/shm/psm_*``).  These accumulate
over repeated crash cycles.

**Fix:** The runtime now auto-cleans orphaned segments before each
launch.  To clean manually: ``rm -f /dev/shm/psm_*``.

See also
:ref:`MuJoCo Training: Subprocess Killed <cleanrl-mujoco-sigkill>`
above for the root cause (PYTHONPATH misconfiguration causing
``sitecustomize.py`` to load in every forked env process).

Curriculum Training Errors
--------------------------

**"No module named 'syllabus'"**

.. code-block:: text

   ModuleNotFoundError: No module named 'syllabus'

**Cause:** Syllabus-RL is not installed.  It is required only for
curriculum training mode and is vendored as a Git submodule.

**Fix:**

.. code-block:: bash

   git submodule update --init 3rd_party/tools/Syllabus
   pip install -e 3rd_party/tools/Syllabus

**``jq: command not found`` in curriculum scripts**

.. code-block:: text

   curriculum_babyai_goto.sh: line 78: jq: command not found

**Cause:** The ``jq`` command-line JSON processor is not installed.
Curriculum scripts use ``jq`` to build the schedule JSON from the base
MOSAIC config.

**Fix:**

.. code-block:: bash

   sudo apt-get update && sudo apt-get install -y jq

See also :doc:`/documents/tutorials/installation/common_errors/workers/index`
for the full write-up.

**"Task space mismatch" or unexpected environment switching**

**Cause:** The ``curriculum_schedule`` contains environment IDs that
are not installed or have incompatible observation/action spaces.
All environments in a curriculum must share the same observation and
action space shapes.

**Fix:** Ensure every ``env_id`` in the schedule is installed and
that all environments produce observations of the same shape.
For MiniGrid/BabyAI curricula, all environments use the standard 7x7x3
observation space by default.

Weights & Biases (W&B) Errors
------------------------------

**"wandb.errors.UsageError: api_key not configured"**

.. code-block:: text

   wandb.errors.UsageError: api_key not configured

**Cause:** W&B tracking is enabled but no API key is available.

**Fixes:**

- Set ``WANDB_API_KEY`` in your ``.env`` file.
- Or enter the API key in the training form's W&B section.
- Or run ``wandb login`` in the terminal before launching training.

**W&B upload failures behind a proxy / VPN**

.. code-block:: text

   requests.exceptions.ConnectionError: HTTPSConnectionPool(host='api.wandb.ai', ...)

**Cause:** Network requests to ``api.wandb.ai`` are blocked by a
corporate proxy or VPN.

**Fixes:**

- Enable the VPN proxy option in the training form and configure the
  proxy URL (e.g. ``https://127.0.0.1:7890``).
- Or set the proxy variables in ``.env``:

  .. code-block:: bash

     WANDB_VPN_HTTPS_PROXY=https://127.0.0.1:7890
     WANDB_VPN_HTTP_PROXY=http://127.0.0.1:7890

- Or disable W&B tracking entirely and rely on TensorBoard for metrics.

gRPC Handshake Failures
------------------------

**"grpc._channel._InactiveRpcError: StatusCode.UNAVAILABLE"**

.. code-block:: text

   grpc._channel._InactiveRpcError: <_InactiveRpcError of RPC that terminated with:
       status = StatusCode.UNAVAILABLE
       details = "failed to connect to all addresses"
   >

**Cause:** The worker subprocess could not reach the Trainer Daemon's
gRPC server (default ``127.0.0.1:50055``).

**Possible causes and fixes:**

- The Trainer Daemon is not running.  Start the MOSAIC GUI, which
  launches the daemon automatically.
- The gRPC port is blocked by a firewall.  Ensure port 50055 is open
  for localhost traffic.
- A different ``grpc_target`` was specified.  Verify the target matches
  the daemon's listening address.
- Set ``GRPC_VERBOSITY=debug`` in ``.env`` for detailed connection
  logs.

**"gRPC handshake timeout"**

**Cause:** The daemon is running but too slow to respond (e.g. under
heavy load with many concurrent runs).

**Fixes:**

- Retry the run -- transient timeouts often resolve on the next
  attempt.
- Reduce the number of concurrent training runs.
- Check system resources (CPU, memory) to ensure the daemon is not
  starved.

MuJoCo Training: Subprocess Killed (SIGKILL / exit code -9)
------------------------------------------------------------

.. code-block:: text

   subprocess.CalledProcessError: Command '...' died with <Signals.SIGKILL: 9>.

or in the GUI, training starts (dry-run succeeds) but TensorBoard shows
"Inactive — No dashboards are active" and the ``cleanrl.stdout.log`` is
empty.

**Cause:** The ``PYTHONPATH`` set by the runtime included the
``cleanrl_worker`` *package directory* itself
(``cleanrl_worker/cleanrl_worker/``).  Python auto-imports any
``sitecustomize.py`` found on ``PYTHONPATH`` at interpreter startup.
When ``gymnasium.vector.SyncVectorEnv`` forks worker processes, **each
fork** loaded ``sitecustomize.py``, which patches ``gym.make()`` to
inject FastLane wrapping and ``render_mode="rgb_array"``.  Every forked
env then created its own shared-memory segment, flooding ``/dev/shm``
with ``psm_*`` tracker entries and ultimately causing the kernel's OOM
killer (or the resource tracker) to SIGKILL the process before the first
training iteration could complete.

**Symptoms:**

- Dry-run succeeds, but the actual training subprocess dies silently.
- ``cleanrl.stdout.log`` is empty (no SPS output).
- ``cleanrl.stderr.log`` contains hundreds of
  ``resource_tracker: There appear to be N leaked shared_memory objects``
  warnings.
- The TensorBoard event file is tiny (< 100 bytes) — only the file
  header, no scalar data.
- Running the same command *directly* in a terminal works fine (because
  the launcher loads ``sitecustomize.py`` once, explicitly).

**Fix (applied in MOSAIC):** The runtime now sets ``PYTHONPATH`` to
the *parent* of the ``cleanrl_worker`` package
(``3rd_party/workers/cleanrl_worker/``) rather than the package
directory itself.  This prevents Python's automatic ``sitecustomize``
import in forked ``SyncVectorEnv`` workers while still allowing the
launcher to import it explicitly.

Additionally, the runtime now calls ``_cleanup_orphaned_shm()`` before
launching a new subprocess and after detecting an abnormal exit, so
stale ``/dev/shm/psm_*`` segments from crashed runs are cleaned up
automatically.

**If you still see leaked segments**, you can clean them manually:

.. code-block:: bash

   rm -f /dev/shm/psm_*

TransformObservation: missing argument 'observation_space'
-----------------------------------------------------------

.. code-block:: text

   TypeError: TransformObservation.__init__() missing 1 required
   positional argument: 'observation_space'

or the inverse error:

.. code-block:: text

   TypeError: _MosaicTransformObservation.__init__() takes 3
   positional arguments but 4 were given

**Cause:** ``gymnasium`` 1.0+ changed
``TransformObservation.__init__()`` to require ``observation_space``
as a mandatory argument.  The upstream CleanRL ``ppo_continuous_action.py``
was written for an older gymnasium version that did not require it.
MOSAIC's ``sitecustomize.py`` provides a compatibility shim
(``_MosaicTransformObservation``) that auto-fills the argument from
``env.observation_space``, but the shim's constructor signature must
accept ``observation_space`` both positionally and as a keyword argument
to support both the upstream and MOSAIC copies of the algorithm.

**Fix (applied in MOSAIC):** The shim now uses
``def __init__(self, env, func, observation_space=None, **kwargs)``
instead of keyword-only syntax.

TensorBoard UI shows "Inactive" despite data existing
------------------------------------------------------

**Symptoms:** Training is running (SPS output in ``cleanrl.stdout.log``,
growing event file), but the TensorBoard web UI shows "Inactive — No
dashboards are active for the current data set."

**Cause:** This is a TensorBoard frontend caching issue, not a data
problem.  The Scalars plugin may not auto-activate on the default
dashboard view.

**Fixes:**

- Hard-refresh the browser (``Ctrl+Shift+R``).
- Click the **Scalars** tab in the left sidebar.
- Navigate directly to ``http://127.0.0.1:6006/#scalars``.
- Verify data exists via the API:
  ``curl http://127.0.0.1:6006/data/plugin/scalars/tags``

**Verifying event files programmatically:**

.. code-block:: python

   from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
   ea = EventAccumulator("path/to/tensorboard/dir")
   ea.Reload()
   print(ea.Tags()["scalars"])  # Should list scalar tag names

Algorithm Registry: upstream vs MOSAIC copies
-----------------------------------------------

The CleanRL worker maintains two copies of each algorithm:

- **Upstream** (``cleanrl.ppo_continuous_action``): the original
  single-file script from the CleanRL repository.  Uses
  ``if __name__ == "__main__":`` — no ``main()`` function.
- **MOSAIC** (``cleanrl_worker.algorithms.ppo_continuous_action``):
  an adapted copy with ``main()`` and ``run()`` functions, MOSAIC
  video-path support, and ``Optional`` type annotations.

The runtime's ``DEFAULT_ALGO_REGISTRY`` maps algorithm names to module
paths.  Algorithms that need ``sitecustomize.py`` patches (FastLane,
TensorBoard redirect, checkpoint resume) **must** use the MOSAIC copy
because:

1. The launcher looks for a ``main()`` function.  If it finds one, it
   runs the module in-process (with ``sitecustomize.py`` already loaded).
2. If no ``main()`` exists, the launcher falls back to
   ``subprocess.call()``, which starts a fresh interpreter where the
   ``sitecustomize.py`` patches may not be active.

For MuJoCo continuous-action algorithms, the MOSAIC copy is required
because the upstream version does not pass ``observation_space`` to
``TransformObservation`` (required by gymnasium 1.0+).

Environment Import Errors
--------------------------

**"No module named 'minigrid'" / "No module named 'ale_py'"**

.. code-block:: text

   ModuleNotFoundError: No module named 'minigrid'

**Cause:** The environment-specific extras are not installed.

**Fix:** Install the appropriate extras for your target environment:

.. code-block:: bash

   pip install -e ".[minigrid]"    # MiniGrid / BabyAI
   pip install -e ".[atari]"      # Atari (ALE)
   pip install -e ".[mujoco]"     # MuJoCo
   pip install -e ".[procgen]"    # Procgen
