Installation and Usage Guide
============================

This guide covers the complete setup and usage of **Microsoft Malmo** (MalmoEnv)
in MOSAIC — from first-time installation to running environments in both
human play and RL training modes.

Quick Start (3 commands)
------------------------

If you just want to get running:

.. code-block:: bash

   # 1. One-time setup (installs Java deps, downloads assets, builds mod):
   ./setup_malmo.sh

   # 2. Start Minecraft (Terminal 1):
   ./run_malmo.sh

   # 3. Start MOSAIC (Terminal 2, after "DORMANT" appears):
   ./run.sh

Then select any **MalmoEnv** environment from the dropdown, click **Load Environment**,
then **Start Game**.

.. important::

   Minecraft must be running BEFORE you load a MalmoEnv environment in MOSAIC.
   If Minecraft is not running, MOSAIC will show:
   ``Cannot connect to Minecraft on localhost:9000``.


Prerequisites
-------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Dependency
     - Notes
   * - **Java 8 (JDK)**
     - **Exactly Java 8** — Gradle 2.14 does not support Java 9+.
       On Ubuntu: ``sudo apt install openjdk-8-jdk``
   * - **xvfb**
     - Required for headless Minecraft (no visible window).
       On Ubuntu: ``sudo apt install xvfb``
   * - **MOSAIC virtual environment**
     - All Python steps assume ``.venv`` is activated:
       ``source .venv/bin/activate``

.. warning::

   The Gradle wrapper bundled with Malmo requires **Java 8 exactly**.
   Running with Java 17 prints ``Could not determine java version`` and aborts.
   The ``setup_malmo.sh`` and ``run_malmo.sh`` scripts handle Java 8 detection
   automatically.


Automated Setup: ``setup_malmo.sh``
------------------------------------

The ``setup_malmo.sh`` script at the repository root automates the entire
first-time setup:

.. code-block:: bash

   ./setup_malmo.sh

What it does (7 steps):

1. Detects Java 8 JDK (searches ``/usr/lib/jvm/``)
2. Verifies ``xvfb`` is installed
3. Sets ``MALMO_XSD_PATH`` environment variable
4. Creates ``version.properties`` if missing
5. Downloads all 1196 Minecraft 1.11.2 assets via HTTPS (bypasses Mojang's broken HTTP CDN)
6. Builds the Malmo Forge mod (``./gradlew build``)
7. Installs the MalmoEnv Python package (``pip install -e``)

.. note::

   The asset download (step 5) takes a few minutes on first run. One asset
   (``nether2.ogg``) always fails because Mojang removed it — this is harmless.

   Subsequent runs of ``setup_malmo.sh`` skip already-completed steps.


Launching Minecraft: ``run_malmo.sh``
--------------------------------------

The ``run_malmo.sh`` script launches Minecraft in headless mode:

.. code-block:: bash

   ./run_malmo.sh                  # Default: headless on port 9000
   ./run_malmo.sh -port 9001       # Custom port
   ./run_malmo.sh --visible         # Show Minecraft window (for debugging)

The script automatically:

- Kills any existing Minecraft process holding ports 9000/9001
- Finds Java 8 and sets ``JAVA_HOME``
- Builds the mod if needed
- Launches Minecraft via ``xvfb-run`` (headless)

When Minecraft is ready, you will see::

   MOSAIC NativeInputHandler listening on port 9001
   ***** Start MalmoEnvServer on port 9000
   CLIENT enter state: DORMANT

Minecraft is now waiting for MOSAIC connections.


Launching MOSAIC: ``run.sh``
-----------------------------

In a **second terminal**, launch MOSAIC:

.. code-block:: bash

   ./run.sh

Then:

1. Select **Family: MalmoEnv** from the dropdown
2. Choose any mission (e.g. ``MalmoEnv-CatchTheMob-v0``)
3. Click **Load Environment** — the Minecraft frame appears in Render View
4. Click **Start Game** — keyboard and mouse input become active


Two Input Modes
---------------

MOSAIC supports two modes for Malmo environments. The mode is selected
automatically based on the **Control Mode** setting:

Human Play Mode (``Human Only``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When you select **Human Only** control mode:

- **Keyboard**: Press/release events go directly to Minecraft via TCP port 9001.
  Press W to walk, release W to stop. Feels like playing Minecraft natively.
- **Mouse**: Raw dx/dy deltas go to Minecraft for smooth mouse look
  (horizontal and vertical).
- **Observations**: The MalmoEnv step loop runs in the background sending
  ``noop`` commands to fetch frames for the Render View.

This mode uses Minecraft's native ``KeyBinding`` system. The Java mod
automatically switches to ``InputType.HUMAN`` when MOSAIC connects to port 9001.

RL Training Mode (``Agent Only``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When you select **Agent Only** control mode (or use a worker like CleanRL):

- **Actions**: The RL agent picks a discrete action integer each step
  (e.g. ``0`` = ``move 1``, ``2`` = ``turn 1``).
- **Step loop**: ``MalmoEnv.step(action)`` sends the command string to
  Minecraft via TCP port 9000.
- **Movement**: For continuous missions, ``move 1`` sets a persistent
  velocity — the agent must learn to send ``move 0`` to stop.
- **No native input**: Port 9001 is not used. The Java mod stays in
  ``InputType.AI`` mode.

.. note::

   The two modes coexist — Minecraft automatically detects which mode to use
   based on whether a human player is connected to port 9001. No configuration
   needed.


TCP Port Reference
------------------

.. list-table::
   :header-rows: 1
   :widths: 15 20 65

   * - Port
     - Service
     - Description
   * - **9000**
     - MalmoEnv TCP
     - Observations, rewards, and discrete action commands.
       Used by BOTH human play and RL training.
       Started by ``MalmoEnvServer`` in the Java mod.
   * - **9001**
     - NativeInputHandler TCP
     - Raw keyboard press/release and mouse dx/dy.
       Used by human play ONLY.
       Started by ``NativeInputHandler`` in the Java mod.


Manual Setup (Advanced)
-----------------------

If you prefer to set up manually instead of using ``setup_malmo.sh``:

Step 1: Install Java 8
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   sudo apt install openjdk-8-jdk
   export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64

Step 2: Install xvfb
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   sudo apt install xvfb

Step 3: Set MALMO_XSD_PATH
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   export MALMO_XSD_PATH=$(pwd)/3rd_party/environments/malmo/Schemas

   # To make permanent:
   echo "export MALMO_XSD_PATH=$(pwd)/3rd_party/environments/malmo/Schemas" >> ~/.bashrc

Step 4: Create version.properties
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   echo "malmomod.version=0.37.0" > \
     3rd_party/environments/malmo_updated/Minecraft/src/main/resources/version.properties

Step 5: Download Minecraft Assets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. warning::

   Mojang dropped HTTP support for their asset CDN. ForgeGradle 2.14 uses HTTP,
   which returns 400 errors. You must download assets via HTTPS manually.

The ``setup_malmo.sh`` script embeds the download logic. Alternatively, save and
run this Python script:

.. code-block:: bash

   python3 -c "
   import json, urllib.request, time
   from pathlib import Path

   ASSETS = Path.home() / '.gradle/caches/minecraft/assets/objects'
   INDEX = 'https://launchermeta.mojang.com/v1/packages/d5a285bdf7b0c8f1dd6e57f36564da0ea17e3c87/1.11.json'
   CDN = 'https://resources.download.minecraft.net'

   with urllib.request.urlopen(INDEX) as r:
       objects = json.load(r)['objects']
   for name, info in objects.items():
       h = info['hash']
       dest = ASSETS / h[:2] / h
       dest.parent.mkdir(parents=True, exist_ok=True)
       if dest.exists() and dest.stat().st_size == info['size']:
           continue
       try:
           urllib.request.urlretrieve(f'{CDN}/{h[:2]}/{h}', dest)
       except Exception:
           pass
   print('Done')
   "

Step 6: Build the Malmo Forge Mod
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   cd 3rd_party/environments/malmo_updated/Minecraft
   export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
   ./gradlew build

Step 7: Install MalmoEnv Python Package
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install --no-build-isolation -e 3rd_party/environments/malmo/MalmoEnv/

Step 8: Launch Minecraft
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Headless (recommended):
   cd 3rd_party/environments/mosaic_malmo
   ./launch_malmo_headless.sh -port 9000 -env

   # Or with visible window (debugging):
   cd 3rd_party/environments/malmo_updated/Minecraft
   export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
   bash launchClient.sh -port 9000 -env

Step 9: Launch MOSAIC
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   ./run.sh


Troubleshooting
---------------

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Error
     - Fix
   * - ``Could not determine java version from '17.0.x'``
     - Java 8 not found. Run ``sudo apt install openjdk-8-jdk``.
       The scripts auto-detect Java 8 in ``/usr/lib/jvm/``.
   * - ``Address already in use`` on port 9000 or 9001
     - Old Minecraft still running. Use ``run_malmo.sh`` which auto-kills old processes,
       or manually: ``kill $(lsof -ti :9000)``
   * - ``Cannot connect to Minecraft on localhost:9000``
     - Minecraft not running. Start it with ``./run_malmo.sh`` and wait for ``DORMANT``.
   * - ``HTTP 400`` errors during asset download
     - Run ``./setup_malmo.sh`` which downloads via HTTPS, or use the manual script above.
   * - ``malmoenv package is not installed``
     - Run ``pip install -e 3rd_party/environments/malmo/MalmoEnv/``
   * - ``MALMO_XSD_PATH not set``
     - Run ``./setup_malmo.sh`` or export manually (see Step 3 above).
   * - ``nether2.ogg does not exist`` warning
     - Harmless — Mojang removed this sound file. Minecraft works fine without it.
   * - Black screen on TreasureHunt
     - This is a **2-agent mission**. The server waits for both agents.
       Single-agent mode is not supported for this mission.
   * - Keys don't move the agent
     - Make sure you clicked **Start Game** after loading the environment.
       Check Java log for ``MOSAIC: Human connected - switching to HUMAN input mode``.
   * - Agent keeps moving after key release (continuous missions)
     - This happens if ``InputType.AI`` is still active. Restart Minecraft
       with ``./run_malmo.sh`` to get the latest Java mod with auto-detection.


Repository Layout
-----------------

.. code-block:: text

   mosaic/
   ├── setup_malmo.sh          ← One-time setup (Java, assets, build, pip)
   ├── run_malmo.sh            ← Launch Minecraft headless (port 9000+9001)
   ├── run.sh                  ← Launch MOSAIC GUI
   │
   ├── 3rd_party/environments/
   │   ├── malmo/              ← Upstream Malmo (git submodule, read-only)
   │   ├── malmo_updated/      ← MOSAIC's modified Malmo (NativeInputHandler)
   │   ├── marLo/              ← Upstream MarLo (git submodule, read-only)
   │   ├── mosaic_malmo/       ← MOSAIC Malmo integration (scripts, docs)
   │   └── mosaic_marLo/       ← MOSAIC MarLo integration (planned)
   │
   └── gym_gui/
       ├── core/adapters/mosaic_malmo.py    ← MalmoEnv adapter (13 missions)
       ├── controllers/malmo_input.py       ← Key resolver (RL path)
       ├── controllers/malmo_interaction.py ← Native input (human path)
       └── rendering/strategies/mosaic_malmo.py ← Minecraft renderer
