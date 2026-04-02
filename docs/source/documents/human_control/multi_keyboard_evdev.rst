Multi-Keyboard Support (Evdev)
==============================

MOSAIC supports true multi-keyboard input so that multiple human players
can each control a separate agent in the same multi-agent environment.
Because the standard Linux display server merges all keyboards into one
virtual device, MOSAIC reads each physical keyboard independently via
the Linux **evdev** subsystem.

Why Multiple Keyboards?
-----------------------

.. raw:: html

   <video style="width:100%; max-width:100%; height:auto; border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,0.15);" controls autoplay muted loop playsinline>
     <source src="../../_static/videos/human_vs_human.mp4" type="video/mp4">
     Your browser does not support the video tag.
   </video>
   <br/><br/>

Multi-agent environments such as MultiGrid-Soccer (2 v 2, 4 agents),
Overcooked (2 cooperating chefs), MeltingPot (up to 16 agents), or
RWARE (2 to 8 warehouse robots) are designed for simultaneous human play.
Each player needs a **dedicated keyboard** so that their key presses
only move their own agent.

A USB hub with at least 4 ports is recommended because:

- Most laptops expose only 1 to 3 USB ports, which are already occupied by
  mouse, headset, or other peripherals.
- A 4-port hub lets you connect 4 inexpensive USB keyboards (one per
  agent) to a single host USB port.
- Larger hubs (7-port) support up to 7 human players for environments
  like MeltingPot that can exceed 4 agents.

Tested hardware combinations:

.. list-table::
   :widths: 25 25 25 25
   :header-rows: 1

   * - Keyboard
     - Vendor ID
     - Product ID
     - USB Port
   * - Logitech USB Keyboard
     - 046d
     - c31c
     - Hub port 1
   * - Logitech USB Keyboard
     - 046d
     - c34b
     - Hub port 2
   * - SIGMACHIP USB Keyboard
     - 1c4f
     - 0002
     - Hub port 3
   * - Lenovo Black Silk
     - 17ef
     - 602d
     - Hub port 4

All four keyboards share the **same key bindings** (WASD for movement,
Space for interact, etc.), the evdev layer distinguishes them by which
physical USB port they are plugged into, not by which keys are pressed.

The X11 Problem
---------------

On Linux with X11 (the most common display server on Ubuntu 22.04), all
connected keyboards are merged into a single logical device called the
**Virtual core keyboard**.  Every key event Qt sees comes from this
virtual device, so ``QInputDevice.systemId()`` returns the same ID
regardless of which physical keyboard was pressed.

Three approaches were evaluated:

.. list-table::
   :widths: 25 50 25
   :header-rows: 1

   * - Approach
     - Details
     - Result
   * - Qt ``QInputDevice.systemId()``
     - Under X11, the display server merges all physical keyboards into a
       single virtual core device.  As a result, ``systemId()`` returns an
       identical value for every keyboard, making per-device discrimination
       impossible at the Qt abstraction layer.
     - Rejected
   * - XInput2 native event filter
     - The XInput2 extension can report distinct device identifiers for
       pointing devices; however, Qt consumes keyboard events at the
       toolkit level before they reach the application's native event
       filter, preventing per-device attribution for key presses.
     - Rejected
   * - **evdev** (direct ``/dev/input``)
     - Opens each ``/dev/input/eventX`` file descriptor independently and
       reads raw ``struct input_event`` data directly from the Linux kernel
       input subsystem.  This approach bypasses the X11 device-merging
       layer entirely, providing reliable per-keyboard identification.
     - **Adopted**

Evidence is preserved in the test suite:

- ``test_qt_keyboard_device_id.py``: all keyboards show ``systemId=3``
- ``test_xinput2_multi_keyboard.py``: mouse events work, keyboard
  events do not
- ``test_usb_hub_keyboards.py``: evdev correctly identifies per-device
  key presses

Architecture
------------

.. mermaid::

   graph TD
       KB1[USB Keyboard 1] --> HUB[USB Hub]
       KB2[USB Keyboard 2] --> HUB
       KB3[USB Keyboard 3] --> HUB
       KB4[USB Keyboard 4] --> HUB
       HUB --> EVDEV["/dev/input/eventX"]
       EVDEV --> MON[EvdevKeyboardMonitor<br/>QThread]
       MON --> |key_pressed signal| HIC[HumanInputController]
       HIC --> |device_path → agent_id| ROUTE{Agent Router}
       ROUTE --> A0[Agent 0 pressed_keys]
       ROUTE --> A1[Agent 1 pressed_keys]
       ROUTE --> A2[Agent 2 pressed_keys]
       ROUTE --> A3[Agent 3 pressed_keys]
       A0 --> RES0[KeyCombinationResolver]
       A1 --> RES1[KeyCombinationResolver]
       A2 --> RES2[KeyCombinationResolver]
       A3 --> RES3[KeyCombinationResolver]
       RES0 --> SC[SessionController]
       RES1 --> SC
       RES2 --> SC
       RES3 --> SC

       style KB1 fill:#4a90d9,stroke:#2e5a87,color:#fff
       style KB2 fill:#4a90d9,stroke:#2e5a87,color:#fff
       style KB3 fill:#4a90d9,stroke:#2e5a87,color:#fff
       style KB4 fill:#4a90d9,stroke:#2e5a87,color:#fff
       style HUB fill:#4a90d9,stroke:#2e5a87,color:#fff
       style MON fill:#ff7f50,stroke:#cc5500,color:#fff
       style HIC fill:#50c878,stroke:#2e8b57,color:#fff
       style SC fill:#50c878,stroke:#2e8b57,color:#fff
       style ROUTE fill:#9370db,stroke:#6a0dad,color:#fff

Data Flow
---------

The complete path of a single key press from hardware to game action:

1. Player presses **W** on Keyboard 2 (plugged into hub port 2).
2. The Linux kernel writes a 24-byte ``input_event`` struct to
   ``/dev/input/event17``.
3. ``EvdevKeyboardMonitor`` (running on a background ``QThread``) wakes
   from ``select()`` and reads the raw bytes.
4. The struct is unpacked:

   .. code-block:: c

      struct input_event {
          struct timeval time;   /* 16 bytes */
          __u16 type;            /*  2 bytes  (EV_KEY = 1) */
          __u16 code;            /*  2 bytes  (KEY_W = 17) */
          __s32 value;           /*  4 bytes  (1 = press)  */
      };
      /* Total: 24 bytes */

5. The monitor emits ``key_pressed(device_path, keycode=17, timestamp)``.
6. ``HumanInputController`` receives the signal and looks up
   ``device_path`` in ``_evdev_device_to_agent`` to find ``agent_id``
   (e.g., ``agent_1``).
7. ``linux_keycode_to_qt_key(17)`` converts Linux keycode 17 to
   ``Qt.Key.Key_W``, then ``int()`` converts the enum to an integer
   (**critical**: the pressed-keys set stores plain integers for
   compatibility with the ``KeyCombinationResolver``).
8. The key is added to ``_agent_pressed_keys["agent_1"]``.
9. The ``KeyCombinationResolver`` for this environment inspects
   agent 1's pressed-key set and returns an action index (e.g.,
   ``FORWARD = 2``).
10. ``perform_human_action(2)`` is emitted **for agent 1 only**.
11. The MultiGrid environment receives agent 1's action; all other agents
    receive ``STILL`` (no-op) unless their keyboards were also pressed.

Implementation Components
-------------------------

EvdevKeyboardMonitor
^^^^^^^^^^^^^^^^^^^^

A ``QObject`` that runs on a dedicated ``QThread``.  Located at
``gym_gui/controllers/evdev_keyboard_monitor.py``.

**Signals:**

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Signal
     - Description
   * - ``key_pressed(str, int, int)``
     - ``(device_path, linux_keycode, timestamp_ms)``
   * - ``key_released(str, int, int)``
     - ``(device_path, linux_keycode, timestamp_ms)``
   * - ``device_connected(object)``
     - A ``KeyboardDevice`` was successfully opened.
   * - ``device_disconnected(str)``
     - A device was removed (hot-unplug or read error).
   * - ``error_occurred(str)``
     - Human-readable error message (e.g., permission denied).

**Key methods:**

- ``discover_keyboards()`` : scans ``/dev/input/by-path/*kbd`` and
  ``/dev/input/by-id/*kbd``, deduplicates by real path, reads device
  names from ``/proc/bus/input/devices``.
- ``add_device(device)`` : opens the event file with
  ``O_RDONLY | O_NONBLOCK``.  On ``PermissionError``, emits an error
  with instructions to run ``sudo usermod -a -G input $USER``.
- ``start_monitoring()`` : moves itself to a ``QThread`` and enters the
  ``_monitor_loop()``.
- ``_monitor_loop()`` : calls ``select()`` with a 0.5 s timeout across
  all open file descriptors; dispatches to ``_process_device_events()``
  for each readable fd.
- ``_process_device_events(fd)`` : reads 24-byte packets, unpacks with
  ``struct.unpack('llHHi', data)``, emits ``key_pressed`` or
  ``key_released`` on ``EV_KEY`` events.

Device Path Strategy
^^^^^^^^^^^^^^^^^^^^

MOSAIC uses ``/dev/input/by-path/*-event-kbd`` paths because they are
**stable across reboots**, the path encodes the USB topology (PCI bus,
hub port number), not a volatile event number.

Example paths for 4 keyboards on one hub::

   /dev/input/by-path/pci-0000:00:14.0-usb-0:5.1:1.0-event-kbd
   /dev/input/by-path/pci-0000:00:14.0-usb-0:5.2:1.0-event-kbd
   /dev/input/by-path/pci-0000:00:14.0-usb-0:5.3:1.0-event-kbd
   /dev/input/by-path/pci-0000:00:14.0-usb-0:5.4:1.0-event-kbd

``/dev/input/by-id/`` is **not** used because two keyboards of the same
make and model produce duplicate by-id names.
``/dev/input/eventX`` numbers change on reboot and are also avoided.

Keycode Translation
^^^^^^^^^^^^^^^^^^^

``gym_gui/controllers/keycode_translation.py`` maps Linux scancodes
(defined in ``/usr/include/linux/input-event-codes.h``) to
``Qt.Key`` enum values.

.. code-block:: python

   LINUX_TO_QT_KEYCODE = {
       17: Qt.Key.Key_W,     # KEY_W
       30: Qt.Key.Key_A,     # KEY_A
       31: Qt.Key.Key_S,     # KEY_S
       32: Qt.Key.Key_D,     # KEY_D
       57: Qt.Key.Key_Space, # KEY_SPACE
       103: Qt.Key.Key_Up,   # KEY_UP
       105: Qt.Key.Key_Left, # KEY_LEFT
       106: Qt.Key.Key_Right,# KEY_RIGHT
       108: Qt.Key.Key_Down, # KEY_DOWN
       # ... 100+ entries covering letters, numbers,
       #     function keys, numpad, modifiers, etc.
   }

After translation, the rest of the input pipeline (resolvers, action
mapping) is identical to single-keyboard mode.

KeyboardAssignmentWidget
^^^^^^^^^^^^^^^^^^^^^^^^

A Qt widget (``gym_gui/ui/widgets/keyboard_assignment_widget.py``) that
provides the user interface for mapping keyboards to agents:

- **Auto-detection:** discovers all connected USB keyboards via the
  evdev monitor.
- **Auto-assign:** maps the first *N* discovered keyboards to agents
  0..*N*-1 in the order they appear on the USB bus.
- **Manual override:** drag-and-drop or drop-down selection to
  reassign any keyboard to any agent.
- **Test mode:** press any key and the widget highlights which agent
  the keyboard controls.
- **Persistence:** assignments are saved to
  ``config/keyboard_assignments.yaml`` and restored on next launch.

Why State-Based Mode Is Required
---------------------------------

Multi-agent environments **force** state-based input mode.  This is
enforced at three layers to prevent misconfiguration:

.. list-table::
   :widths: 20 40 40
   :header-rows: 1

   * - Layer
     - Mechanism
     - Effect
   * - Configuration
     - ``required_input_mode`` property on ``MultiGridConfig``,
       ``MeltingPotConfig``, ``OvercookedConfig``.
     - Declares the requirement in code.
   * - UI
     - ``control_panel.py`` only shows "State-Based (Real-time)" in the
       input mode combo box for multi-agent games.
     - User cannot select the wrong mode.
   * - Controller
     - ``HumanInputController.configure()`` validates at runtime and
       forcibly overrides to state-based if needed.
     - Catches programmatic errors.

The technical reason is that shortcut-based mode uses Qt's global
``QShortcut`` system.  When evdev is active, a key press generates
**two** events:

1. The evdev monitor correctly routes the key to one agent.
2. Qt's ``QShortcut`` **also fires** for all agents (because X11 merged
   the keyboard into the virtual core device).

The result is that one key press moves **every** agent instead of just
the intended one.  State-based mode avoids this because it polls
per-agent ``_agent_pressed_keys`` sets, which are populated exclusively
by evdev signals.

System Requirements
-------------------

Platform
^^^^^^^^

- **Linux only:** evdev is a Linux kernel interface.
- Tested on **Ubuntu 22.04** with X11.
- Kernel 2.6 or later (evdev has been stable since 2003).

Permissions
^^^^^^^^^^^

The user must be a member of the ``input`` group to read
``/dev/input/event*`` files:

.. code-block:: bash

   sudo usermod -a -G input $USER
   # Log out and log back in for the change to take effect

Alternatively, create a udev rule:

.. code-block:: bash

   # /etc/udev/rules.d/99-mosaic-keyboards.rules
   KERNEL=="event*", SUBSYSTEM=="input", MODE="0660", GROUP="plugdev"

Hardware
^^^^^^^^

- A USB hub (4+ ports recommended).
- One USB keyboard per agent.  Any brand works, keyboards do not need
  to match.

User Workflow
-------------

One-Time Setup
^^^^^^^^^^^^^^

1. Plug a USB hub into the host machine.
2. Connect one keyboard per agent to the hub (e.g., 4 keyboards for a
   4-player Soccer match).
3. Add your user to the ``input`` group (see Permissions above).
4. Log out and back in.

In-Game Usage
^^^^^^^^^^^^^

1. Launch the MOSAIC GUI.
2. Select a multi-agent environment (e.g., MultiGrid-Soccer-v0).
3. Set the number of agents (e.g., 4).
4. Click **Load Environment**.
5. The **Keyboard Assignment Widget** appears automatically.
6. Click **Auto-Assign**, keyboards are mapped to agents in USB-port
   order.
7. Click **Apply Assignments**.
8. Start the game.  Each keyboard controls only its assigned agent.

Testing and Verification
^^^^^^^^^^^^^^^^^^^^^^^^^

1. Click **Test** on the Keyboard Assignment Widget.
2. Press any key on each keyboard.
3. The widget highlights which agent the keyboard controls.
4. Verify all keyboards operate independently.

Per-Agent Key Bindings
^^^^^^^^^^^^^^^^^^^^^^

Every keyboard uses the same key layout.  The evdev layer distinguishes
keyboards by USB port, not by key.

.. code-block:: text

   Agent 0 (Keyboard on hub port 1):
     W = Forward, A = Left, S = Backward, D = Right, Space = Interact

   Agent 1 (Keyboard on hub port 2):
     W = Forward, A = Left, S = Backward, D = Right, Space = Interact

   Agent 2 (Keyboard on hub port 3):
     W = Forward, A = Left, S = Backward, D = Right, Space = Interact

   Agent 3 (Keyboard on hub port 4):
     W = Forward, A = Left, S = Backward, D = Right, Space = Interact

Configuration Persistence
^^^^^^^^^^^^^^^^^^^^^^^^^

Keyboard-to-agent assignments are saved to
``config/keyboard_assignments.yaml``:

.. code-block:: yaml

   keyboard_assignments:
     agent_0:
       device_path: /dev/input/by-path/pci-0000:00:14.0-usb-0:5.1:1.0-event-kbd
       device_name: "Logitech USB Keyboard"
       vendor_id: "046d"
       product_id: "c31c"
     agent_1:
       device_path: /dev/input/by-path/pci-0000:00:14.0-usb-0:5.2:1.0-event-kbd
       device_name: "Logitech USB Keyboard"
       vendor_id: "046d"
       product_id: "c34b"

Because paths are based on USB topology, the same physical port always
maps to the same agent, even after a reboot.

Performance
-----------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Metric
     - Value
   * - Input latency
     - < 10 ms (keypress to action)
   * - CPU usage
     - < 2 % for the monitoring thread
   * - Memory overhead
     - ~5 MB for the evdev monitor
   * - Simultaneous keys
     - 20+ keys across 4 keyboards

The ``select()``-based event loop only wakes when there is data to read,
keeping idle CPU usage near zero.

Troubleshooting
---------------

Permission Denied
^^^^^^^^^^^^^^^^^

**Symptom:** ``PermissionError: [Errno 13] Permission denied: '/dev/input/event16'``

**Fix:**

.. code-block:: bash

   sudo usermod -a -G input $USER
   # Log out and back in
   groups | grep input   # verify

No Keyboards Detected
^^^^^^^^^^^^^^^^^^^^^

**Symptom:** The Keyboard Assignment Widget shows "No keyboards detected."

**Diagnosis:**

.. code-block:: bash

   # Verify kernel sees the keyboards
   ls -la /dev/input/by-path/*kbd

   # Check file permissions
   ls -la /dev/input/event*

   # Test direct device access
   sudo evtest /dev/input/event16   # should show key events

One Keyboard Moves All Agents
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Symptom:** Pressing a key on any keyboard moves every agent at once.

**Cause:** evdev monitoring failed to start, so the system fell back to
Qt shortcuts which fire globally.

**Diagnosis:**

.. code-block:: bash

   # Check the MOSAIC log for evdev setup messages
   grep "LOG407" var/logs/gym_gui.log | tail -10
   # LOG4072 = setup started, LOG4073 = success, LOG4074 = failure

   # Verify assignments were applied
   grep "Apply Assignments" var/logs/gym_gui.log

Worker-Based Keyboard Input (Subprocess)
-----------------------------------------

For multi-agent environments where ``adapter.step()`` and
``env.render()`` block the Qt main thread (causing agents to appear
frozen and then "teleport"), MOSAIC provides a subprocess-based
keyboard input mode.

Each physical keyboard is assigned to a **human_worker subprocess**
that reads evdev events in its own process.  The GUI never blocks
on keyboard I/O because the worker communicates via JSON pipes.

.. mermaid::

   graph TD
       KB1[USB Keyboard 1] --> P1[human_worker subprocess<br/>agent_0]
       KB2[USB Keyboard 2] --> P2[human_worker subprocess<br/>agent_1]
       P1 --> |action_selected JSON| GUI[GUI Main Process]
       P2 --> |action_selected JSON| GUI
       GUI --> |select_action JSON| P1
       GUI --> |select_action JSON| P2
       GUI --> ENV["env.step({0: action_0, 1: action_1})"]
       ENV --> RENDER[Render Frame]

       style KB1 fill:#4a90d9,stroke:#2e5a87,color:#fff
       style KB2 fill:#4a90d9,stroke:#2e5a87,color:#fff
       style P1 fill:#ff7f50,stroke:#cc5500,color:#fff
       style P2 fill:#ff7f50,stroke:#cc5500,color:#fff
       style GUI fill:#50c878,stroke:#2e8b57,color:#fff
       style ENV fill:#9370db,stroke:#6a0dad,color:#fff

How It Works
^^^^^^^^^^^^

1. The GUI launches one ``human_worker`` subprocess per agent in
   ``--mode keyboard`` with the evdev device path.
2. Each subprocess opens its ``/dev/input/eventX`` device and creates
   a pure-Python keycode resolver (no Qt dependency).
3. When the GUI sends ``{"cmd": "select_action"}``, the subprocess
   blocks reading the keyboard until a recognized key is pressed.
4. The subprocess returns ``{"type": "action_selected", "action": 3}``.
5. The GUI collects all agents' actions via ``MultiAgentStepState``
   and calls ``env.step(all_actions)`` through the standard pipeline.
6. Rewards, observations, and replay recording are fully preserved.

CLI Usage
^^^^^^^^^

.. code-block:: bash

   human-worker --mode keyboard \
       --device-path /dev/input/by-path/pci-0000:00:14.0-usb-0:5.1:1.0-event-kbd \
       --env-name multigrid \
       --player-name "Player 1"

The subprocess uses pure Python (``select()``, ``os.read()``,
``struct.unpack()``) to read raw evdev events and maps Linux
keycodes directly to action indices without requiring Qt.

Known Limitations
-----------------

- **Linux only:** evdev is not available on Windows or macOS.
- **USB required:** Bluetooth keyboards may have unreliable device
  paths.
- **Same key layout:** all keyboards share the same key bindings
  (WASD, etc.).  Per-keyboard remapping is not yet supported.
