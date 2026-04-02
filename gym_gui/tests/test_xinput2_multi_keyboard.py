#!/usr/bin/env python3
"""
Proof-of-concept: Can we get individual keyboard device IDs using XInput2 in PyQt6?

This test intercepts raw X11 events to extract keyboard device IDs.
If successful, we can use this approach for multi-keyboard support.
"""
import ctypes
import sys

from PyQt6 import QtCore, QtWidgets

# Try to import xcffib for XCB event parsing
try:
    import xcffib
    import xcffib.xinput
    from xcffib.xproto import KeyPressEvent, KeyReleaseEvent
    HAS_XCFFIB = True
except ImportError:
    HAS_XCFFIB = False


class XInput2EventFilter(QtCore.QAbstractNativeEventFilter):
    """Native event filter to intercept XInput2 keyboard events.

    This bypasses Qt's standard event system to access raw X11/XInput2 events,
    which contain device-specific information that Qt doesn't expose.
    """

    def __init__(self, widget):
        super().__init__()
        self.widget = widget
        self.xinput_opcode = None
        self.window_id = None

        if HAS_XCFFIB:
            try:
                # Connect to X11
                self.conn = xcffib.connect()  # type: ignore[possibly-undefined]
                self.widget.append_event("DEBUG: Connected to X11")

                # Query for XInput extension
                self.conn.get_setup()
                xinput_ext = self.conn.core.QueryExtension(len("XInputExtension"), "XInputExtension").reply()

                if not xinput_ext.present:
                    raise Exception("XInput extension not available")

                self.xinput_opcode = xinput_ext.major_opcode

                msg = f"✓ XInput2 extension found: opcode={self.xinput_opcode}"
                print(msg)
                self.widget.append_event(msg)

                self.widget.append_event("✓ XInput extension ready for event capture")

            except Exception as e:
                msg = f"✗ Failed to initialize XInput2: {e}"
                print(msg)
                self.widget.append_event(msg)
                import traceback
                traceback.print_exc()
                self.conn = None

    def select_xinput2_events(self, window_id):
        """Register to receive XInput2 events for this window.

        This is CRITICAL - without calling XISelectEvents, we only get standard
        X11 events which don't contain device information.
        """
        if not self.conn:
            return

        self.window_id = window_id

        try:
            # Use xcffib's proper XInput2 API
            import xcffib.xinput as xinput

            # Get the xinput extension connection
            xi = self.conn(xinput.key)

            # Create event mask for KeyPress and KeyRelease (not Raw - those might need special perms)
            mask_obj = xinput.XIEventMask()
            # Use regular KeyPress/KeyRelease (2, 3) instead of Raw (13, 14)
            event_mask = mask_obj.KeyPress | mask_obj.KeyRelease

            # ALSO try RawKeyPress/RawKeyRelease in case they work
            raw_mask = mask_obj.RawKeyPress | mask_obj.RawKeyRelease

            # Combine both regular and raw events
            combined_mask = event_mask | raw_mask

            # Create EventMask structure using the synthetic class method
            # deviceid=0 means XIAllDevices - we want events from all keyboards
            # mask_len=1 means 1 unit of 4 bytes
            # mask=[combined_mask] is a list of 32-bit integers
            event_mask_struct = xinput.EventMask.synthetic(
                deviceid=0,  # XIAllDevices
                mask_len=1,   # 1 unit of 4 bytes
                mask=[combined_mask]  # List of 32-bit mask values
            )

            # Call XISelectEvents with the window and mask
            xi.XISelectEvents(window_id, 1, [event_mask_struct])
            self.conn.flush()

            self.widget.append_event(f"✓ XISelectEvents called for window {window_id:#x}")
            self.widget.append_event("✓ Requesting KeyPress/KeyRelease + RawKeyPress/RawKeyRelease")
            self.widget.append_event(f"✓ Event mask: regular={event_mask}, raw={raw_mask}, combined={combined_mask}")

        except Exception as e:
            self.widget.append_event(f"✗ Failed to select XInput2 events: {e}")
            import traceback
            traceback.print_exc()

    def nativeEventFilter(self, eventType, message):
        """Intercept native X11 events before Qt processes them.

        This is called for EVERY native event, so we filter for XInput2 keyboard events.
        """
        # Log the first event type we see for debugging
        if not hasattr(self, '_logged_event_type'):
            self._logged_event_type = True
            self.widget.append_event(f"DEBUG: First event type received: {eventType}")

        if not HAS_XCFFIB or not self.conn:
            return False, 0

        # On X11, eventType is "xcb_generic_event_t"
        if eventType != b"xcb_generic_event_t":
            return False, 0

        try:
            # Convert Qt's message pointer to bytes
            # XCB events are 32 bytes (standard) or larger (with extension data)
            # message is a PyQt6.sip.voidptr - need ctypes to read from it
            addr = int(message)  # type: ignore[arg-type]
            event_ptr = ctypes.cast(addr, ctypes.POINTER(ctypes.c_char * 32))
            event_bytes = bytes(event_ptr.contents)

            # First byte: event type (response_type)
            response_type = event_bytes[0] & 0x7F

            # Log first few events for debugging
            if not hasattr(self, '_event_count'):
                self._event_count = 0
            self._event_count += 1

            if self._event_count <= 5:
                self.widget.append_event(f"DEBUG: response_type={response_type}, extension={event_bytes[1] if len(event_bytes) > 1 else '?'}")

            # XInput2 events come through as GenericEvent (type 35)
            if response_type == 35:  # XCB_GE_GENERIC
                # Byte 1: extension opcode
                extension = event_bytes[1]

                if extension == self.xinput_opcode:
                    # This is an XInput2 event!
                    # Bytes 8-9: event type (uint16_t)
                    event_type = int.from_bytes(event_bytes[8:10], 'little')

                    # Log ALL XInput2 event types for first 20 events
                    if not hasattr(self, '_xinput_event_count'):
                        self._xinput_event_count = 0
                    self._xinput_event_count += 1

                    if self._xinput_event_count <= 20:
                        self.widget.append_event(f"DEBUG: XInput2 event_type={event_type}")

                    # XI_KeyPress = 2, XI_KeyRelease = 3
                    # XI_RawKeyPress = 13, XI_RawKeyRelease = 14
                    if event_type in [2, 3, 13, 14]:
                        self.widget.append_event(f"DEBUG: XInput2 KEYBOARD event_type={event_type}")

                        # Extract device ID from the event structure
                        # Note: This is simplified - proper parsing would use xcffib structs
                        # For XIDeviceEvent/XIRawEvent, deviceid is at offset 12-13
                        device_id = int.from_bytes(event_bytes[12:14], 'little')

                        # Also try to get the keycode (detail field)
                        detail = int.from_bytes(event_bytes[16:20], 'little')

                        event_name = {
                            2: "KeyPress",
                            3: "KeyRelease",
                            13: "RawKeyPress",
                            14: "RawKeyRelease"
                        }[event_type]

                        msg = f"🎹 XInput2 {event_name}: Device ID={device_id}, Keycode={detail}"
                        self.widget.append_event(msg)

                        return False, 0  # Don't consume - let Qt handle it too
                    else:
                        # Don't spam the log with non-keyboard events
                        pass

            # Standard X11 KeyPress/KeyRelease (no device info available)
            elif response_type in [2, 3]:
                event_name = "KeyPress" if response_type == 2 else "KeyRelease"
                # Standard events don't have device ID in the structure
                self.widget.append_event(f"Standard X11 {event_name} (no device ID)")

        except Exception as e:
            print(f"Error parsing event: {e}")
            import traceback
            traceback.print_exc()

        return False, 0  # Don't consume events


class XInput2TestWindow(QtWidgets.QWidget):
    """Test window for XInput2 multi-keyboard detection."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("XInput2 Multi-Keyboard Test - MOSAIC")
        self.resize(800, 600)

        layout = QtWidgets.QVBoxLayout()

        # Status header
        if HAS_XCFFIB:
            status = QtWidgets.QLabel("✓ python-xcffib installed - Testing XInput2 events")
            status.setStyleSheet("color: green; font-weight: bold; font-size: 14px;")
        else:
            status = QtWidgets.QLabel(
                "✗ python-xcffib NOT installed - Cannot test XInput2\n\n"
                "To install: .venv/bin/pip install xcffib"
            )
            status.setStyleSheet("color: red; font-weight: bold; font-size: 14px;")
        layout.addWidget(status)

        # Instructions
        instructions = QtWidgets.QLabel(
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "INSTRUCTIONS:\n"
            "1. Click in this window to give it focus\n"
            "2. Press keys on keyboard 1 - note the Device ID\n"
            "3. Press keys on keyboard 2 - check if Device ID is DIFFERENT\n"
            "4. Press keys on keyboard 3 and 4\n\n"
            "SUCCESS: If different keyboards show different Device IDs, we can implement multi-keyboard!\n"
            "FAILURE: If all keyboards show the same Device ID, XInput2 won't help us.\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )
        instructions.setStyleSheet("font-family: monospace;")
        layout.addWidget(instructions)

        # Event log
        self.log = QtWidgets.QTextEdit()
        self.log.setReadOnly(True)
        self.log.setStyleSheet("font-family: monospace; font-size: 12px;")
        layout.addWidget(self.log)

        # Summary label
        self.summary = QtWidgets.QLabel("")
        self.summary.setStyleSheet("font-weight: bold; font-size: 13px; padding: 10px;")
        layout.addWidget(self.summary)

        self.setLayout(layout)

        # Install native event filter
        if HAS_XCFFIB:
            self.event_filter = XInput2EventFilter(self)
            app_instance = QtCore.QCoreApplication.instance()
            assert app_instance is not None
            app_instance.installNativeEventFilter(self.event_filter)
            self.log.append("✓ Native event filter installed")
            self.log.append("✓ Waiting for keyboard events...\n")

            # Track unique device IDs
            self.device_ids = set()
        else:
            self.log.append("✗ Cannot install event filter - python-xcffib not available")
            self.log.append("\nInstall xcffib:")
            self.log.append("  pip install xcffib")

    def showEvent(self, event):  # type: ignore[override]
        """Called when window is shown - register for XInput2 events."""
        super().showEvent(event)

        if HAS_XCFFIB and hasattr(self, 'event_filter') and self.event_filter.conn:
            # Get the X11 window ID from Qt
            window_id = self.winId()
            if window_id:
                self.event_filter.select_xinput2_events(int(window_id))
            else:
                self.log.append("✗ Could not get window ID")

    def append_event(self, text):
        """Thread-safe append to log."""
        QtCore.QMetaObject.invokeMethod(
            self.log, "append", QtCore.Qt.ConnectionType.QueuedConnection,
            QtCore.Q_ARG(str, text)
        )

        # Extract device ID and update summary
        if "Device ID=" in text:
            try:
                device_id_str = text.split("Device ID=")[1].split(",")[0]
                device_id = int(device_id_str)
                self.device_ids.add(device_id)

                if len(self.device_ids) == 1:
                    self.summary.setText(f"📍 Detected 1 device: ID {device_id}")
                    self.summary.setStyleSheet("color: orange; font-weight: bold;")
                elif len(self.device_ids) > 1:
                    ids_str = ", ".join(str(d) for d in sorted(self.device_ids))
                    self.summary.setText(f"✓ SUCCESS! Detected {len(self.device_ids)} different devices: {ids_str}")
                    self.summary.setStyleSheet("color: green; font-weight: bold; font-size: 14px;")
            except:
                pass


def main():
    """Run the XInput2 test."""
    app = QtWidgets.QApplication(sys.argv)
    window = XInput2TestWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
