#!/usr/bin/env python3
"""
Evdev-based keyboard monitoring for multi-keyboard support.

This module provides direct access to Linux input devices via evdev,
bypassing X11/Qt keyboard event merging to enable true multi-keyboard support.
"""
import glob
import logging
import os
import select
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from qtpy.QtCore import QObject, QThread
from qtpy.QtCore import Signal as pyqtSignal

_LOGGER = logging.getLogger(__name__)


@dataclass
class KeyboardDevice:
    """Information about a keyboard input device."""
    device_path: str
    event_path: str
    name: str
    usb_port: Optional[str] = None
    vendor_id: Optional[str] = None
    product_id: Optional[str] = None

    def __hash__(self):
        return hash(self.device_path)

    def __eq__(self, other):
        if not isinstance(other, KeyboardDevice):
            return False
        return self.device_path == other.device_path


class EvdevKeyboardMonitor(QObject):
    """Monitors multiple keyboard devices using Linux evdev."""

    key_pressed = pyqtSignal(str, int, int)
    key_released = pyqtSignal(str, int, int)
    device_connected = pyqtSignal(object)
    device_disconnected = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    EV_KEY = 1
    EV_VALUE_RELEASE = 0
    EV_VALUE_PRESS = 1
    EV_VALUE_REPEAT = 2

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._devices: Dict[str, KeyboardDevice] = {}
        self._file_descriptors: Dict[int, str] = {}
        self._running = False
        self._thread: Optional[QThread] = None
        _LOGGER.info("EvdevKeyboardMonitor initialized")

    def discover_keyboards(self) -> List[KeyboardDevice]:
        """Discover all keyboard input devices."""
        keyboards = []
        device_patterns = ['/dev/input/by-path/*kbd', '/dev/input/by-id/*kbd']
        seen_real_paths = set()

        for pattern in device_patterns:
            for device_path in sorted(glob.glob(pattern)):
                try:
                    real_path = os.path.realpath(device_path)
                    if real_path in seen_real_paths:
                        continue
                    seen_real_paths.add(real_path)

                    name = self._get_device_name(real_path)
                    usb_port = self._extract_usb_port(device_path)
                    vendor_id, product_id = self._get_device_ids(real_path)

                    keyboard = KeyboardDevice(
                        device_path=device_path,
                        event_path=real_path,
                        name=name,
                        usb_port=usb_port,
                        vendor_id=vendor_id,
                        product_id=product_id
                    )
                    keyboards.append(keyboard)
                    _LOGGER.info(f"Discovered keyboard: {name} at {device_path}")
                except Exception as e:
                    _LOGGER.warning(f"Failed to process device {device_path}: {e}")

        return keyboards

    def add_device(self, device: KeyboardDevice) -> bool:
        """Add a keyboard device to monitor."""
        if device.device_path in self._devices:
            return True

        try:
            fd = os.open(device.event_path, os.O_RDONLY | os.O_NONBLOCK)
            self._devices[device.device_path] = device
            self._file_descriptors[fd] = device.device_path
            _LOGGER.info(f"Added device: {device.name} (fd={fd})")
            self.device_connected.emit(device)
            return True
        except PermissionError:
            error_msg = (
                f"Permission denied: Cannot access {device.event_path}\n"
                f"Fix: sudo usermod -a -G input {os.getenv('USER', 'your_username')}\n"
                f"Then log out and log back in."
            )
            _LOGGER.error(error_msg)
            self.error_occurred.emit(error_msg)
            return False
        except Exception as e:
            error_msg = f"Failed to open device {device.event_path}: {e}"
            _LOGGER.error(error_msg)
            self.error_occurred.emit(error_msg)
            return False

    def remove_device(self, device_path: str) -> bool:
        """Remove a keyboard device from monitoring."""
        if device_path not in self._devices:
            return False

        fd_to_close = None
        for fd, path in self._file_descriptors.items():
            if path == device_path:
                fd_to_close = fd
                break

        if fd_to_close is not None:
            try:
                os.close(fd_to_close)
            except Exception as e:
                _LOGGER.warning(f"Error closing fd {fd_to_close}: {e}")
            del self._file_descriptors[fd_to_close]

        del self._devices[device_path]
        _LOGGER.info(f"Removed device: {device_path}")
        self.device_disconnected.emit(device_path)
        return True

    def start_monitoring(self):
        """Start monitoring keyboard devices in a background thread."""
        if self._running:
            _LOGGER.warning("Monitor already running")
            return
        if not self._devices:
            _LOGGER.warning("No devices to monitor")
            return

        self._running = True
        self._thread = QThread()
        self.moveToThread(self._thread)
        self._thread.started.connect(self._monitor_loop)
        self._thread.start()
        _LOGGER.info(f"Started monitoring {len(self._devices)} keyboard(s)")

    def stop_monitoring(self):
        """Stop monitoring keyboard devices."""
        if not self._running:
            return

        self._running = False
        if self._thread and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait(2000)

        for fd in list(self._file_descriptors.keys()):
            try:
                os.close(fd)
            except Exception as e:
                _LOGGER.warning(f"Error closing fd {fd}: {e}")
        self._file_descriptors.clear()
        _LOGGER.info("Stopped monitoring")

    def _monitor_loop(self):
        """Main monitoring loop (runs in background thread)."""
        _LOGGER.debug("Monitoring loop started")

        while self._running:
            try:
                if not self._file_descriptors:
                    break

                readable, _, _ = select.select(
                    self._file_descriptors.keys(), [], [], 0.5
                )

                for fd in readable:
                    self._process_device_events(fd)

            except select.error as e:
                _LOGGER.error(f"Select error: {e}")
                break
            except Exception as e:
                _LOGGER.error(f"Error in monitoring loop: {e}", exc_info=True)
                self.error_occurred.emit(f"Monitoring error: {e}")
                break

        _LOGGER.debug("Monitoring loop ended")

    def _process_device_events(self, fd: int):
        """Process events from a single device."""
        device_path = self._file_descriptors.get(fd)
        if not device_path:
            return

        try:
            while True:
                data = os.read(fd, 24)
                if len(data) == 0:
                    _LOGGER.warning(f"Device disconnected: {device_path}")
                    self.remove_device(device_path)
                    break
                if len(data) != 24:
                    continue

                tv_sec, tv_usec, ev_type, ev_code, ev_value = struct.unpack('llHHi', data)

                if ev_type != self.EV_KEY:
                    continue

                timestamp = (tv_sec * 1000) + (tv_usec // 1000)

                if ev_value == self.EV_VALUE_PRESS:
                    self.key_pressed.emit(device_path, ev_code, timestamp)
                elif ev_value == self.EV_VALUE_RELEASE:
                    self.key_released.emit(device_path, ev_code, timestamp)

        except BlockingIOError:
            pass
        except Exception as e:
            _LOGGER.error(f"Error processing events from {device_path}: {e}", exc_info=True)
            self.error_occurred.emit(f"Error reading from {device_path}: {e}")

    def _get_device_name(self, event_path: str) -> str:
        """Get human-readable name of input device."""
        try:
            event_num = Path(event_path).name.replace('event', '')
            with open('/proc/bus/input/devices', 'r') as f:
                content = f.read()

            for block in content.split('\n\n'):
                if f'event{event_num}' in block:
                    for line in block.split('\n'):
                        if line.startswith('N: Name='):
                            return line.split('Name=')[1].strip('"')
        except Exception as e:
            _LOGGER.debug(f"Could not get device name: {e}")
        return "Unknown Device"

    def _extract_usb_port(self, device_path: str) -> Optional[str]:
        """Extract USB port number from device path."""
        try:
            if 'usb-' in device_path:
                usb_part = device_path.split('usb-')[1].split('-')[0]
                parts = usb_part.split(':')
                if len(parts) >= 2:
                    return parts[1]
        except Exception as e:
            _LOGGER.debug(f"Could not extract USB port: {e}")
        return None

    def _get_device_ids(self, event_path: str):
        """Get vendor and product IDs of device."""
        try:
            event_num = Path(event_path).name.replace('event', '')
            with open('/proc/bus/input/devices', 'r') as f:
                content = f.read()

            for block in content.split('\n\n'):
                if f'event{event_num}' in block:
                    for line in block.split('\n'):
                        if line.startswith('I: Bus='):
                            parts = line.split()
                            vendor_id = None
                            product_id = None
                            for part in parts:
                                if part.startswith('Vendor='):
                                    vendor_id = part.split('=')[1]
                                elif part.startswith('Product='):
                                    product_id = part.split('=')[1]
                            return vendor_id, product_id
        except Exception as e:
            _LOGGER.debug(f"Could not get device IDs: {e}")
        return None, None

    def get_monitored_devices(self) -> List[KeyboardDevice]:
        """Get list of currently monitored devices."""
        return list(self._devices.values())

    def is_monitoring(self) -> bool:
        """Check if monitor is currently running."""
        return self._running
