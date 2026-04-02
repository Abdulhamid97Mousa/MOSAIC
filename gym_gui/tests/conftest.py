"""Shared pytest configuration for gym_gui tests.

Sets environment variables required for headless CI environments before
any Qt modules are imported.  This must happen at conftest load time
(before test collection) because Qt reads these variables once during
QApplication initialization.
"""

import os

# Qt platform: use offscreen rendering (no display server needed)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# QtWebEngine (Chromium): disable GPU and sandbox to prevent SIGABRT
# on headless CI runners without GPU/EGL support
os.environ.setdefault("QTWEBENGINE_DISABLE_SANDBOX", "1")
os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--disable-gpu")

# OpenGL: force Mesa software rendering on headless systems.
# PyQt6 6.9+ loads QtOpenGL/QtOpenGLWidgets internally even for basic
# QWidget subclasses.  Without a GPU or proper EGL, this triggers
# SIGABRT.  LIBGL_ALWAYS_SOFTWARE tells Mesa to use llvmpipe.
os.environ.setdefault("QT_OPENGL", "software")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("MESA_GL_VERSION_OVERRIDE", "3.3")
os.environ.setdefault("QT_QUICK_BACKEND", "software")
