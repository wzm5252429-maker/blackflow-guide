from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import dataclass
import platform


class WindowError(RuntimeError):
    pass


@dataclass(frozen=True)
class ClientArea:
    hwnd: int
    title: str
    left: int
    top: int
    width: int
    height: int


def _windows_only() -> None:
    if platform.system() != "Windows":
        raise WindowError("Real game mode must run on Windows. Use --simulate on other systems.")
    # Keep Win32 client coordinates, screenshots and PyAutoGUI coordinates in the
    # same physical-pixel coordinate space when Windows display scaling is on.
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except AttributeError:
        pass


def find_window(title_keywords: list[str]) -> ClientArea:
    _windows_only()
    user32 = ctypes.windll.user32
    matches: list[tuple[int, str]] = []
    keywords = [item.casefold() for item in title_keywords if item.strip()]

    callback_type = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)

    @callback_type
    def callback(hwnd: int, _lparam: int) -> bool:
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buffer, length + 1)
        title = buffer.value
        folded = title.casefold()
        if any(keyword in folded for keyword in keywords):
            matches.append((int(hwnd), title))
        return True

    user32.EnumWindows(callback, 0)
    if not matches:
        raise WindowError(f"No visible window matched: {title_keywords}")

    hwnd, title = matches[0]
    rect = wintypes.RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
        raise WindowError(f"GetClientRect failed for {title!r}")
    point = wintypes.POINT(0, 0)
    if not user32.ClientToScreen(hwnd, ctypes.byref(point)):
        raise WindowError(f"ClientToScreen failed for {title!r}")
    width = rect.right - rect.left
    height = rect.bottom - rect.top
    if width <= 0 or height <= 0:
        raise WindowError(f"The matched window has an invalid client area: {width}x{height}")
    return ClientArea(hwnd, title, point.x, point.y, width, height)


def focus_window(hwnd: int) -> None:
    _windows_only()
    user32 = ctypes.windll.user32
    SW_RESTORE = 9
    user32.ShowWindow(hwnd, SW_RESTORE)
    if not user32.SetForegroundWindow(hwnd):
        raise WindowError("Could not bring the game window to the foreground. Click it once and retry.")
