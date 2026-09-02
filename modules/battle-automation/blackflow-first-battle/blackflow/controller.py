from __future__ import annotations

import ctypes
import platform
import threading
import time
from typing import Callable

if platform.system() == "Windows":
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except AttributeError:
        pass

import pyautogui
from pynput import keyboard

from .window import ClientArea


class EmergencyStop(RuntimeError):
    pass


class StopMonitor:
    """Global Ctrl+Alt+F12 emergency stop."""

    def __init__(self) -> None:
        self.event = threading.Event()
        self._pressed: set[object] = set()
        self._listener: keyboard.Listener | None = None

    def start(self) -> None:
        def on_press(key: object) -> None:
            self._pressed.add(key)
            ctrl = keyboard.Key.ctrl_l in self._pressed or keyboard.Key.ctrl_r in self._pressed
            alt = keyboard.Key.alt_l in self._pressed or keyboard.Key.alt_r in self._pressed
            if ctrl and alt and key == keyboard.Key.f12:
                self.event.set()

        def on_release(key: object) -> None:
            self._pressed.discard(key)

        self._listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self._listener.daemon = True
        self._listener.start()

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()

    def check(self) -> None:
        if self.event.is_set():
            raise EmergencyStop("Emergency stop requested with Ctrl+Alt+F12.")


class InputController:
    def __init__(self, area: ClientArea, base_resolution: list[int], stop_check: Callable[[], None]) -> None:
        self.area = area
        self.base_width, self.base_height = base_resolution
        self.stop_check = stop_check
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.04

    def to_screen(self, point: list[float] | tuple[float, float]) -> tuple[int, int]:
        x = self.area.left + round(float(point[0]) * self.area.width / self.base_width)
        y = self.area.top + round(float(point[1]) * self.area.height / self.base_height)
        return x, y

    def sleep(self, seconds: float) -> None:
        end = time.monotonic() + max(0.0, seconds)
        while time.monotonic() < end:
            self.stop_check()
            time.sleep(max(0.0, min(0.05, end - time.monotonic())))

    def click(self, point: list[float], button: str = "left") -> None:
        self.stop_check()
        pyautogui.click(*self.to_screen(point), button=button)

    def key(self, key_name: str) -> None:
        self.stop_check()
        pyautogui.press(key_name)

    def drag(self, start: list[float], end: list[float], duration: float = 0.35) -> None:
        self.stop_check()
        sx, sy = self.to_screen(start)
        ex, ey = self.to_screen(end)
        pyautogui.moveTo(sx, sy, duration=0.08)
        pyautogui.dragTo(ex, ey, duration=duration, button="left")

    def deploy(
        self,
        card: list[float],
        tile: list[float],
        direction: str,
        drag_duration: float = 0.45,
        direction_distance: float = 130,
    ) -> None:
        """Drag a card to a tile, then swipe while held to choose facing."""
        self.stop_check()
        sx, sy = self.to_screen(card)
        tx, ty = self.to_screen(tile)
        vectors = {
            "left": (-direction_distance, 0),
            "right": (direction_distance, 0),
            "up": (0, -direction_distance),
            "down": (0, direction_distance),
            "none": (0, 0),
        }
        if direction not in vectors:
            raise ValueError(f"Unknown deployment direction: {direction}")
        dx, dy = vectors[direction]
        dx *= self.area.width / self.base_width
        dy *= self.area.height / self.base_height
        pyautogui.moveTo(sx, sy, duration=0.08)
        pyautogui.mouseDown(button="left")
        pyautogui.moveTo(tx, ty, duration=drag_duration)
        self.sleep(0.18)
        pyautogui.moveTo(tx + dx, ty + dy, duration=0.16)
        pyautogui.mouseUp(button="left")
