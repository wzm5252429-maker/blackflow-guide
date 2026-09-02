from __future__ import annotations

import cv2
import mss
import numpy as np

from .window import ClientArea


class WindowCapture:
    def __init__(self, area: ClientArea) -> None:
        self.area = area
        self._sct = mss.mss()

    def grab(self) -> np.ndarray:
        monitor = {
            "left": self.area.left,
            "top": self.area.top,
            "width": self.area.width,
            "height": self.area.height,
        }
        bgra = np.asarray(self._sct.grab(monitor), dtype=np.uint8)
        return cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)

    def close(self) -> None:
        self._sct.close()

