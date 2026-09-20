"""Lossless coordinate bookkeeping for arbitrary Windows client aspect ratios.

All coordinates are physical pixels, never DPI-scaled logical pixels. Images are
aspect-fitted to a recognizer canvas; the padding is not a clickable part of the
game. No game content is cropped merely to obtain a 16:9 aspect ratio.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time
import uuid

import numpy as np


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    width: float
    height: float

    def __post_init__(self):
        if not all(math.isfinite(v) for v in (self.x, self.y, self.width, self.height)):
            raise ValueError("Rectangle coordinates must be finite")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Rectangle dimensions must be positive")

    @property
    def right(self):
        return self.x + self.width

    @property
    def bottom(self):
        return self.y + self.height

    def contains(self, x: float, y: float) -> bool:
        return self.x <= x < self.right and self.y <= y < self.bottom

    def as_xywh(self):
        return [self.x, self.y, self.width, self.height]


@dataclass(frozen=True)
class WindowGeometry:
    hwnd: int
    pid: int
    title: str
    client_rect: Rect
    dpi: int = 96

    @property
    def width(self):
        return int(self.client_rect.width)

    @property
    def height(self):
        return int(self.client_rect.height)

    @property
    def identity(self):
        # A move changes screen mapping even when the image dimensions agree.
        return self.hwnd, self.pid, self.client_rect, self.dpi

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class ViewportTransform:
    source_size: tuple[int, int]
    output_size: tuple[int, int]
    crop: Rect
    viewport: Rect
    client_origin: tuple[float, float] = (0, 0)

    def recognition_to_client(self, x: float, y: float) -> tuple[float, float]:
        if not self.viewport.contains(x, y):
            raise ValueError("Recognition point lies in padding or outside the frame")
        return (
            self.crop.x + (x - self.viewport.x) * self.crop.width / self.viewport.width,
            self.crop.y + (y - self.viewport.y) * self.crop.height / self.viewport.height,
        )

    def client_to_recognition(self, x: float, y: float) -> tuple[float, float]:
        if not self.crop.contains(x, y):
            raise ValueError("Client point lies outside the observed game content")
        return (
            self.viewport.x + (x - self.crop.x) * self.viewport.width / self.crop.width,
            self.viewport.y + (y - self.crop.y) * self.viewport.height / self.crop.height,
        )

    def recognition_to_screen(self, x: float, y: float) -> tuple[float, float]:
        cx, cy = self.recognition_to_client(x, y)
        return cx + self.client_origin[0], cy + self.client_origin[1]

    to_desktop = recognition_to_screen
    to_client = recognition_to_client

    def screen_to_recognition(self, x: float, y: float) -> tuple[float, float]:
        return self.client_to_recognition(x - self.client_origin[0], y - self.client_origin[1])

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class NormalizedFrame:
    image: np.ndarray
    transform: ViewportTransform
    geometry: WindowGeometry
    frame_id: str
    captured_at: float

    @property
    def bgr(self):
        return self.image

    @property
    def viewport(self):
        return self.transform.viewport


@dataclass(frozen=True)
class CapturedFrame:
    image: np.ndarray
    geometry: WindowGeometry
    frame_id: str = ""
    captured_at: float = 0

    @property
    def bgr(self):
        return self.image

    def __post_init__(self):
        _validate_image(self.image)
        if self.image.shape[:2] != (self.geometry.height, self.geometry.width):
            raise ValueError("Capture pixels do not match the physical client dimensions")
        if not self.frame_id:
            object.__setattr__(self, "frame_id", uuid.uuid4().hex)
        if not self.captured_at:
            object.__setattr__(self, "captured_at", time.time())

    def normalize(self, target_size=(1280, 720), *, detect_bars=True, content_rect=None):
        return normalize_frame(
            self.image, self.geometry, target_size=target_size,
            detect_bars=detect_bars, content_rect=content_rect,
            frame_id=self.frame_id, captured_at=self.captured_at,
        )

    normalized = normalize


def _validate_image(image):
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ValueError("A uint8 BGR screenshot is required")
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 2:
        raise ValueError("A nonempty H×W×3 BGR screenshot is required")


def detect_content_rect(image: np.ndarray, *, black_threshold: int = 8) -> Rect:
    """Remove only symmetric, uniform black bars touching opposite edges.

    An asymmetric dark sidebar, a dark scene, or an all-black loading frame does
    not justify a crop. Explicit content_rect calibration remains available for
    clients with unusual decoration. Detection never assumes a game aspect ratio.
    """
    _validate_image(image)
    height, width = image.shape[:2]
    full = Rect(0, 0, width, height)
    dark = np.max(image, axis=2) <= black_threshold
    if float(np.mean(~dark)) < .08:
        return full

    def margins(flags):
        indices = np.flatnonzero(~flags)
        if not len(indices):
            return 0, 0
        first, last = int(indices[0]), int(len(flags) - 1 - indices[-1])
        # Single-pixel dark edges are common in normal game rendering.
        if min(first, last) < 2 or max(first, last) > len(flags) * .45:
            return 0, 0
        if abs(first - last) > max(2, min(first, last) * .03):
            return 0, 0
        return first, last

    top, bottom = margins(np.all(dark, axis=1))
    left, right = margins(np.all(dark, axis=0))
    cropped = dark[top:height - bottom, left:width - right]
    if not cropped.size or float(np.mean(~cropped)) < .10:
        return full
    return Rect(left, top, width - left - right, height - top - bottom)


def normalize_frame(
    image: np.ndarray, geometry: WindowGeometry,
    target_size: tuple[int, int] = (1280, 720), *,
    detect_bars: bool = True, content_rect: Rect | None = None,
    frame_id: str | None = None, captured_at: float | None = None,
) -> NormalizedFrame:
    """Aspect-fit observed content, recording both resize rounding and padding."""
    _validate_image(image)
    height, width = image.shape[:2]
    if (width, height) != (geometry.width, geometry.height):
        raise ValueError("Image and physical client geometry do not agree")
    out_width, out_height = map(int, target_size)
    if min(out_width, out_height) < 2 or max(out_width, out_height) > 16384:
        raise ValueError("Invalid recognizer canvas size")
    crop = content_rect or (detect_content_rect(image) if detect_bars else Rect(0, 0, width, height))
    if any(int(v) != v for v in crop.as_xywh()):
        raise ValueError("Content crop must use integer pixel boundaries")
    if crop.x < 0 or crop.y < 0 or crop.right > width or crop.bottom > height:
        raise ValueError("Content crop extends beyond the screenshot")
    scale = min(out_width / crop.width, out_height / crop.height)
    fitted_w = max(1, min(out_width, round(crop.width * scale)))
    fitted_h = max(1, min(out_height, round(crop.height * scale)))
    pad_x, pad_y = (out_width - fitted_w) // 2, (out_height - fitted_h) // 2
    from PIL import Image
    cropped = image[int(crop.y):int(crop.bottom), int(crop.x):int(crop.right)]
    # Pillow resizes each channel independently, so BGR does not need swapping.
    resized = np.asarray(Image.fromarray(cropped).resize((fitted_w, fitted_h), Image.Resampling.LANCZOS))
    canvas = np.zeros((out_height, out_width, 3), dtype=np.uint8)
    canvas[pad_y:pad_y + fitted_h, pad_x:pad_x + fitted_w] = resized
    transform = ViewportTransform(
        (width, height), (out_width, out_height), crop,
        Rect(pad_x, pad_y, fitted_w, fitted_h),
        (geometry.client_rect.x, geometry.client_rect.y),
    )
    return NormalizedFrame(canvas, transform, geometry, frame_id or uuid.uuid4().hex, captured_at or time.time())
