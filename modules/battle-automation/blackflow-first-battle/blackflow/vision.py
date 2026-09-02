from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import ProjectConfig


class VisionError(RuntimeError):
    pass


@dataclass(frozen=True)
class Detection:
    name: str
    matched: bool
    score: float
    center: tuple[float, float] | None = None
    bbox: tuple[int, int, int, int] | None = None


class VisionSystem:
    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self.base_width, self.base_height = config.data["base_resolution"]
        self.detectors: dict[str, dict[str, Any]] = config.data.get("detectors", {})
        self._templates: dict[str, np.ndarray] = {}

    def detect_all(self, frame: np.ndarray) -> dict[str, Detection]:
        return {name: self.detect(name, frame) for name in self.detectors}

    def detect(self, name: str, frame: np.ndarray) -> Detection:
        if name not in self.detectors:
            raise VisionError(f"Unknown detector: {name}")
        spec = self.detectors[name]
        detector_type = spec.get("type", "template")
        if detector_type == "template":
            return self._template_detect(name, spec, frame)
        if detector_type == "pixel_range":
            return self._pixel_range_detect(name, spec, frame)
        raise VisionError(f"Unsupported detector type {detector_type!r} for {name}")

    def _scaled_roi(self, roi: list[int] | None, frame: np.ndarray) -> tuple[int, int, int, int]:
        height, width = frame.shape[:2]
        if roi is None:
            return 0, 0, width, height
        x, y, w, h = roi
        sx = width / self.base_width
        sy = height / self.base_height
        x1 = max(0, round(x * sx))
        y1 = max(0, round(y * sy))
        x2 = min(width, round((x + w) * sx))
        y2 = min(height, round((y + h) * sy))
        return x1, y1, x2, y2

    def _load_template(self, name: str, spec: dict[str, Any]) -> np.ndarray:
        if name in self._templates:
            return self._templates[name]
        path = self.config.resolve(spec["template"])
        template = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if template is None:
            raise VisionError(f"Could not load template for {name}: {path}")
        self._templates[name] = template
        return template

    def _template_detect(self, name: str, spec: dict[str, Any], frame: np.ndarray) -> Detection:
        x1, y1, x2, y2 = self._scaled_roi(spec.get("roi"), frame)
        search = frame[y1:y2, x1:x2]
        template = self._load_template(name, spec)
        scale_x = frame.shape[1] / self.base_width
        scale_y = frame.shape[0] / self.base_height
        resized = cv2.resize(template, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_AREA)
        if search.shape[0] < resized.shape[0] or search.shape[1] < resized.shape[1]:
            raise VisionError(f"ROI for {name} is smaller than its template.")
        if spec.get("grayscale", True):
            search_cmp = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY)
            template_cmp = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        else:
            search_cmp = search
            template_cmp = resized
        result = cv2.matchTemplate(search_cmp, template_cmp, cv2.TM_CCOEFF_NORMED)
        _min_score, max_score, _min_loc, max_loc = cv2.minMaxLoc(result)
        threshold = float(spec.get("threshold", 0.86))
        left = x1 + max_loc[0]
        top = y1 + max_loc[1]
        width, height = resized.shape[1], resized.shape[0]
        center_frame = (left + width / 2, top + height / 2)
        center_base = (
            center_frame[0] * self.base_width / frame.shape[1],
            center_frame[1] * self.base_height / frame.shape[0],
        )
        return Detection(name, max_score >= threshold, float(max_score), center_base, (left, top, width, height))

    def _pixel_range_detect(self, name: str, spec: dict[str, Any], frame: np.ndarray) -> Detection:
        x1, y1, x2, y2 = self._scaled_roi(spec.get("roi"), frame)
        crop = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        lower = np.asarray(spec["lower_hsv"], dtype=np.uint8)
        upper = np.asarray(spec["upper_hsv"], dtype=np.uint8)
        ratio = float(np.count_nonzero(cv2.inRange(hsv, lower, upper))) / max(1, crop.shape[0] * crop.shape[1])
        threshold = float(spec.get("ratio", 0.4))
        center = ((x1 + x2) / 2 * self.base_width / frame.shape[1], (y1 + y2) / 2 * self.base_height / frame.shape[0])
        return Detection(name, ratio >= threshold, ratio, center, (x1, y1, x2 - x1, y2 - y1))


def evaluate_condition(
    condition: dict[str, Any] | None,
    detections: dict[str, Detection],
    elapsed: float,
) -> bool:
    if not condition:
        return True
    if "detector" in condition:
        name = condition["detector"]
        expected = bool(condition.get("matched", True))
        return detections[name].matched is expected
    if "elapsed_ge" in condition:
        return elapsed >= float(condition["elapsed_ge"])
    if "all" in condition:
        return all(evaluate_condition(item, detections, elapsed) for item in condition["all"])
    if "any" in condition:
        return any(evaluate_condition(item, detections, elapsed) for item in condition["any"])
    if "not" in condition:
        return not evaluate_condition(condition["not"], detections, elapsed)
    raise VisionError(f"Unsupported condition: {condition}")


def resolve_point(
    spec: list[float] | dict[str, Any],
    points: dict[str, list[float]],
    detections: dict[str, Detection],
) -> list[float]:
    if isinstance(spec, list) and len(spec) == 2:
        return [float(spec[0]), float(spec[1])]
    if isinstance(spec, dict) and "point" in spec:
        return [float(v) for v in points[spec["point"]]]
    if isinstance(spec, dict) and "detector" in spec:
        detection = detections[spec["detector"]]
        if not detection.matched or detection.center is None:
            raise VisionError(f"Detector point is not currently available: {spec['detector']}")
        offset = spec.get("offset", [0, 0])
        return [detection.center[0] + float(offset[0]), detection.center[1] + float(offset[1])]
    raise VisionError(f"Invalid point specification: {spec}")

