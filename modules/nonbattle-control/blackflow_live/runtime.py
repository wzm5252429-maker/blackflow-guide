"""Wire raw Windows frames to real OCR, frozen neural weights and guarded input."""
from __future__ import annotations

def discover_windows():
    from .capture import WindowsGameCapture
    return WindowsGameCapture.discover_windows()


class GameRuntime:
    def __init__(self, hwnd=None, runtime=None):
        from .capture import WindowsGameCapture
        from .controller import WindowsController
        from .policy import CurrentNeuralPolicy
        from .vision import VisionPipeline
        self.capture = WindowsGameCapture(runtime=runtime, hwnd=hwnd)
        self.capture.current_geometry()
        self.controller = WindowsController(self.capture)
        self.vision = VisionPipeline(self.capture.runtime)
        self.policy = CurrentNeuralPolicy()
        self.policy.load()

    def observe(self):
        raw = self.capture.capture()
        height, width = raw.image.shape[:2]
        # Template/OCR scale stays comparable while the full aspect ratio remains
        # visible. UI relocation is handled by image matching, not fixed 16:9 ROIs.
        frame = raw.normalize((max(64, round(width * 720 / height)), 720))
        obs = self.vision.observe(frame.image, frame_id=frame.frame_id, captured_at=frame.captured_at)
        return obs, frame

    def focus(self):
        self.controller.focus()

    def emergency_stop(self):
        return self.controller.emergency_stop()

    def click(self, action, frame):
        self.controller.click(action, frame)

    def preview(self, frame):
        import cv2
        ok, encoded = cv2.imencode('.jpg', frame.image, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            raise RuntimeError('无法生成游戏截图预览')
        return encoded.tobytes()

    def window_info(self, frame):
        result = frame.geometry.to_dict()
        result['recognition_size'] = list(frame.image.shape[1::-1])
        return result

    def close(self):
        self.capture.close()
