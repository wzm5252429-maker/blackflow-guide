"""Wire raw Windows frames to real OCR, frozen neural weights and guarded input."""
from __future__ import annotations

from dataclasses import replace
import time

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
        # Identify the client without requiring capturable geometry yet. An
        # explicitly started control session may restore it in focus(); preview
        # still never restores, activates or sends input to a window.
        self.capture.verified_window()
        self.controller = WindowsController(self.capture)
        self.vision = VisionPipeline(self.capture.runtime)
        self.policy = CurrentNeuralPolicy()
        self.policy.load()

    def observe(self):
        started = time.perf_counter()
        raw = self.capture.capture()
        captured = time.perf_counter()
        height, width = raw.image.shape[:2]
        # Template/OCR scale stays comparable while the full aspect ratio remains
        # visible. UI relocation is handled by image matching, not fixed 16:9 ROIs.
        frame = raw.normalize((max(64, round(width * 720 / height)), 720))
        normalized = time.perf_counter()
        obs = self.vision.observe(frame.image, frame_id=frame.frame_id, captured_at=frame.captured_at)
        recognized = time.perf_counter()
        # Diagnostics must not replace capture provenance or become policy input.
        # In particular, slow inference never makes an old screenshot fresh again.
        obs = replace(obs, metadata={**obs.metadata, 'timing_ms': {
            'capture': round((captured - started) * 1000, 1),
            'normalize': round((normalized - captured) * 1000, 1),
            'perception': round((recognized - normalized) * 1000, 1),
            'total': round((recognized - started) * 1000, 1),
        }})
        return obs, frame

    def focus(self):
        self.controller.focus()

    def emergency_stop(self):
        return self.controller.emergency_stop()

    def click(self, action, frame):
        self.controller.click(action, frame)

    def cancel_pointer_cleanup(self):
        self.controller.cancel_pointer_cleanup()

    def prepare_observation(self, observation, frame, *, still_active=None):
        if (observation.frame_id != frame.frame_id or observation.scene in {
                'battle','battle_start','battle_prepare','squad','combat',
                'ending','ending_complete','failed'}):
            self.cancel_pointer_cleanup()
            return False
        return self.controller.clear_pointer(frame,still_active=still_active)

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
        self.cancel_pointer_cleanup()
        self.capture.close()
