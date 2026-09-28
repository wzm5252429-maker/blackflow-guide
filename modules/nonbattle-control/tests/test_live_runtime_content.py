"""Release-compatible runtime crop geometry checks; no capture or native input."""
from types import SimpleNamespace
import unittest
import numpy as np
from blackflow_live.geometry import CapturedFrame, Rect, WindowGeometry
from blackflow_live.models import LiveObservation
from blackflow_live.runtime import GameRuntime


class RuntimeContentTests(unittest.TestCase):
    def run_case(self, size, content):
        width, height = size
        pixels = np.zeros((height, width, 3), np.uint8)
        pixels[int(content.y):int(content.bottom), int(content.x):int(content.right)] = (60, 90, 120)
        geometry = WindowGeometry(123, 45, 'fixture', Rect(-2800, -200, width, height), 192)
        raw = CapturedFrame(pixels, geometry, 'recorded', 123.)
        seen = []
        def observe(image, *, frame_id, captured_at, **kwargs):
            seen.append(getattr(image, 'image', image))
            return LiveObservation(frame_id, captured_at, 'unknown', 0.)
        runtime = GameRuntime.__new__(GameRuntime)
        runtime.capture = SimpleNamespace(capture=lambda: raw)
        runtime.vision = SimpleNamespace(observe=observe)
        obs, frame = runtime.observe()
        self.assertEqual((obs.frame_id, obs.captured_at), ('recorded', 123.))
        self.assertEqual(frame.transform.crop, content)
        self.assertEqual(seen[0].shape, (720, round(content.width*720/content.height), 3))
        self.assertIs(seen[0], frame.image)
        self.assertEqual(frame.viewport.x, 0)
        self.assertEqual(frame.viewport.y, 0)
        self.assertTrue(np.array_equal(frame.image[0, 0], (60, 90, 120)))
        expected = (-2800+content.x+content.width/2, -200+content.y+content.height/2)
        self.assertEqual(frame.transform.recognition_to_screen(frame.viewport.width/2, frame.viewport.height/2), expected)

    def test_letterbox_pillarbox_and_double_bars_keep_source_offsets(self):
        for size, content in (((1024, 768), Rect(0, 96, 1024, 576)),
                              ((3440, 1440), Rect(440, 0, 2560, 1440)),
                              ((1800, 1200), Rect(100, 150, 1600, 900))):
            with self.subTest(size=size):
                self.run_case(size, content)

    def test_unbarred_aspects_remain_native(self):
        for size in ((1280, 720), (2878, 1659), (3440, 1440), (1024, 768), (720, 1280)):
            with self.subTest(size=size):
                self.run_case(size, Rect(0, 0, *size))


if __name__ == '__main__':
    unittest.main()
