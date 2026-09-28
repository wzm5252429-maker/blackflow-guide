import ctypes as ct
import unittest
from unittest.mock import patch

import numpy as np

from blackflow_live.geometry import (
    CapturedFrame, Rect, WindowGeometry, detect_content_rect, normalize_frame,
)
from blackflow_live.capture import WindowsGameCapture, assert_geometry_current, mat_to_bgr


def geometry(width, height, x=-1400, y=120, dpi=144):
    return WindowGeometry(123, 456, "明日方舟", Rect(x, y, width, height), dpi)


class LiveGeometryTests(unittest.TestCase):
    def test_arbitrary_resolution_content_and_click_roundtrip(self):
        for width, height in ((1920, 1080), (2560, 1440), (2560, 1600), (3440, 1440),
                              (1024, 768), (1600, 900), (899, 517), (720, 1280)):
            with self.subTest(size=(width, height)):
                pixels = np.full((height, width, 3), 70, np.uint8)
                frame = normalize_frame(pixels, geometry(width, height))
                self.assertEqual(frame.image.shape, (720, 1280, 3))
                self.assertEqual(frame.transform.crop, Rect(0, 0, width, height))
                for cx, cy in ((0, 0), (width / 2, height / 2), (width - 1, height - 1)):
                    rx, ry = frame.transform.client_to_recognition(cx, cy)
                    actual = frame.transform.recognition_to_client(rx, ry)
                    self.assertAlmostEqual(actual[0], cx, places=8)
                    self.assertAlmostEqual(actual[1], cy, places=8)
                    sx, sy = frame.transform.to_desktop(rx, ry)
                    self.assertAlmostEqual(sx, cx - 1400, places=8)
                    self.assertAlmostEqual(sy, cy + 120, places=8)
                # Integer resize rounding is recorded, not hidden in one scale.
                ratio = frame.viewport.width / frame.viewport.height
                self.assertLess(abs(ratio - width / height), .01)

    def test_symmetric_letterbox_removed_and_offsets_preserved(self):
        image = np.zeros((1200, 1920, 3), np.uint8)
        image[60:1140] = (60, 110, 160)
        normalized = normalize_frame(image, geometry(1920, 1200, 10, 20, 192))
        self.assertEqual(normalized.transform.crop, Rect(0, 60, 1920, 1080))
        self.assertEqual(normalized.viewport, Rect(0, 0, 1280, 720))
        self.assertEqual(normalized.transform.to_desktop(640, 360), (970, 620))

    def test_pillarbox_removed(self):
        image = np.zeros((720, 1600, 3), np.uint8)
        image[:, 160:1440] = 100
        self.assertEqual(detect_content_rect(image), Rect(160, 0, 1280, 720))

    def test_dark_scene_and_asymmetric_sidebar_not_cropped(self):
        dark = np.zeros((800, 1000, 3), np.uint8)
        dark[390:410, 490:510] = 100
        self.assertEqual(detect_content_rect(dark), Rect(0, 0, 1000, 800))
        sidebar = np.full((800, 1000, 3), 100, np.uint8)
        sidebar[:, :120] = 0
        self.assertEqual(detect_content_rect(sidebar), Rect(0, 0, 1000, 800))

    def test_padding_can_never_be_used_as_game_input(self):
        image = np.full((1440, 3440, 3), 70, np.uint8)
        frame = normalize_frame(image, geometry(3440, 1440))
        with self.assertRaisesRegex(ValueError, "padding"):
            frame.transform.to_desktop(640, 0)
        with self.assertRaises(ValueError):
            frame.transform.to_desktop(float("nan"), 360)
        with self.assertRaises(ValueError):
            frame.transform.to_desktop(1280, 360)

    def test_explicit_content_calibration_is_exact(self):
        image = np.full((900, 1600, 3), 70, np.uint8)
        frame = normalize_frame(image, geometry(1600, 900), content_rect=Rect(40, 10, 1520, 880))
        self.assertEqual(frame.transform.crop, Rect(40, 10, 1520, 880))
        with self.assertRaises(ValueError):
            normalize_frame(image, geometry(1600, 900), content_rect=Rect(50, 0, 1600, 900))

    def test_frame_metadata_survives_normalization(self):
        frame = CapturedFrame(np.full((900, 1600, 3), 70, np.uint8), geometry(1600, 900))
        normalized = frame.normalize()
        self.assertEqual(normalized.frame_id, frame.frame_id)
        self.assertEqual(normalized.captured_at, frame.captured_at)
        self.assertEqual(normalized.geometry, frame.geometry)
        with self.assertRaises(ValueError):
            CapturedFrame(frame.image, geometry(1599, 900))

    def test_window_move_resize_dpi_or_process_change_invalidates_input(self):
        before = geometry(1920, 1080)
        changed = [geometry(1920, 1080, x=1), geometry(1919, 1080), geometry(1920, 1080, dpi=96),
                   WindowGeometry(123, 999, "明日方舟", before.client_rect, 144)]
        for current in changed:
            with patch("blackflow_live.capture.window_geometry", return_value=current):
                with self.assertRaisesRegex(RuntimeError, "recapture"):
                    assert_geometry_current(before)
        with patch("blackflow_live.capture.window_geometry", return_value=before):
            self.assertEqual(assert_geometry_current(before), before)

    def test_resize_in_flight_discards_frame_instead_of_stretching(self):
        old, new = geometry(1280, 720), geometry(1600, 1000)
        class Native:
            def capture(self):
                return np.zeros((1000, 1600, 3), np.uint8)
            def close(self):
                pass
        from contextlib import nullcontext
        with patch("blackflow_live.capture.resolve_runtime", return_value="test"), \
             patch("blackflow_live.capture.physical_pixel_context", side_effect=lambda: nullcontext()), \
             patch("blackflow_live.capture.find_game_windows", return_value=[new]), \
             patch("blackflow_live.capture.MaaWindowCapture", side_effect=lambda *_: Native()):
            capture = WindowsGameCapture(hwnd=123)
            with patch.object(capture, "current_geometry", side_effect=[old, new, new, new]):
                frame = capture.capture()
            self.assertEqual(frame.geometry, new)

    def test_native_mat_stride_is_copied_and_checked(self):
        # Exercise ABI byte copying without a DLL or any GUI input.
        storage = ct.create_string_buffer(128)
        address = ct.addressof(storage)
        for offset, value in ((0, 16), (4, 2), (8, 2), (12, 2)):
            ct.c_int.from_address(address + offset).value = value
        data = (ct.c_ubyte * 16)(1, 2, 3, 4, 5, 6, 99, 99, 7, 8, 9, 10, 11, 12, 99, 99)
        stride = ct.c_size_t(8)
        ct.c_void_p.from_address(address + 16).value = ct.addressof(data)
        ct.c_void_p.from_address(address + 72).value = ct.addressof(stride)
        output = mat_to_bgr(address)
        self.assertEqual(output.tolist(), [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
        data[0] = 88
        self.assertEqual(int(output[0, 0, 0]), 1)
        ct.c_int.from_address(address + 4).value = 3
        with self.assertRaisesRegex(RuntimeError, "ABI"):
            mat_to_bgr(address)


if __name__ == "__main__":
    unittest.main()
