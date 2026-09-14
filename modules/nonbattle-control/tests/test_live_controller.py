"""Mock Win32 calls to verify physical coordinates without moving the mouse."""
from contextlib import contextmanager
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from blackflow_live.controller import WindowsController
from blackflow_live.geometry import CapturedFrame, Rect, WindowGeometry
from blackflow_live.models import ObservedAction


class Function:
    def __init__(self, callback):
        self.callback = callback

    def __call__(self, *args):
        return self.callback(*args)


class LiveControllerTests(unittest.TestCase):
    def setup_controller(self):
        active = [False]
        sent = []
        @contextmanager
        def context():
            self.assertFalse(active[0])
            active[0] = True
            try:
                yield
            finally:
                active[0] = False
        def checked(value):
            def call(*_):
                self.assertTrue(active[0], 'Win32 call escaped physical DPI context')
                return value
            return call
        def metric(index):
            self.assertTrue(active[0])
            return {76: -1920, 77: 0, 78: 4800, 79: 2160}[index]
        def cursor(pointer):
            self.assertTrue(active[0])
            pointer._obj.x, pointer._obj.y = -1920, 0
            return True
        def send(count, inputs, size):
            self.assertTrue(active[0])
            sent.append((count, inputs[0].mi.dx, inputs[0].mi.dy))
            return count
        user = SimpleNamespace(
            GetForegroundWindow=Function(checked(123)),
            GetAncestor=Function(checked(123)),
            IsIconic=Function(checked(False)),
            ShowWindow=Function(checked(True)),
            SetForegroundWindow=Function(checked(True)),
            GetAsyncKeyState=Function(checked(0)),
            GetCursorPos=Function(cursor),
            GetSystemMetrics=Function(metric),
            WindowFromPoint=Function(checked(123)),
            SendInput=Function(send),
        )
        geometry = WindowGeometry(123, 42, '明日方舟', Rect(-1400, 100, 1280, 720), 192)
        def assert_current(snapshot):
            self.assertTrue(active[0])
            self.assertEqual(snapshot, geometry)
        controller = WindowsController.__new__(WindowsController)
        controller.user32 = user
        controller.capture = SimpleNamespace(
            verified_window=lambda: SimpleNamespace(hwnd=123, visible=True, minimized=False),
            current_geometry=lambda: geometry, assert_geometry_current=assert_current)
        frame = CapturedFrame(np.full((720, 1280, 3), 90, np.uint8), geometry).normalize()
        return controller, frame, context, sent, active

    def test_focus_and_emergency_use_physical_dpi_context(self):
        controller, frame, context, sent, active = self.setup_controller()
        with patch('blackflow_live.controller.physical_pixel_context', context):
            controller.focus()
            self.assertTrue(controller.emergency_stop())
        self.assertFalse(active[0])
        self.assertFalse(sent)

    def test_click_uses_physical_virtual_desktop_with_negative_origin(self):
        controller, frame, context, sent, active = self.setup_controller()
        action = ObservedAction('continue', '继续', 'continue', (200, 100, 100, 60))
        with patch('blackflow_live.controller.physical_pixel_context', context):
            controller.click(action, frame)
        # Recognizer center (250,130) => physical (-1150,230), on 4800×2160 desktop.
        self.assertEqual(sent, [(3, round(770 * 65535 / 4799), round(230 * 65535 / 2159))])
        self.assertFalse(active[0])

    def test_occluded_target_never_sends_input(self):
        controller, frame, context, sent, active = self.setup_controller()
        controller.user32.WindowFromPoint = Function(lambda *_: 555)
        controller.user32.GetAncestor = Function(lambda hwnd, _: hwnd)
        action = ObservedAction('continue', '继续', 'continue', (200, 100, 100, 60))
        with patch('blackflow_live.controller.physical_pixel_context', context):
            with self.assertRaises(RuntimeError):
                controller.click(action, frame)
        self.assertFalse(sent)
        self.assertFalse(active[0])

    def test_minimized_client_restored_before_reading_geometry(self):
        controller, frame, context, sent, _ = self.setup_controller()
        events = []
        controller.capture.verified_window = lambda: SimpleNamespace(hwnd=123, visible=True, minimized=True)
        controller.user32.ShowWindow = Function(lambda hwnd, mode: events.append(('restore', hwnd, mode)))
        controller.capture.current_geometry = lambda: events.append('geometry') or frame.geometry
        controller.user32.SetForegroundWindow = Function(lambda hwnd: events.append('focus'))
        with patch('blackflow_live.controller.physical_pixel_context', context):
            controller.focus()
        self.assertEqual(events, [('restore', 123, 9), 'geometry', 'focus'])
        self.assertFalse(sent)

    def test_hidden_or_changed_client_never_activated(self):
        controller, _, context, sent, _ = self.setup_controller()
        events = []
        controller.user32.ShowWindow = Function(lambda *_: events.append('restore'))
        controller.user32.SetForegroundWindow = Function(lambda *_: events.append('focus'))
        controller.capture.verified_window = lambda: SimpleNamespace(hwnd=123, visible=False, minimized=False)
        with patch('blackflow_live.controller.physical_pixel_context', context):
            with self.assertRaisesRegex(RuntimeError, '隐藏'):
                controller.focus()
        controller.capture.verified_window = lambda: SimpleNamespace(hwnd=123, visible=True, minimized=False)
        def changed():
            raise RuntimeError('游戏进程已变化')
        controller.capture.current_geometry = changed
        with patch('blackflow_live.controller.physical_pixel_context', context):
            with self.assertRaisesRegex(RuntimeError, '变化'):
                controller.focus()
        self.assertEqual(events, [])
        self.assertFalse(sent)


if __name__ == '__main__':
    unittest.main()
