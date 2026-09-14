"""Runtime startup and preview are input-free, including minimized games."""
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from blackflow_live.runtime import GameRuntime


class LiveRuntimeTests(unittest.TestCase):
    def test_minimized_client_can_initialize_but_preview_never_restores_it(self):
        capture = MagicMock()
        capture.verified_window.return_value = SimpleNamespace(hwnd=123, minimized=True, visible=True)
        capture.capture.side_effect = RuntimeError('游戏窗口已最小化，请先恢复游戏窗口后重试识别预览')
        controller = MagicMock()
        with patch('blackflow_live.capture.WindowsGameCapture', return_value=capture), \
             patch('blackflow_live.controller.WindowsController', return_value=controller), \
             patch('blackflow_live.vision.VisionPipeline'), \
             patch('blackflow_live.policy.CurrentNeuralPolicy'):
            runtime = GameRuntime()
            capture.verified_window.assert_called_once()
            capture.current_geometry.assert_not_called()
            controller.focus.assert_not_called()
            with self.assertRaisesRegex(RuntimeError, '最小化'):
                runtime.observe()
            controller.focus.assert_not_called()
            controller.click.assert_not_called()
            runtime.close()
            capture.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
