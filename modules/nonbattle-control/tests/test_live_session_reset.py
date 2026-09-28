"""Session restart evidence boundaries; no desktop, capture or native inputs."""
import threading
import unittest
from unittest.mock import Mock

from blackflow_live.engine import LiveEngine


class SessionResetTests(unittest.TestCase):
    def cached_engine(self, factory):
        engine = LiveEngine(factory)
        engine._preview = b'previous-session-jpeg'
        engine._state.update(state='stopped', clicks=7,
                             observation={'frame_id': 'previous-session'},
                             window={'hwnd': 123}, decision={'action': {'label': 'old target'}})
        return engine

    def test_new_run_has_no_old_evidence_while_loading_or_after_failure(self):
        entered, release = threading.Event(), threading.Event()

        def loading(hwnd):
            entered.set()
            if not release.wait(3):
                raise RuntimeError('test synchronization timeout')
            raise RuntimeError('new game window unavailable')

        engine = self.cached_engine(loading)
        try:
            engine.start()
            self.assertTrue(entered.wait(2))
            status = engine.status()
            self.assertEqual(status['state'], 'starting')
            self.assertEqual(status['clicks'], 0)
            self.assertFalse(status['has_preview'])
            self.assertIsNone(engine.preview())
            for field in ('observation', 'window', 'decision'):
                self.assertNotIn(field, status)
            release.set()
            engine._thread.join(2)
            self.assertFalse(engine._thread.is_alive())
            status = engine.status()
            self.assertEqual(status['state'], 'error')
            self.assertFalse(status['has_preview'])
            self.assertNotIn('observation', status)
            self.assertNotIn('decision', status)
        finally:
            release.set()
            engine.stop()
            if engine._thread:
                engine._thread.join(2)

    def test_rejected_start_does_not_destroy_active_session_evidence(self):
        engine = self.cached_engine(Mock(side_effect=AssertionError('factory must not run')))
        engine._thread = Mock(is_alive=Mock(return_value=True))
        before = engine.status()
        with self.assertRaisesRegex(ValueError, '已有接管会话'):
            engine.start()
        self.assertEqual(engine.status(), before)
        self.assertEqual(engine.preview(), b'previous-session-jpeg')

    def test_pause_resume_retains_same_session_image_and_submission_guard(self):
        engine = self.cached_engine(Mock(side_effect=AssertionError('factory must not run')))
        engine._thread = Mock(is_alive=Mock(return_value=True))
        pending = {'frame_id': 'submitted-in-this-session'}
        engine._pending_selected_submit = pending
        engine.pause()
        engine.resume()
        self.assertEqual(engine.preview(), b'previous-session-jpeg')
        self.assertEqual(engine.status()['observation']['frame_id'], 'previous-session')
        self.assertIs(engine._pending_selected_submit, pending)


if __name__ == '__main__':
    unittest.main()
