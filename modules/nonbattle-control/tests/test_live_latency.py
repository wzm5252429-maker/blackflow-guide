"""Slow recognition diagnostics and final deadline checks, with no native input."""
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from blackflow_live.engine import LiveEngine, observation_key
from blackflow_live.models import LiveObservation, ObservedAction, PolicyDecision
from blackflow_live.runtime import GameRuntime


class _AdvancingLock:
    def __init__(self, enter):
        self.enter = enter

    def __enter__(self):
        self.enter()

    def __exit__(self, *_):
        return False


class LiveLatencyTests(unittest.TestCase):
    def test_runtime_reports_stage_timings_without_resetting_capture_time(self):
        image = np.zeros((100, 200, 3), np.uint8)
        frame = SimpleNamespace(image=image, frame_id='captured-frame', captured_at=100.0)
        raw = SimpleNamespace(image=image, normalize=Mock(return_value=frame))
        original = LiveObservation('captured-frame', 100.0, 'unknown', 0.0,
                                   metadata={'perception': 'test-pipeline'})
        runtime = object.__new__(GameRuntime)
        runtime.capture = SimpleNamespace(capture=Mock(return_value=raw))
        runtime.vision = SimpleNamespace(observe=Mock(return_value=original))
        with patch('blackflow_live.runtime.time.perf_counter', side_effect=[10, 10.1, 10.3, 11.5]):
            result, returned_frame = runtime.observe()
        self.assertIs(returned_frame, frame)
        self.assertEqual(result.captured_at, 100.0)
        self.assertEqual(result.frame_id, 'captured-frame')
        self.assertEqual(result.metadata['timing_ms'], {
            'capture': 100.0, 'normalize': 200.0, 'perception': 1200.0, 'total': 1500.0,
        })
        self.assertEqual(original.metadata, {'perception': 'test-pipeline'})
        runtime.vision.observe.assert_called_once_with(image, frame_id='captured-frame', captured_at=100.0)

    def test_performance_diagnostics_do_not_change_policy_state(self):
        first = LiveObservation('first', 100, 'map', .99, resources={'hp': 4},
                                metadata={'timing_ms': {'total': 1200}})
        second = replace(first, frame_id='second', captured_at=101,
                         metadata={'timing_ms': {'total': 11000}})
        self.assertEqual(observation_key(first), observation_key(second))

    def _step_with_elapsed_time(self, *, lock_delay=0, emergency_delay=0):
        clock = [100.0]
        action = ObservedAction('continue', '继续', 'continue', (10, 20, 100, 30), .99)
        initial = LiveObservation('initial', 100, 'event', .99, actions=(action,))
        fresh = replace(initial, frame_id='confirmation')
        frame = SimpleNamespace(frame_id='confirmation', captured_at=100)

        def emergency_stop():
            clock[0] += emergency_delay
            return False

        runtime = SimpleNamespace(
            policy=SimpleNamespace(select=lambda _: PolicyDecision(action, 'test')),
            observe=lambda: (fresh, frame), preview=lambda _: b'preview',
            window_info=lambda _: {}, emergency_stop=emergency_stop, click=Mock(),
        )
        engine = LiveEngine(lambda _: runtime)
        engine._runtime = runtime
        engine._input_lock = _AdvancingLock(lambda: clock.__setitem__(0, clock[0] + lock_delay))
        with patch('blackflow_live.engine.time.time', side_effect=lambda: clock[0]):
            engine._step(initial, object())
        return engine, runtime

    def test_expiry_while_waiting_for_input_lock_blocks_click(self):
        engine, runtime = self._step_with_elapsed_time(lock_delay=16)
        runtime.click.assert_not_called()
        self.assertEqual(engine.status()['state'], 'waiting_observation')
        self.assertIn('点击前截图已过期', engine.status()['message'])

    def test_expiry_during_emergency_check_blocks_click(self):
        engine, runtime = self._step_with_elapsed_time(emergency_delay=16)
        runtime.click.assert_not_called()
        self.assertEqual(engine.status()['state'], 'waiting_observation')

    def test_still_fresh_confirmation_can_click(self):
        engine, runtime = self._step_with_elapsed_time(lock_delay=4, emergency_delay=2)
        runtime.click.assert_called_once()
        self.assertEqual(engine.status()['clicks'], 1)

    def test_expiry_message_identifies_slow_recognition(self):
        engine = LiveEngine(lambda _: None)
        obs = LiveObservation('old', 100, 'map', .99, metadata={'timing_ms': {'total': 17500}})
        self.assertIn('17.5 秒', engine._expired_message(obs))
        self.assertIn('截图已过期', engine._expired_message(obs))
        self.assertEqual(engine._expired_message(replace(obs, metadata={})), '截图已过期，正在重新获取')


if __name__ == '__main__':
    unittest.main()
