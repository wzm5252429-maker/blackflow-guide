"""Pause/resume recovery under slow fake capture, preview and inference calls."""
from dataclasses import replace
import threading
import time
from types import SimpleNamespace
import unittest

from blackflow_live.engine import LiveEngine
from blackflow_live.models import LiveObservation, ObservedAction, PolicyDecision


ACTION = ObservedAction('continue', '继续', 'continue', (100, 100, 180, 60), .99)


class GatedRuntime:
    """The only input sink is a Python list. Gates make lifecycle races explicit."""
    def __init__(self, gates=(), failures=(), make_observation=None):
        self.counts = {}
        self.entered = {key: threading.Event() for key in gates}
        self.release = {key: threading.Event() for key in gates}
        self.failures = set(failures)
        self.trace = []
        self.clicks = []
        self.decisions = []
        self.focus_count = 0
        self.closed = False
        self.make_observation = make_observation
        self.policy = SimpleNamespace(select=self.select)

    def call(self, method, payload=None):
        self.counts[method] = self.counts.get(method, 0) + 1
        key = method, self.counts[method]
        self.trace.append((method, key[1], payload))
        if key in self.entered:
            self.entered[key].set()
            if not self.release[key].wait(3):
                raise RuntimeError('Fake gate timed out')
        if key in self.failures:
            raise RuntimeError('old execution failure: ' + method)
        return key[1]

    def observe(self):
        number = self.counts.get('observe', 0) + 1
        obs = LiveObservation(f'frame-{number}', time.time(), 'event', .99, actions=(ACTION,))
        if self.make_observation:
            obs = self.make_observation(number, obs)
        focus_count = self.focus_count
        self.call('observe', obs.frame_id)
        return obs, SimpleNamespace(frame_id=obs.frame_id, captured_at=obs.captured_at,
                                    focus_count=focus_count)

    def select(self, obs):
        self.call('policy', obs.frame_id)
        self.decisions.append(obs.frame_id)
        return PolicyDecision(obs.actions[0] if obs.actions else None, 'fake', neural=True)

    def preview(self, frame):
        self.call('preview', frame.frame_id)
        return frame.frame_id.encode()

    def window_info(self, frame):
        self.call('window_info', frame.frame_id)
        return {'hwnd': 123, 'frame_id': frame.frame_id}

    def focus(self):
        self.focus_count += 1
        self.trace.append(('focus', self.focus_count, None))

    def emergency_stop(self):
        return False

    def click(self, action, frame):
        self.clicks.append((action, frame))
        self.trace.append(('click', len(self.clicks), frame.frame_id))

    def close(self):
        self.closed = True

    def release_all(self):
        for event in self.release.values():
            event.set()


class ResumeRecoveryTests(unittest.TestCase):
    def engine(self, runtime, **kwargs):
        engine = LiveEngine(lambda hwnd: runtime, interval=.001, lease_seconds=60, **kwargs)
        def cleanup():
            engine.stop()
            runtime.release_all()
            if engine._thread:
                engine._thread.join(3)
            self.assertFalse(engine._thread and engine._thread.is_alive())
        self.addCleanup(cleanup)
        return engine

    def wait_until(self, predicate):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(.002)
        self.fail('Fake execution did not reach expected state')

    def expire_duplicate_guard(self, engine):
        with engine._input_lock:
            engine._last_click_at = time.monotonic() - 19
        self.wait_until(lambda: engine.status()['state'] == 'paused')

    def test_timeout_resume_redecides_once_with_two_new_frames_then_rearms_guard(self):
        runtime = GatedRuntime()
        engine = self.engine(runtime)
        engine.start()
        self.wait_until(lambda: len(runtime.clicks) == 1)
        self.expire_duplicate_guard(engine)
        self.assertEqual(runtime.counts['policy'], 1)
        before_resume = runtime.counts['observe']
        after_retry = ('observe', before_resume + 3)
        runtime.entered[after_retry] = threading.Event()
        runtime.release[after_retry] = threading.Event()
        engine.resume()
        self.assertTrue(runtime.entered[after_retry].wait(3))
        self.assertEqual(len(runtime.clicks), 2)
        self.assertEqual(runtime.counts['policy'], 2)
        self.assertGreater(int(runtime.decisions[1].split('-')[1]), before_resume)
        self.assertNotEqual(runtime.decisions[1], runtime.clicks[1][1].frame_id)
        self.assertEqual(runtime.clicks[1][1].focus_count, 2)
        # Advance only the duplicate deadline while its next observation is
        # gated. No second explicit resume is available to authorize a retry.
        with engine._input_lock:
            engine._last_click_at = time.monotonic() - 19
        runtime.release[after_retry].set()
        self.wait_until(lambda: engine.status()['state'] == 'paused' and engine._paused)
        self.assertEqual(len(runtime.clicks), 2)
        self.assertEqual(runtime.counts['policy'], 2)

    def test_regular_pause_resume_refocuses_before_fresh_decision_and_confirmation(self):
        runtime = GatedRuntime(gates=(('observe', 3), ('observe', 5)))
        engine = self.engine(runtime)
        engine.start()
        self.assertTrue(runtime.entered[('observe', 3)].wait(3))
        self.assertEqual(len(runtime.clicks), 1)
        engine.pause()
        engine.resume()
        runtime.release[('observe', 3)].set()
        self.assertTrue(runtime.entered[('observe', 5)].wait(3))
        self.assertEqual(runtime.decisions, ['frame-1', 'frame-4'])
        self.assertEqual(runtime.focus_count, 2)
        self.assertEqual(len(runtime.clicks), 1)
        runtime.release[('observe', 5)].set()
        self.wait_until(lambda: len(runtime.clicks) == 2)
        self.assertEqual(runtime.clicks[1][1].frame_id, 'frame-5')
        self.assertEqual(runtime.clicks[1][1].focus_count, 2)

    def test_initial_observation_across_pause_resume_never_reaches_policy(self):
        runtime = GatedRuntime(gates=(('observe', 1), ('observe', 3)))
        engine = self.engine(runtime)
        engine.start()
        self.assertTrue(runtime.entered[('observe', 1)].wait(3))
        engine.pause()
        engine.resume()
        runtime.release[('observe', 1)].set()
        self.assertTrue(runtime.entered[('observe', 3)].wait(3))
        self.assertEqual(runtime.decisions, ['frame-2'])
        self.assertEqual(runtime.focus_count, 2)
        self.assertFalse(runtime.clicks)
        runtime.release[('observe', 3)].set()
        self.wait_until(lambda: len(runtime.clicks) == 1)
        self.assertEqual(runtime.clicks[0][1].frame_id, 'frame-3')

    def test_inflight_confirmation_cannot_be_used_after_pause_resume(self):
        runtime = GatedRuntime(gates=(('observe', 2), ('observe', 4)))
        engine = self.engine(runtime)
        engine.start()
        self.assertTrue(runtime.entered[('observe', 2)].wait(3))
        engine.pause()
        engine.resume()
        runtime.release[('observe', 2)].set()
        self.assertTrue(runtime.entered[('observe', 4)].wait(3))
        self.assertEqual(runtime.decisions, ['frame-1', 'frame-3'])
        self.assertFalse(runtime.clicks)
        runtime.release[('observe', 4)].set()
        self.wait_until(lambda: len(runtime.clicks) == 1)
        self.assertEqual(runtime.clicks[0][1].frame_id, 'frame-4')
        self.assertEqual(runtime.clicks[0][1].focus_count, 2)

    def test_inflight_policy_cannot_reuse_its_decision_after_pause_resume(self):
        runtime = GatedRuntime(gates=(('policy', 1), ('observe', 3)))
        engine = self.engine(runtime)
        engine.start()
        self.assertTrue(runtime.entered[('policy', 1)].wait(3))
        engine.pause()
        engine.resume()
        runtime.release[('policy', 1)].set()
        self.assertTrue(runtime.entered[('observe', 3)].wait(3))
        self.assertEqual(runtime.decisions, ['frame-1', 'frame-2'])
        self.assertEqual(runtime.focus_count, 2)
        self.assertFalse(runtime.clicks)
        runtime.release[('observe', 3)].set()
        self.wait_until(lambda: len(runtime.clicks) == 1)
        self.assertEqual(runtime.clicks[0][1].frame_id, 'frame-3')

    def test_slow_preview_or_window_info_cannot_publish_old_observation(self):
        for method in ('preview', 'window_info'):
            with self.subTest(method=method):
                runtime = GatedRuntime(gates=((method, 1), ('observe', 2)))
                engine = self.engine(runtime)
                engine.start()
                self.assertTrue(runtime.entered[(method, 1)].wait(3))
                engine.pause()
                engine.resume()
                runtime.release[(method, 1)].set()
                self.assertTrue(runtime.entered[('observe', 2)].wait(3))
                state = engine.status()
                self.assertNotIn('observation', state)
                self.assertNotIn('window', state)
                self.assertIsNone(engine.preview())
                self.assertFalse(runtime.decisions)
                self.assertFalse(runtime.clicks)
                self.assertEqual(runtime.focus_count, 2)
                engine.stop()
                runtime.release_all()
                engine._thread.join(3)

    def test_old_capture_preview_or_policy_exception_does_not_end_resumed_session(self):
        for method in ('observe', 'preview', 'window_info', 'policy'):
            with self.subTest(method=method):
                runtime = GatedRuntime(gates=((method, 1), ('observe', 2)), failures=((method, 1),))
                engine = self.engine(runtime)
                engine.start()
                self.assertTrue(runtime.entered[(method, 1)].wait(3))
                engine.pause()
                engine.resume()
                runtime.release[(method, 1)].set()
                self.assertTrue(runtime.entered[('observe', 2)].wait(3))
                self.assertEqual(engine.status()['state'], 'running')
                self.assertNotIn('error', [event['state'] for event in engine.status()['events']])
                self.assertFalse(runtime.clicks)
                self.assertEqual(runtime.focus_count, 2)
                runtime.release[('observe', 2)].set()
                self.wait_until(lambda: len(runtime.clicks) == 1)
                engine.stop()
                engine._thread.join(3)

    def test_resume_requires_active_paused_session_and_cannot_override_stop(self):
        runtime = GatedRuntime(gates=(('observe', 1),))
        engine = self.engine(runtime)
        with self.assertRaisesRegex(ValueError, '会话已结束'):
            engine.resume()
        engine.start()
        self.assertTrue(runtime.entered[('observe', 1)].wait(3))
        with self.assertRaisesRegex(ValueError, '已暂停'):
            engine.resume()
        engine.stop()
        with self.assertRaisesRegex(ValueError, '会话已结束'):
            engine.resume()
        with self.assertRaisesRegex(ValueError, '已有接管会话'):
            engine.start()
        runtime.release_all()
        engine._thread.join(3)
        self.assertFalse(runtime.clicks)

    def test_stop_start_uses_new_runtime_and_cannot_emit_old_inflight_work(self):
        first = GatedRuntime(gates=(('observe', 2),))
        second = GatedRuntime(gates=(('observe', 2),))
        engine = self.engine(first)
        runtimes = iter((first, second))
        engine.runtime_factory = lambda hwnd: next(runtimes)
        self.addCleanup(second.release_all)
        engine.start()
        self.assertTrue(first.entered[('observe', 2)].wait(3))
        engine.stop()
        first.release_all()
        engine._thread.join(3)
        self.assertTrue(first.closed)
        self.assertFalse(first.clicks)
        engine.start()
        self.assertTrue(second.entered[('observe', 2)].wait(3))
        self.assertFalse(second.clicks)
        self.assertEqual(second.decisions, ['frame-1'])
        second.release_all()
        self.wait_until(lambda: len(second.clicks) == 1)
        self.assertEqual(engine.status()['clicks'], 1)

    def test_resume_does_not_enable_input_in_observation_only_session(self):
        runtime = GatedRuntime(gates=(('observe', 1), ('observe', 2)))
        engine = self.engine(runtime)
        engine.start(observe_only=True)
        self.assertTrue(runtime.entered[('observe', 1)].wait(3))
        engine.pause()
        engine.resume()
        runtime.release[('observe', 1)].set()
        self.assertTrue(runtime.entered[('observe', 2)].wait(3))
        self.assertEqual(runtime.focus_count, 0)
        runtime.release[('observe', 2)].set()
        self.wait_until(lambda: engine.status()['state'] == 'observing')
        self.assertFalse(runtime.clicks)
        self.assertFalse(runtime.decisions)

    def test_resumed_battle_or_stale_observations_still_never_enter_policy(self):
        for defect in ('battle', 'stale'):
            with self.subTest(defect=defect):
                def make_obs(number, obs):
                    if number == 1:
                        return obs
                    return replace(obs, scene='battle') if defect == 'battle' else replace(obs, captured_at=time.time() - 100)
                runtime = GatedRuntime(gates=(('observe', 1),), make_observation=make_obs)
                engine = self.engine(runtime)
                engine.start()
                self.assertTrue(runtime.entered[('observe', 1)].wait(3))
                engine.pause()
                engine.resume()
                runtime.release_all()
                state = 'waiting_battle' if defect == 'battle' else 'waiting_observation'
                self.wait_until(lambda: engine.status()['state'] == state)
                self.assertFalse(runtime.decisions)
                self.assertFalse(runtime.clicks)
                engine.stop()
                engine._thread.join(3)

    def test_resumed_confirmation_still_rejects_changed_stale_or_ungrounded_targets(self):
        for defect in ('battle', 'stale', 'moved', 'resources', 'action'):
            with self.subTest(defect=defect):
                def make_obs(number, obs):
                    if number != 3:
                        return obs
                    if defect == 'battle':
                        return replace(obs, scene='battle')
                    if defect == 'stale':
                        return replace(obs, captured_at=time.time() - 100)
                    if defect == 'moved':
                        return replace(obs, actions=(replace(ACTION, bbox=(200, 100, 180, 60)),))
                    if defect == 'resources':
                        return replace(obs, resources={'hp': 2})
                    return replace(obs, actions=(replace(ACTION, label='离开'),))
                runtime = GatedRuntime(gates=(('observe', 1), ('observe', 4)), make_observation=make_obs)
                engine = self.engine(runtime)
                engine.start()
                self.assertTrue(runtime.entered[('observe', 1)].wait(3))
                engine.pause()
                engine.resume()
                runtime.release[('observe', 1)].set()
                self.assertTrue(runtime.entered[('observe', 4)].wait(3))
                self.assertEqual(runtime.decisions, ['frame-2'])
                self.assertEqual(engine.status()['state'], 'waiting_observation')
                self.assertFalse(runtime.clicks)
                engine.stop()
                runtime.release_all()
                engine._thread.join(3)


if __name__ == '__main__':
    unittest.main()
