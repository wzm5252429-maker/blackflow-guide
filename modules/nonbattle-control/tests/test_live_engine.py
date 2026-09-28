"""Closed-loop safety regressions, using fake windows and zero native input."""
from dataclasses import replace
from itertools import count
import threading
import time
from types import SimpleNamespace
import unittest

from blackflow_live.engine import LiveEngine, action_is_grounded, observation_key
from blackflow_live.models import LiveObservation, ObservedAction, PolicyDecision


_ids = count()
ACTION = ObservedAction("continue", "继续", "continue", (100, 100, 180, 60), .99)


def observation(**kwargs):
    return replace(LiveObservation(
        f"frame-{next(_ids)}", time.time(), "event", .99, actions=(ACTION,),
    ), **kwargs)


class FakeRuntime:
    def __init__(self, observe=None, *, click_error=None):
        self.next_observation = observe or observation
        self.policy = SimpleNamespace(select=lambda obs: PolicyDecision(obs.actions[0] if obs.actions else None, "fake", neural=True))
        self.clicks = []
        self.focus_count = 0
        self.observe_count = 0
        self.closed = False
        self.click_error = click_error

    def observe(self):
        self.observe_count += 1
        obs = self.next_observation()
        frame = SimpleNamespace(frame_id=obs.frame_id, captured_at=obs.captured_at)
        return obs, frame

    def preview(self, frame):
        return b"image"

    def window_info(self, frame):
        return {"hwnd": 123, "dpi": 192}

    def focus(self):
        self.focus_count += 1

    def emergency_stop(self):
        return False

    def click(self, action, frame):
        if self.click_error:
            raise RuntimeError(self.click_error)
        self.clicks.append((action, frame))

    def close(self):
        self.closed = True


class LiveEngineTests(unittest.TestCase):
    def make_engine(self, runtime, **kwargs):
        engine = LiveEngine(lambda hwnd: runtime, interval=.001, **kwargs)
        self.addCleanup(engine.stop)
        return engine

    def wait_until(self, predicate, timeout=2):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(.005)
        self.fail("Fake runtime did not reach expected state")

    def direct_engine(self, runtime):
        engine = self.make_engine(runtime)
        engine._runtime = runtime
        return engine

    def test_battle_scenes_and_start_labels_never_ground_input(self):
        for scene in ("battle", "battle_start", "combat", "squad", "battle_prepare"):
            self.assertFalse(action_is_grounded(ACTION, observation(scene=scene)))
        for label in ("开始战斗", "开始行动", "开始作战", "进入战斗", "START OPERATION", "开始 战斗", "开始\n行动", "Start  Operation"):
            action = replace(ACTION, label=label)
            with self.subTest(label=label):
                self.assertFalse(action_is_grounded(action, observation(actions=(action,))))
        for kind in ("battle", "start_battle", "battle_start", "combat"):
            action = replace(ACTION, kind=kind)
            self.assertFalse(action_is_grounded(action, observation(actions=(action,))))

    def test_confidence_membership_and_invalid_coordinates_fail_closed(self):
        for action in (replace(ACTION, confidence=.84), replace(ACTION, confidence=float("nan")),
                       replace(ACTION, enabled=False), replace(ACTION, bbox=(-1, 0, 30, 30)),
                       replace(ACTION, bbox=(0, 0, float("inf"), 30)),
                       replace(ACTION, bbox=(0, 0, 0, 30))):
            self.assertFalse(action_is_grounded(action, observation(actions=(action,))))
        self.assertFalse(action_is_grounded(ACTION, observation(actions=())))
        self.assertFalse(action_is_grounded(ACTION, observation(confidence=.84)))

    def test_one_action_requires_second_fresh_observation(self):
        runtime = FakeRuntime()
        engine = self.direct_engine(runtime)
        first = observation()
        engine._step(first, SimpleNamespace(frame_id=first.frame_id))
        self.assertEqual(runtime.observe_count, 1)
        self.assertEqual(len(runtime.clicks), 1)
        self.assertNotEqual(runtime.clicks[0][1].frame_id, first.frame_id)

    def test_changed_resources_or_target_position_causes_new_decision(self):
        for fresh in (observation(resources={"hp": 1}),
                      observation(actions=(replace(ACTION, bbox=(200, 100, 180, 60)),))):
            runtime = FakeRuntime(lambda: fresh)
            engine = self.direct_engine(runtime)
            engine._step(observation(), object())
            self.assertFalse(runtime.clicks)
            self.assertEqual(engine.status()['state'], 'waiting_observation')

    def test_edge_change_alters_decision_state_key(self):
        first = observation(edges=(("a", "b"),))
        changed = replace(first, edges=(("a", "c"),))
        self.assertNotEqual(observation_key(first), observation_key(changed))

    def test_action_metadata_change_alters_decision_state_key(self):
        first = observation(actions=(replace(ACTION, metadata={"price": 3}),))
        changed = replace(first, actions=(replace(ACTION, metadata={"price": 6}),))
        self.assertNotEqual(observation_key(first), observation_key(changed))

    def test_frame_provenance_and_small_action_id_jitter_allow_one_click_only(self):
        def observed_version(version):
            frame_id = f'stable-scene-frame-{version}'
            action = replace(ACTION, action_id=f'ocr:continue-{100 + version}',
                             bbox=(100 + version, 100, 180, 60),
                             metadata={'source_frame_id': frame_id, 'captured_at': time.time(),
                                       'operation': 'event_advance', 'resource_delta': {'gold': 3}})
            return observation(frame_id=frame_id, actions=(action,))
        first, confirmation, after = (observed_version(i) for i in range(3))
        self.assertEqual(observation_key(first), observation_key(confirmation))
        runtime = FakeRuntime(lambda: confirmation)
        engine = self.direct_engine(runtime)
        engine._step(first, object())
        self.assertEqual(len(runtime.clicks), 1)
        self.assertEqual(runtime.clicks[0][0], confirmation.actions[0])
        engine._step(after, object())
        self.assertEqual(len(runtime.clicks), 1)
        self.assertEqual(runtime.observe_count, 1)
        self.assertIn('等待', engine.status()['message'])

    def test_semantic_action_fallback_rejects_ambiguous_targets(self):
        first = observation(actions=(replace(ACTION, action_id='old-coordinate'),))
        fresh = observation(actions=(replace(ACTION, action_id='new-left'),
                                     replace(ACTION, action_id='new-right', bbox=(800, 100, 180, 60))))
        runtime = FakeRuntime(lambda: fresh)
        engine = self.direct_engine(runtime)
        engine._step(first, object())
        self.assertFalse(runtime.clicks)

    def test_jitter_fallback_still_replans_changed_costs_resources_and_uses(self):
        baseline = {'operation': 'purchase', 'price': 3, 'uses_remaining': 2,
                    'resource_delta': {'gold': -3}, 'source_frame_id': 'initial'}
        first = observation(frame_id='initial', actions=(replace(ACTION, action_id='original', metadata=baseline),),
                            resources={'gold': 12, 'hp': 8})
        for changes, resources in (({'price': 7}, first.resources),
                                   ({'uses_remaining': 1}, first.resources),
                                   ({'resource_delta': {'gold': -6}}, first.resources),
                                   ({}, {'gold': 4, 'hp': 8})):
            metadata = {**baseline, **changes, 'source_frame_id': 'confirmation', 'captured_at': time.time()}
            fresh = observation(frame_id='confirmation',
                                actions=(replace(ACTION, action_id='new-id', bbox=(102, 100, 180, 60), metadata=metadata),),
                                resources=resources)
            runtime = FakeRuntime(lambda: fresh)
            engine = self.direct_engine(runtime)
            engine._step(first, object())
            self.assertFalse(runtime.clicks)
            self.assertEqual(engine.status()['state'], 'waiting_observation')

    def test_recursive_provenance_and_diagnostic_jitter_do_not_change_state_key(self):
        first = observation(frame_id='first', metadata={
            'items': [{'item_id': 'vehicle', 'uses_remaining': 2, 'source_frame_id': 'first', 'captured_at': 10}],
            'shop_state': {'shelf_complete': True, 'source_frame_id': 'first'},
            'ocr': [{'text': '继续', 'bbox': [100, 100, 180, 60], 'confidence': .99}],
            'markers': {'continue': {'confidence': .96}},
        }, actions=(replace(ACTION, metadata={'nested': {'source_frame_id': 'first', 'captured_at': 10, 'hope_cost': 1}}),))
        fresh = observation(frame_id='fresh', metadata={
            'items': [{'item_id': 'vehicle', 'uses_remaining': 2, 'source_frame_id': 'fresh', 'captured_at': 11}],
            'shop_state': {'shelf_complete': True, 'source_frame_id': 'fresh'},
            'ocr': [{'text': '继续', 'bbox': [102, 100, 180, 60], 'confidence': .98}],
            'markers': {'continue': {'confidence': .97}},
        }, actions=(replace(ACTION, metadata={'nested': {'source_frame_id': 'fresh', 'captured_at': 11, 'hope_cost': 1}}),))
        self.assertEqual(observation_key(first), observation_key(fresh))

    def test_observed_inventory_shop_and_operator_changes_require_replan(self):
        for field, before, after in (
            ('items', [{'item_id': 'vehicle', 'uses_remaining': 2}], [{'item_id': 'vehicle', 'uses_remaining': 1}]),
            ('inventory', ['relic_a'], ['relic_b']),
            ('shop_state', {'refreshes': 1, 'shelf_complete': True}, {'refreshes': 2, 'shelf_complete': True}),
            ('formal_operator_ids', ['operator_a'], ['operator_a', 'operator_b']),
            ('available_operator_ids', ['operator_a'], []),
            ('promoted_operator_ids', [], ['operator_a']),
            ('pending_recruit_ticket_ids', ['ticket_a'], []),
            ('stored_recruit_ticket_ids', [], ['ticket_a']),
            ('temporary_recruit_offers', ['operator_a'], ['operator_b']),
            ('pending_node_id', 'a', 'b'),
        ):
            with self.subTest(field=field):
                initial = observation(metadata={field: before})
                current = observation(metadata={field: after})
                runtime = FakeRuntime(lambda: current)
                engine = self.direct_engine(runtime)
                engine._step(initial, object())
                self.assertFalse(runtime.clicks)
                self.assertEqual(engine.status()['state'], 'waiting_observation')

    def test_provenance_from_another_frame_cannot_ground_fresh_action(self):
        action = replace(ACTION, metadata={'source_frame_id': 'old-screenshot'})
        self.assertFalse(action_is_grounded(action, observation(actions=(action,))))

    def test_stale_and_future_frames_never_click(self):
        for timestamp in (time.time() - 100, time.time() + 100):
            runtime = FakeRuntime(lambda: observation(captured_at=timestamp))
            engine = self.make_engine(runtime)
            engine.start()
            self.wait_until(lambda: engine.status()['state'] == 'waiting_observation')
            engine.stop()
            engine._thread.join(2)
            self.assertFalse(runtime.clicks)

    def test_recheck_rejects_expired_frame_and_frame_id_mismatch(self):
        for defect in ("expired", "mismatched"):
            runtime = FakeRuntime(lambda: observation(captured_at=time.time() - 100) if defect == "expired" else observation())
            if defect == "mismatched":
                def mismatch():
                    obs = observation()
                    return obs, SimpleNamespace(frame_id="another-frame", captured_at=obs.captured_at)
                runtime.observe = mismatch
            engine = self.direct_engine(runtime)
            engine._step(observation(), object())
            self.assertFalse(runtime.clicks)

    def test_pause_and_stop_during_second_observe_cannot_emit_late_input(self):
        for operation in ("pause", "stop"):
            entered, release = threading.Event(), threading.Event()
            runtime = FakeRuntime()
            def next_obs():
                if runtime.observe_count == 2:
                    entered.set()
                    release.wait(2)
                return observation()
            runtime.next_observation = next_obs
            engine = self.make_engine(runtime)
            engine.start()
            self.assertTrue(entered.wait(2))
            getattr(engine, operation)()
            release.set()
            self.wait_until(lambda: runtime.observe_count >= 2)
            engine.stop()
            engine._thread.join(2)
            self.assertFalse(runtime.clicks)

    def test_geometry_failure_ends_session_without_input(self):
        runtime = FakeRuntime(click_error="Game window moved; recapture before input")
        engine = self.make_engine(runtime)
        engine.start()
        self.wait_until(lambda: engine.status()['state'] == 'error')
        engine._thread.join(2)
        self.assertIn("moved", engine.status()['message'])
        self.assertFalse(runtime.clicks)
        self.assertTrue(runtime.closed)

    def test_duplicate_action_waits_then_pauses_without_retrying_click(self):
        runtime = FakeRuntime()
        engine = self.direct_engine(runtime)
        current = observation()
        engine._step(current, object())
        engine._step(current, object())
        self.assertEqual(len(runtime.clicks), 1)
        engine._last_click_at = time.monotonic() - 19
        engine._step(current, object())
        self.assertEqual(engine.status()['state'], 'paused')
        self.assertEqual(len(runtime.clicks), 1)

    def test_expired_browser_lease_blocks_input_even_after_inference(self):
        runtime = FakeRuntime()
        engine = self.direct_engine(runtime)
        def slow_policy(obs):
            engine._lease = time.monotonic() - engine.lease_seconds - 1
            return PolicyDecision(ACTION, 'fake')
        runtime.policy.select = slow_policy
        engine._step(observation(), object())
        self.assertFalse(runtime.clicks)

    def test_only_first_ending_double_confirmation_completes(self):
        runtime = FakeRuntime(lambda: observation(scene='ending_complete', ending_first_confirmed=True, actions=()))
        engine = self.make_engine(runtime)
        engine.start()
        self.wait_until(lambda: engine.status()['state'] == 'completed')
        engine._thread.join(2)
        self.assertEqual(runtime.observe_count, 2)
        self.assertFalse(runtime.clicks)
        self.assertTrue(runtime.closed)
        runtime2 = FakeRuntime(lambda: observation(scene='ending_complete', ending_first_confirmed=False, actions=()))
        engine2 = self.make_engine(runtime2)
        engine2.start()
        self.wait_until(lambda: runtime2.observe_count >= 2)
        self.assertNotEqual(engine2.status()['state'], 'completed')
        self.assertFalse(runtime2.clicks)

    def test_cancel_during_ending_confirmation_preserves_stopped_state(self):
        entered, release = threading.Event(), threading.Event()
        runtime = FakeRuntime()
        def ending():
            if runtime.observe_count == 2:
                entered.set()
                release.wait(2)
            return observation(scene='ending_complete', ending_first_confirmed=True, actions=())
        runtime.next_observation = ending
        engine = self.make_engine(runtime)
        engine.start()
        self.assertTrue(entered.wait(2))
        engine.stop()
        release.set()
        engine._thread.join(2)
        self.assertEqual(engine.status()['state'], 'stopped')
        self.assertFalse(runtime.clicks)

    def test_identical_frame_cannot_double_confirm_first_ending(self):
        obs = observation(scene='ending_complete', ending_first_confirmed=True, actions=())
        runtime = FakeRuntime(lambda: obs)
        engine = self.make_engine(runtime)
        engine.start()
        self.wait_until(lambda: runtime.observe_count >= 2)
        self.assertNotEqual(engine.status()['state'], 'completed')
        self.assertFalse(runtime.clicks)

    def test_observe_only_never_focuses_or_clicks(self):
        runtime = FakeRuntime()
        engine = self.make_engine(runtime)
        engine.start(observe_only=True)
        self.wait_until(lambda: engine.status()['state'] == 'observing')
        self.assertFalse(runtime.clicks)
        self.assertEqual(runtime.focus_count, 0)


if __name__ == '__main__':
    unittest.main()
