"""Deferred cursor cleanup with mocked Windows; never sends native input."""
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from blackflow_live.models import ObservedAction
from blackflow_live.runtime import GameRuntime
from tests import test_live_controller as controller_support
from tests import test_live_engine as engine_support
Function=controller_support.Function
FakeRuntime=engine_support.FakeRuntime
observation=engine_support.observation
ACTION=engine_support.ACTION


class PointerCleanupTests(unittest.TestCase):
    def setup(self):
        controller,frame,context,_,active=controller_support.LiveControllerTests().setup_controller()
        clock=[100.,1000.]
        position=[-1150,230]
        batches=[]
        def cursor(pointer):
            self.assertTrue(active[0])
            pointer._obj.x,pointer._obj.y=position
            return True
        def send(count,inputs,size):
            self.assertTrue(active[0])
            batches.append([(inputs[i].mi.dx,inputs[i].mi.dy,inputs[i].mi.dwFlags) for i in range(count)])
            return count
        controller.user32.GetCursorPos=Function(cursor)
        controller.user32.SendInput=Function(send)
        self.enterContext(patch('blackflow_live.controller.physical_pixel_context',context))
        self.enterContext(patch('blackflow_live.controller.time.monotonic',lambda:clock[0]))
        self.enterContext(patch('blackflow_live.controller.time.time',lambda:clock[1]))
        controller.click(ObservedAction('continue','继续','continue',(200,100,100,60)),frame)
        return controller,replace(frame,captured_at=1000.5),clock,position,batches

    def test_deferred_move_is_separate_from_click_and_never_contains_button_flags(self):
        controller,frame,clock,_,batches=self.setup()
        self.assertEqual([len(b) for b in batches],[3])
        self.assertFalse(controller.clear_pointer(frame))
        self.assertIsNotNone(controller._pointer_cleanup)
        clock[0]+=.5
        self.assertTrue(controller.clear_pointer(frame))
        self.assertEqual([len(b) for b in batches],[3,1])
        self.assertEqual([entry[2] for entry in batches[0]],[0xC001,2,4])
        self.assertEqual(batches[1],[(round(1799*65535/4799),round(819*65535/2159),0xC001)])
        self.assertFalse(controller.clear_pointer(frame))

    def test_user_intervention_geometry_and_expiration_cancel_without_moving(self):
        # Setup creates only fake calls; every case begins with one completed click.
        for reason in ('position','left','right','middle','x1','x2','escape','cursor_failed',
                       'foreground','geometry','occluded','old_frame','new_dpi','timeout'):
            with self.subTest(reason=reason):
                controller,frame,clock,position,batches=self.setup()
                clock[0]+=.5
                user=controller.user32
                if reason=='position':position[0]+=20
                keys={'left':1,'right':2,'middle':4,'x1':5,'x2':6,'escape':0x1B}
                if reason in keys:user.GetAsyncKeyState=Function(lambda key:0x8000 if key==keys[reason] else 0)
                if reason=='cursor_failed':user.GetCursorPos=Function(lambda *_:False)
                if reason=='foreground':user.GetAncestor=Function(lambda *_:999)
                if reason=='geometry':controller.capture.assert_geometry_current=Mock(side_effect=RuntimeError('changed'))
                if reason=='occluded':
                    user.WindowFromPoint=Function(lambda *_:999)
                    user.GetAncestor=Function(lambda hwnd,_:hwnd)
                if reason=='old_frame':frame=replace(frame,captured_at=999)
                if reason=='new_dpi':frame=replace(frame,geometry=replace(frame.geometry,dpi=144))
                if reason=='timeout':clock[0]+=20
                self.assertFalse(controller.clear_pointer(frame))
                self.assertIsNone(controller._pointer_cleanup)
                self.assertEqual([len(b) for b in batches],[3])

    def test_explicit_cancellation_is_input_free(self):
        controller,frame,clock,_,batches=self.setup()
        controller.cancel_pointer_cleanup()
        clock[0]+=.5
        self.assertFalse(controller.clear_pointer(frame))
        self.assertEqual([len(b) for b in batches],[3])

    def test_cancellation_during_final_hit_test_prevents_move(self):
        controller,frame,clock,_,batches=self.setup()
        clock[0]+=.5
        active=[True]
        count=[0]
        def hit(*_):
            count[0]+=1
            if count[0]==2:active[0]=False
            return 123
        controller.user32.WindowFromPoint=Function(hit)
        self.assertFalse(controller.clear_pointer(frame,still_active=lambda:active[0]))
        self.assertEqual([len(b) for b in batches],[3])

    def test_runtime_never_clears_pointer_from_read_only_observe_or_battle(self):
        runtime=GameRuntime.__new__(GameRuntime)
        runtime.controller=Mock()
        frame=SimpleNamespace(frame_id='frame')
        for scene in ('battle','battle_start','combat','squad','ending','ending_complete','failed'):
            self.assertFalse(runtime.prepare_observation(observation(scene=scene,frame_id='frame'),frame))
        runtime.controller.clear_pointer.assert_not_called()
        runtime.capture=Mock()
        runtime.capture.capture.side_effect=RuntimeError('no capture')
        with self.assertRaises(RuntimeError):runtime.observe()
        runtime.controller.clear_pointer.assert_not_called()


class PointerCleanupEngineTests(unittest.TestCase):
    def engine(self):
        runtime=FakeRuntime()
        runtime.prepare_observation=Mock(return_value=True)
        runtime.cancel_pointer_cleanup=Mock()
        engine=engine_support.LiveEngineTests().direct_engine(runtime)
        self.addCleanup(engine.stop)
        return engine,runtime

    def test_cleanup_preserves_click_history_and_requires_new_decision_frame(self):
        engine,runtime=self.engine()
        obs=observation()
        engine._last_clicked_key='previous'
        engine._last_click_at=123
        engine._state['clicks']=2
        self.assertTrue(engine._prepare_observation(obs,SimpleNamespace(frame_id=obs.frame_id),engine._epoch))
        self.assertEqual(engine._last_clicked_key,'previous')
        self.assertEqual(engine._last_click_at,123)
        self.assertEqual(engine.status()['clicks'],2)
        self.assertFalse(runtime.clicks)
        self.assertTrue(engine._pointer_cleanup_guard)

    def test_preview_battle_and_stale_epoch_never_prepare(self):
        engine,runtime=self.engine()
        obs=observation()
        frame=SimpleNamespace(frame_id=obs.frame_id)
        engine._state['observe_only']=True
        self.assertFalse(engine._prepare_observation(obs,frame,engine._epoch))
        engine._state['observe_only']=False
        self.assertFalse(engine._prepare_observation(obs,frame,engine._epoch-1))
        self.assertFalse(engine._prepare_observation(replace(obs,scene='battle'),frame,engine._epoch))
        runtime.prepare_observation.assert_not_called()
        runtime.cancel_pointer_cleanup.assert_called_once()

    def test_expired_lease_and_emergency_cancel_pending(self):
        for reason in ('lease','escape'):
            engine,runtime=self.engine()
            obs=observation()
            if reason=='lease':engine._lease=0
            else:runtime.emergency_stop=lambda:True
            self.assertFalse(engine._prepare_observation(obs,SimpleNamespace(frame_id=obs.frame_id),engine._epoch))
            runtime.prepare_observation.assert_not_called()
            runtime.cancel_pointer_cleanup.assert_called_once()
            self.assertEqual(engine.status()['state'],'paused')

    def test_newly_visible_ocr_cannot_trigger_retry_of_previous_target(self):
        engine,runtime=self.engine()
        obs=observation(resources={'gold':7})
        engine._last_clicked_key='old obscured observation'
        engine._last_clicked_action=ACTION
        engine._last_clicked_scene=(obs.scene,obs.floor)
        engine._pointer_cleanup_guard=True
        engine._step(obs,SimpleNamespace(frame_id=obs.frame_id))
        self.assertFalse(runtime.clicks)
        self.assertEqual(runtime.observe_count,0)
        self.assertEqual(engine.status()['state'],'paused')

    def test_different_observed_target_can_continue_after_cleanup(self):
        engine,runtime=self.engine()
        obs=observation()
        engine._last_clicked_action=replace(ACTION,label='先前选择')
        engine._last_clicked_scene=(obs.scene,obs.floor)
        engine._pointer_cleanup_guard=True
        engine._step(obs,SimpleNamespace(frame_id=obs.frame_id))
        self.assertEqual(len(runtime.clicks),1)
        self.assertFalse(engine._pointer_cleanup_guard)

    def test_revealed_prices_and_expanded_boxes_do_not_prove_a_new_target(self):
        engine,_=self.engine()
        obs=observation()
        engine._last_clicked_action=ACTION
        engine._last_clicked_scene=(obs.scene,obs.floor)
        x,y,w,h=ACTION.bbox
        for action in (
                replace(ACTION,bbox=(x-10,y,w+20,h)),
                replace(ACTION,metadata={**ACTION.metadata,'hope_cost':0,'resource_costs':{'hope':0}})):
            with self.subTest(action=action):
                self.assertTrue(engine._same_clicked_target(action,obs))
        self.assertFalse(engine._same_clicked_target(replace(ACTION,bbox=(x+w+20,y,w,h)),obs))
