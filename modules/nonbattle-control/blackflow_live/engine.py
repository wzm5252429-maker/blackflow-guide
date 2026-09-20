"""One screenshot-grounded action at a time; never run simulated transitions."""
from __future__ import annotations

from collections import deque
from dataclasses import asdict
import json
import math
import threading
import time
from typing import Any

from .models import LiveObservation, ObservedAction

BATTLE_SCENES = frozenset({'battle', 'battle_start', 'combat', 'squad', 'battle_prepare'})
FORBIDDEN_LABELS = ('开始战斗', '开始行动', '开始作战', '进入战斗', 'Start Operation')
RECRUITMENT_WAIT_MESSAGES = {
    'recruitment_hope_display_may_be_post_selection': '已识别付费招募，但尚未确认当前可用希望；正在等待。',
    'recruitment_selected_cost_unreadable': '已识别选中的干员，但尚未读清招募费用；正在重新识别。',
    'recruitment_hope_display_unreadable': '已识别免费招募，但尚未读清希望状态；正在重新识别。',
    'recruitment_confirm_not_enabled': '尚未确认招募按钮可用；正在重新识别。',
    'recruitment_selection_not_verified': '招募列表已识别，但尚未确认可操作的干员目标；正在重新识别。',
    'initial_recruitment_completion_requires_observation': '招募完成页尚未显示完整，正在等待入场按钮。',
}
POLICY_METADATA_FIELDS = frozenset({
    'items', 'inventory', 'inventory_complete', 'shop_state', 'shop_items', 'pending_node_id',
    'formal_operator_ids', 'available_operator_ids', 'promoted_operator_ids',
    'pending_recruit_ticket_ids', 'stored_recruit_ticket_ids', 'temporary_recruit_offers',
    'image_width', 'image_height',
})


def _stable_metadata(value):
    """Keep decision evidence, omitting per-frame provenance at every depth."""
    if isinstance(value, dict):
        return {key: _stable_metadata(item) for key, item in value.items()
                if key not in {'source_frame_id', 'captured_at'}}
    if isinstance(value, (list, tuple)):
        return [_stable_metadata(item) for item in value]
    return value


def observation_key(obs: LiveObservation) -> str:
    """Compare observed semantics independently of detector enumeration order."""
    nodes = sorted((n.node_id, n.node_type, n.row, n.col, n.revealed, n.completed) for n in obs.nodes)
    # The encoder treats every corridor as undirected. Confidence ordering can
    # reverse both endpoint order and edge enumeration in a later screenshot.
    edges = sorted(tuple(sorted(edge)) for edge in obs.edges)
    actions = [(a.label, a.kind, a.enabled, a.target_node_id, _stable_metadata(a.metadata))
               for a in obs.actions]
    actions.sort(key=lambda item: json.dumps(item, sort_keys=True, ensure_ascii=False))
    # Sort without deduplication: additions, removals and duplicate detections
    # still change the state. Action IDs remain transient pixel-based handles;
    # _step independently re-grounds the selected action in the fresh frame.
    return json.dumps({
        'scene': obs.scene, 'floor': obs.floor, 'current': obs.current_node_id,
        'resources': obs.resources,
        'nodes': nodes,
        'edges': edges,
        'actions': actions,
        # OCR boxes/confidences are diagnostic and can jitter while the same
        # screen is visible. These fields actually affect frozen policy inputs.
        'metadata': _stable_metadata({key: value for key, value in obs.metadata.items()
                                      if key in POLICY_METADATA_FIELDS}),
        'ending': obs.ending_first_confirmed,
    }, sort_keys=True, ensure_ascii=False)


def action_is_grounded(action: ObservedAction, obs: LiveObservation) -> bool:
    if obs.scene in BATTLE_SCENES or obs.scene in {'unknown', 'loading', 'ending_complete'}:
        return False
    if action.kind in {'battle', 'start_battle', 'battle_start', 'combat'}:
        return False
    label_text = ''.join(action.label.split()).casefold()
    if any(''.join(label.split()).casefold() in label_text for label in FORBIDDEN_LABELS):
        return False
    if not action.enabled or not math.isfinite(action.confidence) or action.confidence < .85:
        return False
    if not math.isfinite(obs.confidence) or obs.confidence < .85:
        return False
    if not any(a == action for a in obs.actions):
        return False
    if action.metadata.get('source_frame_id') not in (None, obs.frame_id):
        return False
    x, y, w, h = action.bbox
    return all(math.isfinite(v) for v in action.bbox) and x >= 0 and y >= 0 and w > 0 and h > 0


class LiveEngine:
    """Runtime supplies observe()->(LiveObservation, frame), focus, click, close.

    The runtime must validate the frame's window identity/geometry and foreground
    again at click time. GUI leases stop unattended input after browser disconnect.
    """
    def __init__(self, runtime_factory, *, interval=.7, lease_seconds=20, max_frame_age=15):
        self.runtime_factory = runtime_factory
        self.interval = interval
        self.lease_seconds = lease_seconds
        self.max_frame_age = max_frame_age
        self._lock = threading.RLock()
        self._input_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._runtime = None
        self._paused = False
        self._epoch = 0
        self._focus_requested = False
        self._lease = time.monotonic()
        self._events = deque(maxlen=80)
        self._state: dict[str, Any] = {'state': 'idle', 'message': '本机服务已连接', 'clicks': 0,
                                      'target_ending': 'first', 'battle_control': False}
        self._preview = None
        self._last_clicked_key = None
        self._last_click_at = 0.

    def _set(self, state, message, **extra):
        with self._lock:
            changed = self._state.get('state') != state or self._state.get('message') != message
            self._state.update(state=state, message=message, **extra)
            if changed:
                self._events.append({'time': time.time(), 'state': state, 'message': message})

    def status(self, *, heartbeat=False):
        with self._lock:
            if heartbeat:
                self._lease = time.monotonic()
            return {**self._state, 'events': list(self._events), 'has_preview': self._preview is not None}

    def preview(self):
        with self._lock:
            return self._preview

    def start(self, hwnd=None, *, observe_only=False):
        with self._input_lock:
            if self._thread and self._thread.is_alive():
                raise ValueError('已有接管会话；请先停止后重试')
            self._stop.clear()
            self._epoch += 1
            self._paused = False
            self._lease = time.monotonic()
            self._last_clicked_key = None
            self._last_click_at = 0.
            self._focus_requested = not observe_only
            self._set('starting', '正在连接游戏窗口和识别模型', clicks=0, observe_only=observe_only)
            self._thread = threading.Thread(target=self._run, args=(hwnd, observe_only), daemon=True,
                                            name='blackflow-live-loop')
            self._thread.start()
        return self.status()

    def pause(self, reason='已暂停接管'):
        with self._input_lock:
            self._pause_locked(reason)
        return self.status()

    def _pause_locked(self, reason):
        self._epoch += 1
        self._paused = True
        self._set('paused', reason)

    def resume(self):
        with self._input_lock:
            if not self._thread or not self._thread.is_alive() or self._stop.is_set():
                raise ValueError('会话已结束，请重新点击一键启动自动执行')
            if not self._paused:
                raise ValueError('只能继续已暂停的会话')
            self._epoch += 1
            self._paused = False
            self._focus_requested = not self._state.get('observe_only')
            self._lease = time.monotonic()
            # A deliberate resume permits one newly decided retry, including on
            # an unchanged screen. A successful retry sets the duplicate guard
            # again; the engine must never retry automatically without consent.
            self._last_clicked_key = None
            self._last_click_at = 0.
            self._set('running', '正在重新识别当前画面')
        return self.status()

    def stop(self):
        # Mark cancelled before waiting for the atomic input section.
        self._stop.set()
        with self._input_lock:
            self._epoch += 1
            self._paused = True
            self._set('stopped', '接管已停止')
        return self.status()

    def _epoch_current(self, epoch):
        return epoch == self._epoch and not self._stop.is_set() and not self._paused

    def _set_for_epoch(self, epoch, state, message, **extra):
        with self._input_lock:
            if not self._epoch_current(epoch):
                return False
            self._set(state, message, **extra)
            return True

    def _record_observation(self, obs, frame, epoch):
        preview = self._runtime.preview(frame)
        if not self._epoch_current(epoch):
            return False
        window = self._runtime.window_info(frame)
        observation = asdict(obs)
        with self._input_lock:
            if not self._epoch_current(epoch):
                return False
            with self._lock:
                self._preview = preview
                self._state['observation'] = observation
                self._state['window'] = window
            return True

    def _fresh(self, obs):
        return bool(obs.frame_id) and 0 <= time.time() - obs.captured_at <= self.max_frame_age

    def _expired_message(self, obs):
        duration = obs.metadata.get('timing_ms', {}).get('total')
        if isinstance(duration, (int, float)) and math.isfinite(duration) and duration >= self.max_frame_age * 1000:
            return f'本次识别耗时 {duration / 1000:.1f} 秒，截图已过期；正在重新获取'
        return '截图已过期，正在重新获取'

    def _run(self, hwnd, observe_only):
        try:
            self._runtime = self.runtime_factory(hwnd)
            while not self._stop.is_set():
                if self._paused:
                    self._stop.wait(.1)
                    continue
                epoch = self._epoch
                try:
                    with self._input_lock:
                        if not self._epoch_current(epoch):
                            continue
                        if time.monotonic() - self._lease > self.lease_seconds:
                            self._pause_locked('网站连接中断，接管已暂停')
                            continue
                    emergency = self._runtime.emergency_stop()
                    with self._input_lock:
                        if not self._epoch_current(epoch):
                            continue
                        if emergency:
                            self._pause_locked('检测到 Esc 或鼠标移至左上角，接管已暂停')
                            continue
                        if self._focus_requested:
                            self._runtime.focus()
                            if not self._epoch_current(epoch):
                                continue
                            self._focus_requested = False
                    obs, frame = self._runtime.observe()
                    if not self._epoch_current(epoch):
                        continue
                    if not self._record_observation(obs, frame, epoch):
                        continue
                    if not self._fresh(obs):
                        self._set_for_epoch(epoch, 'waiting_observation', self._expired_message(obs))
                    elif obs.scene in BATTLE_SCENES:
                        self._set_for_epoch(epoch, 'waiting_battle', '请手动开始并完成战斗；战后将重新识别并继续')
                    elif obs.ending_first_confirmed and obs.scene == 'ending_complete' and obs.confidence >= .95:
                        confirm, confirm_frame = self._runtime.observe()
                        with self._input_lock:
                            if not self._epoch_current(epoch) or time.monotonic() - self._lease > self.lease_seconds:
                                continue
                            if (self._fresh(confirm) and confirm.frame_id != obs.frame_id
                                and confirm.frame_id == confirm_frame.frame_id
                                and confirm.ending_first_confirmed and confirm.scene == 'ending_complete' and confirm.confidence >= .95):
                                self._set('completed', '已识别到一结局通关画面，接管结束')
                                break
                    elif observe_only:
                        self._set_for_epoch(epoch, 'observing', '仅识别预览；未启用点击')
                    else:
                        self._step(obs, frame, epoch=epoch)
                except Exception as exc:
                    with self._input_lock:
                        if not self._epoch_current(epoch):
                            # A cancelled OCR/preview/inference may finish or fail
                            # after resume. It belongs to the old execution, and
                            # cannot end the resumed session or overwrite its UI.
                            continue
                        self._set('error', str(exc))
                    break
                self._stop.wait(self.interval)
        except Exception as exc:
            if not self._stop.is_set():
                self._set('error', str(exc))
        finally:
            if self._runtime:
                try:
                    self._runtime.close()
                except Exception:
                    pass
            self._runtime = None

    def _step(self, obs, frame, *, epoch=None):
        epoch = self._epoch if epoch is None else epoch
        if not self._epoch_current(epoch):
            return
        key = observation_key(obs)
        with self._input_lock:
            if not self._epoch_current(epoch):
                return
            if key == self._last_clicked_key:
                if time.monotonic() - self._last_click_at > 18:
                    self._pause_locked('点击后未确认界面变化，请检查游戏画面再继续')
                else:
                    self._set('running', '正在等待上一步操作结果')
                return
        decision = self._runtime.policy.select(obs)
        with self._input_lock:
            if not self._epoch_current(epoch):
                return
            with self._lock:
                self._state['decision'] = asdict(decision)
        action = decision.action
        if action is None:
            message = decision.reason
            if decision.reason == 'no_grounded_scope_legal_action':
                message = next((text for code,text in RECRUITMENT_WAIT_MESSAGES.items()
                                if code in obs.diagnostics),message)
            self._set_for_epoch(epoch, 'waiting_observation', message)
            return
        if not action_is_grounded(action, obs):
            self._set_for_epoch(epoch, 'waiting_observation', '当前动作缺少可靠的画面依据，正在重新识别')
            return
        # Network inference and OCR can take time: reacquire, re-ground, then act.
        fresh, fresh_frame = self._runtime.observe()
        if not self._epoch_current(epoch):
            return
        if not self._record_observation(fresh, fresh_frame, epoch):
            return
        found = next((a for a in fresh.actions if a.action_id == action.action_id), None)
        if found is None:
            matches = [a for a in fresh.actions if (a.label, a.kind, a.target_node_id) ==
                       (action.label, action.kind, action.target_node_id)]
            if len(matches) == 1:
                found = matches[0]
        if (not self._fresh(fresh) or fresh.frame_id != fresh_frame.frame_id or fresh.frame_id == obs.frame_id
            or fresh.scene != obs.scene or found is None or not action_is_grounded(found, fresh)):
            self._set_for_epoch(epoch, 'waiting_observation', '画面已经变化，正在重新决策')
            return
        if observation_key(fresh) != key or found.label != action.label or found.kind != action.kind:
            self._set_for_epoch(epoch, 'waiting_observation', '状态已更新，正在重新决策')
            return
        if any(abs(a-b) > 6 for a, b in zip(found.bbox, action.bbox)):
            self._set_for_epoch(epoch, 'waiting_observation', '目标位置仍在移动，等待画面稳定')
            return
        with self._input_lock:
            if not self._epoch_current(epoch) or time.monotonic() - self._lease > self.lease_seconds:
                return
            if self._runtime.emergency_stop():
                self._pause_locked('已通过紧急暂停停止输入')
                return
            # The frame may have expired during state comparison or while waiting
            # for this lock. Keep the original capture deadline through input.
            if not self._fresh(fresh):
                self._set('waiting_observation', '点击前截图已过期，正在重新识别')
                return
            self._runtime.click(found, fresh_frame)
            self._last_clicked_key = key
            self._last_click_at = time.monotonic()
            self._set('running', f'已点击：{found.label}', clicks=self._state['clicks'] + 1)
