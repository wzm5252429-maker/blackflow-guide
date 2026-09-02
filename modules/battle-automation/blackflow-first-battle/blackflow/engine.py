from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import numpy as np

from .capture import WindowCapture
from .config import ProjectConfig
from .controller import EmergencyStop, InputController, StopMonitor
from .episode_log import EpisodeLogger
from .learner import QLearner
from .vision import Detection, VisionError, VisionSystem, evaluate_condition, resolve_point
from .window import find_window, focus_window


class EngineError(RuntimeError):
    pass


class BattleEnded(RuntimeError):
    def __init__(self, outcome: str, frame: np.ndarray, detections: dict[str, Detection]) -> None:
        super().__init__(outcome)
        self.outcome = outcome
        self.frame = frame
        self.detections = detections


@dataclass
class Snapshot:
    frame: np.ndarray
    detections: dict[str, Detection]
    elapsed: float


class BattleEngine:
    def __init__(self, config: ProjectConfig, training: bool = True) -> None:
        self.config = config
        self.data = config.data
        self.training = training
        window_spec = self.data["window"]
        self.area = find_window(window_spec["title_keywords"])
        focus_window(self.area.hwnd)
        self.stop = StopMonitor()
        self.stop.start()
        self.capture = WindowCapture(self.area)
        self.controller = InputController(self.area, self.data["base_resolution"], self.stop.check)
        self.vision = VisionSystem(config)
        self.points: dict[str, list[float]] = self.data.get("points", {})
        logs_path = config.resolve(self.data.get("logs_dir", "../logs"))
        self.log = EpisodeLogger(logs_path)
        learning = self.data["learning"]
        state_path = config.resolve(learning.get("state_file", "../learning/q_table.json"))
        self.learner = QLearner(
            state_path,
            alpha=float(learning.get("alpha", 0.25)),
            gamma=float(learning.get("gamma", 0.0)),
            epsilon=float(learning.get("epsilon", 0.2)),
            epsilon_min=float(learning.get("epsilon_min", 0.03)),
            epsilon_decay=float(learning.get("epsilon_decay", 0.985)),
        )
        self.poll_interval = float(self.data.get("poll_interval", 0.12))
        self.episode_started = time.monotonic()

    def close(self) -> None:
        self.capture.close()
        self.stop.stop()

    def run(self, episodes: int) -> None:
        self.log.event("run_started", window=self.area.title, episodes=episodes, training=self.training)
        try:
            for episode in range(1, episodes + 1):
                self.stop.check()
                outcome, context, plan_id, reward = self._run_episode(episode)
                print(f"Episode {episode}: {outcome}; plan={plan_id}; reward={reward:.2f}")
                if episode < episodes:
                    reset_actions = self.data.get("episode", {}).get(f"reset_after_{outcome}", [])
                    if not reset_actions:
                        raise EngineError(
                            f"No episode.reset_after_{outcome} actions are configured, so another battle cannot start safely."
                        )
                    self._execute_sequence(reset_actions, allow_outcome=False)
            self.log.event("run_finished")
        finally:
            self.close()

    def _run_episode(self, episode_number: int) -> tuple[str, str, str, float]:
        self.episode_started = time.monotonic()
        self.log.event("episode_started", episode=episode_number)
        prepare = self.data.get("episode", {}).get("prepare_actions", [])
        self._execute_sequence(prepare, allow_outcome=False)
        start_snapshot = self._wait_for_battle()
        # Action timing is relative to the recognized battle start, not loading time.
        self.episode_started = time.monotonic()
        context = self._build_context(start_snapshot.detections)
        plans = self.data["plans"]
        plan_id = self.learner.select(context, [plan["id"] for plan in plans], training=self.training)
        plan = next(plan for plan in plans if plan["id"] == plan_id)
        self.log.event("plan_selected", episode=episode_number, context=context, plan=plan_id)
        print(f"Episode {episode_number}: selected plan {plan_id!r} for context {context!r}")

        try:
            self._execute_sequence(plan["actions"], allow_outcome=True)
            idle_actions = self.data.get("episode", {}).get("idle_actions", [{"type": "wait", "seconds": 0.5}])
            while True:
                self._execute_sequence(idle_actions, allow_outcome=True)
        except BattleEnded as ended:
            outcome = ended.outcome
            final_path = self.log.screenshot(outcome, ended.frame)
            reward = float(self.data["outcome"].get("rewards", {}).get(outcome, 1.0 if outcome == "victory" else -1.0))
            if self.training and self.data["learning"].get("enabled", True):
                self.learner.update(context, plan_id, reward, won=outcome == "victory")
            self.log.event(
                "episode_finished",
                episode=episode_number,
                outcome=outcome,
                context=context,
                plan=plan_id,
                reward=reward,
                screenshot=str(final_path),
            )
            return outcome, context, plan_id, reward

    def _build_context(self, detections: dict[str, Detection]) -> str:
        context_spec = self.data.get("context", {})
        parts = [str(context_spec.get("base", "first_floor_first_battle"))]
        for detector in context_spec.get("feature_detectors", []):
            parts.append(f"{detector}={int(detections[detector].matched)}")
        return "|".join(parts)

    def _snapshot(self, allow_outcome: bool = True) -> Snapshot:
        self.stop.check()
        frame = self.capture.grab()
        detections = self.vision.detect_all(frame)
        elapsed = time.monotonic() - self.episode_started
        snapshot = Snapshot(frame, detections, elapsed)
        if allow_outcome:
            outcome_spec = self.data["outcome"]
            victory = outcome_spec["victory_detector"]
            defeat = outcome_spec["defeat_detector"]
            if detections[victory].matched:
                raise BattleEnded("victory", frame, detections)
            if detections[defeat].matched:
                raise BattleEnded("defeat", frame, detections)
            timeout = float(outcome_spec.get("battle_timeout", 900))
            if elapsed > timeout:
                path = self.log.screenshot("battle_timeout", frame)
                self.log.event("battle_timeout", screenshot=str(path), elapsed=elapsed)
                raise BattleEnded("timeout", frame, detections)
        return snapshot

    def _wait_for_battle(self) -> Snapshot:
        episode = self.data.get("episode", {})
        detector = episode.get("battle_started_detector")
        if not detector:
            return self._snapshot(allow_outcome=False)
        timeout = float(episode.get("battle_start_timeout", 120))
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            snapshot = self._snapshot(allow_outcome=False)
            if snapshot.detections[detector].matched:
                self.log.event("battle_started", detector=detector, score=snapshot.detections[detector].score)
                return snapshot
            self.controller.sleep(self.poll_interval)
        frame = self.capture.grab()
        path = self.log.screenshot("battle_start_timeout", frame)
        raise EngineError(f"Battle start was not recognized within {timeout}s. Screenshot: {path}")

    def _wait_condition(
        self,
        condition: dict[str, Any] | None,
        timeout: float,
        allow_outcome: bool,
    ) -> Snapshot:
        deadline = time.monotonic() + timeout
        while True:
            snapshot = self._snapshot(allow_outcome=allow_outcome)
            if evaluate_condition(condition, snapshot.detections, snapshot.elapsed):
                return snapshot
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Condition timed out after {timeout}s: {condition}")
            self.controller.sleep(self.poll_interval)

    def _execute_sequence(self, actions: list[dict[str, Any]], allow_outcome: bool) -> None:
        for index, action in enumerate(actions):
            self.stop.check()
            label = str(action.get("label", f"action_{index}"))
            timeout = float(action.get("timeout", 60))
            optional = bool(action.get("optional", False))
            retries = int(action.get("retries", 0))
            for attempt in range(retries + 1):
                try:
                    snapshot = self._wait_condition(action.get("when"), timeout, allow_outcome)
                    self._execute_action(action, snapshot)
                    post_delay = float(action.get("post_delay", 0.15))
                    self.controller.sleep(post_delay)
                    verify = action.get("verify")
                    if verify:
                        self._wait_condition(verify, float(action.get("verify_timeout", 3.0)), allow_outcome)
                    self.log.event("action_ok", label=label, type=action.get("type"), attempt=attempt + 1)
                    break
                except (TimeoutError, VisionError) as exc:
                    self.log.event("action_failed", label=label, attempt=attempt + 1, error=str(exc))
                    if attempt >= retries:
                        if optional:
                            self.log.event("action_skipped", label=label)
                            break
                        frame = self.capture.grab()
                        path = self.log.screenshot(f"failed_{label}", frame)
                        raise EngineError(f"Action {label!r} failed: {exc}. Screenshot: {path}") from exc

    def _execute_action(self, action: dict[str, Any], snapshot: Snapshot) -> None:
        kind = action.get("type", "wait")
        if kind == "wait":
            self.controller.sleep(float(action.get("seconds", 0.5)))
            return
        if kind == "click":
            point = resolve_point(action["at"], self.points, snapshot.detections)
            self.controller.click(point, str(action.get("button", "left")))
            return
        if kind == "key":
            self.controller.key(str(action["key"]))
            return
        if kind == "drag":
            start = resolve_point(action["from"], self.points, snapshot.detections)
            end = resolve_point(action["to"], self.points, snapshot.detections)
            self.controller.drag(start, end, float(action.get("duration", 0.35)))
            return
        if kind == "deploy":
            card = resolve_point(action["card"], self.points, snapshot.detections)
            tile = resolve_point(action["tile"], self.points, snapshot.detections)
            self.controller.deploy(
                card,
                tile,
                str(action.get("direction", "none")).lower(),
                float(action.get("duration", 0.45)),
                float(action.get("direction_distance", 130)),
            )
            return
        if kind == "skill":
            operator = resolve_point(action["operator"], self.points, snapshot.detections)
            skill_button = resolve_point(action.get("skill_button", {"point": "skill_button"}), self.points, snapshot.detections)
            self.controller.click(operator)
            self.controller.sleep(float(action.get("open_delay", 0.18)))
            self.controller.click(skill_button)
            return
        if kind == "retreat":
            operator = resolve_point(action["operator"], self.points, snapshot.detections)
            retreat_button = resolve_point(action.get("retreat_button", {"point": "retreat_button"}), self.points, snapshot.detections)
            self.controller.click(operator)
            self.controller.sleep(float(action.get("open_delay", 0.18)))
            self.controller.click(retreat_button)
            return
        raise EngineError(f"Unsupported action type: {kind}")
