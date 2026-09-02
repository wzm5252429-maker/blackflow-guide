"""A small, dependency-free single-player PUCT search implementation.

The search code intentionally depends only on Python's standard library.  An
environment and an evaluator are accepted through structural ``Protocol``
interfaces, which keeps the simulator usable without importing PyTorch (or any
other part of :mod:`blackflow_rl`).

Values returned by an evaluator describe the expected *future* normalized
return from its state.  Edge backups therefore use the single-player equation

``edge_return = immediate_reward / reward_scale + gamma * child_value``.

There is deliberately no alternating-player sign change in selection or
backup.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
import random
from typing import Generic, Protocol, TypeAlias, TypeVar, runtime_checkable


StateT = TypeVar("StateT")
PriorOutput: TypeAlias = Mapping[int, float] | Sequence[float]


@dataclass(frozen=True, slots=True)
class Transition(Generic[StateT]):
    """Convenience transition type accepted by :class:`PUCTMCTS`.

    Environments do not have to construct this exact class.  Any object with
    compatible ``next_state``, ``reward`` and ``terminated`` attributes is
    accepted through :class:`TransitionLike`.
    """

    next_state: StateT
    reward: float
    terminated: bool


@runtime_checkable
class TransitionLike(Protocol[StateT]):
    """Structural result expected from ``environment.transition``."""

    next_state: StateT
    reward: float
    terminated: bool


class MCTSEnvironment(Protocol[StateT]):
    """Minimal environment interface required by PUCT search."""

    @property
    def action_size(self) -> int:
        """Number of IDs in the global action space ``[0, action_size)``."""

        ...

    def legal_action_ids(self, state: StateT) -> tuple[int, ...]:
        """Return the legal, unique global action IDs for ``state``."""

        ...

    def transition(self, state: StateT, action_id: int) -> TransitionLike[StateT]:
        """Apply a legal action without mutating ``state``."""

        ...


class StateEvaluator(Protocol[StateT]):
    """Policy/value evaluator used when a node is first expanded."""

    def evaluate(
        self, state: StateT, legal_action_ids: tuple[int, ...]
    ) -> tuple[PriorOutput, float]:
        """Return action priors and a normalized value in ``[-1, 1]``.

        A mapping is keyed by global action ID.  A sequence with
        ``action_size`` entries is also indexed by global action ID; otherwise
        a sequence with one entry per legal action is aligned with
        ``legal_action_ids``.  Invalid priors are sanitized by the search.
        """

        ...


@dataclass(frozen=True, slots=True)
class MCTSConfig:
    """Configuration for :class:`PUCTMCTS`.

    ``dirichlet_fraction=0`` or ``add_root_dirichlet_noise=False`` disables
    exploration noise.  Supplying a seed makes noise and positive-temperature
    action sampling reproducible.
    """

    num_simulations: int = 200
    c_puct: float = 1.5
    gamma: float = 0.99
    reward_scale: float = 1.0
    max_depth: int = 256
    temperature: float = 1.0
    add_root_dirichlet_noise: bool = False
    dirichlet_alpha: float = 0.3
    dirichlet_fraction: float = 0.25
    seed: int | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.num_simulations, bool)
            or not isinstance(self.num_simulations, int)
            or self.num_simulations <= 0
        ):
            raise ValueError("num_simulations must be a positive integer")
        if (
            isinstance(self.max_depth, bool)
            or not isinstance(self.max_depth, int)
            or self.max_depth <= 0
        ):
            raise ValueError("max_depth must be a positive integer")
        _require_finite_nonnegative("c_puct", self.c_puct)
        if not math.isfinite(self.gamma) or not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be finite and in [0, 1]")
        if not math.isfinite(self.reward_scale) or self.reward_scale <= 0.0:
            raise ValueError("reward_scale must be finite and greater than zero")
        _require_temperature(self.temperature)
        if not math.isfinite(self.dirichlet_alpha) or self.dirichlet_alpha <= 0.0:
            raise ValueError("dirichlet_alpha must be finite and greater than zero")
        if (
            not math.isfinite(self.dirichlet_fraction)
            or not 0.0 <= self.dirichlet_fraction <= 1.0
        ):
            raise ValueError("dirichlet_fraction must be in [0, 1]")
        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, int)
        ):
            raise TypeError("seed must be an integer or None")


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Public result of one root search.

    Both ``visit_policy`` and ``visit_counts`` use the environment's fixed
    global action space and therefore always contain exactly ``action_size``
    entries.  Illegal actions have probability and visit count zero.
    """

    visit_policy: tuple[float, ...]
    visit_counts: tuple[int, ...]
    selected_action: int | None
    root_value: float


class UniformEvaluator(Generic[StateT]):
    """Evaluator with uniform legal priors and a configurable constant value."""

    def __init__(self, value: float = 0.0) -> None:
        self._value = _validated_value(value)

    def evaluate(
        self, state: StateT, legal_action_ids: tuple[int, ...]
    ) -> tuple[Mapping[int, float], float]:
        del state
        if not legal_action_ids:
            return {}, self._value
        probability = 1.0 / len(legal_action_ids)
        return (
            {action_id: probability for action_id in legal_action_ids},
            self._value,
        )


@dataclass(slots=True)
class _Edge(Generic[StateT]):
    prior: float
    visit_count: int = 0
    value_sum: float = 0.0
    immediate_reward: float = 0.0
    child: _Node[StateT] | None = None

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count else 0.0


@dataclass(slots=True)
class _Node(Generic[StateT]):
    state: StateT
    terminated: bool = False
    expanded: bool = False
    visit_count: int = 0
    edges: dict[int, _Edge[StateT]] = field(default_factory=dict)


class PUCTMCTS(Generic[StateT]):
    """Single-player Monte Carlo tree search using the PUCT rule.

    A new tree is built for every :meth:`search` call.  Transitions are lazy:
    an environment action is evaluated only the first time its edge is
    traversed, after which the child and immediate reward are cached.
    """

    def __init__(
        self,
        environment: MCTSEnvironment[StateT],
        evaluator: StateEvaluator[StateT] | None = None,
        config: MCTSConfig | None = None,
        *,
        seed: int | None = None,
    ) -> None:
        self.environment = environment
        self.evaluator: StateEvaluator[StateT] = evaluator or UniformEvaluator()
        self.config = config or MCTSConfig()
        if seed is not None and (
            isinstance(seed, bool) or not isinstance(seed, int)
        ):
            raise TypeError("seed must be an integer or None")
        self._rng = random.Random(self.config.seed if seed is None else seed)

    def search(
        self,
        root_state: StateT,
        *,
        temperature: float | None = None,
        add_root_noise: bool | None = None,
    ) -> SearchResult:
        """Run PUCT from ``root_state`` and return a fixed-size visit policy.

        At temperature zero, action selection is deterministic and one-hot on
        the most visited edge (prior and then lower action ID break ties).  At
        a positive temperature, the returned policy is proportional to
        ``visit_count ** (1 / temperature)`` and ``selected_action`` is sampled
        from it using this search object's seeded RNG.
        """

        action_size = self._action_size()
        selected_temperature = (
            self.config.temperature if temperature is None else temperature
        )
        _require_temperature(selected_temperature)

        root = _Node(root_state)
        root_leaf_value = self._expand(root, action_size)
        use_noise = (
            self.config.add_root_dirichlet_noise
            if add_root_noise is None
            else add_root_noise
        )
        if not isinstance(use_noise, bool):
            raise TypeError("add_root_noise must be a bool or None")
        if use_noise and root.edges and self.config.dirichlet_fraction > 0.0:
            self._add_root_noise(root)

        for _ in range(self.config.num_simulations):
            if root.terminated or not root.edges:
                break
            self._simulate(root, action_size)

        counts = [0] * action_size
        for action_id, edge in root.edges.items():
            counts[action_id] = edge.visit_count
        policy, selected_action = self._policy_and_action(
            root, counts, selected_temperature
        )
        total_visits = sum(counts)
        root_value = (
            sum(edge.value_sum for edge in root.edges.values()) / total_visits
            if total_visits
            else (0.0 if root.terminated else root_leaf_value)
        )
        return SearchResult(
            visit_policy=tuple(policy),
            visit_counts=tuple(counts),
            selected_action=selected_action,
            root_value=root_value,
        )

    def run(
        self,
        root_state: StateT,
        *,
        temperature: float | None = None,
        add_root_noise: bool | None = None,
    ) -> SearchResult:
        """Alias for :meth:`search`, useful in training-loop code."""

        return self.search(
            root_state,
            temperature=temperature,
            add_root_noise=add_root_noise,
        )

    def _simulate(self, root: _Node[StateT], action_size: int) -> None:
        node = root
        path: list[tuple[_Node[StateT], _Edge[StateT]]] = []
        leaf_value = 0.0

        for _depth in range(self.config.max_depth):
            if node.terminated:
                leaf_value = 0.0
                break
            if not node.expanded:
                leaf_value = self._expand(node, action_size)
                break
            if not node.edges:
                # A non-terminated state with no legal actions is a dead end.
                leaf_value = 0.0
                break

            _action_id, edge = self._select_edge(node)
            if edge.child is None:
                outcome = self.environment.transition(node.state, _action_id)
                reward, terminated = _validate_transition(outcome)
                edge.immediate_reward = reward
                edge.child = _Node(outcome.next_state, terminated=terminated)
            path.append((node, edge))
            node = edge.child
        else:
            # Cut off cyclic or unusually long episodes with a fresh value
            # estimate rather than silently pretending that their value is 0.
            if node.terminated:
                leaf_value = 0.0
            else:
                legal_ids = self._legal_ids(node.state, action_size)
                _priors, raw_value = self.evaluator.evaluate(node.state, legal_ids)
                leaf_value = _validated_value(raw_value)

        value = leaf_value
        for parent, edge in reversed(path):
            value = (
                edge.immediate_reward / self.config.reward_scale
                + self.config.gamma * value
            )
            if not math.isfinite(value):
                raise OverflowError(
                    "MCTS backup became non-finite; increase reward_scale"
                )
            edge.visit_count += 1
            edge.value_sum += value
            parent.visit_count += 1

    def _expand(self, node: _Node[StateT], action_size: int) -> float:
        if node.expanded:
            raise RuntimeError("a tree node cannot be expanded twice")
        node.expanded = True
        if node.terminated:
            return 0.0

        legal_ids = self._legal_ids(node.state, action_size)
        if not legal_ids:
            node.terminated = True
            return 0.0
        raw_priors, raw_value = self.evaluator.evaluate(node.state, legal_ids)
        priors = _normalize_priors(raw_priors, legal_ids, action_size)
        node.edges = {
            action_id: _Edge(prior=priors[action_id]) for action_id in legal_ids
        }
        return _validated_value(raw_value)

    def _select_edge(self, node: _Node[StateT]) -> tuple[int, _Edge[StateT]]:
        parent_scale = math.sqrt(max(1, node.visit_count))
        best_action = -1
        best_edge: _Edge[StateT] | None = None
        best_key = (-math.inf, -math.inf, -math.inf)
        for action_id, edge in node.edges.items():
            exploration = (
                self.config.c_puct
                * edge.prior
                * parent_scale
                / (1 + edge.visit_count)
            )
            # Lower action IDs are the final deterministic tie breaker.
            key = (edge.mean_value + exploration, edge.prior, -float(action_id))
            if key > best_key:
                best_key = key
                best_action = action_id
                best_edge = edge
        if best_edge is None:  # guarded by caller; retained as an invariant check
            raise RuntimeError("cannot select an edge from an empty node")
        return best_action, best_edge

    def _add_root_noise(self, root: _Node[StateT]) -> None:
        edges = list(root.edges.values())
        samples = [
            self._rng.gammavariate(self.config.dirichlet_alpha, 1.0)
            for _ in edges
        ]
        sample_sum = sum(samples)
        if not math.isfinite(sample_sum) or sample_sum <= 0.0:
            noise = [1.0 / len(edges)] * len(edges)
        else:
            noise = [sample / sample_sum for sample in samples]
        fraction = self.config.dirichlet_fraction
        for edge, sample in zip(edges, noise, strict=True):
            edge.prior = (1.0 - fraction) * edge.prior + fraction * sample

    def _policy_and_action(
        self, root: _Node[StateT], counts: list[int], temperature: float
    ) -> tuple[list[float], int | None]:
        policy = [0.0] * len(counts)
        if not root.edges:
            return policy, None

        legal_ids = tuple(root.edges)
        if temperature == 0.0:
            selected = max(
                legal_ids,
                key=lambda action_id: (
                    counts[action_id],
                    root.edges[action_id].prior,
                    -action_id,
                ),
            )
            policy[selected] = 1.0
            return policy, selected

        positive_counts = [counts[action_id] for action_id in legal_ids]
        if any(positive_counts):
            largest_count = max(positive_counts)
            largest_log_count = math.log(largest_count)
            log_weights = [
                (math.log(count) - largest_log_count) / temperature
                if count > 0
                else -math.inf
                for count in positive_counts
            ]
            weights = [
                math.exp(weight) if math.isfinite(weight) else 0.0
                for weight in log_weights
            ]
        else:
            weights = [root.edges[action_id].prior for action_id in legal_ids]
        weight_sum = sum(weights)
        if not math.isfinite(weight_sum) or weight_sum <= 0.0:
            weights = [1.0] * len(legal_ids)
            weight_sum = float(len(legal_ids))
        for action_id, weight in zip(legal_ids, weights, strict=True):
            policy[action_id] = weight / weight_sum
        selected = self._sample_action(legal_ids, policy)
        return policy, selected

    def _sample_action(
        self, legal_ids: tuple[int, ...], policy: Sequence[float]
    ) -> int:
        draw = self._rng.random()
        cumulative = 0.0
        for action_id in legal_ids:
            cumulative += policy[action_id]
            if draw < cumulative:
                return action_id
        # Protect against the final cumulative sum being 0.9999999999999999.
        return legal_ids[-1]

    def _action_size(self) -> int:
        action_size = self.environment.action_size
        if isinstance(action_size, bool) or not isinstance(action_size, int):
            raise TypeError("environment.action_size must be an integer")
        if action_size <= 0:
            raise ValueError("environment.action_size must be greater than zero")
        return action_size

    def _legal_ids(self, state: StateT, action_size: int) -> tuple[int, ...]:
        legal_ids = self.environment.legal_action_ids(state)
        if not isinstance(legal_ids, tuple):
            raise TypeError("legal_action_ids(state) must return a tuple")
        seen: set[int] = set()
        for action_id in legal_ids:
            if isinstance(action_id, bool) or not isinstance(action_id, int):
                raise TypeError("legal action IDs must be integers")
            if not 0 <= action_id < action_size:
                raise ValueError(
                    f"legal action ID {action_id} is outside [0, {action_size})"
                )
            if action_id in seen:
                raise ValueError(f"duplicate legal action ID {action_id}")
            seen.add(action_id)
        return legal_ids


# Short name used by callers that do not need to distinguish PUCT variants.
MCTS = PUCTMCTS
Evaluator = StateEvaluator
Environment = MCTSEnvironment


def _normalize_priors(
    raw_priors: PriorOutput,
    legal_ids: tuple[int, ...],
    action_size: int,
) -> dict[int, float]:
    """Discard illegal/invalid mass and normalize over legal actions."""

    values: dict[int, object]
    if isinstance(raw_priors, Mapping):
        values = {action_id: raw_priors.get(action_id, 0.0) for action_id in legal_ids}
    else:
        if isinstance(raw_priors, (str, bytes, bytearray)):
            sequence: list[object] = []
        else:
            try:
                sequence = list(raw_priors)
            except TypeError:
                sequence = []
        if len(sequence) == action_size:
            values = {action_id: sequence[action_id] for action_id in legal_ids}
        elif len(sequence) == len(legal_ids):
            values = dict(zip(legal_ids, sequence, strict=True))
        else:
            values = {action_id: 0.0 for action_id in legal_ids}

    cleaned: dict[int, float] = {}
    for action_id in legal_ids:
        try:
            prior = float(values.get(action_id, 0.0))
        except (TypeError, ValueError, OverflowError):
            prior = 0.0
        cleaned[action_id] = prior if math.isfinite(prior) and prior >= 0.0 else 0.0
    largest = max(cleaned.values())
    if largest <= 0.0:
        uniform = 1.0 / len(legal_ids)
        return {action_id: uniform for action_id in legal_ids}
    # Scaling by the largest entry avoids overflow when several otherwise
    # valid, very large priors are supplied.
    scaled = {action_id: cleaned[action_id] / largest for action_id in legal_ids}
    total = sum(scaled.values())
    return {action_id: scaled[action_id] / total for action_id in legal_ids}


def _validated_value(raw_value: object) -> float:
    try:
        value = float(raw_value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("evaluator value must be a real number") from exc
    if not math.isfinite(value):
        raise ValueError("evaluator value must be finite")
    if not -1.0 <= value <= 1.0:
        raise ValueError("evaluator value must be normalized to [-1, 1]")
    return value


def _validate_transition(outcome: TransitionLike[StateT]) -> tuple[float, bool]:
    try:
        reward = float(outcome.reward)
        terminated = outcome.terminated
        # Access here so malformed outcomes fail at the transition boundary,
        # before an incomplete child is attached to the tree.
        outcome.next_state
    except AttributeError as exc:
        raise TypeError(
            "transition() must return next_state, reward and terminated attributes"
        ) from exc
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("transition reward must be a real number") from exc
    if not math.isfinite(reward):
        raise ValueError("transition reward must be finite")
    if not isinstance(terminated, bool):
        raise TypeError("transition terminated must be a bool")
    return reward, terminated


def _require_finite_nonnegative(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")


def _require_temperature(value: float) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("temperature must be finite and non-negative")


__all__ = [
    "Environment",
    "Evaluator",
    "MCTS",
    "MCTSConfig",
    "MCTSEnvironment",
    "PriorOutput",
    "PUCTMCTS",
    "SearchResult",
    "StateEvaluator",
    "Transition",
    "TransitionLike",
    "UniformEvaluator",
]
