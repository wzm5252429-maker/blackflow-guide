"""Hidden identities and observations of the three existing floor-V fates.

The caller supplies a correctly gated generated map. This module never adds
nodes or increases map density. Unknown server position weights require an
explicit synthetic prior; real identity remains outside MapNode payloads.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import random
from typing import Iterable

from .domain import FloorMap, NodeType, ResourceDelta
from .events import FALSE_FATE_NAME, FATE_SANDBOXES


TRUE_FATE_NAME = "窥视箱中"
FATE_PUBLIC_NAME = "命运所指"
TRUE_FATE_STAGE = "ro6_b_5"


@dataclass(frozen=True, slots=True)
class FateContext:
    node_ids: tuple[str, ...]
    true_node_id: str | None
    observed_scenes: tuple[tuple[str, str], ...] = ()
    true_fate_marked: bool = False
    assumptions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(self.node_ids) != 3 or len(set(self.node_ids)) != 3:
            raise ValueError("floor V requires three distinct existing fate nodes")
        if self.true_node_id is not None and self.true_node_id not in self.node_ids:
            raise ValueError("true fate must be one of the existing fate nodes")
        if self.true_fate_marked and self.true_node_id is None:
            raise ValueError("a marked true fate must have a known coordinate")
        seen = dict(self.observed_scenes)
        if len(seen) != len(self.observed_scenes):
            raise ValueError("duplicate fate observations")
        if sum(s == TRUE_FATE_NAME for s in seen.values()) > 1 or sum(s == FALSE_FATE_NAME for s in seen.values()) > 2:
            raise ValueError("observations violate the one-true/two-false rule")
        for node_id, scene in self.observed_scenes:
            if node_id not in self.node_ids or scene not in (FALSE_FATE_NAME, TRUE_FATE_NAME):
                raise ValueError("invalid fate observation")
            if self.true_node_id is not None and ((node_id == self.true_node_id) != (scene == TRUE_FATE_NAME)):
                raise ValueError("fate observation conflicts with its fixed identity")


def fate_route_available(*, inventory: Iterable[str], first_ending_unlocked: bool) -> bool:
    """PRTS ending flow requires the first ending and at least one sandbox."""
    return first_ending_unlocked and bool(FATE_SANDBOXES.intersection(inventory))


def initialize_fates(
    floor_map: FloorMap, *, inventory: Iterable[str] = (),
    observed_true_node_id: str | None = None,
    rng: random.Random | None = None, allow_synthetic: bool = False,
) -> FateContext | None:
    """Assign one true identity among the three nodes the map already contains.

    Apply ``fate_route_available`` before generating the floor-V layout. An
    absent STORY group returns None; a partial group is an error, not a reason
    to create extra nodes. Floor-VI STORY has a separate ritual interpretation.
    """
    if floor_map.floor != 5:
        if observed_true_node_id is not None:
            raise ValueError("true fate location belongs only to floor V")
        return None
    node_ids = tuple(n.node_id for n in floor_map.nodes if n.node_type == NodeType.STORY)
    if not node_ids:
        if observed_true_node_id is not None:
            raise ValueError("cannot assign a fate absent from the supplied map")
        return None
    if len(node_ids) != 3:
        raise ValueError("supplied map must contain exactly three fate nodes")
    assumptions = ()
    if observed_true_node_id is not None:
        if observed_true_node_id not in node_ids:
            raise ValueError("observed true location is not a fate node")
        true_node_id = observed_true_node_id
    else:
        if not allow_synthetic or rng is None:
            raise ValueError("true fate requires observation or an explicit synthetic prior")
        true_node_id = rng.choice(node_ids)
        assumptions = ("unverified_uniform_true_fate_position",)
    return FateContext(node_ids, true_node_id,
                       true_fate_marked=FATE_SANDBOXES <= frozenset(inventory),
                       assumptions=assumptions)


def enter_fate(context: FateContext, node_id: str) -> tuple[FateContext, str]:
    """Reveal a fixed scene only after the movement/entry has really happened."""
    if node_id not in context.node_ids:
        raise ValueError("not a fate node")
    if context.true_node_id is None:
        raise ValueError("an observation-only context cannot reveal a hidden real scene")
    scene = TRUE_FATE_NAME if node_id == context.true_node_id else FALSE_FATE_NAME
    seen = dict(context.observed_scenes)
    seen[node_id] = scene
    return replace(context, observed_scenes=tuple(sorted(seen.items()))), scene


def mark_true_fate(context: FateContext) -> FateContext:
    """A paid mark, both sandboxes, or leaving true fate reveals its location."""
    if context.true_node_id is None:
        raise ValueError("cannot mark an unobserved real coordinate in a belief context")
    return replace(context, true_fate_marked=True)


def fate_marked_node_ids(context: FateContext | None) -> frozenset[str]:
    return (frozenset({context.true_node_id})
            if context and context.true_fate_marked and context.true_node_id is not None
            else frozenset())


def fate_observed_scene(context: FateContext, node_id: str) -> str | None:
    """Map visibility alone never reveals a fate scene's identity."""
    if node_id not in context.node_ids:
        raise ValueError("not a fate node")
    if context.true_fate_marked and node_id == context.true_node_id:
        return TRUE_FATE_NAME
    return dict(context.observed_scenes).get(node_id)


def fate_belief_context(
    context: FateContext | None, *, revealed_node_ids: Iterable[str] | None = None,
) -> FateContext | None:
    """Remove private truth and optionally the unobserved fate coordinates.

    Live simulator projections must provide their revealed-node set. Unknown
    coordinates become stable unnamed slots, since knowing three fates exist
    does not locate them among all of the map's still-unknown mystery nodes.
    Omitting the set is appropriate only when all three locations are public.
    """
    if context is None:
        return None
    known_true = next((n for n, s in context.observed_scenes if s == TRUE_FATE_NAME), None)
    if context.true_fate_marked:
        known_true = context.true_node_id
    if known_true is None:
        false_nodes = {n for n, s in context.observed_scenes if s == FALSE_FATE_NAME}
        remaining = set(context.node_ids) - false_nodes
        if len(remaining) == 1:
            known_true = next(iter(remaining))
    if revealed_node_ids is None:
        return replace(context, true_node_id=known_true)
    visible = set(revealed_node_ids) | {n for n, _ in context.observed_scenes}
    if context.true_fate_marked and context.true_node_id is not None:
        visible.add(context.true_node_id)
    public_ids = tuple(sorted(n for n in context.node_ids if n in visible))
    private_ids = tuple(n for n in context.node_ids if n not in visible)
    placeholders = tuple(f"__unobserved_fate_{i+1}__" for i in range(len(private_ids)))
    if known_true in private_ids:
        # This occurs only after observing both false fates: the player knows
        # the unnamed remaining slot is true, not its actual grid coordinate.
        known_true = placeholders[private_ids.index(known_true)]
    return replace(context, node_ids=public_ids+placeholders, true_node_id=known_true)


def fate_public_map(floor_map: FloorMap, context: FateContext) -> FloorMap:
    """Remove hidden scene/options/stage payloads even for visible STORY icons.

    This helper only sanitizes fates. The simulator must still mask ordinary
    hidden nodes, future maps, seeds, and the FateContext itself. A node already
    transformed into a boss or empty by settlement is left intact.
    """
    nodes = []
    for node in floor_map.nodes:
        if node.node_type != NodeType.STORY:
            nodes.append(node)
            continue
        scene = (fate_observed_scene(context, node.node_id)
                 if node.node_id in context.node_ids else None)
        if scene is None:
            node = replace(node, event_name=FATE_PUBLIC_NAME, options=(),
                           auto_effect=ResourceDelta(), stage_id=None,
                           requires_observation=True)
        else:
            node = replace(node, event_name=scene)
            if node.node_id not in dict(context.observed_scenes):
                # A marker identifies the scene, not its unopened menu.
                node = replace(node, options=(), auto_effect=ResourceDelta(), stage_id=None,
                               requires_observation=True)
        nodes.append(node)
    return replace(floor_map, nodes=tuple(nodes), fingerprint="")


def sample_fate_belief(
    context: FateContext, *, rng: random.Random, allow_synthetic: bool = False,
) -> FateContext:
    """A search determinization samples only identities consistent with sight.

    Always sanitize its input first, even if a caller accidentally passes the
    real context. This is a declared uniform prior, never a look at real truth.
    """
    belief = fate_belief_context(context)
    if belief.true_node_id is not None:
        return belief
    if not allow_synthetic:
        raise ValueError("unobserved fate belief requires explicit synthetic sampling")
    observed_false = {n for n, s in belief.observed_scenes if s == FALSE_FATE_NAME}
    candidates = tuple(n for n in belief.node_ids if n not in observed_false)
    return replace(belief, true_node_id=rng.choice(candidates),
                   assumptions=tuple(dict.fromkeys(belief.assumptions +
                       ("synthetic_uniform_fate_belief_from_observations",))))


__all__ = ["FateContext", "TRUE_FATE_NAME", "FATE_PUBLIC_NAME", "TRUE_FATE_STAGE",
           "fate_route_available", "initialize_fates", "enter_fate", "mark_true_fate",
           "fate_marked_node_ids", "fate_observed_scene", "fate_belief_context", "fate_public_map",
           "sample_fate_belief"]
