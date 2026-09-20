"""Explicit region economy rules, conditional on a known affected path.

PRTS supplies region footprints and appearance rates. The optional generator
labels its still-unknown center/type selection law as a synthetic prior.
Pass the number of affected grid nodes actually traversed, not merely the
number of MOVE actions.  PRTS distinguishes passing a node from entering it.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from .domain import ResourceDelta, NodeType


SOURCE = "https://prts.wiki/w/沉沦者的黑流树海/黑流数据库#乌托邦幸福论"
EVIDENCE = "B_PRTS_EXPLICIT"
EARLY_MIDDLE_RADIUS = 2
LATE_RADIUS = 3
NORMAL_IDEOLOGIES = ("黑流地脉", "倾斜沙丘", "去温栏", "微型胶囊", "储藏室",
                     "易碎同盟", "停止点", "弥散虚雾")
NEGATIVE_POLICIES = ("改良", "修正", "激进")
REGION_SELECTION_PRIOR = "SYNTHETIC_ONE_REGION_UNIFORM_ELITE_CENTER_IDEOLOGY_POLICY"


@dataclass(frozen=True, slots=True)
class RegionState:
    source_node_id: str
    ideology: str
    policy: str | None
    covered_node_ids: frozenset[str]
    effect_tier: int
    radius: int
    removable: bool = True
    active: bool = True
    selection_evidence: str = "OBSERVED_REGION"

    def affects(self, node_id: str) -> bool:
        return self.active and node_id in self.covered_node_ids

    def clear_source(self, completed_node_id: str):
        return replace(self, active=False) if self.removable and completed_node_id == self.source_node_id else self


def region_coverage(floor_map, source_node_id: str, difficulty: int) -> frozenset[str]:
    """Manhattan 2/3 footprint, established by PRTS' 13/25-square SVGs."""
    tier = effect_tier(difficulty)
    if not tier:
        return frozenset()
    center = floor_map.node(source_node_id)
    radius = LATE_RADIUS if difficulty >= 12 else EARLY_MIDDLE_RADIUS
    return frozenset(node.node_id for node in floor_map.nodes
                     if abs(node.row - center.row) + abs(node.col - center.col) <= radius)


def generate_region(rng, floor_map, difficulty: int, *, eligible_center_ids=None):
    """One-region synthetic placement; event-only beneficial soil is excluded.

    The appearance rate and footprint are sourced. Region count, center,
    ideology and policy draws remain explicitly identified modeling priors.
    Callers must not use this for an event-created or observed region.
    """
    probability = generation_probability(difficulty, floor_map.floor)
    if probability == 0 or rng.random() >= probability:
        return None
    eligible = frozenset(eligible_center_ids) if eligible_center_ids is not None else None
    candidates = [node for node in floor_map.nodes if node.node_type == NodeType.BATTLE_ELITE
                  and (eligible is None or node.node_id in eligible)]
    if not candidates:
        return None
    center = rng.choice(candidates)
    tier = effect_tier(difficulty)
    return RegionState(center.node_id, rng.choice(NORMAL_IDEOLOGIES),
                       rng.choice(NEGATIVE_POLICIES) if difficulty >= 6 else None,
                       region_coverage(floor_map, center.node_id, difficulty), tier,
                       LATE_RADIUS if difficulty >= 12 else EARLY_MIDDLE_RADIUS,
                       selection_evidence=REGION_SELECTION_PRIOR)


def effect_tier(difficulty: int) -> int:
    if not 0 <= difficulty <= 15:
        raise ValueError("difficulty must be 0..15")
    return 0 if difficulty < 2 else 1 if difficulty < 6 else 2 if difficulty < 12 else 3


def generation_probability(difficulty: int, floor: int) -> float:
    """Probability that a normal floor generates a region, not its type law."""
    if floor not in range(1, 7):
        raise ValueError("only normal floors 1..6 have this published law")
    tier = effect_tier(difficulty)
    if not tier:
        return 0.0
    return ((0.0, 0.1, 0.15, 0.2, 0.2, 0.0) if difficulty < 6 else
            (0.1, 0.2, 0.4, 0.4, 0.4, 0.3))[floor - 1]


@dataclass(frozen=True, slots=True)
class RegionShopTerms:
    sell_multiplier: float = 1.0
    slot_reduction: int = 0
    # Once stock has been generated under this region, clearing the source
    # does not restore lost slots. The selling modifier is region-dependent.
    generated_slot_loss_persists: bool = False


def shop_terms(ideology: str | None, difficulty: int) -> RegionShopTerms:
    tier = effect_tier(difficulty)
    if ideology not in ("storage_room", "储藏室", "“储藏室”") or not tier:
        return RegionShopTerms()
    return RegionShopTerms((0.9, 0.7, 0.5)[tier - 1], tier, True)


def extra_battle_scraps(ideology: str | None, *, won: bool) -> int:
    """One additional random scrap for victory inside Hope's Fertile Soil.

    Pool membership/weights are not supplied by this rule. The caller must
    retain its actual observed pool or explicitly identified sampling prior.
    """
    return int(won and ideology in ("hope_soil", "希望的沃土", "“希望的沃土”"))


def apply_policy(engine, state, policy: str | None, *, affected_nodes: int,
                 immune_to_negative_policy: bool = False):
    """Apply a known policy to every actually traversed affected grid node.

    Moving via a vehicle does not establish that intervening grid nodes were
    traversed; supply the real trajectory rather than geometric distance.
    Positive appraisal affects GOODS only and ignores policy immunity.
    """
    if affected_nodes < 0:
        raise ValueError("affected_nodes cannot be negative")
    if not affected_nodes or policy is None:
        return state
    normalized = {"改良": "improve", "修正": "correct", "激进": "radical",
                  "增益": "benefit"}.get(policy, policy)
    if normalized == "benefit":
        state = replace(state, item_instances=tuple(
            replace(item, appraisal=(max(0, min(999, item.appraisal + 2 * affected_nodes))
                if item.item_id == "rogue_6_scrap_G_05" else item.appraisal + 2 * affected_nodes))
            if item.category == "GOODS" else item for item in state.item_instances))
        return engine.entry(state, "region_appraisal", quantity=affected_nodes,
                            source="hope_soil:benefit:" + EVIDENCE)
    if immune_to_negative_policy:
        return state
    if normalized == "improve":
        change = ResourceDelta(gold=-min(state.resources.gold, 2 * affected_nodes))
    elif normalized == "correct":
        change = ResourceDelta(shield=-min(state.resources.shield, affected_nodes))
    elif normalized == "radical":
        change = ResourceDelta(hp=-min(max(0, state.resources.hp - 1), affected_nodes))
    else:
        raise ValueError("unknown region policy: " + str(policy))
    return engine.apply(state, change, "region_policy:" + normalized + ":" + EVIDENCE)
