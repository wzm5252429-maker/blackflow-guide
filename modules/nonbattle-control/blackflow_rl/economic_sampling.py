"""Conditional empirical item sampling and separately named model assumptions.

Neither OCR counts nor unknown slot priors establish a live-game distribution.
Known candidate pools are used before any explicit uniform fallback.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .catalog import ItemCatalog, bonus_shop_slots, synthetic_relic_price


SYNTHETIC_SHOP_LAYOUT = {
    "battle_shop_base_other_relic_slots": 2,
    "battle_shop_training_slots": 1,
    "battle_shop_scrap_slots": 1,
    "battle_shop_recruit_slots": 1,
    "battle_shop_tool_slots": 1,
    "scrap_shop_categories": ("MOVE", "MOVE", "MOVE", "GOODS", "GOODS", "PASSIVE", "PASSIVE"),
    "status": "UNVERIFIED_BASE_LAYOUT_PLUS_CLIENT_CONFIRMED_EXTRA_SLOTS",
}


def battle_shop_categories(*, difficulty: int, full_technology: bool,
                           bank_investment: int, base_relic_slots: int = 2) -> tuple[str, ...]:
    extra = bonus_shop_slots(difficulty=difficulty, full_technology=full_technology,
                            cumulative_bank_investment=bank_investment)
    # One training item and one scrap in each observed page are independently
    # evidenced; the total base count and types of the extra slots are priors.
    return (("TRAINING_RELIC",) + ("RELIC",) * (base_relic_slots + extra)
            + ("PART", "TICKET", "TOOL"))


def candidate_ids(catalog: ItemCatalog, category: str, *, difficulty: int,
                  owned: Iterable[str] = (), exclude: Iterable[str] = (),
                  shop: bool = False, rarity: Iterable[str] | None = None) -> list[str]:
    occupied = {catalog[x].canonical_id if x in catalog.items else x
                for x in tuple(owned) + tuple(exclude)}
    result = []
    categories = ("MOVE", "GOODS", "PASSIVE") if category == "PART" else (category,)
    for kind in categories:
        for item in catalog.for_category(kind, difficulty=difficulty):
            if kind == "RELIC" and item.canonical_id in occupied:
                continue
            if rarity is not None and item.rarity not in rarity:
                continue
            if shop and item.shop_eligible is not True:
                continue
            if shop and item.base_buy_price is None and kind != "RELIC":
                continue
            if kind == "RELIC" and item.rarity not in ("NORMAL", "RARE", "SUPER_RARE"):
                continue
            if kind in ("MOVE", "GOODS", "PASSIVE") and not shop:
                if item.item_id in ("rogue_6_scrap_G_08", "rogue_6_scrap_G_12"):
                    continue
            result.append(item.item_id)
    return sorted(result)


def pool_members(catalog: ItemCatalog, pool_id: str, *, difficulty: int) -> set[str]:
    pool = catalog.observed_pools.get(pool_id, {})
    return {catalog.variant(x, difficulty=difficulty).item_id
            for x in pool.get("item_ids", ()) if x in catalog.items}


def choose_from_pool(catalog: ItemCatalog, rng, candidates: Iterable[str], *,
                     pool_id: str | None, difficulty: int, subpool: str | None = None):
    """Return (item ID, evidence label); None means the candidate pool exhausted.

    Observed positive support is reweighted after inventory exclusions. The
    pool's observed membership is a conservative incomplete set, not a complete
    list of all possibilities. Unknown weights explicitly use a uniform prior.
    """
    available = set(candidates)
    if pool_id is not None and pool_id in catalog.observed_pools:
        members = pool_members(catalog, pool_id, difficulty=difficulty)
        available &= members
    if not available:
        return None, "pool_exhausted"
    weights = defaultdict(int)
    if pool_id is not None and pool_id in catalog.observed_pools:
        for item_id, count in catalog.observed_weights(pool_id, subpool=subpool):
            if item_id not in catalog.items:
                continue
            resolved = catalog.variant(item_id, difficulty=difficulty).item_id
            if resolved in available:
                weights[resolved] += count
    if weights:
        ordered = sorted(weights)
        return rng.choices(ordered, weights=[weights[x] for x in ordered], k=1)[0], "empirical_conditional_frequency"
    label = ("synthetic_uniform_within_known_pool" if pool_id in catalog.observed_pools else
             "synthetic_uniform_catalog_candidates")
    return rng.choice(sorted(available)), label


def source_pool(category: str, source: str) -> str | None:
    if source == "cultivate":
        return "rogue_6:秘境行商：种子培育"
    if source.startswith("passive:") and category == "RELIC":
        return "pool_treasure"
    if source in ("pool_scrap_3", "pool_scrap_7", "pool_scrap_8", "pool_scrap_9"):
        return source
    lower = source.lower()
    if category in ("PART", "MOVE", "GOODS", "PASSIVE"):
        if "boss" in lower or "final" in lower:
            return "rogue_6:node_battle_boss_scrap"
        if "elite" in lower:
            return "rogue_6:node_battle_elite_scrap"
        if "battle" in lower:
            return "rogue_6:node_battle_normal_scrap"
    if category == "RELIC":
        if "boss" in lower:
            return "pool_boss"
        if "elite" in lower:
            return "rogue_6:紧急作战"
        if "wish" in lower:
            return "rogue_6:得偿所愿"
        if "battle" in lower:
            return "rogue_6:额外掉落池"
        return "pool_treasure"
    return None


def source_pool_assumption(category: str, source: str, pool_id: str | None) -> str | None:
    """Pool observations do not prove that an arbitrary reward uses that pool.

    A caller explicitly specifying an audited pool ID does not use this helper.
    The generic treasure-pool fallback, including white/painted concepts, lacks
    a source-to-pool mapping in the topic client and must be disclosed.
    """
    if category == "RELIC" and pool_id == "pool_treasure":
        return "synthetic_source_pool_assignment:" + source + "->pool_treasure"
    return None


def undiscounted_price(catalog: ItemCatalog, item_id: str) -> tuple[int, str]:
    item = catalog[item_id]
    if item.base_buy_price is not None:
        return item.base_buy_price, item.price_evidence
    if item.category == "RELIC" and item.shop_eligible is True:
        return synthetic_relic_price(item), "synthetic_relic_tariff_8_12_16"
    raise ValueError(f"no price or allowed explicit relic prior for {item_id}")
