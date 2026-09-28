"""Concrete rogue_6 items with field-level evidence and explicit unknown prices.

Client ``relics`` is a buff table, not the collection inventory.  In particular,
the 13 difficulty-upgradable relics have four IDs each but only one identity.
The compact snapshot is generated from the pinned complete client table and
public shop observations; empirical frequencies are never server probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


CATALOG_PATH = Path(__file__).resolve().parents[1] / "data" / "evidence" / "rogue6_economy_catalog_v1.json"
SCRAP_CATEGORIES = frozenset({"MOVE", "GOODS", "PASSIVE"})


@dataclass(frozen=True, slots=True)
class ItemDefinition:
    item_id: str
    canonical_id: str
    name: str
    category: str
    rarity: str
    usage: str
    can_sacrifice: bool
    base_buy_price: int | None
    sell_price: int | None
    shop_eligible: bool | None
    price_evidence: str
    eligibility_evidence: str
    unlock_condition: str | None
    minimum_difficulty: int = 0
    move_range: tuple[tuple[int, int], ...] = ()
    move_range_type: str | None = None
    move_uses: int | None = None
    move_ap: int | None = None
    move_target_types: tuple[str, ...] = ()
    random_move: bool = False
    expires_on_floor_change: bool = False
    passive_trigger: str | None = None
    passive_uses: int | None = None
    client_buffs: tuple[Mapping[str, Any], ...] = ()
    unknown_fields: tuple[str, ...] = ()

    @property
    def is_scrap(self) -> bool:
        return self.category in SCRAP_CATEGORIES

    @property
    def counts_as_relic(self) -> bool:
        return self.category == "RELIC"


@dataclass(frozen=True, slots=True)
class ItemCatalog:
    items: Mapping[str, ItemDefinition]
    variants: Mapping[str, tuple[ItemDefinition, ...]]
    observed_pools: Mapping[str, Mapping[str, Any]]
    shop_rules: Mapping[str, Any]
    profile_rules: Mapping[str, Any]
    sources: Mapping[str, Any]

    def __getitem__(self, item_id: str) -> ItemDefinition:
        return self.items[item_id]

    def for_category(self, category: str, *, difficulty: int = 15) -> tuple[ItemDefinition, ...]:
        """Return one applicable variant for each identity at this difficulty."""
        if not 0 <= difficulty <= 15:
            raise ValueError("difficulty must be between 0 and 15")
        result: dict[str, ItemDefinition] = {}
        for item in self.items.values():
            if item.category != category or item.minimum_difficulty > difficulty:
                continue
            old = result.get(item.canonical_id)
            if old is None or item.minimum_difficulty > old.minimum_difficulty:
                result[item.canonical_id] = item
        return tuple(result[key] for key in sorted(result))

    def variant(self, item_id: str, *, difficulty: int = 15) -> ItemDefinition:
        item = self[item_id]
        if not 0 <= difficulty <= 15:
            raise ValueError("difficulty must be between 0 and 15")
        return max((x for x in self.variants[item.canonical_id] if x.minimum_difficulty <= difficulty),
                   key=lambda x: x.minimum_difficulty)

    def shop_items(self, category: str, *, difficulty: int = 15) -> tuple[ItemDefinition, ...]:
        """Only known eligible items; absence from observations remains unknown."""
        return tuple(item for item in self.for_category(category, difficulty=difficulty)
                     if item.shop_eligible is True)

    def observed_weights(self, pool_id: str, *, subpool: str | None = None) -> tuple[tuple[str, int], ...]:
        """Observed occurrence counts, conditional on this pool, not drop rates.

        A shop subpool, or a battle reward screenshot, supplies no evidence for
        whether a reward appears at all.  Callers must model that separately.
        """
        pool = self.observed_pools[pool_id]
        distribution = pool.get("observed_distribution", {})
        if subpool is not None:
            distribution = next((s for s in distribution.get("subpools", ())
                                 if s["id"] == subpool), {})
        return tuple((row["itemId"], int(row["occurrenceCount"]))
                     for row in distribution.get("members", ())
                     if int(row["occurrenceCount"]) > 0)


def synthetic_relic_price(item: ItemDefinition) -> int:
    """Explicit common-tariff prior where rogue_6 per-item quotes are absent.

    Do not call this in evidence/replay execution: use observed shop prices.
    Special/ending relics have no synthetic default price.
    """
    if item.category != "RELIC" or item.rarity not in {"NORMAL", "RARE", "SUPER_RARE"}:
        raise ValueError(f"no common relic tariff for {item.item_id}")
    return {"NORMAL": 8, "RARE": 12, "SUPER_RARE": 16}[item.rarity]


def refreshed_shop_cost(refresh_count: int, *, unlocked: bool = True) -> int | None:
    """Cost of the next refresh after 0..4 already performed at this node."""
    if refresh_count < 0:
        raise ValueError("refresh_count cannot be negative")
    return 4 * (refresh_count + 1) if unlocked and refresh_count < 4 else None


def bonus_shop_slots(*, difficulty: int, full_technology: bool,
                     cumulative_bank_investment: int = 0) -> int:
    """Known additions to BATTLE_SHOP's unknown unmodified slot layout."""
    if not 0 <= difficulty <= 15 or cumulative_bank_investment < 0:
        raise ValueError("invalid difficulty or bank investment")
    technology = 2 if full_technology and difficulty >= 6 else 0
    bank = int(cumulative_bank_investment >= 125) + int(cumulative_bank_investment >= 500)
    return technology + bank


@lru_cache(maxsize=4)
def load_catalog(path: str | Path = CATALOG_PATH) -> ItemCatalog:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("schema_version") != 1:
        raise ValueError("unsupported economy catalog schema")
    items = {}
    for record in raw["items"]:
        row = dict(record)
        for key in ("move_target_types", "client_buffs", "unknown_fields"):
            row[key] = tuple(row.get(key, ()))
        row["move_range"] = tuple(tuple(offset) for offset in row.get("move_range", ()))
        item = ItemDefinition(**row)
        if item.item_id in items:
            raise ValueError(f"duplicate catalog ID: {item.item_id}")
        if item.base_buy_price is not None and (item.base_buy_price < 0 or item.shop_eligible is False):
            raise ValueError(f"invalid price/eligibility for {item.item_id}")
        items[item.item_id] = item
    variants = {}
    for item in items.values():
        variants.setdefault(item.canonical_id, []).append(item)
    return ItemCatalog(
        items=MappingProxyType(items),
        variants=MappingProxyType({key: tuple(value) for key, value in variants.items()}),
        observed_pools=MappingProxyType(raw["observed_pools"]),
        shop_rules=MappingProxyType(raw["shop_rules"]),
        profile_rules=MappingProxyType(raw["profile_rules"]),
        sources=MappingProxyType(raw["sources"]),
    )
