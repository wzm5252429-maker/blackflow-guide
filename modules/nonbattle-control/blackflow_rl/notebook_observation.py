"""Observation adapter for cargo_14, without synthetic layers or probabilities.

Client facts: layer_meet_node_types(max=99) and
drop_extra_pool(id=layer_meet_node_types). Neither buff specifies a per-layer
probability. Node exploration, retroactive credit and same-battle ordering are
not inferred here. Inputs must be actual displayed layers/extra-drop outcomes.
The caller validates item IDs against its catalog before applying grant IDs.
"""
from __future__ import annotations

from dataclasses import dataclass, replace


NOTEBOOK_ID = "rogue_6_relic_cargo_14"
MAX_OBSERVED_LAYER = 99


def _layer(value: int | None) -> None:
    if value is not None and (type(value) is not int or not 0 <= value <= MAX_OBSERVED_LAYER):
        raise ValueError("observed notebook layer must be an integer in 0..99 or unknown")


def _source(value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("an observation source is required")


@dataclass(frozen=True, slots=True)
class NotebookAudit:
    operation: str
    source: str
    layer: int | None = None
    battle_id: str | None = None


@dataclass(frozen=True, slots=True)
class ObservedExtraDrop:
    """Empty item_ids means observed no EXTRA drop, distinct from unobserved.

    A full battle's ordinary reward list cannot be substituted for this input.
    Attribution to the notebook's extra pool must itself have been observed.
    layer_at_roll is optional because UI layer and drop ordering may differ.
    """
    item_ids: tuple[str, ...]
    source: str
    layer_at_roll: int | None = None

    def __post_init__(self):
        _source(self.source)
        _layer(self.layer_at_roll)
        if not isinstance(self.item_ids, tuple) or any(
                not isinstance(item_id, str) or not item_id.strip() for item_id in self.item_ids):
            raise ValueError("observed extra item IDs must be a tuple of nonempty strings")
        if len(set(self.item_ids)) != len(self.item_ids):
            raise ValueError("a unique collectible cannot occur twice in one extra-drop observation")


@dataclass(frozen=True, slots=True)
class ResolvedNotebookBattle:
    battle_id: str
    stage_id: str | None
    outcome: ObservedExtraDrop


@dataclass(frozen=True, slots=True)
class NotebookObservation:
    held: bool = False
    observed_layer: int | None = None
    layer_source: str | None = None
    resolved_battles: tuple[ResolvedNotebookBattle, ...] = ()

    def __post_init__(self):
        _layer(self.observed_layer)
        if self.observed_layer is not None and not self.held:
            raise ValueError("an unheld notebook has no observed layer")
        if self.observed_layer is not None:
            _source(self.layer_source)
        if len({row.battle_id for row in self.resolved_battles}) != len(self.resolved_battles):
            raise ValueError("a battle may only have one resolved extra-drop observation")


@dataclass(frozen=True, slots=True)
class NotebookUpdate:
    state: NotebookObservation
    audit: tuple[NotebookAudit, ...] = ()
    grant_item_ids: tuple[str, ...] = ()


def on_notebook_acquired(*, source: str, observed_layer: int | None = None) -> NotebookUpdate:
    """Initialize ownership without assuming layer zero or retroactive credit."""
    _source(source)
    _layer(observed_layer)
    state=NotebookObservation(True, observed_layer, source if observed_layer is not None else None)
    audits=[NotebookAudit("notebook_acquired", source, observed_layer)]
    if observed_layer is None:
        audits.append(NotebookAudit("unresolved_relic_layer", "notebook:initial_layer_and_retroactivity_unknown"))
    audits.append(NotebookAudit("unresolved_drop_modifier", "notebook:per_layer_extra_pool_probability_unknown", observed_layer))
    return NotebookUpdate(state, tuple(audits))


def observe_notebook_layer(state: NotebookObservation, layer: int, *, source: str) -> NotebookUpdate:
    """Import the displayed counter; never derive it from simulator node types."""
    if not state.held:
        raise ValueError("cannot observe an unheld notebook")
    _layer(layer)
    if layer is None:
        raise ValueError("a layer observation must be concrete")
    _source(source)
    return NotebookUpdate(replace(state, observed_layer=layer, layer_source=source),
        (NotebookAudit("observed_relic_layer", source, layer),))


def after_battle(state: NotebookObservation, *, battle_id: str,
                 stage_id: str | None = None,
                 observed_extra_drop: ObservedExtraDrop | None = None) -> NotebookUpdate:
    """Return only observed extra rewards, at most once per actual battle ID.

    Missing outcomes produce an unresolved-modifier audit, never a Bernoulli
    draw. Observed layer counters are not changed by this callback. A repeated
    identical result is idempotent; conflicting repeats must be reconciled.
    """
    _source(battle_id)
    if not state.held:
        if observed_extra_drop is not None:
            raise ValueError("notebook extra drops require actual notebook ownership")
        return NotebookUpdate(state)
    previous=next((row for row in state.resolved_battles if row.battle_id == battle_id), None)
    if previous is not None:
        if observed_extra_drop is not None and (
                previous.stage_id != stage_id or previous.outcome != observed_extra_drop):
            raise ValueError("conflicting extra-drop observations for the same battle")
        return NotebookUpdate(state)
    if observed_extra_drop is None:
        return NotebookUpdate(state, (NotebookAudit("unresolved_drop_modifier",
            "notebook:extra_pool_result_unobserved_probability_unknown",
            state.observed_layer, battle_id),))
    resolved=ResolvedNotebookBattle(battle_id, stage_id, observed_extra_drop)
    updated=replace(state, resolved_battles=state.resolved_battles+(resolved,))
    return NotebookUpdate(updated, (NotebookAudit("observed_notebook_extra_drop",
        observed_extra_drop.source, observed_extra_drop.layer_at_roll, battle_id),),
        observed_extra_drop.item_ids)
