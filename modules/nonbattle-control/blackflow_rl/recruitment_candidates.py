"""Fixed battle recruitment alternatives which resolve to one actual ticket.

The user observed two alternatives, choose one, after an ordinary battle.
Radio adds one alternative, not one claimable ticket. The base alternative
count for elite/boss battles is not established: their one-option default is
an explicitly labelled model assumption, and an observed count can override
it. Duel, resident and ordinary chase rewards must not borrow this
implementation. Boss chase has a separate entry point with explicit source
and floor assumptions; it must never be disguised as an ordinary map node.

The existing empirical ticket composition supplies a synthetic independent
draw for each slot, with replacement. Neither duplicate behaviour nor the
joint distribution of alternatives is established by those observations.
The boss ticket observations cover floor III only: 125 named direct-ticket
settlements among 128 battles, excluding three radio selection screens whose
candidate names were not saved. The source explicitly forbids extrapolating
those observations to floors V/VI. Existing ordinary-boss callers do not
provide their floor, so their composition label alone must not be presented
as evidence for those later floors. See the original pool notes and the audit
in docs/economic-relic-hook-audit-2026-09-08.md.

Groups contain item IDs, never held inventory instances. Reading a menu or
choosing a slot does not acquire, recruit or retain anything. The engine must
atomically persist the resolved group and acquire exactly the returned ticket
once. Already rolled part rewards and other ticket groups must be preserved.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

from .domain import EventOption, NodeType
from .recruitment import sample_battle_recruit_ticket


ORDINARY_COUNT_EVIDENCE = "USER_OBSERVED_RULE:ordinary_battle_two_choose_one"
UNKNOWN_COUNT_EVIDENCE = "UNVERIFIED_BASE_CANDIDATE_COUNT"
OBSERVED_COUNT_EVIDENCE = "OBSERVED_BATTLE_RECRUIT_BASE_CANDIDATE_COUNT"
JOINT_DISTRIBUTION_ASSUMPTION = "SYNTHETIC_INDEPENDENT_DRAWS_WITH_REPLACEMENT"
RADIO_EVIDENCE = "CLIENT:battle_extra_recruit_ticket:count=1"
BOSS_CHASE_SOURCE_ASSUMPTION = "SYNTHETIC_BOSS_CHASE_SOURCE_MAPPING_FROM_ORDINARY_F3"
BOSS_CHASE_CROSS_FLOOR_ASSUMPTION = "SYNTHETIC_BOSS_CHASE_CROSS_FLOOR_FROM_ORDINARY_F3"


@dataclass(frozen=True, slots=True)
class RecruitTicketCandidateGroup:
    group_id: str
    node_type: NodeType
    candidate_item_ids: tuple[str, ...]
    base_candidate_count: int
    base_count_evidence: str
    radio_owned: bool
    pool_id: str
    sample_count: int
    chosen_index: int | None = None
    declined: bool = False
    sampling_assumption: str = JOINT_DISTRIBUTION_ASSUMPTION
    composition_evidence: str = "EMPIRICAL_CONDITIONAL_TICKET_COMPOSITION"
    source_kind: str | None = None

    def __setstate__(self, values):
        # Older frozen-slot snapshots end at composition_evidence. Populate
        # appended defaults so both property access and dataclasses.replace
        # work after loading those snapshots.
        from dataclasses import MISSING, fields
        definitions = fields(self)
        if len(values) > len(definitions):
            raise ValueError("snapshot has an unsupported newer candidate schema")
        for index, definition in enumerate(definitions):
            if index < len(values):
                value = values[index]
            elif definition.default is not MISSING:
                value = definition.default
            elif definition.default_factory is not MISSING:
                value = definition.default_factory()
            else:
                raise ValueError("snapshot omits a required candidate field")
            object.__setattr__(self, definition.name, value)

    def __post_init__(self):
        if not isinstance(self.group_id, str) or not self.group_id:
            raise ValueError("battle candidate group requires a source occurrence ID")
        if self.source_kind is not None and (
                not isinstance(self.source_kind, str) or not self.source_kind.strip()):
            raise ValueError("source kind must be a nonempty source label or None")
        if self.node_type not in (NodeType.BATTLE_NORMAL, NodeType.BATTLE_ELITE, NodeType.BATTLE_BOSS):
            raise ValueError("no established candidate modifier for this battle source")
        if type(self.base_candidate_count) is not int or self.base_candidate_count < 1:
            raise ValueError("base candidate count must be a positive integer")
        if type(self.radio_owned) is not bool or type(self.declined) is not bool:
            raise ValueError("radio and decline flags must be booleans")
        if (not isinstance(self.candidate_item_ids, tuple)
                or len(self.candidate_item_ids) != self.base_candidate_count + int(self.radio_owned)
                or any(not isinstance(item_id, str) or not item_id for item_id in self.candidate_item_ids)):
            raise ValueError("fixed candidate slots must match the established count plus modifier")
        if self.chosen_index is not None:
            if type(self.chosen_index) is not int or not 0 <= self.chosen_index < len(self.candidate_item_ids):
                raise ValueError("chosen slot is outside the fixed candidate group")
            if self.declined:
                raise ValueError("a candidate group cannot be both selected and declined")

    @property
    def settled(self) -> bool:
        return self.declined or self.chosen_index is not None

    @property
    def source(self) -> str:
        source_kind = getattr(self, "source_kind", None)
        return source_kind if source_kind is not None else self.node_type.value

    @property
    def selected_item_id(self) -> str | None:
        return None if self.chosen_index is None else self.candidate_item_ids[self.chosen_index]


def make_battle_recruit_group(
    rng, group_id: str, node_type: NodeType, *, radio_owned: bool = False,
    observed_base_count: int | None = None,
    existing_group: RecruitTicketCandidateGroup | None = None,
) -> RecruitTicketCandidateGroup:
    """Roll alternatives once; an existing occurrence is never rerolled.

    ``observed_base_count`` is an explicit observation, not a tuning parameter.
    Acquiring/losing radio after the group was rolled cannot change its slots.
    Supplying an existing group returns it unchanged, including after settling.
    """
    if existing_group is not None:
        if existing_group.group_id != group_id or existing_group.node_type != node_type:
            raise ValueError("candidate group belongs to a different source occurrence")
        return existing_group
    if node_type not in (NodeType.BATTLE_NORMAL, NodeType.BATTLE_ELITE, NodeType.BATTLE_BOSS):
        raise ValueError("duel, resident, event and chase rewards have separate rules")
    if type(radio_owned) is not bool:
        raise ValueError("radio ownership must be a boolean")
    if observed_base_count is not None:
        if type(observed_base_count) is not int or observed_base_count < 1:
            raise ValueError("observed base candidate count must be a positive integer")
        base, evidence = observed_base_count, OBSERVED_COUNT_EVIDENCE
    elif node_type == NodeType.BATTLE_NORMAL:
        base, evidence = 2, ORDINARY_COUNT_EVIDENCE
    else:
        base, evidence = 1, UNKNOWN_COUNT_EVIDENCE
    if not isinstance(group_id, str) or not group_id:
        raise ValueError("battle candidate group requires a source occurrence ID")
    draws = tuple(sample_battle_recruit_ticket(rng, node_type)
                  for _ in range(base + int(radio_owned)))
    return RecruitTicketCandidateGroup(
        group_id=group_id, node_type=node_type,
        candidate_item_ids=tuple(draw.item_id for draw in draws),
        base_candidate_count=base, base_count_evidence=evidence,
        radio_owned=radio_owned, pool_id=draws[0].pool_id,
        sample_count=draws[0].sample_count,
    )


def make_normal_battle_group(rng, group_id: str, *, radio_owned: bool = False,
                             existing_group: RecruitTicketCandidateGroup | None = None):
    return make_battle_recruit_group(rng, group_id, NodeType.BATTLE_NORMAL,
        radio_owned=radio_owned, existing_group=existing_group)


def make_boss_chase_recruit_group(
    rng, group_id: str, *, floor: int, radio_owned: bool = False,
    observed_base_count: int | None = None,
) -> RecruitTicketCandidateGroup:
    """Model one boss-chase ticket opportunity, with explicit extrapolation.

    The floor-III ordinary-boss composition is the empirical reference, not
    a directly observed chase pool. Applying it to floor III chase is a
    source mapping assumption; floors V/VI also require a cross-floor prior.
    Ordinary chase floors I/II/IV are rejected before any draw.

    The base candidate count remains the explicit one-option prior unless
    supplied from an actual observation. An observed count does not establish
    the candidate identities or their joint distribution. Radio adds one
    candidate; selecting the immutable result still yields only one ticket.
    The caller must persist this group once per actual chase occurrence and
    reuse it when rendering the menu, rather than invoking this factory again.
    """
    if type(floor) is not int or floor not in (3, 5, 6):
        raise ValueError("boss-chase recruitment requires floor III, V or VI")
    group = make_battle_recruit_group(rng, group_id, NodeType.BATTLE_BOSS,
        radio_owned=radio_owned, observed_base_count=observed_base_count)
    assumption = (BOSS_CHASE_SOURCE_ASSUMPTION if floor == 3
                  else BOSS_CHASE_CROSS_FLOOR_ASSUMPTION)
    return replace(group, source_kind=f"boss_chase:floor={floor}",
        composition_evidence=(f"{assumption}:floor={floor}:"
                              f"EMPIRICAL_REFERENCE:ordinary_boss_floor3:n={group.sample_count}"))


def select_recruit_candidate(group: RecruitTicketCandidateGroup,
                             index: int) -> tuple[RecruitTicketCandidateGroup, str]:
    """Resolve one group to one ticket, without performing an acquisition."""
    if group.settled:
        raise ValueError("this recruitment reward was already resolved")
    if type(index) is not int or not 0 <= index < len(group.candidate_item_ids):
        raise ValueError("a displayed candidate slot is required")
    return replace(group, chosen_index=index), group.candidate_item_ids[index]


def decline_recruit_candidates(group: RecruitTicketCandidateGroup) -> RecruitTicketCandidateGroup:
    if group.settled:
        raise ValueError("this recruitment reward was already resolved")
    return replace(group, declined=True)


def recruit_candidate_options(
    group: RecruitTicketCandidateGroup, *, name_of: Callable[[str], str] | None = None,
) -> tuple[EventOption, ...]:
    """Candidates have no instance ID, so they cannot be stored or recruited."""
    if group.settled:
        return ()
    return tuple(EventOption(
        f"select_recruit_ticket:{group.group_id}:{index}",
        "选择" + (name_of(item_id) if name_of else item_id),
        operation="select_recruit_ticket", item_id=item_id, ends_node=False,
        description="从本组候选中领取一张招募券；领取后可招募或留存。",
    ) for index, item_id in enumerate(group.candidate_item_ids))


def resolve_recruit_candidate_option(
    group: RecruitTicketCandidateGroup, option: EventOption, *,
    name_of: Callable[[str], str] | None = None,
) -> tuple[RecruitTicketCandidateGroup, str]:
    """Validate the exact fixed menu, including duplicate item IDs by slot."""
    options = recruit_candidate_options(group, name_of=name_of)
    if option not in options:
        raise ValueError("a current unmodified candidate option is required")
    return select_recruit_candidate(group, options.index(option))
