"""Observed identities for actual 5/6-star promoted temporary tickets.

The runtime bridge accepts observations but supplies no identity sampler.
The client proves rarity/profession constraints, but not identity weights. A
caller must supply the single identity actually displayed for the actual
ticket; the diagnostic eligibility set must never become a uniform draw pool.

Only a new recruit is supported. An already-owned character is unresolved,
not rerolled or treated as another recruit. Temporary recruitment is distinct
from emergency employment and from the three-star reserve exception.

The bridge persists ObservedTemporaryOffer, invalidates it on storage or
discard, and requires a fresh observation when a stored ticket is reopened.
Part rewards remain available before confirmation to preserve corn-first play.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
import hashlib
import json
from pathlib import Path

from .domain import EventOption
from .operator_economy import recruit_formal


CHARACTER_TABLE_URL = ('https://raw.githubusercontent.com/Kengxxiao/'
    'ArknightsGameData/master/zh_CN/gamedata/excel/character_table.json')
CHARACTER_TABLE_SHA256 = '12030388f7b695062f1e4ce2cab042d8a3be2251d853797bbf74dd2b8b22bcf2'
TOPIC_TABLE_SHA256 = 'aa2b1fc6ba0cc9ee29b9e6a08803550181c3a27189ac449efbad87608880d35b'
TEMPORARY_TICKET_IDS = frozenset({
    'rogue_6_recruit_ticket_temp_5_up', 'rogue_6_recruit_ticket_temp_6_up'})
RULE_SOURCE = 'https://prts.wiki/w/沉沦者的黑流树海'
OBSERVED_CATALOG_PATH = Path(__file__).resolve().parents[1] / 'data' / 'evidence' / 'rogue6_observed_operator_catalog_v1.json'


@dataclass(frozen=True, slots=True)
class TemporaryCharacter:
    char_id: str
    name: str
    rarity: str
    profession: str


@dataclass(frozen=True, slots=True)
class TemporaryCandidateCatalog:
    """Necessary client eligibility constraints, not a sampled game offer."""
    characters: tuple[TemporaryCharacter, ...]
    ticket_constraints: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...]
    character_sha256: str
    topic_sha256: str

    def eligible_ids(self, ticket_item_id: str) -> tuple[str, ...]:
        constraints = next((x for x in self.ticket_constraints if x[0] == ticket_item_id), None)
        if constraints is None:
            return ()
        _, rarities, professions = constraints
        return tuple(x.char_id for x in self.characters
                     if x.rarity in rarities and x.profession in professions)

    def character(self, char_id: str) -> TemporaryCharacter:
        character = next((x for x in self.characters if x.char_id == char_id), None)
        if character is None:
            raise ValueError('character lacks verified ordinary temporary eligibility')
        return character


def load_temporary_candidate_catalog(
    character_table_bytes: bytes, *, expected_character_sha256=CHARACTER_TABLE_SHA256,
    topic_table_bytes: bytes | None = None, expected_topic_sha256=TOPIC_TABLE_SHA256,
) -> TemporaryCandidateCatalog:
    """Validate explicit client inputs; never fetch a changing table at runtime.

    Defaults pin the audited September 8 tables. A caller reviewing another
    revision must explicitly provide its verified hashes. No extraCharIds,
    reserve, collab, Amiya or dedicated-operator exceptions are inferred.
    """
    if topic_table_bytes is None:
        topic_table_bytes = (Path(__file__).resolve().parents[1] / 'source_data' /
                            'roguelike_topic_table_full.json').read_bytes()
    character_hash = hashlib.sha256(character_table_bytes).hexdigest()
    topic_hash = hashlib.sha256(topic_table_bytes).hexdigest()
    if character_hash != expected_character_sha256 or topic_hash != expected_topic_sha256:
        raise ValueError('client table hash differs from the explicitly reviewed revision')
    characters, topic = json.loads(character_table_bytes), json.loads(topic_table_bytes)
    details = topic['details']['rogue_6']
    constraints = []
    for ticket_id in sorted(TEMPORARY_TICKET_IDS):
        item, ticket = details['items'][ticket_id], details['recruitTickets'][ticket_id]
        if item['type'] != 'RECRUIT_TICKET' or item['subType'] != 'TEMP_TICKET':
            raise ValueError('audited ticket is no longer a temporary recruitment ticket')
        constraints.append((ticket_id, tuple(ticket['rarityList']), tuple(ticket['professionList'])))
    eligible = []
    for char_id, predefined in topic['constant']['predefinedChars'].items():
        character = characters.get(char_id)
        if (character is None or predefined.get('canBeFree') is not True
                or predefined.get('recruitType') != 'FREE'
                or character.get('rarity') not in ('TIER_5', 'TIER_6')):
            continue
        eligible.append(TemporaryCharacter(char_id, character['name'],
                                            character['rarity'], character['profession']))
    return TemporaryCandidateCatalog(tuple(sorted(eligible, key=lambda x: x.char_id)),
                                     tuple(constraints), character_hash, topic_hash)


@dataclass(frozen=True, slots=True)
class ObservedTemporaryOffer:
    ticket_instance_id: str
    ticket_item_id: str
    char_id: str
    generation: int
    observation_source: str


@lru_cache(maxsize=1)
def load_pinned_temporary_candidate_catalog():
    """Compact reviewed character fields; membership supplies no draw weights."""
    row = json.loads(OBSERVED_CATALOG_PATH.read_text(encoding='utf-8'))['catalog']
    if row['character_sha256'] != CHARACTER_TABLE_SHA256 or row['topic_sha256'] != TOPIC_TABLE_SHA256:
        raise ValueError('observed character snapshot references another client revision')
    return TemporaryCandidateCatalog(tuple(TemporaryCharacter(**x) for x in row['characters']),
        tuple((item_id,tuple(rarities),tuple(professions)) for item_id,rarities,professions in row['ticket_constraints']),
        row['character_sha256'],row['topic_sha256'])


def _key(ticket_instance_id):
    return 'temporary_recruit_offer:' + ticket_instance_id


def _ticket(state, ticket_instance_id):
    ticket = next((x for x in state.item_instances if x.instance_id == ticket_instance_id), None)
    if (ticket is None or ticket.category != 'RECRUIT_TICKET'
            or ticket.item_id not in TEMPORARY_TICKET_IDS):
        raise ValueError('a held concrete supported temporary ticket is required')
    if (ticket_instance_id not in state.pending_recruit_ticket_ids
            or ticket_instance_id in state.stored_recruit_ticket_ids):
        raise ValueError('temporary ticket must be open in the current recruitment phase')
    return ticket


def observe_temporary_offer(engine, state, catalog: TemporaryCandidateCatalog, *,
                            ticket_instance_id: str, observed_char_id: str,
                            observation_source: str, blocked_operator_ids=frozenset()):
    """Bind one observed identity once; observation itself never recruits."""
    ticket = _ticket(state, ticket_instance_id)
    if not observation_source or not observation_source.strip():
        raise ValueError('an actual observation source is required')
    if observed_char_id not in catalog.eligible_ids(ticket.item_id):
        raise ValueError('observed identity violates supported ticket constraints')
    if observed_char_id in state.formal_operator_ids or observed_char_id in blocked_operator_ids:
        raise ValueError('already-owned or emergency-employed identity requires a separate resolver')
    key = _key(ticket_instance_id)
    if engine.counter(state, key+':active'):
        raise ValueError('the open ticket already has an observed identity; do not reroll it')
    generation = engine.counter(state, key+':generation')+1
    state = engine.set_counter(state, key+':generation', generation)
    state = engine.set_counter(state, key+':active', generation)
    state = engine.set_counter(state, key+':identity:'+observed_char_id, generation)
    offer = ObservedTemporaryOffer(ticket_instance_id, ticket.item_id, observed_char_id,
                                    generation, observation_source.strip())
    state = engine.entry(state, 'temporary_recruit_observed', quantity=1,
        instance_id=ticket_instance_id,
        source=f'OBSERVED_IDENTITY:{observed_char_id}:{generation}:{offer.observation_source}')
    return state, offer


def invalidate_temporary_offer(engine, state, ticket_instance_id: str, *, reason: str):
    """Invalidate before storage/discard; reopening requires a fresh observation."""
    key = _key(ticket_instance_id)
    if not engine.counter(state, key+':active'):
        return state
    state = engine.set_counter(state, key+':active', 0)
    return engine.entry(state, 'temporary_recruit_observation_closed',
                        instance_id=ticket_instance_id, source=reason)


def temporary_recruit_options(engine, state, catalog: TemporaryCandidateCatalog,
                              offers: tuple[ObservedTemporaryOffer, ...], *,
                              blocked_operator_ids=frozenset()) -> tuple[EventOption, ...]:
    """Confirm a displayed real identity, not choose freely from an eligible pool."""
    result = []
    seen_tickets = set()
    for offer in offers:
        if offer.ticket_instance_id in seen_tickets:
            raise ValueError('one temporary identity per actual open ticket')
        seen_tickets.add(offer.ticket_instance_id)
        try:
            ticket = _ticket(state, offer.ticket_instance_id)
        except ValueError:
            continue
        key = _key(offer.ticket_instance_id)
        if (ticket.item_id != offer.ticket_item_id or type(offer.generation) is not int
                or offer.generation <= 0 or engine.counter(state, key+':active') != offer.generation
                or engine.counter(state, key+':identity:'+offer.char_id) != offer.generation
                or offer.char_id not in catalog.eligible_ids(ticket.item_id)
                or offer.char_id in state.formal_operator_ids or offer.char_id in blocked_operator_ids):
            continue
        character = catalog.character(offer.char_id)
        result.append(EventOption(
            f'recruit_temporary:{offer.ticket_instance_id}:{offer.generation}:{offer.char_id}',
            f'临时招募{character.name}（0希望，已进阶）', operation='recruit_temporary',
            item_id=offer.char_id, instance_id=offer.ticket_instance_id, ends_node=False,
            description='确认此券实际显示的干员；消耗一张招募券。'))
    return tuple(result)


def resolve_temporary_recruit(engine, state, catalog: TemporaryCandidateCatalog,
                              offer: ObservedTemporaryOffer, option: EventOption, *,
                              blocked_operator_ids=frozenset()):
    """Consume one ticket and perform one real, free, already-promoted recruit."""
    if option not in temporary_recruit_options(engine, state, catalog, (offer,),
                                               blocked_operator_ids=blocked_operator_ids):
        raise ValueError('active observed temporary recruitment option required')
    source = 'observed_temporary_recruitment:'+offer.observation_source
    state = engine.remove(state, offer.ticket_instance_id, 'consume', source)
    state = replace(state,
        pending_recruit_ticket_ids=tuple(x for x in state.pending_recruit_ticket_ids
                                        if x != offer.ticket_instance_id),
        stored_recruit_ticket_ids=tuple(x for x in state.stored_recruit_ticket_ids
                                       if x != offer.ticket_instance_id))
    state = invalidate_temporary_offer(engine, state, offer.ticket_instance_id, reason='actual_recruitment')
    return recruit_formal(engine, state, offer.char_id, source=source,
                          hope_cost=0, already_promoted=True)
