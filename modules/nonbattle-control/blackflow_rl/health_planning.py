"""Public policy estimates for preparing health before the shadow dance.

This uses the simulator's explicitly synthetic equal-branch prior. It is a
conditional event estimate, not a claim about encountering the event or about
server branch probabilities. It does not apply rewards or inspect any map.
"""
from functools import lru_cache

from .events import SHADOW_DANCE_COSTS, shadow_dance_outcomes


SHADOW_EVENT_NAME = '被歌颂的影子'
SHADOW_REWARD_ID = 'rogue_6_relic_artifact_4'
HEALTH_EXPECTATION_EVIDENCE = 'SYNTHETIC_EQUAL_BRANCH_POLICY_EXPECTATION'
_TRUE_RELIC_REWARD_CHOICES = frozenset({'choice_ro6_normal4_17', 'choice_ro6_normal4_24'})


@lru_cache(maxsize=4096)
def shadow_dance_relic_expectation(hp: int, max_hp: int, *,
                                   had_reward_on_entry: bool = False,
                                   completed_circles: int = 0) -> float:
    """Expected *remaining* true relics while continuing affordable circles.

    Every paid circle must leave at least one actual HP. Shields do not count.
    Healing is capped by max_hp; the dance itself never reduces max_hp. The
    seventh circle grants three relics and the eighth can grant its unique
    reward once. An eighth-circle AP outcome permits another paid attempt.
    """
    if type(hp) is not int or type(max_hp) is not int or not 0 <= hp <= max_hp or max_hp < 1:
        raise ValueError('valid actual HP and HP capacity are required')
    if type(completed_circles) is not int or not 0 <= completed_circles <= 8:
        raise ValueError('completed circle count must be 0..8')
    if completed_circles == 8 or hp <= SHADOW_DANCE_COSTS[completed_circles]:
        return 0.0
    if completed_circles == 7:
        if had_reward_on_entry:
            return 0.0
        # AP-only outcomes repeat this same tier. With the explicit 1:1
        # prior, this closed form avoids recursion for very large HP pools.
        attempts = (hp-1)//SHADOW_DANCE_COSTS[7]
        return 1.0-0.5**attempts
    paid_hp = hp-SHADOW_DANCE_COSTS[completed_circles]
    outcomes = shadow_dance_outcomes(completed_circles+1,
                                    had_relic_on_entry=had_reward_on_entry)
    value = 0.0
    for outcome in outcomes:
        reward = float(outcome.quantity) if outcome.option_id in _TRUE_RELIC_REWARD_CHOICES else 0.0
        healed = min(max_hp, paid_hp+outcome.effect.hp)
        value += reward+shadow_dance_relic_expectation(healed, max_hp,
            had_reward_on_entry=had_reward_on_entry, completed_circles=completed_circles+1)
    return value/len(outcomes)
