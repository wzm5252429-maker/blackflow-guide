"""Documented Blackflow event choices; no generic fallback rewards.

Choice IDs, fixed costs and named items are pinned to the public client
snapshot. Random item identities still need an explicit sampling profile in
the economy engine. A ``needs_observation`` operation is never a free reward or
an instruction to complete the node. Multistage events are deliberately not
flattened into invented payouts.
"""
from __future__ import annotations

from dataclasses import replace

from .domain import EventOption, GameState, ResourceDelta


def cave_visit_counter_key(state: GameState) -> str:
    """The ordinary region and all Black Ponds have separate cave limits."""
    return "cave:portal_seen" if state.portal_context is not None else "cave:ordinary_seen"


def record_incident_seen(state: GameState, event_name: str) -> GameState:
    """Record ordinary exclusions and the cave's separate once-per-realm cap."""
    counters = dict(state.event_counters)
    if event_name == "洞中宝":
        counters[cave_visit_counter_key(state)] = 1
    return replace(state, seen_event_names=state.seen_event_names | {event_name},
                   event_counters=tuple(sorted(counters.items())))


def eligible_incident_names(
    floor: int, state: GameState, candidate_names: tuple[str, ...] | list[str],
) -> tuple[str, ...]:
    """Filter a floor's already-unlocked event pool before drawing an incident.

    The supplied pool must apply save-level ending unlocks; full technology
    alone does not establish those unlocks. Repeatable occurrences still have
    only one transaction per node. Cave occurs once in ordinary regions and
    once in Black Ponds; completion must call ``record_incident_seen``.
    """
    owned = state.inventory | frozenset(x.item_id for x in state.item_instances)
    seen = state.seen_event_names
    counters = dict(state.event_counters)
    repeatable_occurrences = frozenset(("色味不同源", "货从口出", "敲动杠杆"))
    result = []
    for name in candidate_names:
        if name == "洞中宝":
            if counters.get(cave_visit_counter_key(state), 0):
                continue
            # Preserve conservative exclusions for older ordinary saves that
            # predate the realm counters, while allowing their first portal.
            if (name in seen and state.portal_context is None and
                    not any(key.startswith("cave:") for key in counters)):
                continue
        elif name in seen and name not in repeatable_occurrences:
            continue
        if name == "泪之聚落" and not owned.intersection(("rogue_6_relic_final_3", "beacon")):
            continue
        if name == "呼吸的红苔" and "沉寂之屋" not in seen:
            continue
        if name == "线人" and floor not in (2, 3, 4):
            continue
        if name == "愈创之心" and floor != 4:
            continue
        if name not in result:
            result.append(name)
    return tuple(result)


# Client stage IDs and PRTS revision 424917 base settlements. Ranges are
# observed bounds, not a claim that each integer has equal probability.
# Special-enemy rewards depend on defeating the reward carrier and must be
# settled by the combat engine separately from these base coins/experience.
SPECIAL_BATTLES = {
    "ro6_t_1": {"name": "陌生旅伴", "gold": 2, "gold_max": 3, "command_exp": 10, "is_elite": False, "relic_pool": None},
    "ro6_t_2": {"name": "安保措施", "gold": 2, "gold_max": 2, "command_exp": 12, "is_elite": False, "relic_pool": None},
    "ro6_t_3": {"name": "开业剪彩", "gold": 2, "gold_max": 2, "command_exp": 13, "is_elite": False, "relic_pool": None},
    "ro6_e_t_3": {"name": "开业剪彩（紧急）", "gold": 3, "gold_max": 3, "command_exp": 24, "is_elite": True, "relic_pool": None},
    "ro6_t_4": {"name": "合伙人会议", "gold": 3, "gold_max": 3, "command_exp": 25, "is_elite": False, "relic_pool": None},
    "ro6_e_t_4": {"name": "合伙人会议（紧急）", "gold": 3, "gold_max": 3, "command_exp": 36, "is_elite": True, "relic_pool": None},
    "ro6_t_5": {"name": "湖中魇", "gold": 4, "gold_max": 4, "command_exp": 25, "is_elite": False, "relic_pool": "rogue_6:不期而遇：湖中仙女"},
    "ro6_e_t_5": {"name": "湖中魇（紧急）", "gold": 8, "gold_max": 8, "command_exp": 36, "is_elite": True, "relic_pool": "rogue_6:不期而遇：湖中仙女（紧急）"},
    **{f"ro6_t_{n}": {"name": name, "gold": 3, "gold_max": 3, "command_exp": 30, "is_elite": False, "relic_pool": None}
       for n, name in ((13, "闹乐"), (14, "纵怒"), (15, "灭身"))},
}


def _battle(choice_id: str, title: str, stage_id: str) -> EventOption:
    return EventOption(choice_id, title, operation="special_battle", item_id=stage_id, battle=True)


def _stage(state: GameState | None) -> int:
    if state is None:
        return 0
    node_id = state.pending_node_id or state.current_node_id
    return dict(state.event_counters).get(f"{node_id}:stage", 0)


SHADOW_DANCE_COSTS = (1, 1, 2, 2, 2, 3, 3, 3)
SHADOW_DANCE_CHOICE_IDS = (
    "choice_ro6_normal4_1", "choice_ro6_normal4_5", "choice_ro6_normal4_9",
    "choice_ro6_normal4_12", "choice_ro6_normal4_16", "choice_ro6_normal4_19",
    "choice_ro6_normal4_23", "choice_ro6_normal4_25",
)


def cave_draw_outcomes(round_number: int, *, portal: bool = False) -> tuple[EventOption, ...]:
    """Possible random outcomes of one dig, never a player-selectable menu.

    Client scene links permit four digs. Coins and empty results continue for
    the first three digs; a relic or battle ends this occurrence immediately.
    Sampling weights and the next-region normal-stage pool are engine inputs.
    """
    if round_number not in (1, 2, 3, 4):
        raise ValueError("cave draw must be 1..4")
    prefix = "choice_ro6_bat6b_" if portal else "choice_ro6_bat6_"
    gold, relic, empty, battle = ((3, 4, 5, 6), (9, 10, 11, 12),
                                (14, 15, 16, 17), (19, 20, 21, 22))[round_number-1]
    return (
        EventOption(prefix+str(gold), "获得5源石锭", ResourceDelta(gold=5),
                    operation="event_reward", ends_node=round_number == 4),
        _reward(prefix+str(relic), "获得1件收藏品", item_id="pool:RELIC"),
        EventOption(prefix+str(empty), "空的树洞", operation="event_reward", ends_node=round_number == 4),
        _battle(prefix+str(battle), "遭遇下一层普通作战", "stage_pool:cave_next_floor_normal"),
    )


def shadow_dance_outcomes(round_number: int, *, had_relic_on_entry: bool = False) -> tuple[EventOption, ...]:
    """Return possible *random* outcomes after a paid circle, never choices.

    The caller must supply its explicit synthetic/observed probability model.
    ``quantity`` remains the grant count. The AP outcome in round eight stays
    on stage seven (scene_ro6_normal4_24); the named relic ends the event.
    """
    outcomes = {
        1: (_reward("choice_ro6_normal4_3", "4源石锭", ResourceDelta(gold=4)),
            _reward("choice_ro6_normal4_4", "回复1生命", ResourceDelta(hp=1))),
        2: (_reward("choice_ro6_normal4_6", "2希望", ResourceDelta(hope=2)),
            _reward("choice_ro6_normal4_7", "回复1生命", ResourceDelta(hp=1))),
        3: (_reward("choice_ro6_normal4_10", "1自然物", item_id="pool:rogue_6:不期而遇：被歌颂的影子（随机自然物）"),
            _reward("choice_ro6_normal4_11", "回复2生命", ResourceDelta(hp=2))),
        4: (_reward("choice_ro6_normal4_13", "1加工品", item_id="pool:rogue_6:不期而遇：被歌颂的影子（随机加工品）"),
            _reward("choice_ro6_normal4_14", "12源石锭", ResourceDelta(gold=12)),
            _reward("choice_ro6_normal4_15", "回复2生命", ResourceDelta(hp=2))),
        5: (_reward("choice_ro6_normal4_17", "1收藏品", item_id="pool:rogue_6:不期而遇：被歌颂的影子（随机收藏品）"),
            _reward("choice_ro6_normal4_18", "回复5生命", ResourceDelta(hp=5))),
        6: (_reward("choice_ro6_normal4_20", "8希望", ResourceDelta(hope=8)),
            EventOption("choice_ro6_normal4_21", "可携带干员+1", operation="event_squad_capacity"),
            _reward("choice_ro6_normal4_22", "回复3生命", ResourceDelta(hp=3))),
        7: (_reward("choice_ro6_normal4_24", "3收藏品", item_id="pool:rogue_6:不期而遇：被歌颂的影子（随机收藏品）", quantity=3),),
        8: (
            _reward("choice_ro6_normal4_26", "犬植浆", item_id="rogue_6_relic_artifact_4"),
            EventOption("choice_ro6_normal4_27", "1行动力", ResourceDelta(action_points=1), operation="event_reward", ends_node=False),
        ),
    }
    if round_number not in outcomes:
        raise ValueError("shadow dance round must be 1..8")
    if round_number == 8 and had_relic_on_entry:
        return outcomes[8][1:]
    return outcomes[round_number]


def _reward(
    choice_id: str,
    title: str,
    effect: ResourceDelta = ResourceDelta(),
    *,
    item_id: str | None = None,
    quantity: int = 1,
) -> EventOption:
    return EventOption(
        choice_id, title, effect, operation="event_reward",
        item_id=item_id, quantity=quantity,
    )


def _leave(choice_id: str = "choice_leave", title: str = "离开") -> EventOption:
    return EventOption(choice_id, title, operation="event_reward")


def _unknown(title: str, choice_id: str = "needs_observation") -> EventOption:
    return EventOption(
        choice_id, title, operation="needs_observation", ends_node=False,
        description="该分支需要尚未建模的真实场景、战斗或随机结果；不提供替代收益。",
    )


FALSE_FATE_NAME = "好奇心与死"
FALSE_FATE_RELIC_POOL = "rogue_6:命运所指：好奇心与死"
FATE_MARK_FEATURE = "rogue_6_end_2_trigger_4"
FATE_SANDBOXES = frozenset(("rogue_6_relic_final_1", "rogue_6_relic_final_2"))


def documented_fate_options(
    event_name: str, state: GameState | None = None, *,
    real_fate_marked: bool = False,
) -> tuple[EventOption, ...]:
    """The two exclusive outcomes of an observed floor-V false fate node.

    Call after observing its scene; the three STORY icons do not reveal their
    identities in advance. A second false node may pay again even when this
    scene name is in ``seen_event_names``. A completed *same* node cannot.

    ``fate_mark`` must charge its effect, mark the actual true fate coordinate,
    and complete this false node. Its FEATURE ID is a mechanic instruction,
    never a collectible or a teleport. Existing map marks are supplied by the
    caller; the two sandboxes also suppress the redundant paid option. The
    caller must check affordability before executing any returned choice.

    The client says one collectible without a rarity qualifier. The named
    community pool supplies observed identities, not proven server weights.
    True fate and floor-VI rituals belong to their separate ending dispatcher.
    """
    if event_name != FALSE_FATE_NAME:
        return (_unknown(f"{event_name}：需要对应命运场景，不能套用假命运奖励"),)
    if state is not None:
        node_id = state.pending_node_id or state.current_node_id
        if node_id in state.completed:
            return ()
        if state.maps and state.floor != 5:
            raise ValueError("false fate occurs only in ordinary floor V")
        if state.portal_context is not None:
            raise ValueError("false fate is not a Black Pond event")
        real_fate_marked = real_fate_marked or FATE_SANDBOXES <= state.inventory
    options = ()
    if not real_fate_marked:
        options += (EventOption(
            "choice_ro6_end1_1", "修正结果，必须找到目标",
            ResourceDelta(gold=-50), operation="fate_mark", item_id=FATE_MARK_FEATURE,
            description="消耗50源石锭，标记本层真正的命运所指；不获得收藏品。",
        ),)
    return options + (_reward(
        "choice_ro6_end1_2", "这是什么戏法？",
        item_id="pool:" + FALSE_FATE_RELIC_POOL,
    ),)


def documented_true_fate_options(
    *, already_entered: bool = False, committed_to_battle: bool = False,
) -> tuple[EventOption, ...]:
    """Observed true fate: leave/reenter, or commit to the same-node boss.

    ``fate_commit`` must change the current node into the ro6_b_5 boss and
    refresh this menu with committed_to_battle=True. No departure is allowed
    after commitment. Neither commitment nor the battle is another movement:
    battle-node arrival concepts must not fire on this originally STORY node.
    ``fate_leave`` marks this true node and preserves repeatable reentry; it
    grants no collectible, hope, or other resource.
    """
    if committed_to_battle:
        return (EventOption(
            "choice_ro6_end2_3", "你将见证一场毁灭，抑或新生",
            operation="fate_battle", item_id="ro6_b_5", battle=True,
        ),)
    return (
        EventOption(
            "choice_ro6_end2_4" if already_entered else "choice_ro6_end2_1",
            "保护证据，查缴那些机器！", operation="fate_commit",
            item_id="ro6_b_5", ends_node=False,
        ),
        EventOption(
            "choice_ro6_end2_2", "她在说什么疯话？", operation="fate_leave",
            item_id="rogue_6_end_2_trigger_2",
        ),
    )


def documented_incident_options(
    event_name: str, state: GameState | None = None,
) -> tuple[EventOption, ...]:
    """Build currently representable choices from the entered event and state.

    This is an entry-scene dispatcher, not an event occurrence distribution.
    In particular, res5 is a single choice between one and three purchases;
    it does not create a repeatable shop or an unlimited purchasing loop.
    """
    hp = state.resources.hp if state is not None else 0
    owned = (
        state.inventory | {item.item_id for item in state.item_instances}
        if state is not None else frozenset()
    )
    cage = "rogue_6_scrap_G_12" in owned or "cage" in owned
    if event_name == "桑尼的邀请":
        return (
            _reward("choice_ro6_res1_1", "帮它梳理绒毛：3希望", ResourceDelta(hope=3)),
            _reward("choice_ro6_res1_2", "唱一首摇篮曲：10源石锭", ResourceDelta(gold=10)),
        )
    if event_name == "色味不同源":
        return (
            _reward("choice_ro6_res2_1", "拿走苹果：生命上限+3", ResourceDelta(max_hp=3)),
            _reward("choice_ro6_res2_2", "拿走鱼：5护盾", ResourceDelta(shield=5)),
            _leave("choice_ro6_res2_3", "我不饿"),
        )
    if event_name == "货从口出":
        return tuple(
            _reward(
                f"choice_ro6_res3_{index}", f"4源石锭购买1件{label}",
                ResourceDelta(gold=-4), item_id=f"pool:{kind}",
            )
            for index, label, kind in (
                (1, "加工品", "MOVE"), (2, "概念体", "PASSIVE"),
                (3, "自然物", "GOODS"),
            )
        ) + (_leave("choice_ro6_res3_4", "没什么想买的"),)
    if event_name == "沉重的契约":
        return (
            _reward("choice_ro6_res4_1", "上去帮忙：2行动力", ResourceDelta(action_points=2)),
            _reward(
                "choice_ro6_res4_2", "推翻石磨：消耗一半生命，获得5行动力",
                # The client proves a proportion, not a fixed HP-3. Odd-value
                # rounding is the documented simulator convention; see audit.
                ResourceDelta(hp=-(hp // 2), action_points=5),
            ) if state is not None else _unknown("推翻石磨：需计算当前生命的一半", "choice_ro6_res4_2"),
            _leave("choice_ro6_res4_3", "不关我的事"),
        )
    if event_name == "敲动杠杆":
        return (
            _reward("choice_ro6_res5_1", "敲一次：4源石锭换1藏", ResourceDelta(gold=-4), item_id="pool:RELIC"),
            _reward("choice_ro6_res5_2", "敲三次：10源石锭换3藏", ResourceDelta(gold=-10), item_id="pool:RELIC", quantity=3),
            _leave("choice_ro6_res5_3", "不感兴趣"),
        )
    if event_name == "沉寂之屋":
        return (
            # Non-combat HP loss preserves at least one HP. The client has no
            # explicit HP gate on this choice; clamp this event's actual cost
            # without weakening battle-death or generic affordability rules.
            _reward("choice_ro6_normal1_1", "清理藤蔓：2生命换笼控器",
                    ResourceDelta(hp=-min(2, max(0, hp - 1))), item_id="rogue_6_scrap_G_12")
            if state is not None else _reward("choice_ro6_normal1_1", "清理藤蔓：2生命换笼控器",
                                              ResourceDelta(hp=-2), item_id="rogue_6_scrap_G_12"),
            _reward("choice_ro6_normal1_2", "试图交流：8源石锭换笼控器", ResourceDelta(gold=-8), item_id="rogue_6_scrap_G_12"),
            _leave("choice_ro6_normal1_3"),
        )
    if event_name == "线人":
        return (
            _reward("choice_ro6_bomb1_1", "受雇于梅兰德：沙盘α", item_id="rogue_6_relic_final_1"),
            _reward("choice_ro6_bomb1_2", "独立探险家：1件珍贵加工品", item_id="pool:MOVE:SUPER_RARE"),
            _leave("choice_ro6_bomb1_3", "我是联合政府派来的"),
        )
    if event_name == "血衣之下":
        return (
            _reward("choice_ro6_relic1_1", "拽出衣服：他缚", item_id="rogue_6_relic_fight_11"),
            _reward("choice_ro6_relic1_2", "割下怪花：剑锤", item_id="rogue_6_relic_legacy_142"),
            _reward("choice_ro6_relic1_3", "砍断藤蔓：一串钱伥", item_id="rogue_6_relic_legacy_141"),
            _reward("choice_ro6_relic1_4", "离开：生命上限+2", ResourceDelta(max_hp=2)),
        )
    if event_name == "擒与缚":
        return (
            _reward("choice_ro6_relic2_1", "帮助蛇：翱翼", item_id="rogue_6_relic_fight_18"),
            _reward("choice_ro6_relic2_2", "帮助鹰：虬蜕", item_id="rogue_6_relic_fight_19"),
            _leave("choice_ro6_relic2_3", "不关我的事"),
        )
    if event_name == "泪之聚落":
        if state is not None and not ("rogue_6_relic_final_3" in owned or "beacon" in owned):
            return (_unknown("神明缺席的剧情标记要求持有怦然信标"),)
        return (
            _reward("choice_ro6_chimera2_1", "倾囊资助：全部源石锭换击坠神明", ResourceDelta(gold=-state.resources.gold), item_id="rogue_6_relic_final_4")
            if state is not None else _unknown("倾囊资助：需计算当前全部源石锭", "choice_ro6_chimera2_1"),
            _leave("choice_ro6_chimera2_2", "无能为力"),
        )
    if event_name == "黑诞":
        choices = (
            EventOption("choice_ro6_normal2_2", "和猎狗战斗到底", operation="special_battle", item_id="stage_pool:black_birth", battle=True),
            _reward("choice_ro6_normal2_3", "落荒而逃：消耗3生命，至少保留1", ResourceDelta(hp=-min(3, max(0, hp - 1))))
            if state is not None else _unknown("落荒而逃：需计算保留至少1生命的消耗", "choice_ro6_normal2_3"),
        )
        if cage:
            choices = (
                _reward("choice_ro6_normal2_1", "使用笼控器：猎印", item_id="rogue_6_relic_cargo_12"),
            ) + choices
        return choices

    if event_name == "思乡心切":
        return (_battle("choice_ro6_bat1_3", "强行驱赶：陌生旅伴", "ro6_t_1"),
                _leave("choice_ro6_bat1_2", "安慰它"))
    if event_name == "划算买卖":
        return (_battle("choice_ro6_bat2_3", "为客户发声：安保措施", "ro6_t_2"),
                _leave("choice_ro6_bat2_2", "装没听到"))
    if event_name == "鸭托邦":
        return (
            _battle("choice_ro6_bat3_3", "区区打手：开业剪彩", "ro6_t_3"),
            _battle("choice_ro6_bat3_5", "区区商人：紧急开业剪彩", "ro6_e_t_3"),
            _leave("choice_ro6_bat3_4", "兴趣缺缺"),
        )
    if event_name == "传奇团伙":
        options = [_battle("choice_ro6_bat4_3", "阻拦到底：合伙人会议", "ro6_t_4")]
        if state is not None and state.resources.gold > 50:
            options.append(_battle("choice_ro6_bat4_5", "扎穿车胎：紧急合伙人会议", "ro6_e_t_4"))
        options.append(
            _reward("choice_ro6_bat4_4", "果断求饶：消耗一半源石锭", ResourceDelta(gold=-(state.resources.gold // 2)))
            if state is not None else _unknown("果断求饶：需计算当前源石锭的一半", "choice_ro6_bat4_4")
        )
        return tuple(options)
    if event_name == "湖中仙女":
        stage = _stage(state)
        if stage >= 3:
            return (_unknown("第3次供奉后的随机结果尚未结算"),)
        choice_id = ("choice_ro6_bat5_3", "choice_ro6_bat5_7", "choice_ro6_bat5_8")[stage]
        return (
            EventOption(choice_id, f"第{stage+1}次供奉：1源石锭", ResourceDelta(gold=-1),
                        operation="lake_resolve" if stage == 2 else "event_advance", ends_node=False),
            _battle("choice_ro6_bat5_5", "揭穿骗局：湖中魇", "ro6_t_5"),
            _leave("choice_ro6_bat5_6", "离开"),
        )
    if event_name == "被歌颂的影子":
        completed_circles = _stage(state)
        if completed_circles >= 8:
            return (_leave("choice_ro6_normal4_8", "夜晚已经结束"),)
        return (
            EventOption(
                SHADOW_DANCE_CHOICE_IDS[completed_circles], f"跳第{completed_circles+1}圈",
                ResourceDelta(hp=-SHADOW_DANCE_COSTS[completed_circles]),
                operation="shadow_dance", quantity=completed_circles+1, ends_node=False,
            ),
            _leave("choice_ro6_normal4_2" if completed_circles == 0 else "choice_ro6_normal4_8", "不想再跳了"),
        )
    if event_name in {"独活", "和平守卫者"}:
        is_dog = event_name == "独活"
        prefix = "choice_ro6_task2" if is_dog else "choice_ro6_task1"
        item_id = "rogue_6_relic_cargo_10" if is_dog else "rogue_6_relic_cargo_11"
        target = "EMPLOY" if is_dog else "BATTLE_SAVAGE"
        if _stage(state) == 0:
            return (
                EventOption(prefix+"_1", "去看看小狗：同行者" if is_dog else "接取悬赏：厄运火杆",
                            operation="event_advance", item_id=item_id, ends_node=False),
                _leave(prefix+"_2", "小心为上" if is_dog else "离开队伍"),
            )
        return (
            EventOption(prefix+"_3", "跟随小狗前往营地" if is_dog else "立即前往最近居民据点",
                        operation="event_move", item_id="node:"+target),
            EventOption(prefix+"_4", "仅标记位置，暂不前往", operation="event_mark", item_id="node:"+target),
        )
    if event_name == "愈创之心":
        choices = (
            _reward("choice_ro6_normal5_1", "紧紧拥抱愈创木：3行动力换追忆",
                    ResourceDelta(action_points=-3), item_id="rogue_6_relic_artifact_5"),
        )
        if state is not None and sum(x.category == "MOVE" for x in state.item_instances) >= 2:
            choices += (EventOption("choice_ro6_normal5_2", "制作秋千：随机消耗2件加工品换追忆",
                                    operation="remembrance_parts", item_id="rogue_6_relic_artifact_5"),)
        if "rogue_6_relic_cargo_13" in owned:
            choices += (EventOption("choice_ro6_normal5_3", "比对像素小猫：消耗源私钥换追忆",
                                    operation="remembrance_key", item_id="rogue_6_relic_artifact_5",
                                    remove_items=("rogue_6_relic_cargo_13",)),)
        choices += (_reward("choice_ro6_normal5_5", "闭眼感受宁静：1行动力", ResourceDelta(action_points=1)),)
        if cage:
            choices = (
                _reward("choice_ro6_normal5_4", "笼控器检测危险：5行动力", ResourceDelta(action_points=5)),
            ) + choices
        return choices

    if event_name == "洞中宝":
        stage = _stage(state)
        if stage >= 4:
            return (_unknown("树洞的四次抽取已经结束，等待结算"),)
        prefix = "choice_ro6_bat6b_" if state is not None and state.portal_context is not None else "choice_ro6_bat6_"
        choice_id = prefix+str((1, 7, 13, 18)[stage])
        return (
            EventOption(choice_id, "掏一下" if stage == 0 else "继续掏",
                        operation="cave_draw", quantity=stage+1, ends_node=False),
            _leave(prefix+str(2 if stage == 0 else 8)),
        )
    if event_name == "呼吸的红苔":
        choices = ()
        if cage:
            choices += (EventOption("choice_ro6_normal3_1", "驱赶猎狗：30源石锭，下一区域生成希望的沃土",
                                    ResourceDelta(gold=30), operation="red_moss",
                                    item_id="rogue_6_next_weather_1"),)
        seed = next((x for x in state.item_instances if x.item_id == "rogue_6_scrap_G_01"), None) if state is not None else None
        if seed is not None:
            choices += (EventOption("choice_ro6_normal3_2", "种下1枚种子，下一区域生成希望的沃土",
                                    operation="red_moss", item_id="rogue_6_next_weather_1",
                                    instance_id=seed.instance_id),)
        return choices + (_leave("choice_ro6_normal3_3", "快逃"),)

    if event_name == "临时中介所":
        # The four client variants have the same title/reward family, but the
        # displayed profession pair and its distribution have not been observed.
        # This explicitly limited path retains only their common guaranteed
        # free-reserve choice; it does not display four selectable professions.
        return (
            EventOption("broker_guaranteed_reserve", "可靠的中介值得信任（保证预备干员的经济模拟）",
                        ResourceDelta(hope=2, tickets=1), operation="broker_reserve", ends_node=False,
                        description="获得2希望与一次五星及以下招募机会；职业组未观测，仅支持各组都保证的免费预备干员。"),
            EventOption("choice_ro6_hire1_2", "合格的帮手应当仔细挑选：随机6星临时招募",
                        operation="broker_random_six", ends_node=False,
                        description="真实随机六星身份及重复持有结算需观测；此券不是直升临招券。"),
            _leave("choice_ro6_hire1_3", "前行的速度更重要"),
        )

    # These events have a client-confirmed entry leave choice. Their other
    # branch needs a stage-specific combat model, assignment, or scene graph.
    leaves = {
        "思乡心切": "choice_ro6_bat1_2",
        "划算买卖": "choice_ro6_bat2_2",
        "鸭托邦": "choice_ro6_bat3_4",
        "湖中仙女": "choice_ro6_bat5_2",
        "被歌颂的影子": "choice_ro6_normal4_2",
        "独活": "choice_ro6_task2_2",
        "和平守卫者": "choice_ro6_task1_2",
    }
    if event_name in leaves:
        return (_unknown("继续事件：需要后续场景结算"), _leave(leaves[event_name]))
    return (_unknown(f"{event_name}：需要真实事件分支"),)


__all__ = ["documented_incident_options", "eligible_incident_names", "record_incident_seen",
           "cave_visit_counter_key", "cave_draw_outcomes", "SPECIAL_BATTLES",
           "SHADOW_DANCE_COSTS", "shadow_dance_outcomes", "documented_fate_options",
           "documented_true_fate_options", "FALSE_FATE_NAME", "FALSE_FATE_RELIC_POOL",
           "FATE_MARK_FEATURE", "FATE_SANDBOXES"]
