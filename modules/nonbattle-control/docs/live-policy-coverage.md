# 真实截图策略覆盖情况

本表描述 `blackflow_live` 的实现边界。它区分“现有网络能够接收这种已识别动作”与“视觉模块已经在实机稳定识别全部相关界面”。当前单元测试及截图回放不能证明一次真实游戏已经自主通关第一结局。

## 决策链路

`LiveObservation` 是当前截图的观察结果。`ObservedMenuMetadataAdapter` 只对动作卡片已关联的信息做身份标准化；`ObservedFeatureEncoder` 将其填入 CURRENT 检查点的原特征位置；路线使用 CURRENT 宏动作网络，其他菜单使用 CURRENT 冻结菜单网络。最终返回的动作仍是原观察中的对象，点击位置不由神经网络生成。

权重缺失、摘要不一致、特征模式不一致时停止。没有随机权重、教师策略或模拟器策略替代。代码不创建 PPO 训练器，不创建游戏模拟状态，不执行 `reset()` 或 `transition()`，不预测点击后的余额、物品、战利品或胜利。

## 动作覆盖表

| 界面/任务 | 策略接收的操作 | 必需的真实证据 | 当前实现及边界 |
| --- | --- | --- | --- |
| 地图缩放 | `event_advance` | 当前缩放按钮模板、点击框 | 可由菜单网络选择；不要求尚未出现的地图资源。 |
| 地图节点预览 | `move`，包括 `kind=map_node` | 当前楼层、当前位置、节点/连线；行动力、生命/上限、源石锭、希望、零件、收藏品 | 使用宏动作网络。未显示的移动成本保持未观察，节点预览后重新截图；不会预扣行动力。 |
| 事件分支 | `event`、`event_reward`、`event_advance`、`event_mark`、`event_move`、`lake_resolve` | 当前选项文字、绑定的 `choice_id` 或明确效果、点击框 | 客户端选项 ID 使用原 16 位词表编码。未知效果不生成；不同选项缺少区分特征时无法保证决策质量。 |
| 普通继续/确认/返回/离开 | `advance`、`event_advance`、`leave` | 当前识别场景和按钮 | 使用菜单网络；开始战斗、放弃探索等禁用按钮仍被过滤。泛化到每种未见界面未经完整实机验收。 |
| 商店商品预览 | `event` + `preview_only=true` | 当前商品卡片内的精确名称/ID、点击框 | 名称唯一匹配公开目录后填入物品类别、稀有度和身份；不拿别处的名称或价格拼接成商品。 |
| 商店购买 | `purchase` | 当前实际价格 `price` 与当前源石锭 `gold`；商品身份应来自同一卡片 | 未识别价格或余额、买不起时不可选。目录基础价格只作静态物品属性，不能替代实际售价。 |
| 出售/刷新/交换/培养/许愿刷新 | `sell`、`refresh`、`exchange`、`cultivate`、`wish_refresh` | 当前目标及按钮；已显示资源成本、数量、估值、剩余次数 | 特征接口已支持；视觉必须分别完成目标关联与数量识别。无法仅凭一个通用“刷新”按钮重建未显示的价格和候选。 |
| 战后战利品/零件/藏品领取 | `take`、`claim_scrap`、`remembrance_parts`、`remembrance_key` | 当前奖励卡片、实际候选、点击框 | 逐次选择、逐次重拍。候选不提前计入持有物品。 |
| 招募券候选 | `select_recruit_ticket`、`retain_recruit_ticket`、`decline_recruitment` | 当前招募券 ID/精确名称、选项位置 | 填入招募券可用于机械师的训练特征；不从招募池抽取候选。候选券不会提前成为已拥有的券。 |
| 实际干员预览/临时招募/应急招募 | `event` + `preview_only=true`、`recruit_reserve`、`recruit_temporary`、`emergency_hire` | 当前干员名称/ID；实际希望或其他资源成本；候选点击框 | 保留干员身份、填写机械师身份位和已显示成本。原网络没有通用干员身份嵌入，不能宣称已经学会所有干员间的选择。 |
| 进阶 | `mechanist_promote` | 当前可用目标、当前券和实际成本 | 支持原训练中的机械师语义、进阶券类型位；不能将其他干员映射成机械师。 |
| 装备/丢弃/零件容量 | `equip`、`discard`、`parts_capacity` | 当前具体物品、剩余次数/估值、可点击按钮 | 编码接口支持实际物品属性；不根据上一次点击就认定装备或丢弃已成功。 |
| 门户/普通远征 | `portal_enter`、`portal_return`、`expedition`、`expedition_inside` | 当前真实入口/候选、所需物品及按钮 | 操作词表支持；完整所有分支的视觉覆盖和实机连续流程未验证。 |
| 银行/补给/容量等 | `bank_withdraw`、`supply_voucher`、`squad_capacity`、`red_moss`、`shadow_dance`、`broker_reserve`、`broker_random_six`、`employment_refresh` | 当前按钮、余额/费用和相关目标 | 不合成银行余额、刷新次数、干员数量或事件阶段；缺失输入显式保留为未知。 |
| 其他原菜单操作 | `starting_reward`、`remembrance_parts`、`remembrance_key`、`fate_mark`、`fate_leave`、`fate_commit`、`cave_draw` | 对应场景与真实绑定的选项 | 操作编码存在不等于该场景已通过实机验收。 |
| 开始战斗及战斗过程 | `battle_start`、`special_battle`、`fate_battle` 等 | 无 | 按用户任务范围停止，交由用户完成。战后重新观察后才可能继续。 |
| 其他结局、进入第六层 | `expedition_source`、其他结局物品/标记、已识别第六层 | 无 | 第一结局范围约束拒绝。 |
| 未知动作/未识别场景 | `unknown`、`needs_observation` | 无 | 停止并报告识别不足，不以固定坐标补点。 |

## 视觉到策略的元数据契约

`ObservedAction.bbox` 使用当前识别图像的像素坐标 `(x, y, width, height)`。动作可携带以下字段：

| 字段 | 含义及输入限制 |
| --- | --- |
| `operation` | 原模型 `OPTION_OPERATIONS` 中的操作；地图移动为 `move`。 |
| `item_id` / `item_name` | 当前卡片的物品身份或精确名称。名称只去除空白、引号与显示标签，不使用模糊匹配。名称/ID 冲突会阻止该动作。 |
| `operator_id` / `operator_name` | 当前实际干员候选；精确名称查固定客户端词表。候选身份只用于当前选项。 |
| `ticket_item_id` | 打开干员候选所使用的已观察券；与干员身份分开记录。 |
| `choice_id` | 当前事件选项的客户端 ID，必须由当前文字/上下文识别确定。 |
| `price` | 当前商品实际售价；不得填目录均价、估计值或合成商店价。 |
| `hope_cost` / `resource_costs` | 当前显示的希望成本，或 `{resource: cost}`。实际消费动作需要当前观察余额足够；预览按钮本身不消费。 |
| `resource_delta` | 当前选项明确显示的增减量，用作网络输入，绝不直接写回游戏资源。 |
| `quantity` / `uses_remaining` / `remaining_uses` / `appraisal` | 当前显示的数量、剩余使用次数或估值；后者为同义输入字段。 |
| `ends_node` / `parts_capacity_delta` | 当前界面明确表达的结束节点/容量变化语义。 |
| `preview_only` / `selection_stage` | 区分打开详情与实际执行消费/招募。预览后必须重新获取截图。 |
| `slot_id` | 当前货架槽位；同一槽位多个按钮不能被统计成多件商品。 |
| `source_frame_id` | 可选的明确帧来源；若提供且不等于当前帧，则动作被拒绝。 |

物品目录的类别、固定身份、稀有度、目录买卖参考属性可以补充被识别出的真实物品。目录本身不能产生出售中的物品、数量、价格或资源余额。

## 当前货架、持有物品和招募状态

`LiveObservation.metadata.items` 仅接收真正观察到的持有物品快照，每项可包含 `item_id`、`category`、`uses_remaining`、`appraisal`、`equipped`。`formal_operator_ids`、`available_operator_ids`、`promoted_operator_ids`、`pending_recruit_ticket_ids`、`stored_recruit_ticket_ids` 也必须是已观察的实际状态。不能把候选卡片加入这些列表。`inventory_complete=true` 仅表示视觉已经完成真实完整扫描，不能由策略自行设置。

`metadata.shop_state` 可以包含当前节点 `node_id`、`source_frame_id`、`shelf_complete`、实际 `refreshes`、`remaining_refreshes`、`visits`、`sold_parts`、`sold_slots`、`lost_slots`、`trade_bonus_paid`、`total_withdrawn`、`entry_withdrawn`。只有同帧完整货架才能填入节点的货架汇总特征；看不见或已过期的货架不会被补全。已经显示但买不起的商品仍进入菜单观察，不进入可执行动作。

## 为什么没有跨帧资源自动记忆

本次没有添加跨帧资源缓存。一次点击可能触发事件效果、天气、自动奖励、价格变化或界面跳转；仅凭场景/楼层/节点名称相同，无法证明旧余额和旧库存仍然有效。缓存旧值会让策略把过时的资源当作真实状态。

每次决策只使用该帧直接观察的数据及静态身份词表。未知的输入位置保持空缺编码并在 `CurrentNeuralPolicy.readiness()` 中报告。由于原模型没有所有字段的独立缺失标记，空缺编码属于迁移限制，不能解释为“真实库存等于零”或“已经证明策略同训练环境等价”。

## 可复核验证与尚缺的验收

2026-09-20 招募约束：同名普通券与 `_candle` 券保留身份歧义，只编码共有操作、稀有度及机械师适用性，不猜 `item_id`。当前网络没有招募券类别或职业身份嵌入；初始先锋、辅助、特种三券在原特征中相同，真实权重前向概率均为 1/3。`readiness()` 明确报告此范围，不能解释为学会了职业偏好。

实际 `recruit`、`recruit_reserve`、`recruit_temporary`、`emergency_hire` 及确认招募文字都要求当前 recruitment 场景、同帧来源、精确身份、选中身份一致、已观察到按钮启用以及明确费用/足够余额。免费必须明确观察到 0 和余额。预览标记不能绕过检查。v9 普通地灵零费页面已满足完整确认合同，但当前网络在确认与梓兰预览间选择梓兰。保留这个真实选择及所有合法候选，不将它描述为完成招募或已证实死循环；后续需要实际换卡后的画面证明是否推进。付费希望显示语义、助战确认及连续招募仍有缺口。

`tests/test_live_policy.py` 覆盖实际 CURRENT 检查点加载、路线/菜单前向、权重输出与动作对应、战斗排除、缺失资源与价格、商品精确身份、机械师/招募券/进阶券特征、当前货架和过期货架隔离。测试不产生任何游戏输入。

奖励领取现已将同卡片的高置信标题关联到实际“收下”按钮，并移除该卡片重复的预览候选。交易确认弹层以同弹层完整问句关联物品及实际价格，取消仅关闭弹层；背景商品不参与当前选择。`tests/test_live_rewards.py` 与 `tests/test_live_shop_dialog.py` 包含已有真实录像像素回归，购买叠影/价格缺失保持不可选。这些证据仅证明保存画面上的识别与当前网络前向，没有执行录像中的交易或证明当前游戏已推进。

仍需真实连续运行验收：每个非战斗界面的按钮/候选覆盖；完整持有物品扫描；多步事件和商品详情返回路径；刷新后重识别；干员/招募券流程；第一结局胜利页判定。通用干员选择还受原检查点输入表达能力限制，若要学习这些身份差异，需要明确新增训练特征和对应新检查点，而非在现有权重旁添加随机分类器。
