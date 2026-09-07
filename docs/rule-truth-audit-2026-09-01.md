# 黑流树海地图与非作战事件真实性审计（2026-09-01）

## 审计结论

当前仓库已经做到“没有证据就不冒充真实规则”，但尚不能声称完整复刻官服生成器或全部事件执行逻辑。

- `evidence` 是默认 profile：未知模板/节点概率不抽样，未知事件进入后返回 `NEEDS_OBSERVATION` 语义并停步，不结算零收益，也不生成替代奖励。
- `synthetic` 必须显式启用：它只用于测试算法和域随机化，输出、文档与 checkpoint 都不能称为真实规则训练。
- 在服务端概率、事件执行表和真实回放补齐前，`verified_training_ready=false`。

这条边界是有意设计的：错着继续训练会让网络稳定地学会错误偏好；停止并要求观测则不会污染真实规则数据。

## 地图规则

| 项目 | 结果 | 可用范围 |
|---|---|---|
| I～V 的 43 个拓扑 | 路标与影语集交叉核验 43/43 一致 | 可作结构验证 |
| VI 的唯一拓扑 | 路标原始片段证明 6×5、14 格、17 边、起点、BOSS、AP 5 和 4 个固定内容格 | 可作结构验证 |
| I～VI 初始 AP | 客户端数据直接证明为 5/6/7/8/8/5 | 可用于真实规则 |
| I～V 节点数量/距离 | 与路标 `2026-08-31` 表逐项一致 | 只能称“路标兼容” |
| 模板/节点/门/小径概率 | 未公开 | 仅 synthetic |
| VI 其余 8 格 | 类型和分布未公开 | evidence 拒绝生成 |
| VI 入层全揭示 | 当前固定拓扑来源未证明 | 默认关闭 |

路标与当前影语集的节点数量/距离存在 26 项实质差异，完整清单保存在 `data/evidence/map_rule_conflicts_v1.json`。客户端 topic 表没有服务端生成约束，故真实对局日志裁决前不能把任一社区表命名为“官方规则”。运行时明确使用 `lubiao-2026-08-31` 兼容 profile。

第 VI 层的原始来源片段及 SHA 固定在 `tests/golden/lubiao_floor6_v1.json`。测试会从原始 `[x,y]` 数据重新执行 `[row=y,col=x]` 转换，再比较格位、边和固定节点，避免只用手抄后的本地 JSON 自证。

## 非作战事件与收益

客户端快照固定于：

- SHA-256：`aa2b1fc6ba0cc9ee29b9e6a08803550181c3a27189ac449efbad87608880d35b`
- Git blob SHA：`723f15432e989b6d0d402c38548a74a317f2f97c`
- 21 种节点、396 个 choice、338 个 choice scene、105 个关卡

客户端能直接证明 choice 的 ID、标题、描述、类型、`nextSceneId` 和显示提示；不能证明完整的 `scene -> choices` 可用图、服务端条件/成本/效果、随机池与概率。

PRTS 事件页 revision `422994` 补充了 38 个事件组的公开分支与结果，但该页面自己标注“编辑中”，因此作为 B 级交叉证据使用。完整客户端显示快照在 `data/evidence/rogue6_client_choice_snapshot_v1.json`，语义分支目录在 `data/evidence/rogue6_noncombat_event_catalog_v1.json`。

目录中的 102 条选项是人工归纳的语义分支，不是从服务端执行表逐项导出的完整 truth table；当前 golden tests 只锁定若干高风险动态成本、指定物品和离开分支，不能据此声称 102 条都已由独立来源逐项证明。它们继续保持 B 级、不得进入 evidence 模式自动结算。

审计确认并隔离了原运行时的硬错误，包括：

- “沉重的契约”把当前生命减半错误压成固定 `HP-3`；
- “线人”错加希望且漏掉珍贵加工品；
- “泪之聚落”把全部源石锭错误压成固定 12 锭；
- “擒与缚”把指定收藏品错误压成泛零件/收藏品计数；
- 多阶段事件被替换为统一作战或通用补给；
- REST、WISH、SACRIFICE、EXPEDITION、PORTAL、DUEL、商店、应急助力等均有结构性缺失；
- 未知事件池被错误当作等概率。

这些旧近似没有被包装成“修好”：它们现在只能在显式 `synthetic` profile 中运行。默认 profile 不消费它们。

## 可复核命令

```powershell
py -3 -m blackflow_rl audit-evidence
py -3 -m blackflow_rl validate-data
py -3 -m unittest discover -s tests -v
```

研究用合成采样必须显式写出：

```powershell
py -3 -m blackflow_rl sample-map --profile synthetic --seed 42
py -3 -m blackflow_rl train --profile synthetic --output artifacts/blackflow_policy.pt
```

第 VI 层研究先验还需同时选择三结局；“全揭示”只有在明确接受该未证假设时才启用：

```powershell
py -3 -m blackflow_rl sample-map --profile synthetic `
  --ending-route third --floor 6 --seed 42

py -3 -m blackflow_rl sample-map --profile synthetic `
  --ending-route third --floor 6 --assume-floor6-full-reveal --seed 42
```

## 下一步验真门槛

要把 `verified_training_ready` 变成 `true`，至少需要真实对局记录下列信息：

1. 每层完整地图、节点类型、坐标、边、AP、路线状态与版本；
2. 进入事件前的资源/背包/旗标；显示的 `scene_id` 与全部 `choice_id`；
3. 选择后的资源、具体物品、旗标、下一场景、节点变化和随机结果；
4. 商店库存、刷新、随机池和重复交互；
5. VI 的 8 个非固定格，以及入层究竟揭示哪些节点。

真实回放应进入独立 golden fixtures，并以 `前置状态 -> 合法选择 -> 后置状态` 做测试。概率未知时允许回放或外部注入，不允许默认等概率。
