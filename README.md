# 黑流树海节点规则与收益核验

该项目使用 LangGraph 对记录《明日方舟》集成战略「沉沦者的黑流树海」节点规则和收益的文档进行事实核验，不再使用金融、法律或通用文档审查标准。

## 核验流程

```text
待核验文档
  ├─ 客户端数据与官方资料核验员 ─┐
  ├─ PRTS / 影语集 / 路标核验员 ─┼─→ 独立审查 ─→ 两轮交叉讨论 ─→ 最终报告
  └─ 玩家实测与统计核验员 ──────┘
```

每一路先独立搜索和取证，再独立判断。讨论阶段三个 Agent 能看到彼此的结论和原始证据包，可以质疑来源、识别版本差异、撤回误判或补充遗漏。

报告会针对具体节点记录给出：

- 核验状态：一致、不一致、部分一致、版本差异、无法确认；
- 文档记录值与参考值；
- 节点名、类型、选项、触发条件、固定/随机收益；
- 来源 URL、证据等级、置信度；
- 尚未解决的资料冲突和优先修改项。

## 证据优先级

1. 当前版本客户端公开数据和鹰角官方公告；
2. PRTS 等可追溯 Wiki；
3. 影语集 `arkrog.com`、路标档案馆 `lubiao.wiki` 等攻略数据库的交叉记录；
4. 带游戏版本、难度、前置条件、样本量和原始视频/截图的玩家实测；
5. 玩家个人经验或主观评级。

不同版本的数值不能直接判错。随机事件不会因为一两局结果就被判定为固定收益。攻略站互相转载时也不会被当成两个独立证据。

## 在 PyCharm 中运行

建议使用 Python 3.11 或 3.12。在 PyCharm Terminal 中执行：

```powershell
python -m pip install -r requirements.txt
python document_validator.py "C:\文档\黑流树海节点规则.md"
```

程序默认读取项目内：

```text
source_data/roguelike_topic_table_full.json
```

并使用其中的 `rogue_6` 数据。该数据表的节点类型、事件选项、事件场景、关卡、区域、零件等相关字段会成为客户端数据证据包。

## 加入玩家真实数据

复制 `player_data_template.csv`，删除示例行后填写逐局记录：

```powershell
python document_validator.py "C:\文档\黑流树海节点规则.md" `
  --player-data "C:\数据\黑流树海实测.csv"
```

有效实测最好包含：游戏版本、难度、节点名、选择、前置藏品/分队/零件、进入前资源、离开后资源、奖励、视频或截图链接和记录日期。缺少这些条件的社区帖子只能作为经验参考。

## 其他参数

```powershell
# 仅核验非战斗节点
python document_validator.py "节点规则.md" --rubric "只核验非战斗节点、选项和资源收益"

# 使用其他数据表或主题
python document_validator.py "节点规则.md" `
  --game-data "C:\GameData\roguelike_topic_table_full.json" `
  --topic-id rogue_6

# 降低讨论次数和 API 调用量
python document_validator.py "节点规则.md" --rounds 1
```

默认模型为 `gpt-5.6-luna`，可用 `--model` 或环境变量 `OPENAI_MODEL` 修改。程序自动读取 `OPENAI_API_KEY`。

默认两轮讨论大约包含 13 次模型请求，其中三次使用联网搜索。一次讨论约为 10 次请求。实际费用取决于文档、客户端参考数据长度、搜索结果和模型。

输出目录中会生成：

- `.md`：最终核验报告；
- `.audit.json`：三路检索证据、三份独立审查及全部讨论记录。

支持 `.txt`、`.md`、`.json`、`.csv`、`.yaml`、`.pdf`、`.docx`。扫描 PDF 需要先 OCR。

## 路线决策训练内核

项目现已包含第二阶段所需的路线决策核心：I～V 层保存 43 个固定拓扑，三结局另有 1 个第 VI 层固定拓扑（合计 44 个模板），并配有约束求解、单玩家 PUCT-MCTS、图 policy/value 网络、planner-guided rollout、固定种子评估和 checkpoint 恢复。默认 `evidence` profile 不会用未知模板/节点权重生成随机地图；要运行当前约束兼容但非官方概率的模拟，必须显式选择 `synthetic`。VI 只有拓扑和 4 个固定节点有来源，其余 8 格的类型/分布及“全图揭示”都没有独立证据。它能对已建模或人工给定结果的状态做路线规划；未知真实事件目前仍由人选择，系统只接收结算后状态，尚不能对真实 UI 的未知候选分支直接给建议。截图识别、背包 OCR 和客户端点击仍应由独立适配器接入。

```text
显式 synthetic 地图/事件 -> 部分可观测状态 -> MCTS 搜索 -> 动作
                                               ^          |
                                               |          v
                                           图神经网络 <- 回放训练
```

快速验证：

```powershell
py -3.13 -m pip install -r requirements.txt
py -3.13 -m blackflow_rl validate-data
py -3.13 -m blackflow_rl audit-evidence
py -3.13 -m blackflow_rl sample-map --seed 42 --profile synthetic
py -3.13 -m blackflow_rl sample-map --seed 42 --floor 6 --ending-route third --profile synthetic
py -3.13 -m blackflow_rl plan --seed 42 --simulations 64 --profile synthetic
py -3.13 -m blackflow_rl simulate --seed 42 --policy mcts --profile synthetic
py -3.13 -m unittest discover -s tests -v
```

启动一轮适合普通 CPU 的短训练并比较基线：

```powershell
py -3.13 -m blackflow_rl train --profile synthetic --output artifacts/blackflow_policy.pt
py -3.13 -m blackflow_rl evaluate `
  --profile synthetic `
  --checkpoint artifacts/blackflow_policy.pt `
  --episodes 20 --simulations 32
```

`evidence` 是 CLI 默认值，会在真实随机权重未知时明确拒绝自行生成地图；训练器也只接受显式 `synthetic` profile。上述命令只用于打通合成管线，不代表已经收敛，更不代表真实通关率。正式研究若扩大 `--episodes`、`--simulations` 与网络宽度，也必须保留 synthetic 标签并使用独立训练/验证种子。

`data/rules/blackflow_map_templates_v1.json` 保存 I～V 的 43 个普通模板和 VI 的 1 个三结局固定模板；第 VI 层不会进入普通路线，只有 `enable_third_ending=True`（CLI 为 `--ending-route third`）才追加。路标与影语集的节点数量/距离规则仍有 26 项未解决冲突，当前 JSON 只是版本化的路标兼容配置。事件目录中的 38 组、102 条选项只是 PRTS B 级语义摘要，不是服务端可执行 truth table；严格事件模式不会把未知事件当成无效果：到达后返回 `needs_observation`，对应证据状态为 `NEEDS_OBSERVATION`。checkpoint v2 同时校验 profile、规则 SHA 和覆盖关键生成/模拟语义的 environment SHA。地图证据边界见 `docs/random-map-generation.md`，决策架构见 `docs/decision-core.md`。
