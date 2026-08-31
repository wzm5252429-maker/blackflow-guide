# 黑流树海路线决策训练内核

## 交付范围

本模块实现 README 第二阶段中“根据已识别地图与背包给出动态路线建议”的决策核心：

- 按种子从 43 个已核验普通拓扑中抽样并填充 I～V 层；三结局配置再追加第 VI 层唯一固定、全揭示拓扑，并对来源未声明的 8 个内容槽使用显式合成先验；
- 自动结算作战节点和带作战标记的事件选项；
- 对非作战节点暴露合法分支，由玩家策略选择；
- 以部分可观测状态运行单玩家 PUCT-MCTS；
- 用图 policy/value 网络模仿搜索分布并估计回报；
- 支持回放池、训练恢复、固定种子策略对比与规则版本校验。

本模块不读取游戏窗口，也不执行点击。真实使用链路应为：

```text
截图/OCR/模板识别
       |
       v
ObservationAdapter -> GameState -> MCTS + Network -> action_id
                                                    |
                                                    v
                                         UIActionAdapter/路线高亮
```

## 目录

| 文件 | 职责 |
|---|---|
| `blackflow_rl/domain.py` | 不可变状态、地图、节点、事件选项和动作类型 |
| `blackflow_rl/rules.py` | 读取并校验带 SHA-256 的版本化规则 |
| `blackflow_rl/client_data.py` | 校验完整 `rogue_6` 客户端表并拒绝截断文件 |
| `blackflow_rl/map_templates.py` | 读取并严格校验 43 个普通拓扑与 1 个 VI 层固定拓扑 |
| `blackflow_rl/mapgen.py` | 模板抽样、配额求解、带权回溯与合成事件/收益 |
| `blackflow_rl/simulator.py` | 合法动作、纯状态转移、追猎、楼层切换与观测模型 |
| `blackflow_rl/features.py` | 固定上限的图、资源、背包和候选动作张量 |
| `blackflow_rl/mcts.py` | 不依赖 PyTorch 的单玩家 PUCT |
| `blackflow_rl/network.py` | 有向消息传递图网络与 policy/value heads |
| `blackflow_rl/training.py` | planner-guided rollout、回放训练、评估和 checkpoint |
| `blackflow_rl/cli.py` | 数据校验、地图预览、规划、训练和评估入口 |
| `data/rules/blackflow_sim_v1.json` | 所有可调地图与收益规则及来源说明 |
| `data/rules/blackflow_map_templates_v1.json` | I～V 层 43 个普通拓扑及 VI 层 1 个三结局固定拓扑 |
| `docs/random-map-generation.md` | 地图生成规则、证据等级和 CSP 算法 |

## 规则真实性边界

当前规则集 ID 是 `blackflow-template-csp-v1`。它有意把已核验事实和合成训练参数放在同一份可版本化配置中，但不声称逐字节复刻服务端生成器。

| 内容 | 当前来源/状态 |
|---|---|
| 网格尺寸 | I～V：5×3、5×4、7×5、8×5、10×5；VI 固定为 6×5 |
| 区域行动力 | I～V：5、6、7、8、8；三结局 VI：5 |
| NORMAL 0 初始基础资源 | 客户端：8 生命/上限、6 希望、8 锭、6 编队上限、0 护盾；零件默认 0 |
| 节点类型目录 | 客户端 `details.rogue_6.nodeTypeData` 的 21 类 |
| 地图拓扑 | I～V 为 43 个普通模板；路标工具另给出 VI 唯一固定模板 `floor-6-01` |
| VI 节点内容 | 路标工具明确 4 个固定内容节点；其余 8 格的类型范围和权重是待真实日志替换的合成训练先验 |
| 节点距离与数量范围 | 路标规则版本 2026-08-31，经版本化后写入 JSON |
| 移动语义 | 模板物理边；完成节点可通行；成对密道另作 0 费虚拟连接 |
| 事件名目录 | 用户仓库资料与客户端目录 |
| 地图模板概率、节点采样权重 | 合成参数，可替换 |
| 事件 scene→choices、随机概率 | 客户端表不完整，当前为合成选项 |
| 作战基础收益与掉落池 | 客户端表未给出，当前为合成收益 |
| FINAL/EVACUATE 跨层 AP 细节 | FINAL 剩余 AP 转希望；EVACUATE 按“基础 AP + 保留 AP”建模 |

公开参考：

- [影语集黑流地图工具](https://arkrog.com/tool/blackflowmap)
- [路标档案馆黑流路线](https://www.lubiao.wiki/tools/blackstream-route)
- [PRTS 黑流树海页面](https://prts.wiki/w/沉沦者的黑流树海)

加载器只接受 `source_data/roguelike_topic_table_full.json`。当前 golden SHA-256 为 `aa2b1fc6ba0cc9ee29b9e6a08803550181c3a27189ac449efbad87608880d35b`；较短的 `roguelike_topic_table.json` 和 `part_*`/`rem_*` 传输片段会被拒绝，避免静默用残缺数据训练。

## 状态、动作和奖励

模拟器保留完整真实地图；编码器只向策略暴露已揭示的节点类型，其他节点只显示“未知凶险/未知诡谲”。MCTS 搜索 `belief_state()`：当前层未揭示内容只按可见大类替换为期望代表，已看见但尚未进入的节点也不会泄露其具体事件、选项或随机掉落；未来楼层整体替换为仅由规则版本决定的 canonical determinization。真实未来拓扑、隐藏收益、生成 seed 和 RNG 都不会进入搜索。第 VI 层按游戏语义在进入时立即揭示全部节点，是这一观测规则的显式例外。

动作空间固定为 56：

- `0..49`：移动到对应节点；
- `50..55`：选择当前事件的第 1–6 个选项。

合法动作 mask 处理可达性、剩余行动力、事件资源门槛和终局状态。每次移动按穿过的新路径边数扣除 AP；走过已完成节点不重复领奖。AP 不足且无可达前沿时自动触发并默认赢得追猎。

收益由资源势函数差分、层完成奖励、整局完成奖励和追猎惩罚组成：

```text
r_t = Φ(resources_after) - Φ(resources_before)
      + floor_clear + run_clear + chase_penalty
```

`Φ` 的每个资源权重都在规则 JSON 中。所谓“正确策略”因此是相对于该显式目标函数的最优策略，而不是脱离玩家偏好的唯一答案。后续可为结局优先、藏品优先、稳健通关等目标分别维护规则 profile。

## 为什么不是 AlphaZero 式自我对弈

这是单玩家、随机且部分可观测的规划问题，没有对手回合。因此 MCTS 回传不翻转符号，边值使用：

```text
Q(s,a) <- reward(s,a) / value_scale + gamma * V(s')
```

网络给出合法动作先验和 `[-1, 1]` value。训练数据来自 MCTS 引导的单局 rollout，而不是双方 self-play。当前随机内容在生成地图时确定；得到真实概率表后，应把 belief 近似升级为 root belief sampling 或显式 chance node。

## 训练与评估

安装并验证：

```powershell
py -3.13 -m pip install -r requirements.txt
py -3.13 -m blackflow_rl validate-data
py -3.13 -m unittest discover -s tests -v
```

短训练：

```powershell
py -3.13 -m blackflow_rl train `
  --episodes 5 --simulations 8 --hidden-dim 64 `
  --output artifacts/blackflow_policy.pt
```

正式训练示例：

```powershell
py -3.13 -m blackflow_rl train `
  --episodes 500 --simulations 96 --batch-size 128 `
  --updates 8 --hidden-dim 128 --device cuda `
  --output artifacts/blackflow_policy.pt
```

相同种子对比 random、启发式和 MCTS：

```powershell
py -3.13 -m blackflow_rl evaluate `
  --checkpoint artifacts/blackflow_policy.pt `
  --episodes 100 --seed-start 10000 --simulations 64
```

checkpoint 原子写入，包含网络、优化器、回放池、Python/NumPy/Torch RNG 状态、训练配置和规则 SHA。规则被修改后旧 checkpoint 会拒绝加载，防止不同收益定义之间误续训。`value_scale=256` 按当前合成回报范围校准，使高收益轨迹仍可区分；替换收益 profile 后应重新统计回报分位数并同步调整。

## 接入真实第二阶段还需要什么

1. `ObservationAdapter`：把地图截图、节点类别/坐标、当前位置、剩余 AP、资源和背包识别结果转换为状态。
2. 地图身份与历史跟踪：跨截图维持节点 `completed/revealed` 状态，处理识别置信度和人工纠错。
3. `UIActionAdapter`：将动作映射为路线高亮或点击；建议先只显示建议，积累离线回放后再启用自动点击。
4. 真实规则补全：模板真实权重、节点类型先验、事件 scene→choice、选项条件、概率、战斗掉落、商店池和高难天气。
5. 评估门槛：独立测试种子上稳定优于 random 与现有 `scoreRoute` 启发式，再用真实局日志做离线回放验证。

当前背包是训练接口占位：数值型资源使用聚合计数，关键物品使用字符串集合，编码器只内置六个关键物品标志；它还不能表达具体收藏品效果、同名物品叠层、招募券类别或零件被动。真实背包识别接入前应改为 item-ID 集合/多重集编码器。

这些缺口不会被合成训练掩盖：规则集标题、CLI 输出和 checkpoint 都带规则 ID/SHA；替换真实数据时可以保持模拟器、MCTS 与网络接口不变。
