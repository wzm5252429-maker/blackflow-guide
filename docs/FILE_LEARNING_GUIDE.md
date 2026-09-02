# Blackflow 核心代码学习指南

> 目标：只学习足以**大致读懂、跟踪运行、修改小功能**的核心知识，不追求先学完整门课。
>
> 优先级：**MCTS、神经网络 > 模拟器 > 地图识别与计算机视觉 > AnchoredTouch > 网站搭建**。

## 导航

- [1. 30 分钟快速开始](#1-30-分钟快速开始)
- [2. 学习路线与项目边界](#2-学习路线与项目边界)
- [3. 蒙特卡洛树搜索（MCTS）](#3-蒙特卡洛树搜索mcts)
- [4. 神经网络与图神经网络（GNN）](#4-神经网络与图神经网络gnn)
- [5. 模拟器](#5-模拟器)
- [6. 地图识别与计算机视觉](#6-地图识别与计算机视觉)
- [7. AnchoredTouch](#7-anchoredtouch)
- [8. 网站搭建](#8-网站搭建)
- [9. 六周执行表](#9-六周执行表)
- [10. 固定的代码阅读方法](#10-固定的代码阅读方法)
- [11. 术语速查与暂缓内容](#11-术语速查与暂缓内容)

## 1. 30 分钟快速开始

先不要看课程。用半小时建立对决策核心的第一印象。

### 第一步：进入决策核心并安装最小依赖

从完整 GitHub 仓库根目录执行：

```powershell
cd modules/decision-core/blackflow-research
py -3.13 -m pip install -r requirements-core.txt
```

如果本机 Python 版本不同，把 `py -3.13` 换成 `python`。

### 第二步：运行最小测试和一次搜索

```powershell
py -3.13 -m unittest discover -s tests -p "test_mcts.py" -v
py -3.13 -m blackflow_rl plan --seed 42 --simulations 8 --profile synthetic
```

成功标准：测试显示 `OK`，规划命令打印候选动作、访问次数或建议动作，而不是证据不足错误。

### 第三步：只打开四个文件

按顺序快速浏览：

1. [test_mcts.py](../modules/decision-core/blackflow-research/tests/test_mcts.py)：先看程序必须满足什么行为。
2. [domain.py](../modules/decision-core/blackflow-research/blackflow_rl/domain.py)：找到状态与动作的数据结构。
3. [simulator.py](../modules/decision-core/blackflow-research/blackflow_rl/simulator.py)：只找 `legal_action_ids()` 和 `transition()`。
4. [mcts.py](../modules/decision-core/blackflow-research/blackflow_rl/mcts.py)：只找 `search()`、`_simulate()`、`_expand()`、`_select_edge()`。

此时看不懂细节是正常的。第一遍只回答一个问题：**MCTS 怎样反复询问模拟器，再选出下一步？**

## 2. 学习路线与项目边界

### 2.1 建议时间分配

建议总投入约 **55～70 小时**。前四项合计应占总时间的 80% 左右。

| 顺序 | 模块 | 建议时间 | 达标产出 |
|---:|---|---:|---|
| 0 | 最低限度 Python、数学和模拟器接口 | 6～8 小时 | 能读类型标注、数组形状、状态与动作 |
| 1A | MCTS | 12～16 小时 | 能手算一次选择、扩展和回传 |
| 1B | 神经网络与 GNN | 14～18 小时 | 能解释输入、两个输出头和训练损失 |
| 2 | 模拟器深入阅读 | 7～9 小时 | 能追踪一次完整状态转移 |
| 3 | 地图识别与计算机视觉 | 8～10 小时 | 能从截图追到节点图 JSON |
| 4 | AnchoredTouch | 4～5 小时 | 能解释窗口、坐标、原生调用和校验 |
| 5 | 网站搭建 | 4～6 小时 | 能找到页面、状态和样式并改小功能 |

优先级不等于严格阅读顺序。MCTS 依赖模拟器接口，所以先用 2～4 小时理解接口，再集中学习搜索；模拟器内部规则可以之后深入。

### 2.2 核心依赖关系

```text
GameState（当前状态）
    │
    ├── Simulator.legal_action_ids / transition ──┐
    │                                              │
    └── belief_state → FeatureEncoder → GNN ──┐   │
                                               │   │
                                      policy 先验 + value 估值
                                               │   │
                                               ▼   ▼
                                             PUCT-MCTS
                                               │
                               visit_policy + selected_action
                                      │                    │
                                      ▼                    ▼
                                  训练样本             执行下一步
```

现实接入链路是另一条线：

```text
游戏窗口截图
  → 地图/状态识别
  → 适配为 GameState（目前尚未完整接通）
  → MCTS 给出动作
  → AnchoredTouch/执行层操作窗口（目前尚未完整接通）
```

### 2.3 当前代码的真实边界

- 模拟器、MCTS、GNN、训练循环和相应测试已经存在。
- 端到端训练目前必须使用 `synthetic` profile；生成的数据含未验证先验，不能当成真实游戏策略效果。
- 地图识别器已经能输出 `map_result.json` 和 `map_graph.json`，但到 `FloorMap` / `GameState` 的正式适配器仍未完成。
- AnchoredTouch 工具负责输入、安装与校验，目前尚未和路线决策组成完整自动闭环。
- 网站目前主要是独立的路线参谋界面，不等于模型已经在网页中实时推理。

还要区分项目里的两个同名概念：

- 决策核心的 [blackflow_rl/simulator.py](../modules/decision-core/blackflow-research/blackflow_rl/simulator.py) 是供 MCTS 使用的路线世界模型，是本文“模拟器”章节的主体。
- 第一场战斗工程的 [blackflow/simulator.py](../modules/battle-automation/blackflow-first-battle/blackflow/simulator.py) 只是按人工战术的设定胜率模拟输赢；其 [learner.py](../modules/battle-automation/blackflow-first-battle/blackflow/learner.py) 用 epsilon-greedy Q-learning 在整套人工战术之间选择，与 GNN/MCTS 不是同一套学习系统。
- Windows PC 客户端或安卓模拟器是游戏运行载体，也不是上述 Python 状态模拟器。

### 2.4 只补这些前置知识

Python 只需会函数、类、模块、`dataclass`、`Enum`、类型标注、异常、列表与字典，以及用断点或日志看变量。

数学只需会向量、矩阵、点积、概率分布、期望、`softmax`、梯度下降、交叉熵和均方误差的直觉。不需要先完成高等数学或概率论整门课程。

## 3. 蒙特卡洛树搜索（MCTS）

| 项目 | 内容 |
|---|---|
| 目标 | 能从 `search()` 追到选择、扩展、评估、回传和最终选动作 |
| 预计时间 | 12～16 小时 |
| 首个入口 | [test_mcts.py](../modules/decision-core/blackflow-research/tests/test_mcts.py) |
| 核心实现 | [mcts.py](../modules/decision-core/blackflow-research/blackflow_rl/mcts.py) |

### 3.1 必须懂的 20%

先把环境抽象成四件事：

- **状态 `state`**：当前地图、资源、生命、已走节点等信息。
- **动作 `action`**：下一步选择哪个节点或事件选项。
- **状态转移 `transition`**：执行动作后得到新状态、即时奖励和终止标记。
- **合法动作 `legal_action_ids`**：当前真正允许选择的动作集合。

MCTS 每次模拟包含四步：

1. **选择（selection）**：按 PUCT 分数沿树向下。
2. **扩展（expansion）**：给第一次到达的状态创建合法动作边。
3. **评估（evaluation）**：由启发式或神经网络给出先验概率与叶子价值。
4. **回传（backup）**：把折扣后的价值累积到路径上的边。

常见教材记号与本项目字段的对应关系：

| 含义 | 常见记号 | 代码字段 |
|---|---|---|
| 访问次数 | `N` | `edge.visit_count` |
| 累计价值 | `W` | `edge.value_sum` |
| 平均价值 | `Q = W / N` | `edge.mean_value` |
| 先验概率 | `P` | `edge.prior` |

`P` 由当前评估器提供，并不必然来自神经网络。未加载 checkpoint 的 `plan` 使用 `HeuristicEvaluator`；加载训练 checkpoint 后，才由 `training.py` 中的 `TorchPolicyValueEvaluator` 把网络输出转换为先验和价值。

本项目实际使用的选择分数是：

```text
score = edge.mean_value
      + c_puct × edge.prior × sqrt(max(1, parent.visit_count))
        / (1 + edge.visit_count)
```

本项目实际使用的回传是：

```text
value = immediate_reward / reward_scale + gamma × downstream_value
```

这是单人规划，不是双方轮流行动的零和棋类，因此回传时**不逐层翻转正负号**。

### 3.2 对照代码的阅读顺序

| 顺序 | 文件 | 只看什么 |
|---:|---|---|
| 1 | [domain.py](../modules/decision-core/blackflow-research/blackflow_rl/domain.py) | `GameState`、`Action`、资源和节点结构 |
| 2 | [simulator.py](../modules/decision-core/blackflow-research/blackflow_rl/simulator.py) | `action_size`、`legal_action_ids()`、`transition()` |
| 3 | [test_mcts.py](../modules/decision-core/blackflow-research/tests/test_mcts.py) | 三动作玩具环境和“回传不翻号”测试 |
| 4 | [mcts.py](../modules/decision-core/blackflow-research/blackflow_rl/mcts.py) | `_Edge`、`_Node`、`search()` 和四个搜索阶段 |
| 5 | [agents.py](../modules/decision-core/blackflow-research/blackflow_rl/agents.py) | `HeuristicEvaluator`、随机策略和启发式基线 |
| 6 | [cli.py](../modules/decision-core/blackflow-research/blackflow_rl/cli.py) | `plan` 和 `simulate` 怎样组装环境与搜索器 |

阅读时留意三个实现细节：每次 `search()` 都重新建树；状态转移第一次走过某条边时才计算并缓存；温度为 0 时按访问次数选动作，再用先验和较小动作 ID 打破平局。

### 3.3 最有效的练习

1. 在纸上画一个只有 3 个动作、2 层深的小树，手算前 5 次模拟后的 `visit_count`、`value_sum`、`mean_value` 和 `prior`。
2. 临时输出根节点各动作统计，把模拟次数从 8 改成 64，比较访问分布。
3. 屏蔽一个动作，确认它不会出现在根节点合法边中。
4. 修改 `gamma` 或 `c_puct`，先预测结果方向，再运行测试验证。

### 3.4 必修资料

- [Sutton、Barto《Reinforcement Learning: An Introduction》开放版](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)：只读第 3 章的状态、动作、奖励、回报，以及第 5 章的蒙特卡洛估计。
- [Browne 等：A Survey of Monte Carlo Tree Search Methods](https://cs.gettysburg.edu/~tneller/cs371/mcts-survey.pdf)：只读基本算法、树策略和 UCT/PUCT 相关部分。

<details>
<summary>可选中文视频与延伸资料</summary>

- [浙江大学：蒙特卡洛树搜索 MCTS 入门（B 站）](https://www.bilibili.com/video/BV1kT411G7nE/)
- [强化学习课程：MDP 与蒙特卡洛方法（B 站）](https://www.bilibili.com/video/BV1dV4y1n7Sn/)
- [AlphaZero 论文](https://arxiv.org/abs/1712.01815)：只看 policy/value 网络如何与搜索配合。

</details>

### 3.5 达标检查

- [ ] 能解释为什么既需要 `mean_value` 又需要 `prior`。
- [ ] 能指出非法动作在哪里被过滤。
- [ ] 能解释访问次数怎样影响探索。
- [ ] 能说明为什么单人回传不翻转价值符号。
- [ ] 能解释搜索结束后为什么通常按访问次数而非瞬时网络概率选动作。

## 4. 神经网络与图神经网络（GNN）

| 项目 | 内容 |
|---|---|
| 目标 | 能解释特征编码、消息传递、策略头、价值头和训练目标 |
| 预计时间 | 14～18 小时 |
| 首个入口 | [test_network_and_training.py](../modules/decision-core/blackflow-research/tests/test_network_and_training.py) |
| 核心实现 | [features.py](../modules/decision-core/blackflow-research/blackflow_rl/features.py)、[network.py](../modules/decision-core/blackflow-research/blackflow_rl/network.py)、[training.py](../modules/decision-core/blackflow-research/blackflow_rl/training.py) |

### 4.1 必须懂的 20%

普通神经网络只需掌握张量、线性层、激活函数、前向传播、损失、反向传播和优化器。

GNN 在本项目中的核心动作可以概括为：

```text
节点的新表示
= 自身特征
+ 出边邻居信息
+ 入边邻居信息
```

网络有两个输出目标：

- **策略头（policy head）**：为节点动作和事件选项动作输出分数。
- **价值头（value head）**：估计当前状态的归一化长期回报，末层 `Tanh` 把输出限制在 `[-1, 1]`。

不同状态的合法动作不同，因此必须使用**合法动作掩码（action mask）**。实现把非法动作的 logit 设为当前数据类型可表示的最小有限值；经过 `softmax` 后，它们的概率近似为 0。

### 4.2 先分清单样本与批量形状

| 数据 | `predict()` 单样本 | `forward()` / 训练批量 |
|---|---|---|
| 节点特征 | `[N, F]` | `[B, N, F]` |
| 邻接矩阵 | `[N, N]` | `[B, N, N]` |
| 节点掩码 | `[N]` | `[B, N]` |
| 全局特征 | `[G]` | `[B, G]` |
| 事件选项特征 | `[O, OF]` | `[B, O, OF]` |
| 动作掩码 | `[N + O]` | `[B, N + O]` |
| 策略输出 | `[N + O]` | `[B, N + O]` |
| 价值输出 | 标量 | `[B]` |

`N` 是填充后的最大节点数，`O` 是最大事件选项数，`B` 是 batch size。真正节点和合法动作分别由 `node_mask`、`action_mask` 标记。

### 4.3 当前训练链路

```text
真实 episode state
  → belief_state（只保留可观测信息）
  → MCTS 搜索
  → visit_policy 作为策略目标
  → 折扣累计回报 / value_scale，再裁剪到 [-1, 1] 作为价值目标
  → ReplaySample
  → policy loss + value MSE
  → AdamW 更新，并裁剪梯度
```

代码还计算 entropy 供观察，但**没有把 entropy 加进总损失**。checkpoint 同时记录环境摘要，环境 SHA 不匹配时拒绝恢复，避免把模型错误地用于另一套规则。

> 重要边界：`Trainer` 明确拒绝非 `synthetic` 模拟器。这里学到的是训练管线，不是已经验证的真实游戏模型。

### 4.4 对照代码的阅读顺序

| 顺序 | 文件 | 只看什么 |
|---:|---|---|
| 1 | [test_network_and_training.py](../modules/decision-core/blackflow-research/tests/test_network_and_training.py) | 输入输出形状、非法动作、单批训练和 checkpoint 往返 |
| 2 | [features.py](../modules/decision-core/blackflow-research/blackflow_rl/features.py) | `EncodedState`、`FeatureEncoder.encode()`、`stack_encoded()` |
| 3 | [network.py](../modules/decision-core/blackflow-research/blackflow_rl/network.py) | 三个编码器、消息传递层、两个策略分支和价值头 |
| 4 | [training.py](../modules/decision-core/blackflow-research/blackflow_rl/training.py) | `TorchPolicyValueEvaluator`、`collect_episode()`、`train_batch()` |
| 5 | [mcts.py](../modules/decision-core/blackflow-research/blackflow_rl/mcts.py) | 网络返回的 `priors`、`value` 如何进入搜索 |

### 4.5 最有效的练习

先运行：

```powershell
py -3.13 -m unittest discover -s tests -p "test_network_and_training.py" -v
```

然后依次做：

1. 给编码结果打印所有数组的 `shape` 和 `dtype`。
2. 屏蔽一个动作，确认其 logit 极小且 softmax 概率近似为 0。
3. 用测试中的单个 `ReplaySample` 重复训练，观察总损失下降。
4. 保存并重新加载 checkpoint，确认同一输入的输出一致。

### 4.6 必修资料

- [PyTorch 官方 Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)：只学 Tensors、Build Model、Autograd、Optimization、Save & Load。
- [《动手学深度学习》中文开放版](https://zh.d2l.ai/)：选读 2.1、2.3、2.5、3.4、4.1、4.7、5.5；不必从头通读。

<details>
<summary>可选 GNN 资料与中文视频</summary>

- [PyTorch Geometric 官方 Colab 教程](https://pytorch-geometric.readthedocs.io/en/latest/get_started/colabs.html)：用于理解消息传递；本项目并不依赖 PyG。
- [Stanford CS224W 官方课程页](https://web.stanford.edu/class/cs224w/)：只看图表示与 GNN 消息传递。
- [同济子豪兄：CS224W 图机器学习中文精讲（B 站）](https://www.bilibili.com/video/BV1pR4y1S7GA/)
- [李宏毅课程助教：图神经网络 GNN（B 站）](https://www.bilibili.com/video/BV1G54y1971S/)

</details>

### 4.7 达标检查

- [ ] 能画出 `GameState → EncodedState → network → policy/value`。
- [ ] 能说出每个输入张量的维度含义。
- [ ] 能指出出边与入边消息在哪里聚合。
- [ ] 能说明策略损失、价值损失分别在纠正什么。
- [ ] 能解释为什么当前训练结果不能直接宣称适用于真实游戏。

## 5. 模拟器

| 项目 | 内容 |
|---|---|
| 目标 | 能给定状态和动作，追到合法性、下一状态、奖励和终止条件 |
| 预计时间 | 接口 2～4 小时，内部实现再用 7～9 小时 |
| 首个入口 | [test_simulator.py](../modules/decision-core/blackflow-research/tests/test_simulator.py) |
| 核心实现 | [simulator.py](../modules/decision-core/blackflow-research/blackflow_rl/simulator.py) |

### 5.1 必须懂的 20%

- **马尔可夫决策过程（MDP）**：状态、动作、转移、奖励、终止。
- **确定性与随机种子（seed）**：相同规则、状态、动作和种子应可复现。
- **部分可观测性**：真实隐藏内容不应泄漏给规划器，所以训练先生成 `belief_state`。
- **规则与数据分离**：代码负责怎样变化，JSON 和模板描述具体规则与拓扑。
- **未知机制处理**：证据不足时显式拒绝、暂停或降级，不能静默编造规则。

### 5.2 对照代码的阅读顺序

| 顺序 | 文件 | 作用 |
|---:|---|---|
| 1 | [test_simulator.py](../modules/decision-core/blackflow-research/tests/test_simulator.py) | 从可执行例子理解期望行为 |
| 2 | [domain.py](../modules/decision-core/blackflow-research/blackflow_rl/domain.py) | 状态、动作、资源、节点和转移对象 |
| 3 | [rules.py](../modules/decision-core/blackflow-research/blackflow_rl/rules.py) | 从规则 JSON 建立运行时规则 |
| 4 | [map_templates.py](../modules/decision-core/blackflow-research/blackflow_rl/map_templates.py) | 固定拓扑加载与不变量校验 |
| 5 | [mapgen.py](../modules/decision-core/blackflow-research/blackflow_rl/mapgen.py) | 约束、回溯、随机种子和节点填充 |
| 6 | [simulator.py](../modules/decision-core/blackflow-research/blackflow_rl/simulator.py) | `reset()`、合法动作、转移、跨层和 belief state |
| 7 | [decision-core.md](../modules/decision-core/blackflow-research/docs/decision-core.md) | 设计意图和真实接入边界 |

第一遍只读公开方法：`reset()`、`legal_action_ids()`、`decode_action()`、`transition()`、`belief_state()`。以下划线开头的内部方法放到第二遍。

### 5.3 最有效的练习

1. 选择一个测试，逐字段写出动作前后的 `GameState`。
2. 固定 seed 重跑两次，确认地图和结果一致。
3. 添加一个极小奖励规则，并先补测试再改实现。
4. 制造未知事件，观察 evidence 与 synthetic profile 的行为差异。

### 5.4 必修资料

- [Sutton、Barto 教材第 3 章](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)：只学 MDP 五要素。
- [Gymnasium 官方：创建自定义环境](https://gymnasium.farama.org/main/tutorials/environment_creation/)：重点看 observation、action、`reset()`、`step()`、terminated。

<details>
<summary>可选中文视频</summary>

- [Stable Baselines3 / Gym 自定义环境教程（B 站）](https://www.bilibili.com/video/BV1ty4y197JE/)：只看环境接口，不必继续学整套算法。

</details>

### 5.5 达标检查

- [ ] 能解释模拟器为何可以被 MCTS 重复调用。
- [ ] 能找到一个动作的合法性检查、奖励和终局判断。
- [ ] 能区分真实 episode state 与规划使用的 belief state。
- [ ] 能说明 evidence 和 synthetic profile 的用途不同。

## 6. 地图识别与计算机视觉

| 项目 | 内容 |
|---|---|
| 目标 | 能从窗口截图追到视觉检测结果、节点和道路图 |
| 预计时间 | 8～10 小时 |
| 地图入口 | [BFMapRecognizer README](../modules/map-recognition/BFMapRecognizer/README_zh-CN.md) |
| 战斗视觉入口 | [vision.py](../modules/battle-automation/blackflow-first-battle/blackflow/vision.py) |

### 6.1 先区分两条视觉管线

| 管线 | 主要技术 | 当前输出 |
|---|---|---|
| BFMapRecognizer | MAA FramePool、OCR、节点识别、ONNX CorridorNet、图后处理 | 地图原图、节点/道路诊断图、`map_result.json`、`map_graph.json` |
| 第一场战斗自动化 | MSS 截图、OpenCV 模板匹配/颜色范围、ROI、坐标标定 | 战斗状态检测、胜负检测、点位和动作条件 |

两者都处理截图，但并不是同一套实现。先分别看懂输入输出，再考虑复用。

### 6.2 必须懂的 20%

- 屏幕坐标、窗口坐标、客户区坐标和 DPI 缩放。
- 截图、感兴趣区域（ROI）裁剪，以及 RGB、BGR、灰度图。
- 模板匹配分数、阈值、误检和漏检。
- OCR 适合文字，模板匹配适合外观相对固定的图标。
- ONNX Runtime 的输入预处理、推理、输出后处理。
- 节点检测结果怎样经过去重、道路判断和图结构导出。

### 6.3 对照代码的阅读顺序

第一场战斗视觉链路：

1. [window.py](../modules/battle-automation/blackflow-first-battle/blackflow/window.py)：找到窗口和客户区。
2. [capture.py](../modules/battle-automation/blackflow-first-battle/blackflow/capture.py)：截取 BGRA 并转换为 BGR。
3. [vision.py](../modules/battle-automation/blackflow-first-battle/blackflow/vision.py)：ROI、模板匹配、颜色范围和条件组合。
4. [calibrate.py](../modules/battle-automation/blackflow-first-battle/blackflow/calibrate.py)：交互裁模板和标坐标。
5. [test_config_and_vision.py](../modules/battle-automation/blackflow-first-battle/tests/test_config_and_vision.py)：看最小可验证行为。

地图识别链路：

1. [README_zh-CN.md](../modules/map-recognition/BFMapRecognizer/README_zh-CN.md)：先弄清输入、产物和安全边界。
2. 本机发布包中的 `MapRecognizer.ps1`：完整脚本没有上传到 GitHub，阅读本机副本。
3. 一次成功输出中的 `map_original.png`、`map_result.json`、`map_graph.json` 和 `maa_events.jsonl`。
4. [domain.py](../modules/decision-core/blackflow-research/blackflow_rl/domain.py)：确认适配器最终必须构造的 `FloorMap`、`MapNode` 和 `GameState`。

### 6.4 最有效的练习

1. 固定一张截图并裁出一个模板。
2. 用 OpenCV `matchTemplate` 得到分数并画出识别框。
3. 上下调整阈值，记录误检与漏检怎样变化。
4. 在截图上手工核对 `map_graph.json` 的每个节点和边。
5. 最后再看 ONNX 推理；第一遍可以把 CorridorNet 当作“输入候选道路图像，输出有路概率”的黑盒。

### 6.5 必修资料

- [OpenCV 官方模板匹配教程](https://docs.opencv.org/4.x/de/da9/tutorial_template_matching.html)
- [ONNX Runtime 官方 Python 快速开始](https://onnxruntime.ai/docs/get-started/with-python.html)

<details>
<summary>可选中文视频</summary>

- [OpenCV 课程：含模板匹配与 OCR（B 站）](https://www.bilibili.com/video/BV1xgW3zUEnt/)
- [OpenCV 入门：含模板匹配（B 站）](https://www.bilibili.com/video/BV1BenMz9ESE/)

</details>

### 6.6 达标检查

- [ ] 能从截图追到 ROI 和识别结果。
- [ ] 能解释阈值、缩放和坐标系为什么会导致失败。
- [ ] 能读懂 `map_graph.json` 的节点与边。
- [ ] 能说明地图识别器与 `GameState` 之间还缺什么数据和适配。

## 7. AnchoredTouch

实际技术名称是 **AnchoredTouch**。部分文件仍保留早期的 `AnchoredTorch` 拼写，但它们不是 PyTorch 的 `torch`。

| 项目 | 内容 |
|---|---|
| 目标 | 能理解窗口定位、坐标换算、C# 原生调用以及安装校验 |
| 预计时间 | 4～5 小时 |
| 首个入口 | [maa-pc-anchored-touch.md](../modules/decision-core/blackflow-research/docs/maa-pc-anchored-touch.md) |
| 最小实验 | [AnchoredTouchSmoke.cs](../modules/decision-core/blackflow-research/tools/anchored_touch_smoke/AnchoredTouchSmoke.cs) |

### 7.1 必须懂的 20%

- `HWND` 窗口句柄，以及前台与后台输入的区别。
- 屏幕坐标和客户区坐标的换算。
- Windows 消息队列、输入消息与目标窗口。
- C# 通过平台调用（P/Invoke）访问 Win32 API。
- DLL、ABI、32/64 位和导出函数的基本概念。
- 安装前的备份、哈希校验、清单、恢复与冒烟测试（smoke test）。

当前兼容方案会在界面配置中保留 `SendMessageWithWindowPos` 这个槽位，但补丁在运行时把它映射为 AnchoredTouch 数值 `1024`。核心验证不是普通鼠标消息发送，而是通过 MaaFramework ControlUnit 的 `touch_down` / `touch_up` 产生 `WM_POINTER*` 行为，同时确保系统光标、目标窗口位置和前台窗口不发生变化。

### 7.2 对照代码的阅读顺序

| 顺序 | 文件 | 作用 |
|---:|---|---|
| 1 | [maa-pc-anchored-touch.md](../modules/decision-core/blackflow-research/docs/maa-pc-anchored-touch.md) | 目标、边界、风险和整体流程 |
| 2 | [Start-MaaAnchoredTouch.ps1](../modules/decision-core/blackflow-research/tools/Start-MaaAnchoredTouch.ps1) | 启动前检查与入口 |
| 3 | [maa_anchored_touch.ps1](../modules/decision-core/blackflow-research/tools/maa_anchored_touch.ps1) | 安装、检查与恢复 |
| 4 | [Test-MaaAnchoredTouch.ps1](../modules/decision-core/blackflow-research/tools/Test-MaaAnchoredTouch.ps1) | 验证 `WM_POINTER*` 和前台/光标/窗口位置不变量 |
| 5 | [AnchoredTouchSmoke.cs](../modules/decision-core/blackflow-research/tools/anchored_touch_smoke/AnchoredTouchSmoke.cs) | 直接调用 `touch_down` / `touch_up` 的最小样例 |
| 6 | [MaaAnchoredTorchLauncher.cs](../modules/decision-core/blackflow-research/tools/MaaAnchoredTorchLauncher.cs) | 启动器校验与历史命名 |
| 7 | [MaaAnchoredTouchInstaller.cs](../modules/decision-core/blackflow-research/tools/MaaAnchoredTouchInstaller.cs) | 图形安装、备份与恢复 |

### 7.3 最有效的练习

1. 只运行无害的 smoke test，不先操作真实战斗。
2. 画出“PowerShell → 启动器 → ControlUnit → `touch_down/up` → `WM_POINTER*`”调用链。
3. 核对报告中的光标、窗口矩形和前台窗口是否保持不变。
4. 阅读安装器中的哈希、备份清单和回滚路径。

### 7.4 必修资料

- [MaaFramework 官方：控制方法](https://maafw.com/docs/2.4-ControlMethods/)
- [Microsoft Learn：.NET P/Invoke](https://learn.microsoft.com/en-us/dotnet/standard/native-interop/pinvoke)
- [Microsoft Learn：Win32 窗口消息](https://learn.microsoft.com/en-us/windows/win32/learnwin32/window-messages)

<details>
<summary>可选资料与中文视频</summary>

- [MaaFramework 官方：集成接口概览](https://github.com/MaaXYZ/MaaFramework/blob/main/docs/en_us/2.2-IntegratedInterfaceOverview.md)
- [C# Hook 与 EasyHook（B 站）](https://www.bilibili.com/video/BV1kg411L7pJ/)：只看第 2～5 节的环境、Win32 API、P/Invoke 和消息观察；后续 Hook/逆向内容不属于本项目必修。

</details>

### 7.5 达标检查

- [ ] 能解释配置槽位为什么显示 `SendMessageWithWindowPos`，运行时却是 AnchoredTouch `1024`。
- [ ] 能解释 C# 为什么可以调用原生 Windows API 或 DLL。
- [ ] 能说明 smoke test 验证的输入送达与三个“不改变”不变量。
- [ ] 能说明哈希、备份和恢复分别防什么风险。

## 8. 网站搭建

当前网站的主线是 TypeScript、React、Next.js 接口、Tailwind CSS，以及面向 Cloudflare 的 vinext/Vite 构建。第一遍不需要把所有依赖都学一遍。

| 项目 | 内容 |
|---|---|
| 目标 | 能找到页面、状态、组件和样式并完成小修改 |
| 预计时间 | 4～6 小时 |
| 首个入口 | [app/page.tsx](../app/page.tsx) |
| 配置入口 | [package.json](../package.json) |

### 8.1 当前代码现状

- [app/page.tsx](../app/page.tsx) 是带 `"use client"` 的主要页面，核心阅读点是 JSX、`useState` 和 `useMemo`。
- [app/layout.tsx](../app/layout.tsx) 定义全局布局与页面 metadata。
- [app/globals.css](../app/globals.css) 和组件中的 Tailwind 类共同控制视觉样式。
- [db/schema.ts](../db/schema.ts) 当前是空 schema；现在不用先学 Drizzle ORM。
- [worker/index.ts](../worker/index.ts) 是 Cloudflare/vinext 入口，放到最后阅读。
- [package.json](../package.json) 要求 Node.js `>=22.13.0`；现有脚本使用 Bash/POSIX 环境写法。

### 8.2 必须懂的 20%

- HTML 结构与 CSS 的盒模型、Flex/Grid。
- TypeScript 的对象、数组、函数、联合类型和异步。
- React 的 JSX、组件、props、state、事件、条件渲染和列表渲染。
- `useState` 保存交互状态，`useMemo` 缓存派生结果。
- Tailwind 类名怎样控制间距、颜色、布局和响应式样式。
- Next.js 中 `app`、`page`、`layout` 的基本约定。

### 8.3 对照代码的阅读顺序

1. [package.json](../package.json)：只看 `scripts`、Node 版本和核心依赖。
2. [app/layout.tsx](../app/layout.tsx)：看页面外壳。
3. [app/page.tsx](../app/page.tsx)：从状态声明追到 JSX 使用位置。
4. [app/globals.css](../app/globals.css)：看全局主题和基础样式。
5. `components/`：仅在 `page.tsx` 实际导入时进入对应组件。
6. [worker/index.ts](../worker/index.ts)：最后理解部署适配。

### 8.4 本地运行

现有 `npm` 脚本含 Bash 和 POSIX 环境变量写法。最省事的方式是在 **Git Bash 或 WSL** 中运行：

```bash
npm install
npm run dev
npm test
```

若只想在 PowerShell 启动开发服务器，可以使用：

```powershell
npm install
$env:WRANGLER_LOG_PATH = ".wrangler/wrangler.log"
npx vite
```

`npm test` 会先调用 Bash 构建脚本，因此仍建议在 Git Bash 或 WSL 中执行。

### 8.5 最有效的练习

1. 修改一段文字和一个 Tailwind 颜色类。
2. 找到一个 `useState`，追踪它如何改变页面。
3. 给现有组件增加一个 prop。
4. 新增一个静态页面并添加入口。

### 8.6 必修资料

- [React 官方 Learn](https://react.dev/learn)：只看 Quick Start、State、Responding to Events。
- [TypeScript 官方 Handbook](https://www.typescriptlang.org/docs/handbook/)：只看 Everyday Types、Functions、Object Types。
- [Next.js 官方 Learn](https://nextjs.org/learn)：只看布局、页面与导航部分。

<details>
<summary>可选样式资料与中文视频</summary>

- [Tailwind CSS 官方文档](https://tailwindcss.com/docs)
- [React + Next.js 实战入门（B 站）](https://www.bilibili.com/video/BV1NV4y1B73Y/)
- [Next.js 系列教程（B 站）](https://www.bilibili.com/video/BV1mnhJzbECu/)
- [React/Next.js 所需 TypeScript（B 站）](https://www.bilibili.com/video/BV1Fu34zHEsb/)

</details>

### 8.7 达标检查

- [ ] 能从页面找到对应状态、事件和 JSX。
- [ ] 能修改文字、布局或颜色并看到结果。
- [ ] 能解释 `page.tsx`、`layout.tsx` 和 `worker/index.ts` 的职责区别。
- [ ] 知道当前数据库 schema 为空，不把 Drizzle 当成必修前置。

## 9. 六周执行表

按每天约 1～1.5 小时安排。若时间不足，优先完成前四周。

| 周次 | 学习重点 | 必须完成的产出 |
|---:|---|---|
| 第 1 周 | 模拟器最小接口 + MCTS 基本流程 | 画一棵搜索树；标出代码中的 `N/W/Q/P` 对应字段 |
| 第 2 周 | PUCT、回传、温度和根节点选择 | 给根动作打印统计；比较 8 与 64 次模拟 |
| 第 3 周 | PyTorch、特征编码和张量形状 | 写出全部单样本/批量 shape；验证非法动作 mask |
| 第 4 周 | GNN、policy/value 和训练循环 | 画训练数据流；完成单批过拟合与 checkpoint 往返 |
| 第 5 周 | 模拟器内部 + 地图识别 | 追踪一次状态转移；核对一份 `map_graph.json` |
| 第 6 周 | AnchoredTouch + 网站 | 画输入调用链；完成一个网页文字与状态小修改 |

## 10. 固定的代码阅读方法

所有模块统一使用这套方法：

1. **先读测试**：确定输入、输出和边界条件。
2. **只追一条最小路径**：从一个入口追到一个结果。
3. **记录数据**：张量写 shape，模拟器写动作前后字段。
4. **运行验证**：用断点、日志或临时输出检查理解。
5. **只改一个变量**：参数、小规则或小组件均可。
6. **重新运行测试**：把“我觉得懂了”变成可验证结果。
7. **最后读旁支**：兼容层、错误恢复和部署工具放到主流程之后。

建议为每个文件只记四栏：

| 文件 | 输入 | 输出 | 它解决的唯一核心问题 |
|---|---|---|---|
| `mcts.py` | 状态、模拟器、评估器 | 动作与搜索统计 | 怎样分配有限搜索预算 |
| `network.py` | 图、全局特征、选项和 mask | policy logits、value | 怎样评价动作与状态 |
| `simulator.py` | 状态、动作 | 新状态、奖励、终局 | 游戏规则怎样推进一步 |
| `vision.py` | 截图/ROI | 检测结果 | 画面上发生了什么 |

## 11. 术语速查与暂缓内容

### 11.1 术语速查

| 术语 | 在本项目中的含义 |
|---|---|
| MDP | 用状态、动作、转移、奖励和终止描述决策问题 |
| MCTS | 通过反复模拟建立局部搜索树并选动作 |
| PUCT | 同时利用平均价值和神经网络先验的选择规则 |
| GNN | 沿地图边聚合节点信息的神经网络 |
| policy | 对候选动作的偏好或概率分布 |
| value | 对当前状态长期回报的估计 |
| mask | 标记真实节点或合法动作，排除填充和非法项 |
| belief state | 只使用当前可观测信息构造的规划状态 |
| ROI | 从整张截图裁出的感兴趣区域 |
| checkpoint | 模型、优化器、回放数据和环境信息的保存点 |
| seed | 控制伪随机过程、帮助复现实验的随机种子 |
| smoke test | 验证最小调用链能否工作的冒烟测试 |

### 11.2 当前暂时不用学

- PPO、DQN 等强化学习算法全谱系。
- Transformer、扩散模型和分布式训练。
- 从零训练目标检测、OCR 或 CorridorNet。
- CUDA 内核和神经网络严格数学证明。
- Windows 驱动、内核注入和逆向工程。
- React/Next.js 框架源码、复杂状态管理和完整云运维。
- Drizzle ORM；当前 `db/schema.ts` 为空，等网站真正接数据库再学。

当前最佳节奏始终是：**运行测试 → 追一条执行链 → 做一个小修改 → 再补对应理论**。

---

官方资料链接最后核验日期：2026-09-02。B 站会对自动访问返回限制，视频均只作为可选中文补充；若链接失效，按本文给出的标题、BV 号或核心关键词搜索同主题课程即可。

