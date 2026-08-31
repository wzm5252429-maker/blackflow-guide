# Blackflow Decision Core

《明日方舟》“沉沦者的黑流树海”路线决策研究内核，包含可复现地图模拟、单玩家 PUCT-MCTS、图 policy/value 网络和 planner-guided 训练管线。

它负责“状态已经录入以后该怎么选”，目前不包含稳定的截图识别、客户端点击或自主战斗。仓库没有随代码发布的已训练 checkpoint；现阶段是可运行、可测试的训练框架，不是已经证明能在真实游戏通关的成品策略。

## 已实现

- I～V 层 43 个公开固定拓扑的抽样与约束填充；
- 三结局条件启用的第 VI 层“源流交汇处”：固定、全揭示拓扑，4 个来源明确的固定节点，其余 8 格按合成先验随机填充；
- 纯状态转移、行动力寻路、事件选择、跨层与追猎；
- 隐藏未揭示节点和未来真实地图的 belief state；
- 单玩家 PUCT-MCTS；
- 带节点/选项 policy head、value head 和合法动作 mask 的 GNN；
- MCTS 引导 rollout、回放训练、固定种子评估和 checkpoint 恢复；
- 规则与模板 SHA 校验及完整单元测试。

## 快速开始

推荐 Python 3.11～3.13。

```powershell
python -m pip install -r requirements.txt
python -m blackflow_rl sample-map --seed 42
python -m blackflow_rl plan --seed 42 --simulations 64
python -m blackflow_rl simulate --seed 42 --policy mcts
python -m unittest discover -s tests -v
```

查看三结局第六层：

```powershell
python -m blackflow_rl sample-map --seed 42 --floor 6 --ending-route third
```

在程序中生成包含第六层的完整三结局路线：

```python
from blackflow_rl.mapgen import MapGenerator, MapGeneratorConfig
from blackflow_rl.simulator import BlackflowSimulator

generator = MapGenerator(
    config=MapGeneratorConfig(enable_third_ending=True),
)
simulator = BlackflowSimulator(map_generator=generator)
state = simulator.reset(seed=42)
assert [floor.floor for floor in state.maps] == [1, 2, 3, 4, 5, 6]
```

普通/一结局默认只生成 I～V 层。VI 层不是另一个随机**拓扑**池：公开资料只给出一个拓扑，并明确 4 个内容节点；剩余 8 格会随 seed 变化。由于来源没有给出这些格位的真实数量分布，当前规则把它们明确标记为可替换的合成训练先验，而不冒充游戏概率。

## 短训练与评估

```powershell
python -m blackflow_rl train `
  --episodes 5 --simulations 8 `
  --output artifacts/blackflow_policy.pt

python -m blackflow_rl evaluate `
  --checkpoint artifacts/blackflow_policy.pt `
  --episodes 20 --simulations 32
```

默认训练参数只用于验证管线。合成环境中的 reward 提升不能直接解释为真实通关率。

## 目录

| 路径 | 作用 |
|---|---|
| `blackflow_rl/mapgen.py` | 模板选择、约束填充和地图验证 |
| `blackflow_rl/simulator.py` | 状态、合法动作、转移和观测模型 |
| `blackflow_rl/mcts.py` | 单玩家 PUCT |
| `blackflow_rl/features.py` | 图、资源、候选动作编码 |
| `blackflow_rl/network.py` | 图 policy/value 网络 |
| `blackflow_rl/training.py` | rollout、训练、评估和 checkpoint |
| `data/rules/` | 版本化规则与 44 个拓扑（43 个 I～V + 1 个 VI） |
| `tests/` | 地图、模拟器、MCTS、网络与训练测试 |
| `docs/` | 算法、证据边界与 sim-to-real 路线图 |

## 证据边界

- I～V 拓扑与公开节点规则来自版本化社区工具/公开客户端数据；
- VI 层拓扑与 4 个固定内容节点来自[路标档案馆](https://www.lubiao.wiki/tools/blackstream-route) 2026-08-31 构建数据；其余 8 格的当前先验是合成参数；
- 模板真实抽取权重、部分节点先验、事件概率和基础作战掉落未公开，代码中的对应值是可替换的合成假设；
- 完整客户端表体积较大，没有复制到这个子目录。`validate-data` 需要显式传入 `roguelike_topic_table_full.json`；相关黄金测试在数据不存在时会跳过。

## 真实游戏落地

最短可行目标不是先自动战斗，而是“真人作战与点击、系统做战略决策”。下一阶段需要真实局状态导入、失败终态、按玩家校准的战斗结果模型、完整日志回放和风险敏感 value。详见 [`docs/sim-to-real-roadmap.md`](docs/sim-to-real-roadmap.md)。
