# Blackflow Decision Core

《明日方舟》“沉沦者的黑流树海”路线决策研究内核，包含可复现地图模拟、单玩家 PUCT-MCTS、图 policy/value 网络和 planner-guided 训练管线。

它能对“规则已建模或结果已由人工给定”的状态做路线规划，目前不包含稳定的截图识别、客户端点击或自主战斗。严格模式遇到未知真实事件时只能停步并接收结算后的绝对状态，还不能注入真实 UI 候选后让 MCTS 选择该事件分支。仓库没有随代码发布的已训练 checkpoint；现阶段是可运行、可测试的训练框架，不是已经证明能在真实游戏通关的成品策略。

## 已实现

- I～V 层 43 个公开固定拓扑及约束求解器；默认 `evidence` profile 因真实模板/节点抽取权重未公开而拒绝随机生成，只有显式 `synthetic` profile 才按版本化先验抽样；
- 三结局第 VI 层“源流交汇处”的唯一固定拓扑和 4 个来源明确的固定节点；其余 8 格只在显式 `synthetic` profile 中填充；
- 纯状态转移、行动力寻路、合成事件选择、跨层与追猎；严格事件模式遇到未验证场景/结算时返回 `NEEDS_OBSERVATION`，可回填绝对状态并追加下一张真实观测地图，不会把未知效果偷换成零收益；
- 隐藏未揭示节点和未来真实地图的 belief state；
- 单玩家 PUCT-MCTS；
- 带节点/选项 policy head、value head 和合法动作 mask 的 GNN；
- MCTS 引导 rollout、回放训练、固定种子评估和 checkpoint 恢复；
- 证据完整性审计、规则/模板 SHA、checkpoint v2 environment SHA 校验及完整单元测试。

## 快速开始

推荐 Python 3.11～3.13。

```powershell
python -m pip install -r requirements.txt
python -m blackflow_rl audit-evidence
python -m blackflow_rl sample-map --seed 42 --profile synthetic
python -m blackflow_rl plan --seed 42 --simulations 64 --profile synthetic
python -m blackflow_rl simulate --seed 42 --policy mcts --profile synthetic
python -m unittest discover -s tests -v
```

CLI 默认使用 `evidence` profile。由于公开资料没有给出真实随机权重，`sample-map`、`plan`、`simulate` 等需要自行生成地图的命令在默认 profile 下会明确失败；上面的 `synthetic` 仅用于约束兼容的研究扰动，不代表官服 RNG。

若另行下载了完整客户端表，可运行 `python -m blackflow_rl validate-data --path <文件>` 重新推导摘要；发布到本仓库的决策内核只携带固定散列的紧凑证据快照，不重复提交约 18 MB 的上游表。

查看三结局第六层：

```powershell
python -m blackflow_rl sample-map --seed 42 --floor 6 --ending-route third --profile synthetic
```

在程序中生成包含第六层的完整三结局路线：

```python
from blackflow_rl.mapgen import MapGenerator, MapGeneratorConfig
from blackflow_rl.simulator import BlackflowSimulator

generator = MapGenerator(
    config=MapGeneratorConfig(
        enable_third_ending=True,
        allow_synthetic_map_sampling=True,
        allow_synthetic_event_effects=True,
        allow_synthetic_floor6_contents=True,
    ),
)
simulator = BlackflowSimulator(map_generator=generator)
state = simulator.reset(seed=42)
assert [floor.floor for floor in state.maps] == [1, 2, 3, 4, 5, 6]
```

普通/一结局路线只包含 I～V 层。VI 层不是另一个随机**拓扑**池：公开资料只给出一个拓扑，并明确 4 个内容节点；剩余 8 格的类型与分布未知。`synthetic` profile 会让这 8 格随 seed 变化，`evidence` profile 则拒绝生成，不能把未列出的格位解释成固定空地。

当前固定来源也没有独立证明“进入 VI 后全图揭示”。因此默认不会开启这一行为；研究实验若确实需要该假设，还必须额外传入 CLI 的 `--assume-floor6-full-reveal`，并在结果中保留 synthetic 标记。

## 短训练与评估

```powershell
python -m blackflow_rl train `
  --profile synthetic --episodes 5 --simulations 8 `
  --output artifacts/blackflow_policy.pt

python -m blackflow_rl evaluate `
  --profile synthetic `
  --checkpoint artifacts/blackflow_policy.pt `
  --episodes 20 --simulations 32
```

训练器会拒绝 `evidence` 或混合 profile；端到端 rollout 必须显式选择 `synthetic`。这些参数只用于验证管线，合成环境中的 reward 提升不能直接解释为真实通关率。

训练 checkpoint 格式为 v2。除模型、优化器、回放池和 RNG 状态外，还保存 simulation profile 与 environment SHA；后者覆盖规则/模板、生成配置、关键模拟语义代码和事件证据目录。profile 或环境语义不一致时拒绝恢复，避免把旧合成语义静默续训。

## 目录

| 路径 | 作用 |
|---|---|
| `blackflow_rl/mapgen.py` | 模板选择、约束填充和地图验证 |
| `blackflow_rl/simulator.py` | 状态、合法动作、转移和观测模型 |
| `blackflow_rl/mcts.py` | 单玩家 PUCT |
| `blackflow_rl/features.py` | 图、资源、候选动作编码 |
| `blackflow_rl/network.py` | 图 policy/value 网络 |
| `blackflow_rl/training.py` | rollout、训练、评估和 checkpoint |
| `blackflow_rl/evidence.py` | 固定来源快照、冲突清单与训练阻断项审计 |
| `data/rules/` | 版本化规则与 44 个拓扑（43 个 I～V + 1 个 VI） |
| `data/evidence/` | 客户端快照、非作战事件目录和 26 项地图规则冲突 |
| `tests/` | 地图、模拟器、MCTS、网络与训练测试 |
| `docs/` | 算法、证据边界与 sim-to-real 路线图 |

## 证据边界

- I～V 的固定拓扑已固定来源快照；但路标与影语集的节点数量/距离约束仍有 **26 项未解决差异**。当前兼容配置采用路标 2026-08-31，不能称为官方规则或“两站一致”；
- VI 层拓扑与 4 个固定内容节点来自[路标档案馆](https://www.lubiao.wiki/tools/blackstream-route)的构建数据；其余 8 格和全图揭示行为都没有独立证据；
- 模板真实抽取权重、节点先验、事件 scene→choice、事件概率、效果执行和基础作战掉落未公开。38 个事件组中的 102 条语义摘要只有 PRTS B 级证据，不是完整服务端执行表；证据目录将未知概率/效果标为 `NEEDS_OBSERVATION`；
- 严格模式的 `ingest_external_observation()` 能接续结算和下一层，但没有“注入真实候选选项及未知后果，再由 MCTS 选分支”的接口；当前真人辅助只能让人处理未知事件后回填结果，不能声称会给全部真实事件选项建议；
- 完整客户端表由 SHA-256 与 Git blob SHA 固定；本地存在 `source_data/roguelike_topic_table_full.json` 时，`audit-evidence` 会重新推导并校验它。发布版没有复制约 18 MB 上游文件，因此审计会报告 `client_source_present=false`，但仍校验紧凑快照、事件目录、VI 原始片段和 26 项冲突清单，并保持 `verified_training_ready=false`。

## 真实游戏落地

最短可行目标不是先自动战斗，而是“真人作战与点击、系统做战略决策”。下一阶段需要真实局状态导入、失败终态、按玩家校准的战斗结果模型、完整日志回放和风险敏感 value。详见 [`docs/sim-to-real-roadmap.md`](docs/sim-to-real-roadmap.md)。
