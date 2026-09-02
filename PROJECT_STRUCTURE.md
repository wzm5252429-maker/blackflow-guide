# blackflow-guide 功能目录

本仓库同时包含网页、地图识别、第一场战斗自动化，以及路线模拟与决策研究。为避免把性质不同的程序混成一个“全自动成品”，新增内容按功能独立存放，原有远端目录保持不动。

## 新增目录

| GitHub 路径 | 来源 | 功能 | 当前边界 |
|---|---|---|---|
| `modules/map-recognition/BFMapRecognizer/` | `<MAP_RECOGNIZER_SOURCE>` | MAA FramePool 截图、OCR、地图节点/道路识别与图导出 | 本次只收录修订后的说明文档；识别脚本与 MAA 运行包仍在本机目录 |
| `modules/battle-automation/blackflow-first-battle/` | `<FIRST_BATTLE_SOURCE>` | 视觉模板、Windows 输入控制、首战策略执行、结果识别与 Q-learning | 框架可运行；真实作战前仍须录入模板、坐标和合法战术 |
| `modules/decision-core/blackflow-research/` | 本次 Codex 工作区 | 规则证据、约束地图模拟、部分可观测状态、PUCT-MCTS、GNN policy/value、训练评估与 MAA AnchoredTouch 工具 | 合成训练管线不等于真实服规则或真实通关能力 |
| `docs/FILE_LEARNING_GUIDE.md` | 三个模块的联合盘点 | 逐文件学习知识地图 | 核心文件逐个解释；重复缓存、分页渲染和同构数据按模式覆盖 |

## 三个核心模块的关系

```text
游戏地图画面
  -> BFMapRecognizer：截图 / OCR / 节点与道路识别
  -> map_graph.json
  -> 尚待实现的状态适配器
  -> decision-core：模拟器 / MCTS / GNN / 路线建议
  -> 尚待实现的动作编排接口
  -> first-battle：视觉验证 / 点击与拖动 / 首战闭环 / Q-learning
```

地图识别器、路线决策核心和首战执行器都是真实存在的模块，但目前是三个可以分别运行的工程；仓库不把它们描述成已经完全接通的端到端系统。

## 决策核心包含的功能

- `blackflow_rl/mapgen.py` 与 `map_templates.py`：固定拓扑、约束填充和合成地图生成；
- `blackflow_rl/simulator.py`：资源、节点、事件、跨层与部分可观测状态转移；
- `blackflow_rl/mcts.py`：单玩家 PUCT-MCTS 搜索；
- `blackflow_rl/features.py` 与 `network.py`：图特征和 PyTorch 图 policy/value 网络；
- `blackflow_rl/training.py`：MCTS 引导 rollout、回放训练、评估和 checkpoint；
- `data/`、`source_data/` 与 `tests/`：规则、来源证据、上游数据、黄金样本和自动化测试；
- `tools/` 与 `artifacts/`：MAA AnchoredTouch 的 Windows 适配、安装、冒烟测试和可复现实验产物。

## 上传口径

- 当前工作区的项目文件全部纳入，包括 `.gitignore` 忽略的 `artifacts/`、`__pycache__/` 和 `*.pyc`；
- 第一场战斗工程的 39 个文件全部纳入；
- 地图识别工程按本次要求只上传修订后的 `README_zh-CN.md`，不复制 9,000 余个 MAA 第三方资源与运行日志；
- `.git/` 是版本库内部对象与索引，不属于项目内容，不嵌套上传；
- 目标仓库为公开仓库，本机专属配置中的个人标识、绝对路径和窗口参数使用明确占位符替换，文件结构与字段保留。

## 推荐阅读顺序

1. `docs/FILE_LEARNING_GUIDE.md`：先看完整知识地图；
2. 地图识别器 README：理解真实画面如何变成图；
3. 决策核心的 `domain.py`、`mapgen.py`、`simulator.py`、`mcts.py`：理解图如何变成决策；
4. `features.py`、`network.py`、`training.py`：理解神经网络与训练；
5. 首战工程的 `vision.py`、`controller.py`、`engine.py`、`learner.py`：理解真实画面识别、执行与强化更新。
