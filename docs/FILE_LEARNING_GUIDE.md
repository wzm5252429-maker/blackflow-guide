# 文件学习知识地图

本文按功能梳理三个交付范围中的文件，并说明读懂它们需要学习的知识。

## 盘点范围

- 地图识别器：`BFMapRecognizer/README_zh-CN.md`，1 个文件；
- 第一场战斗自动化：`blackflow_first_battle/`，43 个文件；
- 决策核心原工作区：175 个文件，不包含 `.git/` 内部对象库；
- 本次新增：`PROJECT_STRUCTURE.md` 与本指南，共 2 个文件；
- 最终合计覆盖 221 个文件。

核心源码、配置、测试和文档逐文件列出；同类缓存、分块数据、逐页渲染图和二进制构建产物按明确路径模式成组列出。`__pycache__`、`.pyc`、安装结果、备份 DLL、QA 页面图等虽然通常应由 `.gitignore` 排除，本次仍纳入上传范围。

## 一、整体功能关系

```text
地图截图与识别
BFMapRecognizer
    │ 输出节点、道路和有向图
    ▼
路线决策核心
blackflow_rl
    ├─ 地图模板与约束生成
    ├─ 状态转移模拟器
    ├─ PUCT-MCTS
    └─ GNN policy/value 网络
    │ 输出路线或事件动作建议
    ▼
真实执行层
blackflow_first_battle / MAA AnchoredTouch
    ├─ 截图与模板识别
    ├─ 点击、拖动、技能动作
    ├─ 胜负检测与日志
    └─ 战术级 Q-learning
```

证据快照、客户端数据和规则审计共同约束决策核心，防止程序把未知游戏机制伪装成已验证规则。

## 二、知识编号

| 编号 | 需要学习的知识 |
|---|---|
| K01 | Python 基础、模块、类型标注、异常处理、`pathlib`、命令行参数 |
| K02 | JSON、CSV、JSONL、数据模式、序列化、版本兼容 |
| K03 | 游戏领域建模、状态机、不可变状态、纯状态转移 |
| K04 | 图论、邻接表、BFS、最短路、连通性、图拓扑 |
| K05 | 约束满足问题、回溯搜索、可行性剪枝、随机种子与可复现性 |
| K06 | MDP、奖励、Q-learning、MCTS、PUCT、探索与利用 |
| K07 | NumPy、张量、特征工程、批处理、合法动作掩码 |
| K08 | PyTorch、图神经网络、policy/value head、反向传播、checkpoint |
| K09 | 单元测试、golden fixture、边界测试、性质与不变量测试 |
| K10 | 证据工程、来源追踪、SHA-256、Git blob SHA、真实性边界 |
| K11 | HTML、JavaScript 构建包、网页数据提取、客户端数据逆向阅读 |
| K12 | Word、PDF、`python-docx`、OpenXML、文档生成与渲染验收 |
| K13 | Windows 窗口、Win32 API、DPI、坐标转换、进程与全局热键 |
| K14 | OpenCV、OCR、ONNX、模板匹配、ROI、屏幕截图与图像坐标 |
| K15 | PowerShell、Windows Batch、依赖安装、构建与部署脚本 |
| K16 | C#/.NET、P/Invoke、COM、WinForms、资源嵌入、PE/DLL/EXE |
| K17 | MAA/MaaFramework、FramePool、Win32 ControlUnit、AnchoredTouch |
| K18 | PNG、ICO、DOCX、PDF、EXE、DLL、PYC 等文件格式和构建产物 |
| K19 | Git/GitHub、目录组织、忽略规则、版本与发布管理 |
| K20 | 自动化安全、急停、日志、重试、哈希校验、可恢复安装 |
| K21 | 《明日方舟》集成战略、黑流树海节点、资源、事件和结局机制 |
| K22 | LangGraph、LangChain、OpenAI API、多 Agent 事实核验 |
| K23 | 开源许可证、第三方文件再分发、隐私配置与凭据清理 |

## 三、地图截图与识别

### `BFMapRecognizer/README_zh-CN.md`

用途：地图识别器的中文总说明。它描述窗口查找、MAA `FramePool` 截图、层数和区域 OCR、地图节点识别、`BlackFlow_corridor_net.onnx` 道路判断，以及 `blackflow-directed-graph-v1` 有向图导出格式。

需要学习：K02、K04、K10、K14、K17、K20、K21。

阅读重点：

- 截图、节点检测和道路检测是感知层；
- ONNX `CorridorNet` 用于判断相邻节点之间是否有路；
- MAA 原始无向道路会被转换为双向有向边；
- 识别器可能恢复、聚焦窗口，并在需要时点击一次缩小地图按钮；它不会点击路线节点或执行事件、背包和战斗；
- 导出的节点图仍需适配为决策核心的 `FloorMap`、`MapNode` 和 `GameState`；
- 地图截图本身无法提供生命、行动力、希望、背包等完整玩家状态。

## 四、第一场战斗自动化工程

目录：`blackflow_first_battle/`。它是“用户编写合法战术 → 视觉检测 → 实时执行 → 动作验证 → 判断胜负 → 更新战术价值”的首战闭环。这里的强化学习是对整套战术方案做选择，不是神经网络自动创造未知打法。

### 4.1 使用说明、依赖与许可证

| 文件 | 用途 | 需要学习 |
|---|---|---|
| `README.md` | 工程总说明、模板采集、坐标标定、动作格式、急停和 Q-learning 边界 | K02、K06、K13、K14、K20、K21 |
| `requirements.txt` | NumPy、OpenCV、MSS、PyAutoGUI、pynput 依赖范围 | K01、K14、K19 |
| `LICENSE` | MIT 许可证 | K23 |

### 4.2 Windows 启动脚本

| 文件 | 用途 | 需要学习 |
|---|---|---|
| `01_install.bat` | 安装 Python 依赖，并在配置不存在时复制示例配置 | K15、K19 |
| `02_run_simulation.bat` | 运行不操作游戏的模拟学习 | K01、K06、K15 |
| `03_capture_window.bat` | 截取目标窗口，生成标定底图 | K13、K14、K15 |
| `03b_crop_template.bat` | 输入检测器名称并交互裁剪模板 | K14、K15 |
| `03c_get_point.bat` | 从标定截图选择位置并取得基准坐标 | K13、K14、K15 |
| `04_run_first_battle.bat` | 执行一局真实首战 | K13、K15、K20 |
| `05_train_repeatedly.bat` | 输入局数并连续执行真实对局学习 | K06、K13、K15、K20 |

### 4.3 Python 核心源码

| 文件 | 用途 | 需要学习 |
|---|---|---|
| `blackflow/__init__.py` | Python 包声明和版本号 | K01 |
| `blackflow/calibrate.py` | 截图、OpenCV 交互式 ROI 裁剪、模板缩放和坐标标定 | K01、K02、K13、K14 |
| `blackflow/capture.py` | 使用 MSS 截取客户区并把 BGRA 转为 OpenCV BGR 图像 | K07、K13、K14 |
| `blackflow/cli.py` | 解析真实运行、模拟运行、局数和是否学习等参数 | K01、K20 |
| `blackflow/config.py` | 加载、验证策略 JSON，处理相对路径和配置错误 | K01、K02 |
| `blackflow/controller.py` | PyAutoGUI 点击/拖动、坐标换算、DPI 处理、全局急停监听 | K13、K20 |
| `blackflow/engine.py` | 编排画面轮询、战术选择、动作、重试、结果判断和学习更新 | K03、K06、K13、K14、K20 |
| `blackflow/episode_log.py` | 写入 UTC 时间的 JSONL 事件日志和失败截图 | K02、K14、K20 |
| `blackflow/learner.py` | 持久化 epsilon-greedy Q-learning，记录 Q 值、访问次数和胜率 | K02、K06 |
| `blackflow/simulator.py` | 按战术胜率模拟对局，验证学习器能否偏向较优方案 | K02、K06 |
| `blackflow/vision.py` | 模板匹配、像素范围、ROI、检测条件组合和点位解析 | K07、K14 |
| `blackflow/window.py` | 枚举窗口、获取客户区物理像素坐标、处理 DPI 并聚焦窗口 | K13、K16 |

### 4.4 策略与运行数据

| 文件 | 用途 | 需要学习 |
|---|---|---|
| `configs/strategy_first_battle.example.json` | 完整策略示例：检测器、点位、战术、奖励和学习参数 | K02、K06、K13、K14、K21 |
| `configs/strategy_first_battle.json` | 当前可编辑运行配置；盘点时内容与示例一致 | K02、K06、K13、K14、K21、K23 |
| `calibration/window.png` | 窗口标定截图，用于裁模板和取坐标 | K14、K18、K23 |
| `learning/README.txt` | 解释真实和模拟 Q 表的位置与删除后果 | K02、K06 |
| `learning/simulation_q_table.json` | 模拟对局产生的 Q 值、访问次数、胜率和 epsilon | K02、K06 |
| `templates/README.txt` | 说明真实运行所需的战斗、胜负和干员卡模板 | K14 |

当前 `templates/` 只有说明文件，没有真实模板 PNG；也没有 `learning/q_table.json`，因此没有真实对局学习记录。真实运行前仍须按用户画面重新标定配置、坐标和模板。

### 4.5 测试

| 文件 | 用途 | 需要学习 |
|---|---|---|
| `tests/test_config_and_vision.py` | 验证示例配置、条件组合和检测点解析 | K02、K09、K14 |
| `tests/test_learner.py` | 验证学习器会偏向更优战术并正确持久化 | K06、K09 |
| `tests/test_simulator.py` | 验证完整模拟闭环会偏向安全方案 | K06、K09 |

### 4.6 Python 缓存

以下 12 个文件是 CPython 3.13 自动生成的字节码缓存：

```text
blackflow/__pycache__/__init__.cpython-313.pyc
blackflow/__pycache__/calibrate.cpython-313.pyc
blackflow/__pycache__/capture.cpython-313.pyc
blackflow/__pycache__/cli.cpython-313.pyc
blackflow/__pycache__/config.cpython-313.pyc
blackflow/__pycache__/learner.cpython-313.pyc
blackflow/__pycache__/simulator.cpython-313.pyc
blackflow/__pycache__/vision.cpython-313.pyc
blackflow/__pycache__/window.cpython-313.pyc
tests/__pycache__/test_config_and_vision.cpython-313.pyc
tests/__pycache__/test_learner.cpython-313.pyc
tests/__pycache__/test_simulator.cpython-313.pyc
```

需要学习：K01、K18、K19。它们不是应当直接维护的源码，可由对应 `.py` 文件重新生成。

## 五、路线决策、MCTS 与神经网络工作区

### 5.1 根目录文件

| 文件 | 用途 | 需要学习 |
|---|---|---|
| `.gitignore` | 忽略虚拟环境、缓存、训练 checkpoint、输出和部署产物 | K19 |
| `README.md` | LangGraph 三路事实核验流程，以及路线决策核心入口 | K10、K19、K21、K22 |
| `README_DECISION_CORE.md` | 决策核心功能、命令、真实性边界和真实接入缺口 | K03、K04、K06、K08、K10、K21 |
| `PROJECT_STRUCTURE.md` | 本次 GitHub 功能目录、来源映射、模块关系与上传口径 | K19、K23 |
| `requirements.txt` | 核验 Agent、文档解析、NumPy 和 PyTorch 的完整依赖 | K01、K08、K12、K22 |
| `requirements-core.txt` | 决策核心的最小 NumPy、PyTorch 依赖 | K07、K08、K19 |
| `build_blackflow_spec.py` | 生成规则库 JSON 和 Word 规范，包含表格、样式和页码 | K01、K02、K10、K12、K21 |
| `inspect_rogue6.py` | 从完整客户端表抽取 `rogue_6` 节点、关卡、零件和藏品摘要 | K01、K02、K11、K21 |
| `player_data_template.csv` | 玩家逐局实测数据录入模板 | K02、K10、K21、K23 |
| `黑流树海模拟器规则库.json` | 节点、事件、收益、资源池、未知项和来源的机器可读规范 | K02、K03、K10、K21 |
| `黑流树海节点事件与收益模拟器规范.docx` | 面向人的完整规则规范文档 | K10、K12、K21 |
| `tree.json` | 外部游戏数据 GitHub 仓库的递归 tree API 快照 | K02、K10、K11、K19 |

### 5.2 `blackflow_rl` 领域模型与接口

| 文件 | 用途 | 需要学习 |
|---|---|---|
| `blackflow_rl/__init__.py` | 暴露公共 API，并保持基础模拟器不强制导入 PyTorch | K01 |
| `blackflow_rl/__main__.py` | 支持 `python -m blackflow_rl` 入口 | K01 |
| `blackflow_rl/domain.py` | 定义节点、资源、事件选项、地图、动作、状态和转移对象 | K02、K03、K04、K21 |
| `blackflow_rl/rules.py` | 把规则 JSON 解析成层规则、节点规则和目标函数 | K02、K03、K21 |
| `blackflow_rl/cli.py` | 数据校验、证据审计、抽图、规划、模拟、训练和评估入口 | K01、K02、K06、K08 |
| `blackflow_rl/client_data.py` | 验证完整客户端表的结构、数量和文件哈希 | K02、K10、K11 |

### 5.3 地图模板与约束生成

| 文件 | 用途 | 需要学习 |
|---|---|---|
| `blackflow_rl/map_templates.py` | 加载并严格验证 44 个固定拓扑 | K02、K04、K09、K10 |
| `blackflow_rl/mapgen.py` | 按层、距离、结局和特殊节点门控执行 CSP/回溯填图 | K03、K04、K05、K21 |
| `blackflow_rl/evidence.py` | 固定证据哈希、重算来源摘要并判断真实训练条件 | K02、K10 |
| `blackflow_rl/simulator.py` | 确定性状态转移、合法动作、跨层、事件、追猎和 belief state | K03、K04、K06、K10、K21 |

### 5.4 搜索、特征和模型训练

| 文件 | 用途 | 需要学习 |
|---|---|---|
| `blackflow_rl/agents.py` | 可解释启发式评估器，以及随机/启发式动作基线 | K03、K06、K21 |
| `blackflow_rl/mcts.py` | 与具体游戏解耦的单玩家 PUCT-MCTS | K04、K06 |
| `blackflow_rl/features.py` | 把地图、资源、可见节点和事件选项编码成 NumPy 特征 | K03、K07 |
| `blackflow_rl/network.py` | 有向消息传递 GNN、节点/选项 policy head 和 value head | K04、K07、K08 |
| `blackflow_rl/training.py` | MCTS 引导 rollout、回放池、损失、优化、评估和 checkpoint | K06、K07、K08、K20 |

### 5.5 规则与证据数据

| 文件 | 用途 | 需要学习 |
|---|---|---|
| `data/rules/blackflow_map_templates_v1.json` | I～V 层 43 个拓扑及第 VI 层 1 个固定拓扑 | K02、K04、K10、K21 |
| `data/rules/blackflow_sim_v1.json` | 层参数、节点数量/距离、目标函数、事件池和证据状态 | K02、K03、K05、K10、K21 |
| `data/evidence/map_rule_conflicts_v1.json` | 路标与影语集之间 26 项规则冲突及兼容策略 | K02、K10、K21 |
| `data/evidence/rogue6_client_choice_snapshot_v1.json` | 客户端表的节点、选项和场景紧凑快照 | K02、K10、K11 |
| `data/evidence/rogue6_noncombat_event_catalog_v1.json` | 非作战事件目录、证据等级、未知概率和运行时策略 | K02、K03、K10、K21 |

### 5.6 设计文档

| 文件 | 用途 | 需要学习 |
|---|---|---|
| `docs/FILE_LEARNING_GUIDE.md` | 本文：所有交付文件所需知识与建议学习顺序 | K19 |
| `docs/decision-core.md` | 状态、动作、奖励、MCTS/GNN 架构和真实接入 | K03、K06、K08、K21 |
| `docs/random-map-generation.md` | 固定拓扑、两套距离、节点门控、CSP 和失败策略 | K04、K05、K10、K21 |
| `docs/rule-truth-audit-2026-09-01.md` | 地图和非作战事件真实性审计 | K10、K11、K21 |
| `docs/sim-to-real-roadmap.md` | 从合成训练到真人顾问模式的阶段路线和验收门槛 | K03、K06、K08、K20、K21 |
| `docs/maa-pc-anchored-touch.md` | MAA PC 后台输入回移方案与安全目标 | K13、K16、K17、K20 |

### 5.7 决策核心测试

| 文件 | 用途 | 需要学习 |
|---|---|---|
| `tests/test_evidence.py` | 证据完整性、哈希、动态成本、冲突配置和原始片段 | K09、K10 |
| `tests/test_map_templates.py` | 模板数量、第 VI 层拓扑、图不变量和重复边拒绝 | K04、K09、K10 |
| `tests/test_mapgen.py` | evidence 拒绝策略、种子复现、结局门控、CSP 和不变量 | K04、K05、K09、K10 |
| `tests/test_mcts.py` | 单玩家 MCTS 回传不进行双人博弈符号翻转 | K06、K09 |
| `tests/test_network_and_training.py` | 编码/掩码形状、训练批次和 checkpoint 往返 | K07、K08、K09 |
| `tests/test_rules_and_data.py` | 行动力规则、客户端表摘要和损坏片段拒绝 | K02、K09、K10 |
| `tests/test_simulator.py` | 严格未知事件、外部观测、belief、门节点、确定性和终止性 | K03、K06、K09、K10 |
| `tests/golden/lubiao_floor6_v1.json` | 第 VI 层来源构建包的固定片段和坐标变换 | K02、K04、K09、K10 |
| `tests/golden/lubiao_map_rules_v1.json` | 节点数量、距离和索引的固定来源夹具 | K02、K05、K09、K10 |

以下 7 个测试缓存由对应测试源码自动生成：

```text
tests/__pycache__/test_evidence.cpython-313.pyc
tests/__pycache__/test_map_templates.cpython-313.pyc
tests/__pycache__/test_mapgen.cpython-313.pyc
tests/__pycache__/test_mcts.cpython-313.pyc
tests/__pycache__/test_network_and_training.cpython-313.pyc
tests/__pycache__/test_rules_and_data.cpython-313.pyc
tests/__pycache__/test_simulator.cpython-313.pyc
```

需要学习：K01、K18、K19。

### 5.8 原始来源数据

| 文件或文件组 | 用途 | 需要学习 |
|---|---|---|
| `source_data/roguelike_topic_table_full.json` | 完整客户端肉鸽主题表；证据审计主要上游数据 | K02、K10、K11、K21 |
| `source_data/part_00`～`part_11` | 完整 JSON 的 12 个连续分块 | K02、K10、K11 |
| `source_data/rem_00`、`rem_02`、`rem_04`、`rem_06`、`rem_07`、`rem_09` | 下载或区间恢复中间片段，不是独立有效 JSON | K10、K11、K18 |
| `source_data/roguelike_topic_table.json` | 不完整或截断的主题表，不能作为完整 JSON 解析 | K02、K10、K11 |
| `source_data/roguelike_table.json` | 肉鸽常量、物品、关卡、区域、选项、场景和结局表 | K02、K11、K21 |
| `source_data/rogue6_inspection.json` | `inspect_rogue6.py` 生成的 `rogue_6` 摘要 | K02、K10、K21 |
| `source_data/ISEvent.js` | 保存的网页 JavaScript 构建模块 | K10、K11 |
| `source_data/prts_events_raw.html` | 一次 PRTS 请求返回的 403 HTML，属于失败取证记录 | K10、K11 |

`part_00`～`part_11` 顺序拼接后的 SHA-256 与完整表一致；`roguelike_topic_table.json` 则是截断或损坏文件。

### 5.9 证据构建工具

| 文件 | 用途 | 需要学习 |
|---|---|---|
| `tools/build_evidence_snapshots.py` | 从完整客户端表和规则库生成紧凑、可复现证据快照 | K01、K02、K10 |
| `tools/extract_lubiao_rule_fixture.py` | 从固定哈希的路标 JS 构建包解析数量和距离规则 | K01、K02、K10、K11 |
| `tools/normalize_template_catalog.py` | 清除模板目录中损坏的旧说明文本，不修改图数据 | K01、K02、K10 |

对应的 3 个 CPython 3.13 缓存位于 `tools/__pycache__/`，文件名分别对应上述脚本。需要学习：K01、K18、K19。

### 5.10 MAA AnchoredTouch 安装、启动与验证工具

| 文件 | 用途 | 需要学习 |
|---|---|---|
| `tools/maa_anchored_touch.ps1` | 安装、恢复或检查 AnchoredTouch 兼容补丁 | K10、K15、K16、K17、K20 |
| `tools/Build-MaaAnchoredTouchInstaller.ps1` | 验证负载哈希并编译嵌入式安装程序 | K15、K16、K18、K20 |
| `tools/MaaAnchoredTouchInstaller.cs` | 图形安装器：检测、备份、替换、配置、快捷方式和恢复 | K13、K16、K17、K20 |
| `tools/MaaAnchoredTorchLauncher.cs` | 启动前校验版本、DLL 哈希、配置和备份清单 | K10、K13、K16、K17、K20 |
| `tools/RegisterMaaAnchoredTorchShortcut.cs` | 通过 COM ShellLink 创建 Windows 快捷方式 | K13、K16 |
| `tools/Start-MaaAnchoredTouch.ps1` | 校验版本、文件和鼠标方法后启动 MAA | K10、K15、K17、K20 |
| `tools/Test-MaaAnchoredTouch.ps1` | 用内嵌 C# Win32 窗口验证触控和鼠标行为 | K13、K15、K16、K17 |
| `tools/启动 MAA（AnchoredTouch 校验）.cmd` | 面向双击使用的 PowerShell 7 包装器 | K15、K17、K20 |
| `tools/anchored_touch_smoke/AnchoredTouchSmoke.cs` | 两个无害 Win32 窗口上的 ControlUnit 触控冒烟测试 | K13、K16、K17、K20 |
| `tools/anchored_touch_smoke/Run-AnchoredTouchSmoke.ps1` | 编译并运行上述 C# 测试 | K15、K16、K17 |
| `tools/anchored_touch_smoke/README.md` | 冒烟测试目标、依赖和运行方法 | K16、K17、K20 |

### 5.11 启动器构建产物

以下 8 个文件是图标或已编译程序：

```text
tools/launcher-build/launcher-icon-final.png
tools/launcher-build/launcher-icon.png
tools/launcher-build/MAA-full.ico
tools/launcher-build/MAA.ico
tools/launcher-build/MAA(AnchoredTorch).exe
tools/launcher-build/original-maa-icon-final.png
tools/launcher-build/original-maa-icon.png
tools/launcher-build/register-shortcut.exe
```

需要学习：K16、K17、K18、K19。维护时应优先阅读生成它们的 C# 和 PowerShell 源码。

### 5.12 安装夹具、备份和结果记录

以下 11 个文件属于部署测试或真实安装产生的材料：

```text
artifacts/maa-anchored-touch-fixture/MAA.exe
artifacts/maa-anchored-touch-fixture/MAA.dll
artifacts/maa-anchored-touch-fixture/MaaWin32ControlUnit.dll
artifacts/maa-anchored-touch-fixture/config/gui.new.json
artifacts/maa-anchored-touch-fixture/install-result.json
artifacts/maa-anchored-touch-fixture/codex-backups/maa-pc-anchored-touch-v6.16.8/gui.new.json.original
artifacts/maa-anchored-touch-fixture/codex-backups/maa-pc-anchored-touch-v6.16.8/MAA.dll.original
artifacts/maa-anchored-touch-fixture/codex-backups/maa-pc-anchored-touch-v6.16.8/MaaWin32ControlUnit.dll.original
artifacts/maa-anchored-touch-fixture/codex-backups/maa-pc-anchored-touch-v6.16.8/manifest.json
artifacts/maa-anchored-touch-live-install.json
artifacts/maa-anchored-touch-live-install-2.json
```

需要学习：K02、K10、K13、K16、K17、K18、K20、K23。公开上传的 JSON 保留模式和字段，但本机绝对路径、个人标识、窗口位置、进程/句柄等值须脱敏。

### 5.13 Word/PDF 视觉验收产物

以下两组各包含 1 个 PDF 和 22 张逐页 PNG，共 46 个文件：

```text
qa_render_word/黑流树海节点事件与收益模拟器规范.pdf
qa_render_word/page-01.png ～ page-22.png

qa_render_word_v2/黑流树海节点事件与收益模拟器规范.pdf
qa_render_word_v2/page-01.png ～ page-22.png
```

用途：把 DOCX 转为 PDF 和逐页图片，对分页、表格、字号、裁切和版式做视觉检查。需要学习：K12、K18、K19。

### 5.14 工作区根缓存

```text
__pycache__/build_blackflow_spec.cpython-313.pyc
__pycache__/document_validator.cpython-313.pyc
```

需要学习：K01、K18、K19。`document_validator.cpython-313.pyc` 对应的 `document_validator.py` 源码当前不在工作区；README 仍引用该程序，应优先找回源码，而不是长期维护字节码。

## 六、建议学习顺序

1. **基础与配置**：K01、K02、K19；先读两个工程 README、地图识别 README 和 requirements。
2. **决策世界模型**：`domain.py` → `rules.py` → `data/rules/*.json` → `simulator.py`；对应 K03、K21。
3. **地图生成**：图论/BFS → `map_templates.py` → `random-map-generation.md` → CSP/回溯 → `mapgen.py` 与测试；对应 K04、K05、K09。
4. **MCTS**：MDP、奖励、UCB/PUCT → `mcts.py` → `agents.py` → `test_mcts.py`；对应 K06。
5. **神经网络**：NumPy/张量/mask → PyTorch → GNN → `features.py` → `network.py` → `training.py`；对应 K07、K08。
6. **识别与真实执行**：OpenCV/OCR/ONNX → Win32/DPI → MAA FramePool/ControlUnit → `vision.py`、`capture.py`、`window.py`、`controller.py`、`engine.py`；对应 K13、K14、K17、K20。
7. **真实性与发布**：`evidence.py` → `data/evidence/*.json` → 规则审计 → 哈希/golden fixture → 隐私和许可证；对应 K10、K23。

## 七、当前最重要的工程缺口

- 地图识别器能输出节点和道路图，但尚缺到 `FloorMap` / `GameState` 的正式适配器；
- 决策核心具备模拟器、MCTS、GNN 和训练框架，但真实概率、事件执行表和真实局模型仍不完整；
- 第一场战斗工程具备视觉与动作闭环，但真实模板、真实坐标和真实战术仍需用户标定；
- `document_validator.py` 源码缺失，只留下 CPython 缓存；
- 当前没有已训练的决策核心 checkpoint，也没有真实对局 `q_table.json`；
- 缓存、备份二进制、失败下载片段和脱敏配置虽按本次要求上传，仍与核心源码分区说明。
