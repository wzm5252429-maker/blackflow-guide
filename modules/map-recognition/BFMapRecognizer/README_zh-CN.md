# 黑流树海地图与节点识别器

这是一个面向《明日方舟》Windows PC 客户端的独立地图识别程序。它直接复用 MAA `v6.17.0-beta.6` 的 Windows x64 核心、黑流树海任务、模板、OCR 与 CorridorNet 模型，不需要安装 Python。

当前工作目录以 `v1.0.2` 发布包为基础，并继续加入了 PC 客户端兼容、回调解析、识别后立即停止和图结构导出修订。因此，本目录中的 `MapRecognizer.ps1` 比原始 `BFMapRecognizer_v1.0.2_Windows.zip` 更新。

## 能做什么

- 自动寻找标题为“明日方舟”或包含 `Arknights` 的游戏窗口；
- 恢复并聚焦游戏窗口，使用 MAA `FramePool` 截取客户区；
- OCR 识别当前层数和区域名称；
- 识别地图节点、节点类型、显隐状态、网格位置和当前节点；
- 使用 `BlackFlow_corridor_net.onnx` 判断相邻节点之间的道路；
- 保留道路置信度、CNN 判定和连通性约束等诊断信息；
- 把 MAA 的无向道路转换成双向有向边，导出 `blackflow-directed-graph-v1`；
- 识别成功后立即停止 MAA 任务，不选择路线节点、不处理背包、不进入战斗。

## 输入安全边界

程序不是完全零输入：为了让整张地图进入可识别状态，当前兼容脚本会连接 MAA 的 PC 输入方式，并在检测到地图尚未缩小时点击一次“缩小地图”按钮。

除此之外，它不会：

- 点击任何路线节点；
- 点击“进入”或确认按钮；
- 处理事件、商店、招募或背包；
- 执行战斗；
- 持续接管鼠标操作。

识别流程到达 `BlackFlowMapSummary` 后会立即停止；运行日志也会记录这一边界。

## 使用方法

1. 将整个压缩包解压到新文件夹，不要在 ZIP 内直接运行。
2. 启动《明日方舟》Windows PC 客户端，进入黑流树海地图页。
3. 尽量让顶部区域名称和完整地图可见；程序可在需要时自动点击缩小地图。
4. 双击 `StartMapRecognizer.cmd`。
5. 等待约 5～30 秒。成功后程序会打开本次输出目录；若 MAA 保存了道路诊断图，还会同时打开该图。

`OpenLatestOutput.cmd` 可以重新打开最近一次输出目录。

## 输出文件

每次运行都会创建 `output/YYYYMMDD_HHMMSS/`。成功时可能包含：

| 文件 | 内容 |
|---|---|
| `map_result.json` | MAA 原始识别快照，包含节点、无向道路、置信度和当前位置 |
| `map_graph.json` | `blackflow-directed-graph-v1` 图结构；每条有效无向道路导出为两条有向边 |
| `map_original.png` | MAA 实际截取的原始客户区画面 |
| `map_normalized.png` | MAA 保存了归一化诊断图时生成 |
| `map_nodes.png` | MAA 保存了节点诊断图时生成 |
| `map_edges.png` | MAA 保存了道路诊断图时生成 |
| `summary.txt` | 层数、区域、节点数、道路数和产物编号摘要 |
| `run_log.txt` | 启动器简明日志 |
| `maa_events.jsonl` | MAA 原始回调事件，供失败诊断和结果复核 |

若识别失败，程序会尽可能保存 `last_capture.png` 及日志。诊断图片由 MAA 是否在该次运行中实际产生决定，并非每次成功都保证全部存在。

## 已验证记录

本目录已有一次成功实机记录 `output/20260829_191344/`：

- 第 1 层；
- 区域“玻利瓦尔肤层”；
- 11 个节点；
- 11 条 MAA 无向道路；
- 22 条导出的有向边。

这证明当前版本已经完成“窗口截图 → 层数 OCR → 节点识别 → 道路识别 → 图结构导出”，但不代表它已经与路线模拟器、MCTS 或自动点击器接通。

## 与路线决策核心的关系

`map_graph.json` 是给路线决策系统使用的中间数据。目前仍需要一个适配器将：

- `battle_normal`、`hide_invisible` 等 MAA 节点类型映射为决策核心的 `NodeType`；
- MAA 数值节点 ID、行列坐标和有向边转换成 `FloorMap` / `MapNode`；
- 当前节点、显隐状态和置信度转换成真实对局 `GameState`；
- 生命、行动力、希望、资源和背包等地图截图中没有的信息补入状态。

在完成这个适配器之前，地图识别器和 MCTS 是两个可分别运行、但尚未形成闭环的模块。

## 系统要求与故障处理

- Windows 10/11 x64；
- 《明日方舟》PC 客户端或兼容的模拟器窗口；
- 管理员权限：PC 客户端可能以提升权限运行，启动器会在需要时请求相同权限；
- 若提示缺少 `MSVCP140.dll` 或 `VCRUNTIME140.dll`，运行 `InstallVCRuntimeIfNeeded.cmd`；
- 若层数已识别但没有地图摘要，请确认完整地图已经缩小并可见；
- 失败反馈至少应包含本次目录中的 `run_log.txt` 与 `maa_events.jsonl`。

## 来源与许可证

MAA 二进制、任务、模板、OCR 模型和 CorridorNet 来自 MaaAssistantArknights，按照其 AGPL-3.0 许可证和相关条款再分发。详见：

- `SOURCE_AND_LICENSE.txt`
- `MAA_LICENSE_AGPL-3.0.txt`

本目录的启动与导出脚本说明以 `SOURCE_AND_LICENSE.txt` 中的许可声明为准。

