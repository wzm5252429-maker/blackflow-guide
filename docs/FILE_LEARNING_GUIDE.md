# Blackflow 核心代码最短学习路线

这份指南的目标不是让你系统学完人工智能、计算机视觉或前端开发，而是用尽量少的前置知识，达到以下程度：

- 能说清楚代码每一层在做什么；
- 能沿着一次运行过程追踪数据；
- 能看懂主要类、函数、张量形状和状态变化；
- 能修改参数或小功能，并用测试确认没有改坏。

## 一、优先级与时间分配

你的优先级是：

1. **MCTS、神经网络**
2. **模拟器**
3. **地图识别与计算机视觉**
4. **AnchoredTouch**
5. **网站搭建**

建议总投入约 **55～70 小时**：

| 模块 | 建议时间 | 学到什么程度就够 |
|---|---:|---|
| 最低限度的 Python、数学与模拟器接口 | 6～8 小时 | 能看懂类型、数组形状、状态与动作 |
| MCTS | 12～16 小时 | 能手算一次选择、扩展、回传，并读懂 `mcts.py` |
| 神经网络与训练 | 14～18 小时 | 能解释输入、policy/value 输出、损失和训练循环 |
| 模拟器深入阅读 | 7～9 小时 | 能追踪一次完整状态转移并添加小规则 |
| 地图识别与计算机视觉 | 8～10 小时 | 能从截图追到识别结果和图结构 |
| AnchoredTouch | 4～5 小时 | 能理解窗口句柄、坐标、消息注入和安全校验 |
| 网站搭建 | 4～6 小时 | 能找到页面、组件、样式和数据入口并改小功能 |

> 优先级不完全等于阅读顺序。MCTS 必须调用模拟器，因此先用 2～4 小时只学模拟器的接口，再集中攻 MCTS；模拟器内部实现可以之后再深入。

## 二、开始前只补这些基础

### Python：掌握到能读代码即可

只需要会：

- 函数、类、模块导入；
- `list`、`dict`、集合、循环和推导式；
- `dataclass`、`Enum`、类型标注；
- NumPy/PyTorch 数组的 `shape`、索引、拼接和广播；
- 用断点、日志或 `print` 查看变量。

不必先学：元类、装饰器原理、异步框架、复杂设计模式。

### 数学：只保留会在代码中出现的部分

只需要会：

- 向量、矩阵、点积；
- 概率分布、期望、均值；
- `softmax` 把分数变成概率；
- 导数和梯度下降的直觉；
- 交叉熵与均方误差分别在惩罚什么。

先看 [PyTorch 官方基础教程](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)，边运行边学。不要先花数周补完高等数学。

## 三、第一优先级：MCTS

### 必须懂的核心知识

先把程序抽象成四件事：

- **状态 `state`**：当前地图、资源、生命、已走节点等信息；
- **动作 `action`**：下一步选择哪个节点或事件选项；
- **状态转移 `transition`**：执行动作后得到新状态和奖励；
- **终局 `terminal`**：本局是否结束。

然后掌握 MCTS 的四步：

1. **选择 Selection**：从根节点按 PUCT 分数向下走；
2. **扩展 Expansion**：遇到未展开状态时创建子节点；
3. **评估 Evaluation**：用规则或神经网络估计该状态价值；
4. **回传 Backup**：把价值写回经过的节点。

代码中最重要的节点统计量：

- `N`：访问次数；
- `W`：累计价值；
- `Q = W / N`：平均价值；
- `P`：神经网络给出的先验概率。

PUCT 可以先按下面这个直觉理解：

```text
选择分数 = 当前看起来有多好 Q
         + 这个动作原本多有希望 P × 对尚未充分探索的奖励
```

单人路线规划和双人棋类有一个关键差异：回传时通常不需要像零和棋局那样每层把价值正负翻转。阅读实现时要专门确认这一点。

### 对照代码的阅读顺序

1. `blackflow_rl/domain.py`：状态、动作和结果的数据结构；
2. `blackflow_rl/simulator.py`：只先看 `legal_action_ids`、状态转移和终局判断；
3. `tests/test_mcts.py`：先看测试希望 MCTS 做成什么；
4. `blackflow_rl/mcts.py`：按“选择 → 扩展 → 评估 → 回传”标注函数；
5. `blackflow_rl/agents.py`：MCTS 如何被包装成可调用的决策器；
6. `blackflow_rl/cli.py`：命令行怎样启动规划。

### 最有效的练习

1. 在纸上画一个只有 3 个动作、2 层深的小树；
2. 手算前 3～5 次搜索后每条边的 `N/Q/P`；
3. 在 `mcts.py` 临时输出根节点各动作统计；
4. 把搜索次数从 8 改到 64，观察最终动作和访问次数；
5. 人为屏蔽一个非法动作，确认它永远不会被选中。

可先运行：

```powershell
py -3.13 -m unittest discover -s tests -p "test_mcts.py" -v
py -3.13 -m blackflow_rl plan --seed 42 --simulations 8 --profile synthetic
```

如果本机 Python 版本不同，把 `py -3.13` 换成 `python`。

### 学习资料

主线资料：

- [Sutton 与 Barto《Reinforcement Learning: An Introduction》开放 PDF](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)：先看第 3～5 章，建立状态、动作、奖励、回报和蒙特卡洛估计概念；
- [Browne 等人的 MCTS 综述](https://cs.gettysburg.edu/~tneller/cs371/mcts-survey.pdf)：重点读基本流程、UCT 和树策略，不必通读全部变体；
- [AlphaZero 论文](https://arxiv.org/abs/1712.01815)：只看 policy/value 网络怎样与 MCTS 配合，不必复现围棋系统。

中文视频补充：

- [浙江大学：蒙特卡洛树搜索 MCTS 入门（B 站）](https://www.bilibili.com/video/BV1kT411G7nE/)；
- [强化学习课程：MDP 与蒙特卡洛方法（B 站）](https://www.bilibili.com/video/BV1dV4y1n7Sn/)。

### 学完的判断标准

不看资料也能回答：为什么既要 `Q` 又要 `P`、访问次数如何影响探索、非法动作在哪里被屏蔽、搜索结束后为什么通常按访问次数选动作。

## 四、第一优先级：神经网络与 GNN

### 必须懂的核心知识

普通神经网络部分只需掌握：

- 张量以及 batch 维、节点维、特征维；
- 线性层、激活函数和前向传播；
- 训练时的前向、计算损失、反向传播、更新参数；
- `model.train()`、`model.eval()`、保存与加载 checkpoint。

本项目还需要理解 GNN 的一个核心动作：

```text
每个地图节点的新表示
= 自己原来的特征
+ 邻居节点信息的聚合
```

policy/value 双头网络的含义：

- **policy head**：给每个候选动作一个分数；
- **value head**：估计当前状态最终能获得多大收益；
- MCTS 用 policy 作为先验 `P`，用 value 作为叶子节点评估。

必须特别关注 **mask**：不同状态的合法动作数量不同，非法动作不能进入 softmax 后获得概率。

训练通常可先理解为：

```text
总损失 = policy 预测与搜索访问分布之间的误差
       + value 预测与最终回报之间的误差
       + 可选的正则项
```

### 对照代码的阅读顺序

1. `tests/test_network_and_training.py`：先弄清输入输出和必须满足的形状；
2. `blackflow_rl/features.py`：状态如何变成节点特征、图连接和 mask；
3. `blackflow_rl/network.py`：GNN 主干、policy head、value head；
4. `blackflow_rl/training.py`：样本、损失、优化器、checkpoint；
5. 回看 `blackflow_rl/mcts.py`：网络输出如何进入搜索；
6. `blackflow_rl/agents.py`：训练好的模型如何用于实际决策。

阅读每个张量时，在纸上写形状，例如：

```text
node_features: [batch, nodes, features]
policy_logits: [batch, actions]
value:         [batch, 1]
legal_mask:    [batch, actions]
```

实际形状以代码和测试为准，不要只靠变量名猜。

### 最有效的练习

1. 运行网络测试，给关键张量打印 `shape`；
2. 手工构造一个只有 3 个节点的小图，检查输出维度；
3. 把一个动作 mask 掉，确认其最终概率接近 0；
4. 用一个样本反复训练到过拟合，确认 loss 能下降；
5. 保存 checkpoint 后重新加载，确认同一输入的输出一致。

```powershell
py -3.13 -m unittest discover -s tests -p "test_network_and_training.py" -v
```

### 学习资料

主线资料：

- [PyTorch 官方 Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)：按 Tensors → Build Model → Autograd → Optimization 顺序；
- [PyTorch Geometric 官方 Colab 教程](https://pytorch-geometric.readthedocs.io/en/latest/get_started/colabs.html)：先完成 Introduction 与 Node Classification；
- [Stanford CS224W 官方课程页](https://web.stanford.edu/class/cs224w/)：只补图表示、消息传递和 GNN 三块。

中文视频补充：

- [同济子豪兄：斯坦福 CS224W 图机器学习中文精讲（B 站）](https://www.bilibili.com/video/BV1pR4y1S7GA/)；
- [李宏毅课程助教：图神经网络 GNN（B 站）](https://www.bilibili.com/video/BV1G54y1971S/)。

### 暂时不用学

Transformer、扩散模型、目标检测网络训练、CUDA 内核、分布式训练、PPO、DQN 都不是读懂当前决策核心的前置条件。

### 学完的判断标准

能指出一个状态怎样变成网络输入、邻居信息在哪里聚合、两个输出头分别服务谁、两个损失分别在纠正什么，以及非法动作怎样被 mask。

## 五、第二优先级：模拟器

### 必须懂的核心知识

- **确定性状态转移**：相同状态、动作和随机种子应产生可复现结果；
- **MDP**：当前状态应包含决定下一步所需的信息；
- **部分可观测性**：真实游戏中有未知信息，因此代码可能保留 belief/unknown 状态；
- **规则与数据分离**：规则代码决定怎样变化，模板和数据描述具体数值；
- **终局与奖励**：结束条件和奖励定义会直接改变 MCTS 偏好；
- **未知机制处理**：不确定的数据应暂停、降级或显式标注，不能假装已知。

### 对照代码的阅读顺序

1. `tests/test_simulator.py`；
2. `blackflow_rl/domain.py`；
3. `blackflow_rl/rules.py`；
4. `blackflow_rl/map_templates.py`；
5. `blackflow_rl/mapgen.py`；
6. `blackflow_rl/simulator.py`；
7. `tests/test_rules_and_data.py`、`tests/test_mapgen.py`、`tests/test_map_templates.py`；
8. `docs/decision-core.md` 与 `docs/random-map-generation.md`。

### 最有效的练习

- 选一个测试，逐字段写出动作前后的状态变化；
- 固定 seed 重跑两次，确认地图与结果一致；
- 增加一个极小的奖励或节点规则，并先补测试；
- 故意制造未知事件，观察系统选择报错、暂停还是降级。

```powershell
py -3.13 -m unittest discover -s tests -p "test_simulator.py" -v
py -3.13 -m unittest discover -s tests -p "test_mapgen.py" -v
```

### 学习资料

- [Sutton 与 Barto 教材第 3～5 章](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)；
- [Gymnasium 官方：创建自定义环境](https://gymnasium.farama.org/main/tutorials/environment_creation/)：重点看 observation、action、reset、step、terminated；
- [Stable Baselines3 / Gym 自定义环境教程（B 站）](https://www.bilibili.com/video/BV1ty4y197JE/)：只看环境接口部分，不必继续学全部算法。

### 学完的判断标准

给你一个状态和动作，你能顺着代码找到合法性检查、状态变化、奖励和终止条件，并解释它为什么可被 MCTS 反复调用。

## 六、第三优先级：地图识别与计算机视觉

### 必须懂的核心知识

- 屏幕坐标、窗口坐标、客户区坐标及 DPI 缩放；
- 截图、ROI 裁剪和颜色空间（RGB/BGR/灰度）；
- 模板匹配、相似度分数、阈值和误检/漏检；
- OCR 与模板匹配各自适合识别什么；
- ONNX 模型的输入预处理、推理和输出后处理；
- 从识别框/节点到 `map_graph.json` 的图结构转换。

### 对照代码的阅读顺序

先看第一场战斗自动化中的视觉链路：

1. `modules/battle-automation/blackflow-first-battle/blackflow_first_battle/window.py`；
2. `modules/battle-automation/blackflow-first-battle/blackflow_first_battle/capture.py`；
3. `modules/battle-automation/blackflow-first-battle/blackflow_first_battle/vision.py`；
4. `modules/battle-automation/blackflow-first-battle/blackflow_first_battle/calibrate.py`；
5. 对应 `tests/` 与模板图片。

再看地图识别器：

1. GitHub 中的 `modules/map-recognition/BFMapRecognizer/README_zh-CN.md`；
2. 本机 `D:\ArknightsAuto\BFMapRecognizer_v1.0.2_Windows\BFMapRecognizer\MapRecognizer.ps1`；
3. 输出的 `map_result.json`、`map_graph.json` 和标注图；
4. 看这些图数据怎样交给 `blackflow_rl/map_templates.py` 或适配层。

地图识别器在 GitHub 中目前主要收录说明文档；完整运行程序仍以本机目录为准。

### 最有效的练习

1. 固定一张截图，裁出一个节点模板；
2. 用 OpenCV `matchTemplate` 得到分数并画出识别框；
3. 上下调整阈值，记录误检与漏检；
4. 打开 `map_graph.json`，在截图上手工核对每个节点和边；
5. 最后再看 ONNX 推理，把模型暂时当作“输入图像、输出候选节点”的黑盒。

### 学习资料

- [OpenCV 官方模板匹配教程](https://docs.opencv.org/4.x/de/da9/tutorial_template_matching.html)；
- [ONNX Runtime 官方 Python 快速开始](https://onnxruntime.ai/docs/get-started/with-python.html)；
- [OpenCV 计算机视觉课程：含模板匹配与 OCR（B 站）](https://www.bilibili.com/video/BV1xgW3zUEnt/)；
- [OpenCV 入门课程：含模板匹配（B 站）](https://www.bilibili.com/video/BV1BenMz9ESE/)。

### 暂时不用学

先不要训练自己的 CNN，也不用先学完整的图像信号处理数学。当前目标是理解截图、预处理、推理、后处理和图结构输出这条管线。

### 学完的判断标准

能从一张窗口截图一路追到识别框、节点类型、边和 JSON 输出，并知道阈值、缩放和坐标系为什么会导致识别失败。

## 七、第四优先级：AnchoredTouch

### 先纠正名称

实际技术名称是 **AnchoredTouch**。仓库中的 `MaaAnchoredTorchLauncher.cs`、`RegisterMaaAnchoredTorchShortcut.cs` 和部分产物保留了早期的 `AnchoredTorch` 拼写，但它们仍属于 AnchoredTouch 启动与校验工具，不是 PyTorch 的 `torch`。

### 必须懂的核心知识

- `HWND` 窗口句柄，以及前台/后台窗口输入的区别；
- 屏幕坐标与客户区坐标换算；
- Windows 消息队列和输入消息；
- C# 通过 P/Invoke 调用 Win32 API；
- DLL、ABI、位数与导出函数的基本概念；
- 安装前备份、哈希校验、清单、恢复和 smoke test。

### 对照代码的阅读顺序

1. `docs/maa-pc-anchored-touch.md`；
2. `tools/Start-MaaAnchoredTouch.ps1`；
3. `tools/maa_anchored_touch.ps1`；
4. `tools/Test-MaaAnchoredTouch.ps1`；
5. `tools/anchored_touch_smoke/AnchoredTouchSmoke.cs`；
6. `tools/MaaAnchoredTorchLauncher.cs`；
7. `tools/MaaAnchoredTouchInstaller.cs`。

### 最有效的练习

- 先只运行无破坏性的 smoke test，确认窗口查找与坐标计算；
- 画出“脚本 → 启动器 → DLL/API → 模拟器窗口”的调用链；
- 给定一个屏幕坐标，手算客户区坐标后和日志对照；
- 阅读安装器中的备份、哈希和回滚路径，不要一开始就在真实战斗中测试。

### 学习资料

- [MaaFramework 官方：控制方法](https://maafw.com/docs/2.4-ControlMethods/)；
- [MaaFramework 官方：集成接口概览](https://github.com/MaaXYZ/MaaFramework/blob/main/docs/en_us/2.2-IntegratedInterfaceOverview.md)；
- [Microsoft Learn：.NET P/Invoke](https://learn.microsoft.com/en-us/dotnet/standard/native-interop/pinvoke)；
- [Microsoft Learn：Win32 窗口消息](https://learn.microsoft.com/en-us/windows/win32/learnwin32/window-messages)；
- [C# Win32、P/Invoke 与 Hook 示例课（B 站）](https://www.bilibili.com/video/BV1kg411L7pJ/)。

### 学完的判断标准

能解释程序如何找到目标窗口、如何换算坐标、C# 为什么能调用原生 DLL，以及为什么安装与启动前要做哈希、备份和校验。

## 八、第五优先级：网站搭建

网站当前主要技术栈可从根目录 `package.json` 确认，包括 TypeScript、React、Next.js、Tailwind CSS、Drizzle ORM，以及面向 Cloudflare 的构建工具。

### 必须懂的核心知识

- HTML 结构与 CSS 布局；
- JavaScript/TypeScript 的对象、数组、函数、异步和类型；
- React 组件、props、state、事件与 JSX；
- Next.js 的 `app` 路由、`page`、`layout` 和服务端/客户端组件；
- Tailwind 类名如何控制样式；
- 表单校验、数据库 schema 和请求/响应的基本数据流；
- `npm` 脚本、开发服务器、构建与部署。

### 对照代码的阅读顺序

1. 根目录 `package.json`：先认依赖和脚本；
2. `app/`：从首页 `page` 和全局 `layout` 开始；
3. `components/`：找到首页直接使用的组件，再逐层进入；
4. 样式文件和 Tailwind 类名；
5. `db/`：只看 schema 与页面需要的数据；
6. `worker/` 与 Cloudflare 配置：最后再看部署入口；
7. 测试与构建脚本。

### 最有效的练习

1. 修改一段页面文字和一个卡片样式；
2. 新增一个静态页面并从导航进入；
3. 给现有组件增加一个 prop；
4. 最后再追踪一次“页面 → 校验 → 数据库/接口 → 页面更新”的完整数据流。

```powershell
npm install
npm run dev
npm test
```

若项目脚本调用 `.sh`，请在 Git Bash 或 WSL 中执行对应构建脚本。

### 学习资料

- [React 官方 Learn](https://react.dev/learn)；
- [Next.js 官方 Learn](https://nextjs.org/learn)；
- [TypeScript 官方 Handbook](https://www.typescriptlang.org/docs/handbook/)；
- [Tailwind CSS 官方文档](https://tailwindcss.com/docs)；
- [React + Next.js 实战入门（B 站）](https://www.bilibili.com/video/BV1NV4y1B73Y/)；
- [Next.js 系列教程（B 站）](https://www.bilibili.com/video/BV1mnhJzbECu/)；
- [React/Next.js 所需 TypeScript（B 站）](https://www.bilibili.com/video/BV1Fu34zHEsb/)。

### 学完的判断标准

看到一个页面时，能找到对应路由、组件、样式和数据来源，并独立完成文字、布局或小字段修改。

## 九、建议的 6 周执行表

按每天约 1～1.5 小时安排：

| 周次 | 主任务 | 本周产出 |
|---|---|---|
| 第 1 周 | 2 天补模拟器接口，随后学 MCTS | 手画搜索树；给根动作打印 `N/Q/P` |
| 第 2 周 | MCTS 深入并结合项目代码 | 能完整讲解一次搜索；修改模拟次数并解释结果 |
| 第 3 周 | PyTorch、GNN、policy/value 网络 | 标注全部关键张量形状；让一个小样本过拟合 |
| 第 4 周 | 训练循环 + 模拟器内部 | 追踪一个训练样本；添加一条小规则和测试 |
| 第 5 周 | 地图识别与计算机视觉 | 对固定截图做模板匹配；核对一份地图图结构 |
| 第 6 周 | AnchoredTouch + 网站 | 画输入调用链；修改并运行一个网页小功能 |

如果时间不够，优先完成前 4 周。那已经足以理解整个决策核心为什么能运行。

## 十、读代码时固定使用的方法

每个模块都按同一套方法，不要从第一个文件第一行硬啃到最后一行：

1. **先读测试**：确认输入、输出和边界条件；
2. **只追一条路径**：选一个最小示例，从入口一路追到结果；
3. **写下数据形状/状态**：张量写 `shape`，模拟器写动作前后字段；
4. **运行并观察**：用断点或日志确认自己的理解；
5. **只改一个变量**：参数、小规则、小组件均可；
6. **重新运行测试**：让理解变成可验证结果；
7. **最后再读旁支**：错误处理、兼容层和部署工具放到主流程之后。

推荐为每个核心文件做一张四栏笔记：

| 文件 | 输入 | 输出 | 它负责的唯一核心问题 |
|---|---|---|---|
| `mcts.py` | 状态、模拟器、网络评估 | 根动作与搜索统计 | 怎样把有限模拟预算分配给候选动作 |
| `network.py` | 图特征与合法动作 mask | policy、value | 怎样评价动作和状态 |
| `simulator.py` | 状态、动作 | 新状态、奖励、终局 | 游戏规则怎样推进一步 |
| `vision.py` | 截图/ROI | 匹配结果 | 画面上发生了什么 |

## 十一、明确不在当前学习范围内的内容

为了尽快看懂代码，暂时跳过：

- 强化学习所有算法的完整谱系；
- 神经网络严格数学证明和从零实现自动微分；
- 自己训练目标检测或 OCR 大模型；
- Windows 驱动、内核注入和逆向工程；
- React/Next.js 框架源码与复杂前端架构；
- 云原生平台的完整运维体系。

这些内容只有在你准备重写相关模块时才需要深入。当前最佳路线是：**测试验证 → 单条执行链 → 小修改 → 再补理论**。

