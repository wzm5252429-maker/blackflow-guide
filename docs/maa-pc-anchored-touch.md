# MAA PC 后台输入：AnchoredTouch 回移

本机 MAA v6.16.8 捆绑 MaaFramework v5.9.2，只在界面中暴露会抢系统鼠标的
`SendMessageWithCursorPos` 和会移动游戏窗口的 `SendMessageWithWindowPos`。

`tools/maa_anchored_touch.ps1` 将 MaaFramework v5.13.0-beta.5 的官方
`AnchoredTouch` 控制单元回移到该版本。它使用 Windows 合成触控完成点击和滑动：

- 不改变系统鼠标坐标；
- 不激活游戏为前台窗口；
- 不改变游戏窗口坐标；
- 保留 `FramePool` 后台截图和 `SendMessage` 键盘输入。

为了不重编整个 MAA，安装器只做两处定点修改：

1. 用官方 v5.13.0-beta.5 的 `MaaWin32ControlUnit.dll` 替换 v5.9.2 控制单元；
2. 在 `MAA.dll` 的 Win32 连接入口把界面的 WindowPos 值 `128` 映射为
   AnchoredTouch 值 `1024`，因此 MaaCore 和控制单元都不会进入移动窗口路径。

安装器会核对 MAA 版本、所有原文件和下载包的 SHA-256，并在
`D:\明日方舟\codex-backups\maa-pc-anchored-touch-v6.16.8` 保存原始 DLL 与配置备份。
它只支持已验证的 MAA v6.16.8；MAA 更新后会拒绝修改未知二进制。

状态、安装和恢复命令：

```powershell
pwsh -File .\tools\maa_anchored_touch.ps1 -Action Status
pwsh -File .\tools\maa_anchored_touch.ps1 -Action Install -StopRunning -Restart
pwsh -File .\tools\maa_anchored_touch.ps1 -Action Restore -StopRunning -Restart
```

## 校验启动器

开始菜单中的入口名为 `MAA(AnchoredTorch)`，它使用原版 MAA 图标并指向
`D:\明日方舟\MAA(AnchoredTorch).exe`。也可以双击 MAA 安装目录中的
`启动 MAA（AnchoredTouch 校验）.cmd`。启动器会在运行
MAA 前检查当前版本、`MAA.exe`、`MAA.dll`、`MaaWin32ControlUnit.dll`、活动配置
以及备份清单。只有全部关键检查通过时才会以管理员身份启动 MAA。

启动器只校验，不会在检测到更新后盲目覆盖新版文件。被拦截时需先为新版本重新适配。
命令行只检查而不启动：

```powershell
pwsh -NoProfile -File .\tools\Start-MaaAnchoredTouch.ps1 -CheckOnly -Json
```

`AnchoredTouch` 要求 Windows 10 1809 或更高版本，支持点击和滑动。目标点击点被其他
窗口遮挡时，上游实现会短暂调整目标窗口的 Z 序和透明度以完成系统命中测试，随后恢复；
它不会移动目标窗口。无遮挡时不会改动目标窗口状态。触控事件期间鼠标图标可能短暂闪烁，
但系统鼠标坐标和控制权不会改变。

无害实机验证不会定位或点击游戏，只创建两个临时 Win32 窗口，并直接检查触控事件、
鼠标消息、鼠标坐标、窗口矩形和前台窗口：

```powershell
pwsh -NoProfile -File .\tools\anchored_touch_smoke\Run-AnchoredTouchSmoke.ps1
```

上游依据：

- <https://github.com/MaaXYZ/MaaFramework/releases/tag/v5.13.0-beta.5>
- <https://github.com/MaaXYZ/MaaFramework/pull/1456>
