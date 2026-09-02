[CmdletBinding()]
param(
    [string]$MaaDir = 'D:\明日方舟',

    [switch]$CheckOnly,

    [switch]$Json
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$Expected = [ordered]@{
    ProductVersion = 'v6.16.8+5ee4315f7d9d79f28a3a76dd6a75eb452ac1ff66'
    MaaExeSha256 = '7377234bf379de7cf0d40612ce7eff0f88aa8a0dfb1913ac083e3b662e9bcc1e'
    MaaDllSha256 = 'ec1b400b234dc03cd65cc14c3e881a4d7689ff968c36be421dcd9a312b4a6bcf'
    ControlUnitSha256 = '6744c36a3e6e18630cc88224f4e7fc9d71a7eea482f8ec6875cb994cb81bf0e4'
    MouseMethod = 'SendMessageWithWindowPos'
    RuntimeMouseMethod = 1024
}

function Get-Sha256([string]$Path)
{
    return (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
}

function Get-ConfiguredMouseMethod([string]$ConfigPath)
{
    $config = [IO.File]::ReadAllText($ConfigPath) | ConvertFrom-Json
    $profileName = [string]$config.Current
    if (-not $profileName)
    {
        $profileName = 'Default'
    }

    $profileProperty = $config.Configurations.PSObject.Properties[$profileName]
    if ($null -eq $profileProperty)
    {
        throw "Current MAA profile '$profileName' was not found."
    }
    return [string]$profileProperty.Value.Gui.ConnectSettings.Extras.Win32Extra.MouseMethod
}

function Show-LauncherMessage([string]$Message, [bool]$IsError)
{
    try
    {
        Add-Type -AssemblyName System.Windows.Forms
        $icon = if ($IsError)
        {
            [Windows.Forms.MessageBoxIcon]::Error
        }
        else
        {
            [Windows.Forms.MessageBoxIcon]::Information
        }
        [void][Windows.Forms.MessageBox]::Show(
            $Message,
            'MAA AnchoredTouch 校验启动器',
            [Windows.Forms.MessageBoxButtons]::OK,
            $icon
        )
    }
    catch
    {
        if ($IsError) { Write-Error $Message } else { Write-Host $Message }
    }
}

function Add-LauncherLog([string]$Path, [object]$Record)
{
    try
    {
        $directory = [IO.Path]::GetDirectoryName($Path)
        if ($directory)
        {
            [IO.Directory]::CreateDirectory($directory) | Out-Null
        }
        $line = $Record | ConvertTo-Json -Compress -Depth 5
        [IO.File]::AppendAllText($Path, $line + [Environment]::NewLine, [Text.UTF8Encoding]::new($false))
    }
    catch
    {
        # Logging must never turn a successful integrity check into a launch failure.
    }
}

$maaRoot = [IO.Path]::GetFullPath($MaaDir).TrimEnd(
    [IO.Path]::DirectorySeparatorChar,
    [IO.Path]::AltDirectorySeparatorChar
)
$maaExe = Join-Path $maaRoot 'MAA.exe'
$maaDll = Join-Path $maaRoot 'MAA.dll'
$controlDll = Join-Path $maaRoot 'MaaWin32ControlUnit.dll'
$configPath = Join-Path $maaRoot 'config\gui.new.json'
$backupManifest = Join-Path $maaRoot 'codex-backups\maa-pc-anchored-touch-v6.16.8\manifest.json'
$logPath = Join-Path $maaRoot 'codex-backups\anchored-touch-launcher.log'

$failures = [Collections.Generic.List[string]]::new()
$warnings = [Collections.Generic.List[string]]::new()
$actual = [ordered]@{}

foreach ($required in @($maaExe, $maaDll, $controlDll, $configPath))
{
    if (-not (Test-Path -LiteralPath $required -PathType Leaf))
    {
        $failures.Add("缺少必要文件：$required")
    }
}

if ($failures.Count -eq 0)
{
    try
    {
        $actual.ProductVersion = [string](Get-Item -LiteralPath $maaExe).VersionInfo.ProductVersion
        $actual.MaaExeSha256 = Get-Sha256 $maaExe
        $actual.MaaDllSha256 = Get-Sha256 $maaDll
        $actual.ControlUnitSha256 = Get-Sha256 $controlDll
        $actual.MouseMethod = Get-ConfiguredMouseMethod $configPath

        if ($actual.ProductVersion -ne $Expected.ProductVersion)
        {
            $failures.Add("检测到 MAA 版本变化：当前 $($actual.ProductVersion)，已验证版本 $($Expected.ProductVersion)。")
        }
        if ($actual.MaaExeSha256 -ne $Expected.MaaExeSha256)
        {
            $failures.Add('MAA.exe 已被更新或修改。')
        }
        if ($actual.MaaDllSha256 -ne $Expected.MaaDllSha256)
        {
            $failures.Add('MAA.dll 校验不匹配，AnchoredTouch 映射可能已被更新覆盖。')
        }
        if ($actual.ControlUnitSha256 -ne $Expected.ControlUnitSha256)
        {
            $failures.Add('MaaWin32ControlUnit.dll 校验不匹配，AnchoredTouch 控制组件可能已被更新覆盖。')
        }
        if ($actual.MouseMethod -ne $Expected.MouseMethod)
        {
            $failures.Add("当前鼠标方式为 '$($actual.MouseMethod)'，应为 '$($Expected.MouseMethod)'（运行时映射到 1024）。")
        }
    }
    catch
    {
        $failures.Add("读取 MAA 状态失败：$($_.Exception.Message)")
    }
}

if (-not (Test-Path -LiteralPath $backupManifest -PathType Leaf))
{
    $warnings.Add("未找到补丁备份清单：$backupManifest")
}

$status = if ($failures.Count -gt 0) { 'BLOCKED' } elseif ($warnings.Count -gt 0) { 'PASS_WITH_WARNING' } else { 'PASS' }
$record = [ordered]@{
    timestamp = [DateTimeOffset]::Now.ToString('o')
    status = $status
    maa_directory = $maaRoot
    expected_runtime_mouse_method = $Expected.RuntimeMouseMethod
    actual = $actual
    failures = @($failures)
    warnings = @($warnings)
    check_only = [bool]$CheckOnly
}

Add-LauncherLog $logPath $record

if ($Json)
{
    $record | ConvertTo-Json -Depth 5
}

if ($failures.Count -gt 0)
{
    if (-not $Json)
    {
        $details = ($failures | ForEach-Object { "• $_" }) -join [Environment]::NewLine
        $message = @"
AnchoredTouch 校验未通过，MAA 已被阻止启动。

$details

这通常表示 MAA 已更新或补丁文件被替换。不要把旧版 DLL 强行覆盖到新版 MAA，请重新适配后再启动。

日志：$logPath
"@
        Show-LauncherMessage $message $true
    }
    exit 10
}

if ($CheckOnly)
{
    if (-not $Json)
    {
        $message = "校验通过：当前补丁会把界面的 WindowPos 选项映射为 AnchoredTouch (1024)。"
        if ($warnings.Count -gt 0)
        {
            $message += [Environment]::NewLine + (($warnings | ForEach-Object { "• $_" }) -join [Environment]::NewLine)
        }
        Write-Host $message
    }
    exit 0
}

$running = @(Get-Process -Name 'MAA' -ErrorAction SilentlyContinue).Count -gt 0
if ($running)
{
    if (-not $Json)
    {
        Show-LauncherMessage '校验通过。MAA 已经在运行，因此没有重复启动。' $false
    }
    exit 0
}

try
{
    Start-Process -FilePath $maaExe -WorkingDirectory $maaRoot -Verb RunAs
}
catch
{
    if (-not $Json)
    {
        Show-LauncherMessage "校验通过，但启动 MAA 失败：$($_.Exception.Message)" $true
    }
    exit 20
}

exit 0
