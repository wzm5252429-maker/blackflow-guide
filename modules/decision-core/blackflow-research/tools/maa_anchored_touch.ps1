[CmdletBinding()]
param(
    [ValidateSet('Install', 'Restore', 'Status')]
    [string]$Action = 'Status',

    [string]$MaaDir = 'D:\明日方舟',

    [string]$FrameworkZip = '',

    [string]$CecilPackage = '',

    [switch]$StopRunning,

    [switch]$Restart,

    [string]$LogPath = ''
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$FrameworkVersion = 'v5.13.0-beta.5'
$FrameworkZipUrl = 'https://github.com/MaaXYZ/MaaFramework/releases/download/v5.13.0-beta.5/MAA-win-x86_64-v5.13.0-beta.5.zip'
$FrameworkZipSha256 = '77885cac17dcf9334e2c6ad18df3fc504afa322845c87cfb611c01b9b8169c3b'
$ControlUnitSha256 = '6744c36a3e6e18630cc88224f4e7fc9d71a7eea482f8ec6875cb994cb81bf0e4'
$CecilVersion = '0.11.6'
$CecilPackageUrl = 'https://api.nuget.org/v3-flatcontainer/mono.cecil/0.11.6/mono.cecil.0.11.6.nupkg'
$CecilPackageSha256 = 'd2a23832aaa948ba9a01acc42b5726e34c5f995958f1b30d45c0e7c70b3a72d5'
$ExpectedMaaVersionPrefix = 'v6.16.8+'
$ExpectedOriginalMaaDllSha256 = '94050652d294ff36a756ad8389e8e20992d3c29403c952cb1e1ccaf105ef3f5a'
$ExpectedOriginalControlUnitSha256 = 'a47e5364305aa0d40c3720d6486b8ddd6215b24dce99717de53344a47c9b4805'
$WindowPosInputMethod = [uint64]128
$AnchoredTouchInputMethod = [uint64]1024

function Get-Sha256([string]$Path)
{
    return (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
}

function Assert-Sha256([string]$Path, [string]$Expected)
{
    $actual = Get-Sha256 $Path
    if ($actual -ne $Expected.ToLowerInvariant())
    {
        throw "SHA-256 mismatch for '$Path': expected $Expected, got $actual"
    }
}

function Get-OrDownloadFile([string]$RequestedPath, [string]$DefaultName, [string]$Uri, [string]$ExpectedSha256)
{
    if ($RequestedPath)
    {
        $resolved = [IO.Path]::GetFullPath($RequestedPath)
        if (-not (Test-Path -LiteralPath $resolved -PathType Leaf))
        {
            throw "Payload not found: $resolved"
        }
        Assert-Sha256 $resolved $ExpectedSha256
        return $resolved
    }

    $cacheRoot = Join-Path ([Environment]::GetFolderPath('LocalApplicationData')) 'Codex\MaaAnchoredTouch'
    [IO.Directory]::CreateDirectory($cacheRoot) | Out-Null
    $destination = Join-Path $cacheRoot $DefaultName
    if (-not (Test-Path -LiteralPath $destination -PathType Leaf) -or (Get-Sha256 $destination) -ne $ExpectedSha256)
    {
        Invoke-WebRequest -Uri $Uri -OutFile $destination
    }
    Assert-Sha256 $destination $ExpectedSha256
    return $destination
}

function Expand-ZipEntry([string]$ArchivePath, [string]$EntryName, [string]$DestinationPath)
{
    Add-Type -AssemblyName System.IO.Compression
    $archive = [IO.Compression.ZipFile]::OpenRead($ArchivePath)
    try
    {
        $entry = $archive.Entries | Where-Object FullName -eq $EntryName | Select-Object -First 1
        if ($null -eq $entry)
        {
            throw "Entry '$EntryName' is missing from '$ArchivePath'"
        }
        $input = $entry.Open()
        try
        {
            $output = [IO.File]::Open($DestinationPath, [IO.FileMode]::Create, [IO.FileAccess]::Write, [IO.FileShare]::None)
            try
            {
                $input.CopyTo($output)
            }
            finally
            {
                $output.Dispose()
            }
        }
        finally
        {
            $input.Dispose()
        }
    }
    finally
    {
        $archive.Dispose()
    }
}

function Get-LocalVariable([Mono.Cecil.Cil.MethodBody]$Body, [Mono.Cecil.Cil.Instruction]$Instruction)
{
    if ($Instruction.Operand -is [Mono.Cecil.Cil.VariableDefinition])
    {
        return $Instruction.Operand
    }

    switch ($Instruction.OpCode.Code.ToString())
    {
        'Stloc_0' { return $Body.Variables[0] }
        'Stloc_1' { return $Body.Variables[1] }
        'Stloc_2' { return $Body.Variables[2] }
        'Stloc_3' { return $Body.Variables[3] }
        'Ldloc_0' { return $Body.Variables[0] }
        'Ldloc_1' { return $Body.Variables[1] }
        'Ldloc_2' { return $Body.Variables[2] }
        'Ldloc_3' { return $Body.Variables[3] }
        default { return $null }
    }
}

function Find-MouseMethodStore([Mono.Cecil.MethodDefinition]$Method)
{
    foreach ($instruction in $Method.Body.Instructions)
    {
        if ($instruction.Operand -isnot [Mono.Cecil.MethodReference] -or $instruction.Operand.Name -ne 'get_MouseMethod')
        {
            continue
        }

        $candidate = $instruction.Next
        while ($null -ne $candidate -and $candidate.OpCode.Code.ToString() -notlike 'Stloc*')
        {
            $candidate = $candidate.Next
        }
        if ($null -eq $candidate)
        {
            throw 'Could not find the local variable that stores MouseMethod.'
        }
        return $candidate
    }
    throw 'Could not find get_MouseMethod in AsstAttachWindowConnect.'
}

function Test-AnchoredTouchIlPatch([Mono.Cecil.MethodDefinition]$Method)
{
    $store = Find-MouseMethodStore $Method
    $mouseVariable = Get-LocalVariable $Method.Body $store
    if ($null -eq $mouseVariable)
    {
        return $false
    }

    $cursor = $store.Next
    if ($null -eq $cursor -or (Get-LocalVariable $Method.Body $cursor) -ne $mouseVariable)
    {
        return $false
    }
    $cursor = $cursor.Next
    if ($null -eq $cursor -or $cursor.OpCode.Code.ToString() -ne 'Ldc_I4' -or [int]$cursor.Operand -ne 128)
    {
        return $false
    }
    $cursor = $cursor.Next
    if ($null -eq $cursor -or $cursor.OpCode.Code.ToString() -ne 'Conv_I8')
    {
        return $false
    }
    $cursor = $cursor.Next
    if ($null -eq $cursor -or $cursor.OpCode.Code.ToString() -notlike 'Bne_Un*')
    {
        return $false
    }
    $cursor = $cursor.Next
    if ($null -eq $cursor -or $cursor.OpCode.Code.ToString() -ne 'Ldc_I4' -or [int]$cursor.Operand -ne 1024)
    {
        return $false
    }
    $cursor = $cursor.Next
    if ($null -eq $cursor -or $cursor.OpCode.Code.ToString() -ne 'Conv_I8')
    {
        return $false
    }
    $cursor = $cursor.Next
    return $null -ne $cursor -and (Get-LocalVariable $Method.Body $cursor) -eq $mouseVariable
}

function Open-MaaAssembly([string]$Path)
{
    $assembly = [Mono.Cecil.AssemblyDefinition]::ReadAssembly($Path)
    $type = $assembly.MainModule.Types | Where-Object FullName -eq 'MaaWpfGui.Main.AsstProxy' | Select-Object -First 1
    if ($null -eq $type)
    {
        $assembly.Dispose()
        throw 'MaaWpfGui.Main.AsstProxy was not found. This patch only supports the verified MAA WPF build.'
    }
    $method = $type.Methods | Where-Object Name -eq 'AsstAttachWindowConnect' | Select-Object -First 1
    if ($null -eq $method -or -not $method.HasBody)
    {
        $assembly.Dispose()
        throw 'AsstAttachWindowConnect was not found or has no IL body.'
    }
    return [PSCustomObject]@{ Assembly = $assembly; Method = $method }
}

function Add-AnchoredTouchIlPatch([string]$InputPath, [string]$OutputPath)
{
    $opened = Open-MaaAssembly $InputPath
    try
    {
        if (Test-AnchoredTouchIlPatch $opened.Method)
        {
            $opened.Assembly.Write($OutputPath)
            return
        }

        $store = Find-MouseMethodStore $opened.Method
        $mouseVariable = Get-LocalVariable $opened.Method.Body $store
        if ($null -eq $mouseVariable)
        {
            throw 'Could not resolve the MouseMethod local variable.'
        }
        $continueTarget = $store.Next
        if ($null -eq $continueTarget)
        {
            throw 'Unexpected end of AsstAttachWindowConnect IL body.'
        }

        $processor = $opened.Method.Body.GetILProcessor()
        $instructions = @(
            [Mono.Cecil.Cil.Instruction]::Create([Mono.Cecil.Cil.OpCodes]::Ldloc, $mouseVariable),
            [Mono.Cecil.Cil.Instruction]::Create([Mono.Cecil.Cil.OpCodes]::Ldc_I4, [int]$WindowPosInputMethod),
            [Mono.Cecil.Cil.Instruction]::Create([Mono.Cecil.Cil.OpCodes]::Conv_I8),
            [Mono.Cecil.Cil.Instruction]::Create([Mono.Cecil.Cil.OpCodes]::Bne_Un, $continueTarget),
            [Mono.Cecil.Cil.Instruction]::Create([Mono.Cecil.Cil.OpCodes]::Ldc_I4, [int]$AnchoredTouchInputMethod),
            [Mono.Cecil.Cil.Instruction]::Create([Mono.Cecil.Cil.OpCodes]::Conv_I8),
            [Mono.Cecil.Cil.Instruction]::Create([Mono.Cecil.Cil.OpCodes]::Stloc, $mouseVariable)
        )
        foreach ($instruction in $instructions)
        {
            $processor.InsertBefore($continueTarget, $instruction)
        }
        $opened.Assembly.Write($OutputPath)
    }
    finally
    {
        $opened.Assembly.Dispose()
    }

    $verify = Open-MaaAssembly $OutputPath
    try
    {
        if (-not (Test-AnchoredTouchIlPatch $verify.Method))
        {
            throw 'The generated MAA.dll failed AnchoredTouch IL verification.'
        }
    }
    finally
    {
        $verify.Assembly.Dispose()
    }
}

function Get-ConfiguredMouseMethod([string]$ConfigPath)
{
    $json = [IO.File]::ReadAllText($ConfigPath) | ConvertFrom-Json
    $profileName = [string]$json.Current
    if (-not $profileName)
    {
        $profileName = 'Default'
    }
    $profile = $json.Configurations.PSObject.Properties[$profileName].Value
    if ($null -eq $profile)
    {
        throw "Current MAA profile '$profileName' was not found."
    }
    return [string]$profile.Gui.ConnectSettings.Extras.Win32Extra.MouseMethod
}

function Set-ConfiguredMouseMethod([string]$ConfigPath, [string]$NewValue)
{
    $oldValue = Get-ConfiguredMouseMethod $ConfigPath
    if ($oldValue -eq $NewValue)
    {
        return $oldValue
    }

    $text = [IO.File]::ReadAllText($ConfigPath)
    $pattern = '("MouseMethod"\s*:\s*")' + [regex]::Escape($oldValue) + '(")'
    $matches = [regex]::Matches($text, $pattern)
    if ($matches.Count -ne 1)
    {
        throw "Expected exactly one active MouseMethod value '$oldValue', found $($matches.Count)."
    }
    $replacement = '${1}' + $NewValue + '${2}'
    $updated = [regex]::Replace($text, $pattern, $replacement, 1)
    [IO.File]::WriteAllText($ConfigPath, $updated, [Text.UTF8Encoding]::new($false))
    if ((Get-ConfiguredMouseMethod $ConfigPath) -ne $NewValue)
    {
        throw 'MouseMethod verification failed after updating gui.new.json.'
    }
    return $oldValue
}

function Get-TargetMaaProcesses([string]$MaaExe)
{
    $target = [IO.Path]::GetFullPath($MaaExe)
    $matches = @()
    foreach ($process in Get-Process -Name 'MAA' -ErrorAction SilentlyContinue)
    {
        try
        {
            if ([IO.Path]::GetFullPath($process.Path) -eq $target)
            {
                $matches += $process
            }
        }
        catch
        {
            continue
        }
    }
    return $matches
}

function Stop-TargetMaa([string]$MaaExe)
{
    $processes = @(Get-TargetMaaProcesses $MaaExe)
    if ($processes.Count -eq 0)
    {
        return
    }
    if (-not $StopRunning)
    {
        throw "MAA is running. Re-run with -StopRunning after stopping its current task."
    }
    foreach ($process in $processes)
    {
        Stop-Process -Id $process.Id -Force
    }

    $deadline = (Get-Date).AddSeconds(20)
    do
    {
        if (@(Get-TargetMaaProcesses $MaaExe).Count -eq 0)
        {
            return
        }
        Start-Sleep -Milliseconds 250
    }
    while ((Get-Date) -lt $deadline)

    if (@(Get-TargetMaaProcesses $MaaExe).Count -ne 0)
    {
        throw 'MAA did not stop; no files were changed.'
    }
}

function Load-Cecil([string]$PackagePath, [string]$WorkDir)
{
    $cecilPath = Join-Path $WorkDir 'Mono.Cecil.dll'
    Expand-ZipEntry $PackagePath 'lib/netstandard2.0/Mono.Cecil.dll' $cecilPath
    Add-Type -Path $cecilPath
}

$maaRoot = [IO.Path]::GetFullPath($MaaDir).TrimEnd([IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar)
$maaExe = Join-Path $maaRoot 'MAA.exe'
$maaDll = Join-Path $maaRoot 'MAA.dll'
$controlDll = Join-Path $maaRoot 'MaaWin32ControlUnit.dll'
$configPath = Join-Path $maaRoot 'config\gui.new.json'
foreach ($required in @($maaExe, $maaDll, $controlDll, $configPath))
{
    if (-not (Test-Path -LiteralPath $required -PathType Leaf))
    {
        throw "Required MAA file not found: $required"
    }
}

$productVersion = (Get-Item -LiteralPath $maaExe).VersionInfo.ProductVersion
if (-not $productVersion.StartsWith($ExpectedMaaVersionPrefix, [StringComparison]::OrdinalIgnoreCase))
{
    throw "Unsupported MAA version '$productVersion'. Expected the verified v6.16.8 build."
}

$backupRoot = Join-Path $maaRoot 'codex-backups\maa-pc-anchored-touch-v6.16.8'
$backupMaaDll = Join-Path $backupRoot 'MAA.dll.original'
$backupControlDll = Join-Path $backupRoot 'MaaWin32ControlUnit.dll.original'
$backupConfig = Join-Path $backupRoot 'gui.new.json.original'
$backupManifest = Join-Path $backupRoot 'manifest.json'

$workRoot = Join-Path ([IO.Path]::GetTempPath()) "maa-anchored-touch-$PID-$([Guid]::NewGuid().ToString('N'))"
[IO.Directory]::CreateDirectory($workRoot) | Out-Null

try
{
    $cecilPackagePath = Get-OrDownloadFile $CecilPackage "mono.cecil.$CecilVersion.nupkg" $CecilPackageUrl $CecilPackageSha256
    Load-Cecil $cecilPackagePath $workRoot

    if ($Action -eq 'Status')
    {
        $opened = Open-MaaAssembly $maaDll
        try
        {
            $isPatched = Test-AnchoredTouchIlPatch $opened.Method
        }
        finally
        {
            $opened.Assembly.Dispose()
        }
        $status = [ordered]@{
            action = 'Status'
            maa_version = $productVersion
            # A non-elevated status shell cannot read the executable path of an
            # elevated MAA process, so report the observable process name here.
            # Install/Restore still use the exact resolved path before stopping.
            maa_running = @(Get-Process -Name 'MAA' -ErrorAction SilentlyContinue).Count -gt 0
            gui_il_maps_window_pos_to_anchored_touch = $isPatched
            control_unit_sha256 = Get-Sha256 $controlDll
            control_unit_is_v5_13_beta_5 = (Get-Sha256 $controlDll) -eq $ControlUnitSha256
            configured_mouse_method = Get-ConfiguredMouseMethod $configPath
            backup_path = $backupRoot
            backup_exists = (Test-Path -LiteralPath $backupManifest -PathType Leaf)
        }
    }
    elseif ($Action -eq 'Install')
    {
        Stop-TargetMaa $maaExe

        $currentMaaHash = Get-Sha256 $maaDll
        $currentControlHash = Get-Sha256 $controlDll
        if (-not (Test-Path -LiteralPath $backupManifest -PathType Leaf))
        {
            if ($currentMaaHash -ne $ExpectedOriginalMaaDllSha256)
            {
                throw "MAA.dll does not match the verified original build ($currentMaaHash)."
            }
            if ($currentControlHash -ne $ExpectedOriginalControlUnitSha256)
            {
                throw "MaaWin32ControlUnit.dll does not match the verified v5.9.2 build ($currentControlHash)."
            }
            [IO.Directory]::CreateDirectory($backupRoot) | Out-Null
            [IO.File]::Copy($maaDll, $backupMaaDll, $false)
            [IO.File]::Copy($controlDll, $backupControlDll, $false)
            [IO.File]::Copy($configPath, $backupConfig, $false)
            $manifest = [ordered]@{
                created_at = [DateTimeOffset]::Now.ToString('o')
                maa_version = $productVersion
                maa_dll_sha256 = $currentMaaHash
                control_unit_sha256 = $currentControlHash
                config_sha256 = Get-Sha256 $configPath
                original_mouse_method = Get-ConfiguredMouseMethod $configPath
                framework_payload = $FrameworkVersion
            }
            [IO.File]::WriteAllText($backupManifest, ($manifest | ConvertTo-Json), [Text.UTF8Encoding]::new($false))
        }
        else
        {
            Assert-Sha256 $backupMaaDll $ExpectedOriginalMaaDllSha256
            Assert-Sha256 $backupControlDll $ExpectedOriginalControlUnitSha256
        }

        $frameworkZipPath = Get-OrDownloadFile $FrameworkZip "MAA-win-x86_64-$FrameworkVersion.zip" $FrameworkZipUrl $FrameworkZipSha256
        $newControlDll = Join-Path $workRoot 'MaaWin32ControlUnit.dll'
        Expand-ZipEntry $frameworkZipPath 'bin/MaaWin32ControlUnit.dll' $newControlDll
        Assert-Sha256 $newControlDll $ControlUnitSha256

        $patchedMaaDll = Join-Path $workRoot 'MAA.dll'
        Add-AnchoredTouchIlPatch $backupMaaDll $patchedMaaDll

        Copy-Item -LiteralPath $patchedMaaDll -Destination $maaDll -Force
        Copy-Item -LiteralPath $newControlDll -Destination $controlDll -Force
        Set-ConfiguredMouseMethod $configPath 'SendMessageWithWindowPos' | Out-Null

        $opened = Open-MaaAssembly $maaDll
        try
        {
            if (-not (Test-AnchoredTouchIlPatch $opened.Method))
            {
                throw 'Installed MAA.dll does not contain the AnchoredTouch mapping.'
            }
        }
        finally
        {
            $opened.Assembly.Dispose()
        }
        Assert-Sha256 $controlDll $ControlUnitSha256

        $status = [ordered]@{
            action = 'Install'
            installed = $true
            maa_version = $productVersion
            framework_control_unit = $FrameworkVersion
            mouse_method_ui_slot = 'SendMessageWithWindowPos'
            mouse_method_passed_to_core = $AnchoredTouchInputMethod
            control_unit_sha256 = Get-Sha256 $controlDll
            configured_mouse_method = Get-ConfiguredMouseMethod $configPath
            backup_path = $backupRoot
        }
    }
    else
    {
        Stop-TargetMaa $maaExe
        if (-not (Test-Path -LiteralPath $backupManifest -PathType Leaf))
        {
            throw "Backup manifest not found: $backupManifest"
        }
        $manifest = [IO.File]::ReadAllText($backupManifest) | ConvertFrom-Json
        Assert-Sha256 $backupMaaDll ([string]$manifest.maa_dll_sha256)
        Assert-Sha256 $backupControlDll ([string]$manifest.control_unit_sha256)

        Copy-Item -LiteralPath $backupMaaDll -Destination $maaDll -Force
        Copy-Item -LiteralPath $backupControlDll -Destination $controlDll -Force
        Set-ConfiguredMouseMethod $configPath ([string]$manifest.original_mouse_method) | Out-Null

        $status = [ordered]@{
            action = 'Restore'
            restored = $true
            maa_version = $productVersion
            control_unit_sha256 = Get-Sha256 $controlDll
            configured_mouse_method = Get-ConfiguredMouseMethod $configPath
            backup_path = $backupRoot
        }
    }

    if ($Restart -and $Action -ne 'Status')
    {
        Start-Process -FilePath $maaExe -WorkingDirectory $maaRoot
        $status.restarted = $true
    }

    $result = $status | ConvertTo-Json -Depth 5
    if ($LogPath)
    {
        $resolvedLog = [IO.Path]::GetFullPath($LogPath)
        [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($resolvedLog)) | Out-Null
        [IO.File]::WriteAllText($resolvedLog, $result, [Text.UTF8Encoding]::new($false))
    }
    Write-Output $result
}
catch
{
    $failure = [ordered]@{
        action = $Action
        installed = $false
        error = $_.Exception.Message
        error_type = $_.Exception.GetType().FullName
    } | ConvertTo-Json
    if ($LogPath)
    {
        $resolvedLog = [IO.Path]::GetFullPath($LogPath)
        [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($resolvedLog)) | Out-Null
        [IO.File]::WriteAllText($resolvedLog, $failure, [Text.UTF8Encoding]::new($false))
    }
    throw
}
finally
{
    foreach ($file in Get-ChildItem -LiteralPath $workRoot -File -ErrorAction SilentlyContinue)
    {
        # Add-Type keeps Mono.Cecil.dll mapped until this PowerShell process exits.
        # Cleanup is best-effort so a successful install/status run never turns into
        # a failure merely because Windows still has that exact temporary DLL open.
        try
        {
            Remove-Item -LiteralPath $file.FullName -Force -ErrorAction Stop
        }
        catch
        {
            continue
        }
    }
    if ((Test-Path -LiteralPath $workRoot -PathType Container) -and (Get-ChildItem -LiteralPath $workRoot -Force | Measure-Object).Count -eq 0)
    {
        Remove-Item -LiteralPath $workRoot -Force
    }
}
