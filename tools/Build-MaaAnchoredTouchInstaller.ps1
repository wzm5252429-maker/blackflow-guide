[CmdletBinding()]
param(
    [string]$OutputDirectory = (
        Join-Path ([Environment]::GetFolderPath('UserProfile')) 'Downloads\MAA-v5.24.1-win-x64'
    )
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$toolRoot = $PSScriptRoot
$source = Join-Path $toolRoot 'MaaAnchoredTouchInstaller.cs'
$payloadRoot = 'D:\明日方舟'
$icon = Join-Path $toolRoot 'launcher-build\MAA-full.ico'
$launcher = Join-Path $payloadRoot 'MAA(AnchoredTorch).exe'
$patchedMaa = Join-Path $payloadRoot 'MAA.dll'
$control = Join-Path $payloadRoot 'MaaWin32ControlUnit.dll'
$compiler = 'C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe'
$output = Join-Path $OutputDirectory 'MAA-AnchoredTouch-Setup-v6.16.8.exe'

$expected = @{
    $patchedMaa = 'EC1B400B234DC03CD65CC14C3E881A4D7689FF968C36BE421DCD9A312B4A6BCF'
    $control = '6744C36A3E6E18630CC88224F4E7FC9D71A7EEA482F8EC6875CB994CB81BF0E4'
    $launcher = '54D426380300CB10E85ABF6C628386E9172FBA3961CE85F64664EDF55A3F8E92'
}
foreach ($entry in $expected.GetEnumerator())
{
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $entry.Key).Hash
    if ($actual -ne $entry.Value) { throw "Payload hash mismatch: $($entry.Key)" }
}

[IO.Directory]::CreateDirectory($OutputDirectory) | Out-Null
$compilerArgs = @(
    '/nologo', '/target:winexe', '/optimize+', '/platform:anycpu',
    '/reference:System.dll', '/reference:System.Core.dll', '/reference:System.Windows.Forms.dll',
    '/reference:System.Drawing.dll', '/reference:System.Web.Extensions.dll',
    "/win32icon:$icon",
    "/resource:$patchedMaa,Payload.MAA.dll",
    "/resource:$control,Payload.MaaWin32ControlUnit.dll",
    "/resource:$launcher,Payload.Launcher.exe",
    "/out:$output", $source
)
& $compiler $compilerArgs
if ($LASTEXITCODE -ne 0) { throw "csc.exe failed with exit code $LASTEXITCODE" }

$hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $output).Hash
[PSCustomObject]@{ Path = $output; Sha256 = $hash; Size = (Get-Item -LiteralPath $output).Length }
