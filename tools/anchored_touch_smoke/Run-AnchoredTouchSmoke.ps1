param(
    [Parameter(Mandatory = $false)]
    [string]$ControlDll = 'D:\明日方舟\MaaWin32ControlUnit.dll'
)

$ErrorActionPreference = 'Stop'

$resolvedDll = (Resolve-Path -LiteralPath $ControlDll).Path
$source = Join-Path $PSScriptRoot 'AnchoredTouchSmoke.cs'

Add-Type -Path $source
[void][AnchoredTouchSmoke.Program]::Run($resolvedDll)
