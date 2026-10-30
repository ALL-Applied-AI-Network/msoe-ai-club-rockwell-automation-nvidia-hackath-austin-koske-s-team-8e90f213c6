param([string]$Out, [int]$Port = 8767, [switch]$Gui, [switch]$Walkthrough, [ValidateSet('isaac','prerender','frames')][string]$Source = 'isaac', [string]$SourceId = 'local-fillcam', [string]$PrerenderCache)
$ErrorActionPreference = 'Stop'
$toolRoot = if ($env:AI_CLUB_HOME) { $env:AI_CLUB_HOME } else { Join-Path $env:USERPROFILE 'ai-club-tools' }
$pythonPath = Join-Path $toolRoot 'envs/nvidia-pilot-windows/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'The existing nvidia-pilot-windows GPU environment is required for this local simulator.' }
if (-not $Out) { $Out = Join-Path $toolRoot ('results/operator-isaac-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff')) }
$module = if ($Walkthrough) { 'replay.operator.twin' } else { 'replay.operator.station' }
$arguments = @('-u', '-m', $module, '--out', $Out, '--port', "$Port")
if (-not $Walkthrough) { $arguments += @('--source', $Source, '--source-id', $SourceId) }
if ($PrerenderCache -and -not $Walkthrough) { $arguments += @('--prerender-cache', $PrerenderCache) }
if ($Gui) { $arguments += '--gui' }
Push-Location (Split-Path $PSScriptRoot -Parent)
try {
    & $pythonPath @arguments
    if ($LASTEXITCODE -ne 0) { throw 'Isaac operator workspace stopped with an error. Preserve its output/logs.' }
} finally { Pop-Location }
